"""Transformers Trainer construction and JSONL research logging."""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

from sonomed_vlm.config import ExperimentConfig


def build_training_arguments(config: ExperimentConfig, run_dir: Path) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": config.training.epochs,
        "max_steps": config.training.max_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_ratio": config.training.warmup_ratio,
        "lr_scheduler_type": config.training.scheduler,
        "per_device_train_batch_size": config.training.per_device_batch_size,
        "per_device_eval_batch_size": config.training.eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "bf16": config.training.bf16,
        "tf32": config.training.tf32,
        "max_grad_norm": config.training.max_grad_norm,
        "save_strategy": "steps",
        "save_steps": config.training.save_steps,
        "eval_steps": config.training.eval_steps,
        "logging_steps": config.training.logging_steps,
        "save_total_limit": config.training.save_total_limit,
        "seed": config.training.seed,
        "data_seed": config.training.seed,
        "report_to": config.training.report_to,
        "logging_dir": str(run_dir / "tensorboard"),
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "load_best_model_at_end": False,
        "torch_compile": config.performance.torch_compile,
        "torch_compile_backend": config.performance.torch_compile_backend,
        "torch_compile_mode": config.performance.torch_compile_mode,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[strategy_name] = "steps"
    return TrainingArguments(**kwargs)


class JsonlMetricsCallback:
    """Callback implemented without importing Transformers at module import time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any
    ) -> None:
        if not state.is_world_process_zero or not logs:
            return
        record = {
            "step": state.global_step,
            "elapsed_seconds": time.monotonic() - self.started,
            **logs,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        return None


class BenchmarkTimingCallback:
    """Measure steady-state throughput after a fixed optimizer-step warmup."""

    def __init__(self, path: str | Path, warmup_steps: int) -> None:
        self.path = Path(path)
        self.warmup_steps = warmup_steps
        self.started: float | None = None

    @staticmethod
    def _synchronize() -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_step_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        if self.started is None and state.global_step >= self.warmup_steps:
            self._synchronize()
            self.started = time.perf_counter()

    def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        if self.started is None:
            return
        self._synchronize()
        elapsed = time.perf_counter() - self.started
        measured_steps = max(0, state.global_step - self.warmup_steps)
        examples = (
            measured_steps
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.world_size
        )
        if state.is_world_process_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "warmup_steps": self.warmup_steps,
                        "measured_steps": measured_steps,
                        "measured_examples": examples,
                        "elapsed_seconds": elapsed,
                        "examples_per_second": examples / elapsed if elapsed else None,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


class TorchProfilerCallback:
    """Short scheduled CPU/CUDA profile for controlled benchmark jobs."""

    def __init__(self, run_dir: str | Path, config: ExperimentConfig) -> None:
        self.run_dir = Path(run_dir)
        self.config = config.performance
        self.profiler: Any | None = None
        self.rank = int(os.environ.get("RANK", "0"))

    def _trace_ready(self, profiler: Any) -> None:
        if not self.config.profiler_export_trace or self.rank != 0:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(self.run_dir / f"trace_rank{self.rank}.json"))

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.config.profiler_wait_steps,
                warmup=self.config.profiler_warmup_steps,
                active=self.config.profiler_active_steps,
                repeat=1,
            ),
            on_trace_ready=self._trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        )
        self.profiler.__enter__()

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        if self.profiler is None:
            return
        self.profiler.__exit__(None, None, None)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        averages = self.profiler.key_averages(group_by_input_shape=True)
        sections = [
            averages.table(sort_by="self_cpu_time_total", row_limit=75),
            averages.table(sort_by="self_cuda_time_total", row_limit=75),
        ]
        (self.run_dir / f"profile_rank{self.rank}.txt").write_text(
            "\n\n".join(sections) + "\n", encoding="utf-8"
        )


def build_metrics_callback(path: str | Path) -> Any:
    from transformers import TrainerCallback

    class _Callback(JsonlMetricsCallback, TrainerCallback):
        pass

    return _Callback(path)


def build_benchmark_timing_callback(path: str | Path, warmup_steps: int) -> Any:
    from transformers import TrainerCallback

    class _Callback(BenchmarkTimingCallback, TrainerCallback):
        pass

    return _Callback(path, warmup_steps)


def build_profiler_callback(run_dir: str | Path, config: ExperimentConfig) -> Any:
    from transformers import TrainerCallback

    class _Callback(TorchProfilerCallback, TrainerCallback):
        pass

    return _Callback(run_dir, config)


def build_trainer(
    *,
    model: Any,
    processor: Any,
    collator: Any,
    train_dataset: Any,
    eval_dataset: Any,
    config: ExperimentConfig,
    run_dir: Path,
) -> Any:
    from transformers import Trainer

    callbacks = [
        build_metrics_callback(run_dir / "train_log.jsonl"),
        build_benchmark_timing_callback(
            run_dir / "benchmark_metrics.json", config.performance.timing_warmup_steps
        ),
    ]
    if config.performance.profiler_enabled:
        callbacks.append(build_profiler_callback(run_dir, config))
    return Trainer(
        model=model,
        args=build_training_arguments(config, run_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        processing_class=processor,
    )
