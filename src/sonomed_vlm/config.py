"""Typed configuration loading with environment interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise ValueError(f"Required environment variable {name!r} is not set")
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    name: str = "google/medgemma-1.5-4b-it"
    revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = False
    require_hf_token: bool = True
    attn_implementation: str | None = None
    image_max_pixels: int | None = Field(default=None, ge=65536)


class DataConfig(StrictModel):
    root: Path
    train_manifest: Path = Path("data/manifests/train_1pct.jsonl")
    val_manifest: Path = Path("data/manifests/val.jsonl")
    num_workers: int = Field(default=4, ge=0)
    task_filters: list[str] = Field(default_factory=list)
    source_filters: list[str] = Field(default_factory=list)
    max_length: int | None = Field(default=None, ge=1)


class LoraConfigModel(StrictModel):
    enabled: bool = True
    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    train_projector: bool = False
    train_vision: bool = False
    qlora: bool = False
    qlora_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


class TrainingConfig(StrictModel):
    epochs: float = Field(default=1.0, gt=0)
    max_steps: int = -1
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    scheduler: str = "cosine"
    per_device_batch_size: int = Field(default=1, ge=1)
    eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    max_grad_norm: float = Field(default=1.0, gt=0)
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    seed: int = 42
    resume_from_checkpoint: str | None = None
    require_cuda: bool = True
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])


class PerformanceConfig(StrictModel):
    """Optional benchmark instrumentation; disabled for normal training."""

    timing_warmup_steps: int = Field(default=0, ge=0)
    profiler_enabled: bool = False
    profiler_wait_steps: int = Field(default=5, ge=0)
    profiler_warmup_steps: int = Field(default=5, ge=0)
    profiler_active_steps: int = Field(default=10, ge=1)
    profiler_export_trace: bool = True
    torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = None


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=256, ge=1)
    do_sample: bool = False
    temperature: float | None = Field(default=None, gt=0)


class OutputConfig(StrictModel):
    root: Path
    run_name: str | None = None


class ExperimentConfig(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig

    @model_validator(mode="after")
    def validate_precision(self) -> ExperimentConfig:
        if self.training.bf16 and self.model.dtype != "bfloat16":
            raise ValueError("training.bf16=true requires model.dtype=bfloat16")
        if self.lora.qlora and not self.lora.enabled:
            raise ValueError("QLoRA requires lora.enabled=true")
        if self.lora.train_vision:
            raise ValueError(
                "Selective vision adaptation is an explicit later ablation and is not enabled in v0.1"
            )
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load, interpolate, validate, and path-resolve an experiment config."""
    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml_with_base(config_path, seen=set())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    config = ExperimentConfig.model_validate(_expand_env(raw))
    project_root = Path(os.environ.get("SONOMED_PROJECT_ROOT", config_path.parent.parent)).resolve()
    for field_name in ("train_manifest", "val_manifest"):
        value = getattr(config.data, field_name)
        if not value.is_absolute():
            setattr(config.data, field_name, project_root / value)
    if not config.output.root.is_absolute():
        config.output.root = project_root / config.output.root
    return config


def _load_yaml_with_base(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"Cyclic config inheritance detected at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    parent = _load_yaml_with_base(parent_path, seen)
    return _deep_merge(parent, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
