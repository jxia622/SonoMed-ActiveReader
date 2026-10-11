"""Reuse the existing Qwen processor/model loader with explicit local ultrasound assets."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from sonomed_vlm.active_reader.io import file_hash, versions
from sonomed_vlm.active_reader.models.constraints import ActionGrammar, restrict_logits
from sonomed_vlm.active_reader.models.prompts import PROMPT_HASH, build_messages, prompt_hash
from sonomed_vlm.config import ExperimentConfig
from sonomed_vlm.models.medgemma import load_medgemma


def resolve_assets(config):
    required = {"base": config.base, "processor": config.processor}
    if config.checkpoint_kind == "base_adapter":
        required["adapter"] = config.adapter
    elif config.adapter:
        raise ValueError("Merged checkpoint must not also specify an adapter")
    paths = {}
    for key, value in required.items():
        path = Path(value).expanduser() if value else None
        if path is None or not path.is_dir():
            raise ValueError(
                f"model.{key} must be an existing local directory for the ultrasound checkpoint. No stock-model fallback or automatic download is allowed."
            )
        paths[key] = path.resolve()
    base_config = json.loads((paths["base"] / "config.json").read_text())
    if base_config.get("model_type") != "qwen3_vl":
        raise ValueError("Expected the existing Qwen3-VL checkpoint (model_type=qwen3_vl)")
    if "adapter" in paths:
        if not (paths["adapter"] / "adapter_config.json").is_file() or not any(
            (paths["adapter"] / n).is_file()
            for n in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise ValueError("Ultrasound adapter directory lacks adapter config or weights")
        cfg = json.loads((paths["adapter"] / "adapter_config.json").read_text())
        if cfg.get("peft_type") != "LORA":
            raise ValueError("This integration requires the existing LoRA adapter")
    return paths


def inventory(config):
    paths = resolve_assets(config)
    hashes = {
        key: {
            p.name: file_hash(p)
            for p in path.iterdir()
            if p.is_file()
            and (p.suffix in (".json", ".jinja") or p.name.startswith("adapter_model"))
        }
        for key, path in paths.items()
    }
    return {
        "model_family": "qwen3_vl",
        "expected_base": "Qwen/Qwen3-VL-4B-Instruct",
        "checkpoint_kind": config.checkpoint_kind,
        "paths": {k: str(v) for k, v in paths.items()},
        "base_revision": config.revision,
        "inference_configuration": config.model_dump(mode="json"),
        "artifact_hashes": hashes,
        "inference_entrypoint": "sonomed_vlm.models.medgemma.load_medgemma + active_reader.models.qwen3_adapter",
        "prompt_template_hash": PROMPT_HASH,
        "supports_ordered_multi_image_input": None,
        "training_stack_versions": versions(),
    }


@dataclass
class Generation:
    text: str
    record: dict
    inputs: dict = field(default_factory=dict, repr=False)
    tokens: object = field(default=None, repr=False)
    behavior_log_probs: object = field(default=None, repr=False)


class QwenPolicy:
    def __init__(self, config, for_training=False):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor

        paths = resolve_assets(config)
        if for_training and not torch.cuda.is_available():
            raise ValueError(
                "Qwen RL requires the configured CRC CUDA environment; CPU smoke uses the test fixture"
            )
        if for_training and config.checkpoint_kind != "base_adapter":
            raise ValueError(
                "RL currently requires base + existing ultrasound LoRA; merged checkpoints support frozen baselines only"
            )
        inherited = ExperimentConfig.model_validate(
            {
                "model": {
                    "name": str(paths["base"]),
                    "revision": None,
                    "require_hf_token": False,
                    "dtype": config.dtype,
                    "attn_implementation": "sdpa",
                    "image_max_pixels": config.image_max_pixels,
                },
                "data": {"root": "."},
                "lora": {"qlora": False},
                "training": {
                    "gradient_checkpointing": for_training,
                    "bf16": config.dtype == "bfloat16",
                },
                "output": {"root": "artifacts/active_reader"},
            }
        )
        model, processor = load_medgemma(inherited, for_training=for_training)
        if paths["processor"] != paths["base"]:
            processor = AutoProcessor.from_pretrained(
                str(paths["processor"]), local_files_only=True
            )
            if "longest_edge" not in processor.image_processor.size:
                raise ValueError("Processor does not expose the existing Qwen image-area control")
            processor.image_processor.size["longest_edge"] = config.image_max_pixels
        if "adapter" in paths:
            model = PeftModel.from_pretrained(
                model, paths["adapter"], adapter_name="policy", is_trainable=for_training
            )
            if for_training:
                # Same ultrasound SFT weights, separate immutable adapter. Never disable SFT.
                model.load_adapter(paths["adapter"], adapter_name="reference", is_trainable=False)
                model.to("cuda")
                model.enable_input_require_grads()
        self.model, self.processor, self.config = model, processor, config
        self.for_training = for_training
        if for_training:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
        self.model.eval()  # Disable all dropout; sampling and likelihoods must agree.
        self.use_adapter("policy")

    def use_adapter(self, name):
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(name)
        for param_name, parameter in self.model.named_parameters():
            language_lora = (
                "lora_" in param_name
                and ".policy." in param_name
                and not any(
                    x in param_name.lower() for x in ("visual", "vision", "projector", "connector")
                )
            )
            parameter.requires_grad = self.for_training and name == "policy" and language_lora
        self.model.eval()

    def _move(self, inputs):
        device = next(self.model.parameters()).device
        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        return {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()
        }

    def generate(self, observation, sample=False, seed=42, final_only=False, retain=False):
        import torch
        from transformers import GenerationConfig

        call_started = time.perf_counter()
        self.use_adapter("policy")
        messages = build_messages(observation, final_only)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move(dict(inputs))
        p = inputs["input_ids"].shape[-1]
        original = self.model.generation_config
        grammar = None
        if getattr(self.config, "constrain_actions", False):
            grammar = ActionGrammar(
                observation, self.processor.tokenizer, original.eos_token_id, final_only
            )
        sampling = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=sample,
            temperature=1.0,
            top_k=0 if sample else 50,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=original.eos_token_id,
            pad_token_id=original.pad_token_id,
            bos_token_id=original.bos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=retain,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            grammar_kwargs = (
                {"prefix_allowed_tokens_fn": lambda _, ids: grammar.allowed(ids[p:].tolist())}
                if grammar
                else {}
            )
            result = self.model.generate(**inputs, generation_config=sampling, **grammar_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tokens = result.sequences[0, p:]
        raw = self.processor.decode(tokens, skip_special_tokens=True)
        behavior = None
        if retain:
            behavior = (
                torch.stack(
                    [
                        score[0].float().log_softmax(-1)[token]
                        for score, token in zip(result.scores, tokens, strict=True)
                    ]
                )
                .detach()
                .cpu()
            )
        image_token = getattr(self.model.config, "image_token_id", None)
        record = {
            "raw_output": raw,
            "prompt_template_hash": prompt_hash(observation),
            "context": observation.public_dict(),
            "final_only": final_only,
            "input_ids": inputs["input_ids"][0].tolist() if retain else None,
            "generated_token_ids": tokens.tolist(),
            "behavior_log_probs": behavior.tolist() if behavior is not None else None,
            "token_constraints": grammar.constraints_for(tokens.tolist())
            if grammar and retain
            else None,
            "constrained_actions": grammar is not None,
            "loss_mask": [0] * p + [1] * len(tokens) if retain else None,
            "image_presentations": len(observation.images),
            "visible_image_indices": list(observation.image_indices),
            "visual_input_tokens": int((inputs["input_ids"] == image_token).sum())
            if image_token is not None
            else None,
            "generated_tokens": len(tokens),
            "latency_seconds": time.perf_counter() - started,
            "generation_seconds": time.perf_counter() - started,
            "call_seconds": time.perf_counter() - call_started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
        }
        if observation.protocol != "joint":
            import hashlib

            record["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            record["processed_pixels_sha256"] = hashlib.sha256(
                inputs["pixel_values"]
                .detach()
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
            record["clean_image_sha256"] = [
                hashlib.sha256(im.tobytes()).hexdigest() for im in observation.images
            ]
            record["input_ids"] = inputs["input_ids"][0].tolist()
        return Generation(
            raw,
            record,
            {k: v.detach().cpu() for k, v in inputs.items()} if retain else {},
            tokens.detach().cpu() if retain else None,
            behavior,
        )

    def log_probs(self, generation, adapter="policy"):
        import torch

        self.use_adapter(adapter)
        if self.for_training and adapter == "policy":
            # Activation checkpointing requires training mode; all dropout is zero.
            self.model.train()
        inputs = self._move(generation.inputs)
        tokens = generation.tokens.to(inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], tokens[None]], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(tokens[None])], dim=1
        )
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in inputs:
                inputs[key] = torch.cat([inputs[key], torch.zeros_like(tokens[None])], dim=1)
        # Recompute multimodal positions for this exact causal snapshot.
        inputs.pop("position_ids", None)
        logits = (
            self.model(**inputs, use_cache=False, logits_to_keep=len(tokens) + 1)
            .logits[0, -len(tokens) - 1 : -1]
            .float()
        )
        logits = restrict_logits(logits, generation.record.get("token_constraints"))
        return logits.log_softmax(-1).gather(1, tokens[:, None]).squeeze(1)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(directory, selected_adapters=["policy"])
        self.processor.save_pretrained(directory / "processor")

    def load_policy(self, directory: Path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = load_file(str(directory / "policy" / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model, state, adapter_name="policy")
        self.use_adapter("policy")
