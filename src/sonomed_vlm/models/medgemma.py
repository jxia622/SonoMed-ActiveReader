"""Gated MedGemma loading and runtime architecture inspection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sonomed_vlm.config import ExperimentConfig, ModelConfig


def _resolve_token(config: ModelConfig) -> str | None:
    local_model = Path(config.name).expanduser().exists()
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except ImportError:
            token = None
    if config.require_hf_token and not local_model and not token:
        raise RuntimeError(
            "Authenticated Hugging Face access is required for gated MedGemma. Accept the model "
            "terms, then use `hf auth login` or export HF_TOKEN securely; never add the token to "
            "YAML or SLURM files."
        )
    return token


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_medgemma(
    config: ExperimentConfig,
    *,
    for_training: bool,
) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "MedGemma requires the GPU dependencies from pyproject.toml. Install the project "
            "in the A100 environment with `pip install -e .`."
        ) from exc

    token = _resolve_token(config.model)
    dtype = torch_dtype(config.model.dtype)
    common: dict[str, Any] = {
        "revision": config.model.revision,
        "token": token,
        "trust_remote_code": config.model.trust_remote_code,
    }
    model_kwargs: dict[str, Any] = {
        **common,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation
    if config.lora.qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a supported CUDA GPU and bitsandbytes installation")
        compute_dtype = torch_dtype(config.lora.qlora_compute_dtype)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif not for_training:
        model_kwargs["device_map"] = "auto"

    try:
        processor = AutoProcessor.from_pretrained(config.model.name, **common)
        if config.model.image_max_pixels is not None:
            image_processor = processor.image_processor
            if "longest_edge" not in image_processor.size:
                raise ValueError("image_max_pixels requires an area-based Qwen image processor")
            image_processor.size["longest_edge"] = config.model.image_max_pixels
        model = AutoModelForImageTextToText.from_pretrained(config.model.name, **model_kwargs)
    except OSError as exc:
        raise RuntimeError(
            "Unable to load MedGemma. Verify accepted model terms, HF_TOKEN, revision, cache "
            "space, and network access. Original error: " + str(exc)
        ) from exc
    if for_training:
        model.config.use_cache = False
        if config.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    return model, processor


def architecture_report(model: Any, processor: Any | None = None) -> dict[str, Any]:
    modules = [name for name, _ in model.named_modules()]
    vision = [name for name in modules if any(key in name.lower() for key in ("vision", "visual"))]
    projectors = [
        name
        for name in modules
        if any(key in name.lower() for key in ("projector", "connector", "multi_modal"))
    ]
    return {
        "model_class": type(model).__name__,
        "config_class": type(model.config).__name__,
        "processor_class": type(processor).__name__ if processor is not None else None,
        "architectures": list(getattr(model.config, "architectures", []) or []),
        "module_count": len(modules),
        "vision_module_candidates": vision,
        "projector_module_candidates": projectors,
    }


def validate_training_device(config: ExperimentConfig) -> None:
    import torch

    if config.training.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MedGemma training; refusing silent CPU fallback")
    if config.training.bf16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("training.bf16=true but this CUDA device does not support BF16")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    visible_devices = torch.cuda.device_count()
    if world_size > 1 and visible_devices < local_world_size:
        raise RuntimeError(
            f"Distributed world requests {local_world_size} local ranks, but only "
            f"{visible_devices} CUDA devices are visible"
        )
