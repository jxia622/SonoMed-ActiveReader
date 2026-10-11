"""Architecture-aware decoder LoRA injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sonomed_vlm.config import LoraConfigModel

_VISION_MARKERS = ("vision", "visual", "image_encoder")
_PROJECTOR_MARKERS = ("projector", "connector", "multi_modal")
_DECODER_MARKERS = ("language_model", "text_model", "decoder", "model.layers")


@dataclass(frozen=True)
class ParameterReport:
    total: int
    trainable: int
    trainable_percent: float
    trainable_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_parameters": self.total,
            "trainable_parameters": self.trainable,
            "trainable_percent": self.trainable_percent,
            "trainable_parameter_names": self.trainable_names,
        }


def find_decoder_lora_targets(model: Any, requested_suffixes: list[str]) -> list[str]:
    suffixes = set(requested_suffixes)
    all_candidates: list[str] = []
    explicit_decoder: list[str] = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in suffixes:
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in _VISION_MARKERS + _PROJECTOR_MARKERS):
            continue
        if not hasattr(module, "weight"):
            continue
        all_candidates.append(name)
        if any(marker in lowered for marker in _DECODER_MARKERS):
            explicit_decoder.append(name)
    targets = explicit_decoder or all_candidates
    found_suffixes = {name.rsplit(".", 1)[-1] for name in targets}
    missing = sorted(suffixes - found_suffixes)
    if missing:
        available = sorted({name.rsplit(".", 1)[-1] for name, _ in model.named_modules()})
        raise ValueError(
            f"Configured LoRA targets do not exist in the decoder: {missing}. "
            f"Available module leaf names include: {available[:100]}"
        )
    return sorted(targets)


def find_projector_modules(model: Any) -> list[str]:
    candidates = [
        name
        for name, module in model.named_modules()
        if any(marker in name.lower() for marker in _PROJECTOR_MARKERS)
        and any(True for _ in module.parameters(recurse=False))
    ]
    # Save the highest-level trainable projector modules; selecting both a parent
    # and its children can make PEFT wrap the same state more than once.
    return [
        name
        for name in sorted(candidates, key=lambda item: (item.count("."), item))
        if not any(name.startswith(parent + ".") for parent in candidates if parent != name)
    ]


def apply_lora(model: Any, config: LoraConfigModel) -> tuple[Any, list[str]]:
    if not config.enabled:
        return model, []
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "PEFT is required for LoRA; install the project GPU dependencies"
        ) from exc

    for parameter in model.parameters():
        parameter.requires_grad = False
    if config.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = find_decoder_lora_targets(model, config.target_modules)
    projector_modules = find_projector_modules(model) if config.train_projector else []
    if config.train_projector and not projector_modules:
        raise ValueError("train_projector=true, but no projector/connector modules were found")
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=projector_modules or None,
    )
    model = get_peft_model(model, peft_config)
    return model, targets


def parameter_report(model: Any) -> ParameterReport:
    total = 0
    trainable = 0
    names: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            names.append(name)
    return ParameterReport(
        total=total,
        trainable=trainable,
        trainable_percent=(100 * trainable / total) if total else 0.0,
        trainable_names=names,
    )
