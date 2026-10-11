"""Strict configuration; assets are resolved only by commands that need them."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from sonomed_vlm.config import StrictModel, _expand_env


class Data(StrictModel):
    root: str = ""
    annotation_files: list[str] = Field(default_factory=list)
    manifest: Path = Path("artifacts/active_reader/manifest.jsonl")
    audit: Path = Path("artifacts/active_reader/data_audit.json")
    splits: Path = Path("artifacts/active_reader/splits.json")
    sft_manifests: list[Path] = Field(default_factory=list)


class Model(StrictModel):
    checkpoint_kind: Literal["base_adapter", "merged"] = "base_adapter"
    base: str = ""
    adapter: str = ""
    processor: str = ""
    revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    image_max_pixels: int = Field(default=1048576, ge=65536)
    max_new_tokens: int = Field(default=128, ge=16, le=512)
    constrain_actions: bool = True


class Environment(StrictModel):
    max_unique_frames: int = Field(default=24, ge=1)
    max_decision_turns: int = Field(default=8, ge=1)
    initial_images: int = Field(default=4, ge=1, le=4)
    max_evidence_frames: int = Field(default=3, ge=1, le=3)


class Reward(StrictModel):
    relevance_weight: float = Field(default=0.1, ge=0)
    unique_frame_cost_weight: float = Field(default=0.2, ge=0)
    inspection_call_cost: float = Field(default=0.01, ge=0)
    invalid_action_cost: float = Field(default=0.1, ge=0)
    class_balance: Literal["inverse_frequency_reward", "unweighted"] = "inverse_frequency_reward"


class Training(StrictModel):
    group_size: int = Field(default=4, ge=2)
    learning_rate: float = Field(default=1e-6, gt=0)
    steps: int = Field(default=2, ge=1)
    accumulation_groups: int = Field(default=1, ge=1)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    kl_weight: float = Field(default=0.02, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)


class Evaluation(StrictModel):
    budgets: list[int] = Field(default_factory=lambda: [4, 8, 12, 16, 24])
    seeds: list[int] = Field(default_factory=lambda: [7, 19, 43])
    methods: list[str] = Field(default_factory=lambda: ["uniform", "random", "prompted"])
    bootstrap_samples: int = Field(default=1000, ge=0)


class Navigation(StrictModel):
    # Cached generation and teacher forcing need sufficiently precise action logits.
    # The reader continues to use Model.dtype and the original checkpoint unchanged.
    selector_dtype: Literal["float32"] = "float32"
    max_behavior_logprob_drift: float = Field(default=0.001, gt=0, le=0.001)


class Config(StrictModel):
    seed: int = 42
    data: Data = Field(default_factory=Data)
    model: Model = Field(default_factory=Model)
    environment: Environment = Field(default_factory=Environment)
    reward: Reward = Field(default_factory=Reward)
    training: Training = Field(default_factory=Training)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    navigation: Navigation = Field(default_factory=Navigation)
    output: Path = Path("artifacts/active_reader/runs")


def load_config(path: Path) -> Config:
    config = Config.model_validate(_expand_env(yaml.safe_load(path.read_text())))
    if not config.evaluation.budgets or any(x < 1 for x in config.evaluation.budgets):
        raise ValueError("evaluation.budgets must contain positive integers")
    if not config.evaluation.seeds:
        raise ValueError("evaluation.seeds must not be empty")
    return config
