"""Official-chat-template multimodal collation with assistant-only labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from sonomed_vlm.data.schema import NormalizedExample


def build_messages(example: NormalizedExample, include_answer: bool = True) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if example.system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": example.system_prompt}]}
        )
    user_content = [
        {"type": "image", "image": image.open()} for image in example.images
    ]
    user_content.append({"type": "text", "text": example.user_prompt})
    messages.append({"role": "user", "content": user_content})
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example.assistant_response}],
            }
        )
    return messages


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


class AssistantOnlyMultimodalCollator:
    """Batch images/text and mask every token before the assistant response."""

    def __init__(self, processor: Any, max_length: int | None = None) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[NormalizedExample | dict[str, Any]]) -> dict[str, Any]:
        import torch

        normalized = [
            example
            if isinstance(example, NormalizedExample)
            else NormalizedExample(
                **{
                    key: value
                    for key, value in example.items()
                    if key in {field.name for field in fields(NormalizedExample)}
                }
            )
            for example in examples
        ]
        full_conversations = [build_messages(example, include_answer=True) for example in normalized]
        prompt_conversations = [conversation[:-1] for conversation in full_conversations]
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_length is not None:
            template_kwargs.update({"truncation": True, "max_length": self.max_length})
        batch = self.processor.apply_chat_template(
            full_conversations, add_generation_prompt=False, **template_kwargs
        )
        prompt_token_ids: list[list[int]] = []
        for conversation in prompt_conversations:
            prompt_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
            }
            if self.max_length is not None:
                prompt_kwargs.update({"truncation": True, "max_length": self.max_length})
            prompt = self.processor.apply_chat_template(conversation, **prompt_kwargs)
            prompt_token_ids.append(_input_ids(prompt))

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            labels.masked_fill_(attention_mask.eq(0), -100)
        for row, expected_ids in enumerate(prompt_token_ids):
            prompt_length = len(expected_ids)
            active_positions = (
                attention_mask[row].ne(0).nonzero(as_tuple=False).flatten()
                if attention_mask is not None
                else torch.arange(input_ids.shape[1], device=input_ids.device)
            )
            mask_length = min(prompt_length, len(active_positions))
            expected = torch.as_tensor(expected_ids, device=input_ids.device)
            observed = input_ids[row, active_positions[:mask_length]]
            if mask_length and not torch.equal(observed, expected[:mask_length]):
                raise ValueError(
                    "Processor prompt tokens are not a prefix of the full conversation; "
                    "assistant-only masking cannot be guaranteed"
                )
            labels[row, active_positions[:mask_length]] = -100
        if torch.all(labels.eq(-100)):
            raise ValueError(
                "All labels are masked; increase max_length or inspect the processor chat template"
            )
        batch["labels"] = labels
        return dict(batch)
