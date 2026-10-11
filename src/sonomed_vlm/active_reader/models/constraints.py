"""Finite legal action grammar built solely from public observation state."""

import itertools
import json


class ActionGrammar:
    def __init__(self, observation, tokenizer, eos_token_ids, final_only=False):
        actions = []
        if observation.protocol == "navigation":
            actions = list(observation.legal_actions)
        elif observation.protocol == "reader":
            actions = [{"prediction": label} for label in ("benign", "malignant")]
        elif not final_only:
            actions.extend(
                {"action": "inspect", "center_bin": b, "scale": s}
                for b, s in observation.legal_inspections
            )
        for count in range(
            1,
            (
                min(observation.max_evidence_frames, len(observation.indices))
                if observation.protocol == "joint"
                else 0
            )
            + 1,
        ):
            for refs in itertools.combinations(observation.indices, count):
                for label in ("benign", "malignant"):
                    actions.append(
                        {"action": "stop", "prediction": label, "evidence_frames": list(refs)}
                    )
        if not actions:
            raise ValueError("Action grammar needs observed evidence")
        texts = [json.dumps(action, separators=(",", ":")) for action in actions]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        self.root = {}
        eos = [eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids
        if not eos:
            raise ValueError("Action grammar needs the model's EOS token")
        if observation.protocol != "joint":
            eos = eos[:1]
        for tokens in encoded:
            node = self.root
            for token in tokens:
                node = node.setdefault(int(token), {})
            for token in eos:
                node[int(token)] = {}

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            if int(token) not in node:
                raise ValueError("Generated prefix is outside the action grammar")
            node = node[int(token)]
        if not node:
            raise ValueError("Generation continued past grammar EOS")
        return sorted(node)

    def constraints_for(self, tokens):
        return [self.allowed(tokens[:i]) for i in range(len(tokens))]


def restrict_logits(logits, constraints):
    import torch

    if constraints is None:
        return logits
    if len(constraints) != logits.shape[0]:
        raise ValueError("Action grammar mask is not aligned to assistant tokens")
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, allowed in enumerate(constraints):
        if not allowed:
            raise ValueError("Empty action grammar support")
        mask[row, allowed] = True
    return logits.masked_fill(~mask, -torch.inf)
