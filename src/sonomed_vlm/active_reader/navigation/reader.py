"""Fixed ultrasound Qwen reader: only selected pixels and a shared final prompt.

Scores the two complete legal response paths with identical constrained supports.
The normalized probabilities are forced-choice scores, not clinical calibration.
No reader token participates in the selector's gradient.
"""

import copy
import hashlib
import json
import time

import torch

from sonomed_vlm.active_reader.io import digest
from sonomed_vlm.active_reader.models.constraints import ActionGrammar
from sonomed_vlm.active_reader.models.qwen3_adapter import Generation
from sonomed_vlm.active_reader.navigation.prompts import READER_HASH, messages


class FrozenReader:
    def __init__(self, policy):
        if policy.for_training:
            raise ValueError("Frozen reader requires a separate nontrainable Qwen instance")
        self.policy = policy
        self.cache = {}
        if any(p.requires_grad for p in policy.model.parameters()):
            raise ValueError("Reader parameters must all be frozen")

    def score(self, observation):
        if observation.protocol != "reader" or observation.image_indices != observation.indices:
            raise ValueError("Reader must receive exactly the complete selected clean frame set")
        started = time.perf_counter()
        pixels = [hashlib.sha256(im.tobytes()).hexdigest() for im in observation.images]
        key = digest([READER_HASH, observation.indices, pixels])
        if key in self.cache:
            result = copy.deepcopy(self.cache[key])
            result.update(
                cache_hit=True,
                executed_model_calls=0,
                latency_seconds=time.perf_counter() - started,
            )
            return result
        p = self.policy
        p.use_adapter("policy")
        inputs = dict(
            p.processor.apply_chat_template(
                messages(observation),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
        dtype = next(v.dtype for v in p.model.parameters() if v.is_floating_point())
        model_pixels = inputs["pixel_values"].to(dtype=dtype)
        eos = p.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, int) else eos[0]
        grammar = ActionGrammar(observation, p.processor.tokenizer, eos, final_only=True)
        logp = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for label in ("benign", "malignant"):
                text = json.dumps({"prediction": label}, separators=(",", ":"))
                ids = p.processor.tokenizer(text, add_special_tokens=False)["input_ids"] + [eos]
                generation = Generation(
                    text,
                    {"token_constraints": grammar.constraints_for(ids)},
                    inputs,
                    torch.tensor(ids),
                )
                logp[label] = float(p.log_probs(generation).sum())
        scores = torch.tensor([logp["benign"], logp["malignant"]]).softmax(0).tolist()
        probabilities = dict(zip(("benign", "malignant"), scores, strict=True))
        prediction = max(probabilities, key=probabilities.get)
        result = {
            "prediction": prediction,
            "probabilities": probabilities,
            "candidate_log_probabilities": logp,
            "prompt_template_hash": READER_HASH,
            "selected_indices": list(observation.indices),
            "clean_image_sha256": pixels,
            "input_ids": inputs["input_ids"][0].tolist(),
            "input_ids_sha256": digest(inputs["input_ids"].tolist()),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "processed_pixels_sha256": hashlib.sha256(
                model_pixels.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "processed_pixels_dtype": str(dtype),
            "model_calls": 2,
            "executed_model_calls": 2,
            "image_presentations": 2 * len(pixels),
            "cache_hit": False,
            "latency_seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            "score_definition": "normalized likelihood of two complete constrained responses; uncalibrated",
        }
        self.cache[key] = copy.deepcopy(result)
        return result

    def assert_frozen(self):
        if any(p.requires_grad for p in self.policy.model.parameters()):
            raise RuntimeError("Reader unexpectedly became trainable")
