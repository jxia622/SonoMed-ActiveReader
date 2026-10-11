"""Native Qwen observations and exact constrained probabilities for inspections."""

import hashlib
import json
import string
import time

import torch
from transformers import GenerationConfig

from sonomed_vlm.active_reader.acouslic.environment import legal_centers
from sonomed_vlm.active_reader.models.qwen3_adapter import Generation

READER_QUESTION = (
    "These are clean frames from one prenatal ultrasound examination. "
    "Choose the single image that provides the most suitable transverse fetal abdominal "
    "plane for abdominal circumference measurement. Select based only on visible anatomy "
    "and plane quality. Do not estimate a measurement or infer a clinical diagnosis. "
    "Choose the image's letter."
)


class ChoiceGrammar:
    def __init__(self, tokenizer, texts):
        self.texts = list(texts)
        self.encoded = tokenizer(self.texts, add_special_tokens=False)["input_ids"]
        assert len({tuple(x) for x in self.encoded}) == len(self.texts)
        lengths = {len(x) for x in self.encoded}
        assert len(lengths) == 1 and min(lengths) > 0
        self.length = next(iter(lengths))
        self.root = {}
        for tokens in self.encoded:
            node = self.root
            for token in tokens:
                node = node.setdefault(int(token), {})

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            node = node[int(token)]
        if not node:
            raise ValueError("Choice continued after its deterministic termination")
        return sorted(node)

    def constraints(self, tokens):
        assert list(tokens) in self.encoded
        return [self.allowed(tokens[:i]) for i in range(len(tokens))]


def inspection_question(displayed, observed):
    geometry = [
        {
            "image": string.ascii_uppercase[j],
            "global_frame": f"{i:03d}",
            "sweep": i // 140 + 1,
            "frame_in_sweep": i % 140,
        }
        for j, i in enumerate(displayed)
    ]
    return (
        "You are navigating six separate prenatal ultrasound sweeps to find a suitable transverse "
        "fetal abdominal plane for circumference measurement. Use the anatomy in the observations "
        "to decide where to inspect next. Sweep 1 is global frames 000-139; sweep 2 is 140-279; "
        "sweep 3 is 280-419; sweep 4 is 420-559; sweep 5 is 560-699; sweep 6 is 700-839. "
        "You may jump to ANY unseen global frame, including near ANY earlier observation. "
        "The inspection will reveal the three nearest unseen frames in that same sweep. "
        "No candidate images are visible until after your decision.\n"
        f"Displayed image locations: {json.dumps(geometry)}\n"
        f"All previously observed global frames: {json.dumps(sorted(observed))}\n"
        f"Remaining new-frame budget: {12 - len(observed)}\n"
        "Return exactly the three-digit global frame index of the next inspection center, "
        "from 000 to 839, excluding previously observed frames."
    )


class Choices:
    def __init__(self, policy, data):
        self.policy, self.data = policy, data

    def prepare(self, alias, indices, question, texts, purpose, blank=False):
        images, hashes = self.data.images(alias, indices, blank)
        content = []
        for j, image in enumerate(images):
            content.extend(
                [
                    {"type": "text", "text": f"Image {string.ascii_uppercase[j]}:"},
                    {"type": "image", "image": image},
                ]
            )
        content.append({"type": "text", "text": question})
        inputs = dict(
            self.policy.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
        grammar = ChoiceGrammar(self.policy.processor.tokenizer, texts)
        record = {
            "alias": alias,
            "purpose": purpose,
            "question": question,
            "image_indices": list(indices),
            "clean_images": hashes,
            "blank_images": blank,
            "input_ids": inputs["input_ids"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "legal_choices": list(texts),
            "processed_pixels_sha256": hashlib.sha256(
                inputs["pixel_values"].contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "visual_input_tokens": int(
                (inputs["input_ids"] == self.policy.model.config.image_token_id).sum()
            ),
            "image_presentations": len(images),
            "context": {"protocol": "acouslic_choice"},
        }
        return inputs, grammar, record

    def generate(
        self,
        alias,
        indices,
        question,
        texts,
        purpose,
        sample=False,
        seed=0,
        retain=False,
        blank=False,
        target=None,
        adapter="policy",
    ):
        started = time.perf_counter()
        inputs, grammar, record = self.prepare(alias, indices, question, texts, purpose, blank)
        policy = self.policy
        policy.use_adapter(adapter)
        values = policy._move(inputs)
        if target is None:
            config = policy.model.generation_config
            generation_config = GenerationConfig(
                do_sample=sample,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                max_new_tokens=grammar.length,
                eos_token_id=None,
                pad_token_id=config.pad_token_id,
                bos_token_id=config.bos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(seed)
                result = policy.model.generate(
                    **values,
                    generation_config=generation_config,
                    prefix_allowed_tokens_fn=lambda _, ids: grammar.allowed(
                        ids[prompt_length:].tolist()
                    ),
                )
            tokens = result.sequences[0, prompt_length:].detach().cpu()
            old = (
                torch.stack(
                    [
                        score[0].float().log_softmax(-1)[int(token)]
                        for score, token in zip(result.scores, tokens, strict=True)
                    ]
                )
                .detach()
                .cpu()
            )
            text = grammar.texts[grammar.encoded.index(tokens.tolist())]
        else:
            text = target
            tokens = torch.tensor(grammar.encoded[grammar.texts.index(target)], dtype=torch.long)
            old = None
        record.update(
            raw_output=text,
            generated_token_ids=tokens.tolist(),
            token_constraints=grammar.constraints(tokens.tolist()),
            behavior_log_probs=old.tolist() if old is not None else None,
            generated_tokens=len(tokens),
            call_seconds=time.perf_counter() - started,
            forward_calls=grammar.length if target is None else 0,
            deterministic_termination=True,
            adapter=adapter,
        )
        return Generation(
            text,
            record,
            {k: v.detach().cpu() for k, v in inputs.items()} if retain else {},
            tokens if retain else None,
            old if retain else None,
        )

    def inspect(self, alias, observed, latest, mode, seed, sample=False, retain=False):
        assert mode in {"full", "current", "geometry"}
        displayed = sorted(observed) if mode != "current" else list(latest)
        texts = [f"{i:03d}" for i in legal_centers(observed)]
        generation = self.generate(
            alias,
            displayed,
            inspection_question(displayed, observed),
            texts,
            "inspection",
            sample,
            seed,
            retain,
            blank=mode == "geometry",
        )
        generation.record["context"] = {
            "protocol": "acouslic_navigation",
            "mode": mode,
            "observed": list(observed),
            "latest": list(latest),
            "budget_remaining": 12 - len(observed),
        }
        return generation

    def recognize(self, alias, indices, purpose, target=None, retain=False, adapter="policy"):
        texts = list(string.ascii_uppercase[: len(indices)])
        question = READER_QUESTION + f"\nReply with exactly one letter from {', '.join(texts)}."
        return self.generate(
            alias, indices, question, texts, purpose, target=target, retain=retain, adapter=adapter
        )
