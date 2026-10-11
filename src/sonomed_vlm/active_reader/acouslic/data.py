"""Clean-pixel observations, private annotation scores, and explicit split gates."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from sonomed_vlm.active_reader.io import file_hash, read_json


class Cases:
    def __init__(self, root, allow_evaluation=False):
        self.root = Path(root)
        self.partition = read_json(self.root / "partition.json")
        assert self.partition["protocol"] == "acouslic-acquisition-pilot-v1"
        self.rows = read_json(self.root / "development_manifest.json")
        if allow_evaluation:
            self.rows += read_json(self.root / "sealed_evaluation_manifest.json")
        assert len({r["subject_id"] for r in self.rows}) == len(self.rows)
        self.by_alias = {r["alias"]: r for r in self.rows}
        self.allow_evaluation = allow_evaluation
        self.admission = lru_cache(maxsize=60)(self.admission)
        self.quality = lru_cache(maxsize=60)(self.quality)
        self._image = lru_cache(maxsize=1200)(self._image)

    def subset(self, role):
        if role == "evaluation" and not self.allow_evaluation:
            raise ValueError("Held-out evaluation requires an explicit frozen-protocol gate")
        return [r for r in self.rows if r["split"] == role]

    def admission(self, alias):
        assert alias in self.by_alias, "Subject not admitted to this process"
        row = read_json(self.root / "cases" / alias / "admission.json")
        assert row["case"] == self.by_alias[alias]
        return row

    def quality(self, alias):
        meta = self.admission(alias)
        path = self.root / "cases" / alias / "private_labels.npz"
        assert file_hash(path) == meta["private_labels_sha256"]
        with np.load(path) as archive:
            q, labels = archive["quality"].copy(), archive["labels"].copy()
        assert q.shape == labels.shape == (840,) and np.isin(q, [0, 0.6, 1]).all()
        return q, labels

    def _image(self, alias, index):
        meta = self.admission(alias)
        assert 0 <= index < 840
        path = self.root / "cases" / alias / "frames" / f"{index:04d}.png"
        assert file_hash(path) == meta["frames"][str(index)]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            assert rgb.size == (744, 562)
            pixels = rgb.tobytes()
        return pixels, hashlib.sha256(pixels).hexdigest()

    def images(self, alias, indices, blank=False):
        images, hashes = [], []
        for index in indices:
            pixels, original_hash = self._image(alias, index)
            image = (
                Image.new("RGB", (744, 562))
                if blank
                else Image.frombytes("RGB", (744, 562), pixels)
            )
            images.append(image)
            hashes.append(
                {
                    "index": index,
                    "source_rgb_sha256": original_hash,
                    "presented_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
            )
        return images, hashes
