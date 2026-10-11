"""Only this boundary object may enter an agent; no source identifiers or rewards."""

from dataclasses import dataclass, field

from PIL import Image


@dataclass(frozen=True)
class Observation:
    indices: tuple[int, ...]
    images: tuple[Image.Image, ...] = field(repr=False, compare=False)
    returned_indices: tuple[int, ...] = ()
    total_frames: int = 0
    remaining_frames: int = 0
    remaining_turns: int = 0
    legal_inspections: tuple[tuple[int, str], ...] = ()
    history: tuple[dict, ...] = ()
    error: str | None = None
    max_evidence_frames: int = 3
    protocol: str = "joint"
    visible_indices: tuple[int, ...] | None = None
    legal_actions: tuple[dict, ...] = ()
    history_mode: str = "full"

    @property
    def image_indices(self):
        return self.indices if self.visible_indices is None else self.visible_indices

    def public_dict(self):
        result = {
            "observed_indices": self.indices,
            "returned_indices": self.returned_indices,
            "total_frames": self.total_frames,
            "remaining_frames": self.remaining_frames,
            "remaining_turns": self.remaining_turns,
            "legal_inspections": [{"center_bin": b, "scale": s} for b, s in self.legal_inspections],
            "history": self.history,
            "error": self.error,
            "max_evidence_frames": self.max_evidence_frames,
        }
        if self.protocol != "joint":
            result.update(
                protocol=self.protocol,
                visible_indices=self.image_indices,
                legal_actions=self.legal_actions,
                history_mode=self.history_mode,
            )
        return result
