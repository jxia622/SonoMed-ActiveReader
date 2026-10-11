"""Small DDP/SLURM environment helpers."""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorldInfo:
    rank: int
    local_rank: int
    world_size: int
    hostname: str
    slurm_job_id: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def world_info() -> WorldInfo:
    return WorldInfo(
        rank=int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))),
        local_rank=int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))),
        world_size=int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
