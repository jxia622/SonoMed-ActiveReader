#!/usr/bin/env python3
"""Deterministic subject partitions and public-release frame admission for the pilot.

Downloads only declared members of Zenodo's public training ZIP. No challenge or
BreastVid test input is accepted. Raw temporary MHA buffers are not retained.
"""

import argparse
import csv
import hashlib
import io
import os
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import requests
from PIL import Image

from sonomed_vlm.active_reader.io import digest, file_hash, read_json, write_json


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (
            offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        )
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b""
        if n > 700_000_000:
            raise ValueError("Unexpectedly large ZIP member range")
        start, end = self.pos, self.pos + n - 1
        error = None
        for attempt in range(4):
            try:
                response = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 180)
                )
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                ):
                    raise ValueError(f"Incorrect range response: {response.status_code}")
                data = response.content
                if len(data) != n:
                    raise ValueError("Incomplete byte range")
                self.pos += n
                return data
            except (requests.RequestException, ValueError) as exc:
                error = exc
                if attempt < 3:
                    time.sleep(2**attempt)
        raise RuntimeError("Public archive download failed after four attempts") from error


def decode_mha(data):
    stream = io.BytesIO(data)
    header = {}
    for _ in range(80):
        line = stream.readline().decode("ascii").strip()
        if "=" in line:
            key, value = map(str.strip, line.split("=", 1))
            header[key] = value
            if key == "ElementDataFile":
                break
    assert header["ElementDataFile"] == "LOCAL" and header["CompressedData"] == "False"
    assert header["ElementType"] == "MET_UCHAR" and header["DimSize"] == "744 562 840"
    assert len(data) - stream.tell() == 840 * 562 * 744
    return header, np.frombuffer(data, dtype=np.uint8, offset=stream.tell()).reshape(840, 562, 744)


def partition(rows, screened, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject_id"]].append(row["uuid"])
    known = {r["subject_id"]: r["source_id"] for r in screened}
    assert len(known) == 5 and all(v in groups[k] for k, v in known.items())
    ordered = sorted(set(groups) - set(known), key=lambda s: digest([seed, "subject", s]))
    subjects = sorted(known) + ordered[:27] + ordered[27:35] + ordered[35:55]
    assert len(subjects) == len(set(subjects)) == 60
    cases = []
    for i, subject in enumerate(subjects):
        source = known.get(subject) or min(groups[subject], key=lambda s: digest([seed, "scan", s]))
        cases.append(
            {
                "alias": f"P{i + 1:03d}",
                "subject_id": subject,
                "source_id": source,
                "split": "train" if i < 32 else "tune" if i < 40 else "evaluation",
                "screening_exposed": subject in known,
            }
        )
    return cases


def prepare(args):
    screening = args.screening
    csv_path = (
        screening
        / "samples/acouslic/acouslic-ai-train-set/circumferences/fetal_abdominal_circumferences_per_sweep.csv"
    )
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    screened = read_json(screening / "samples/acouslic/screen_manifest.json")
    cases = partition(rows, screened, args.seed)
    entries = read_json(screening / "sources/acouslic_zip_entries.json")
    metadata = read_json(screening / "sources/acouslic_zip_range.json")
    protocol = {
        "protocol": "acouslic-acquisition-pilot-v1",
        "partition_seed": args.seed,
        "public_source": "https://zenodo.org/records/12697994",
        "source_version": "1.1",
        "source_archive_bytes": metadata["size"],
        "source_csv_sha256": file_hash(csv_path),
        "cases": cases,
        "partition_fingerprint": digest(cases),
        "test_policy": "Only public ACOUSLIC training release. BreastVid and challenge tests forbidden.",
    }
    manifest = args.output / "partition.json"
    if manifest.exists():
        assert read_json(manifest) == protocol, "Existing partition differs"
    else:
        write_json(manifest, protocol)
    write_json(
        args.output / "development_manifest.json", [c for c in cases if c["split"] != "evaluation"]
    )
    write_json(
        args.output / "sealed_evaluation_manifest.json",
        [c for c in cases if c["split"] == "evaluation"],
    )
    print(
        "Frozen partition: 32 train, 8 tune, 20 sealed evaluation subjects",
        protocol["partition_fingerprint"],
        flush=True,
    )
    existing = {r["source_id"]: screening / "samples/acouslic" / r["alias"] for r in screened}

    def one(case):
        out = args.output / "cases" / case["alias"]
        if (out / "admission.json").exists():
            record = read_json(out / "admission.json")
            assert record["case"] == case and len(record["frames"]) == 840
            assert all((out / "frames" / f"{i:04d}.png").exists() for i in range(840))
            return record
        out.mkdir(parents=True, exist_ok=True)
        (out / "frames").mkdir(exist_ok=True)
        members = [
            e
            for e in entries
            if Path(e["name"]).stem == case["source_id"] and e["name"].endswith(".mha")
        ]
        assert len(members) == 2
        member_records, hashes = [], {}
        if case["source_id"] in existing:
            old = existing[case["source_id"]]
            for i in range(840):
                src, dst = old / "frames" / f"{i:04d}.png", out / "frames" / f"{i:04d}.png"
                if not dst.exists():
                    os.link(src, dst)
                hashes[str(i)] = file_hash(dst)
            shutil.copyfile(old / "private_labels.npz", out / "private_labels.npz")
            member_records = [
                {"reused_screening_source": str(old), "verified_by_prior_source_roundtrip": True}
            ]
        else:
            remote = RemoteZip(metadata["url"], metadata["size"])
            with zipfile.ZipFile(remote) as archive:
                for member in sorted(members, key=lambda e: "/masks/" not in e["name"]):
                    data = archive.read(member["name"])  # ZipFile verifies member CRC.
                    header, arr = decode_mha(data)
                    member_records.append(
                        {
                            "member": member["name"],
                            "source_sha256": hashlib.sha256(data).hexdigest(),
                            "header": header,
                            "zip_crc32": member["crc32"],
                        }
                    )
                    if "/masks/" in member["name"]:
                        assert set(np.unique(arr)).issubset({0, 1, 2})
                        optimal = np.any(arr == 1, axis=(1, 2))
                        suboptimal = np.any(arr == 2, axis=(1, 2))
                        assert not np.any(optimal & suboptimal)
                        labels = np.where(optimal, 1, np.where(suboptimal, 2, 0))
                        quality = np.where(
                            optimal, 1, np.where(suboptimal, 0.6 if optimal.any() else 1, 0)
                        )
                        np.savez_compressed(
                            out / "private_labels.npz", labels=labels, quality=quality
                        )
                    else:
                        assert "/images/" in member["name"]
                        for i in range(840):
                            path = out / "frames" / f"{i:04d}.png"
                            Image.fromarray(arr[i]).save(path)
                            hashes[str(i)] = file_hash(path)
                    del arr, data
        assert len(hashes) == 840
        record = {
            "case": case,
            "shape": [840, 562, 744],
            "source": member_records,
            "frames": hashes,
            "private_labels_sha256": file_hash(out / "private_labels.npz"),
        }
        write_json(out / "admission.json", record)
        print(f"Admitted {case['alias']} ({case['split']}); 840 clean frames", flush=True)
        return record

    allowed = cases if args.role == "all" else [c for c in cases if c["split"] == args.role]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, allowed))
    write_json(
        args.output / f"admission_{args.role}.json",
        {
            "cases": len(records),
            "frames": len(records) * 840,
            "partition_fingerprint": protocol["partition_fingerprint"],
            "admission_fingerprints": {r["case"]["alias"]: digest(r) for r in records},
        },
    )
    print(f"Complete: {len(records)} cases admitted; no evaluation outcomes inspected", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screening", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    p.add_argument("--role", choices=["all", "train", "tune", "evaluation"], default="all")
    prepare(p.parse_args())
