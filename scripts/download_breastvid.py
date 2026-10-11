#!/usr/bin/env python3
"""Download the author-linked BreastVid archive without installing CVA-Net."""

import argparse
import hashlib
import json
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

FILE_ID = "1LVXK34OJhC2LkqqyMmVFnXsXQavvZdeF"
SOURCE = "https://github.com/jhl-Det/CVA-Net#dataset-preparation"


class Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "hidden" and "name" in attrs:
            self.values[attrs["name"]] = attrs.get("value", "")


def download(destination):
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "breastvid_author.7z"
    if output.exists():
        raise ValueError(
            "Archive already exists; verify its provenance file instead of downloading again"
        )
    partial = output.with_suffix(".7z.partial")
    session = requests.Session()
    page = session.get(
        "https://drive.google.com/uc", params={"export": "download", "id": FILE_ID}, timeout=60
    )
    page.raise_for_status()
    form = Form()
    form.feed(page.text)
    params = form.values or {"id": FILE_ID, "export": "download", "confirm": "t"}
    offset = partial.stat().st_size if partial.exists() else 0
    response = session.get(
        "https://drive.usercontent.google.com/download",
        params=params,
        headers={"Range": f"bytes={offset}-"} if offset else {},
        stream=True,
        timeout=(30, 120),
    )
    response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        raise ValueError("Author Drive download returned HTML; access or quota requires attention")
    if offset and (
        response.status_code != 206
        or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
    ):
        raise ValueError(
            "Server did not honor resume range; preserve partial archive and retry explicitly"
        )
    started, last = time.monotonic(), time.monotonic()
    with partial.open("ab" if offset else "wb") as stream:
        for block in response.iter_content(4 * 1024 * 1024):
            stream.write(block)
            if time.monotonic() - last > 20:
                print(f"Downloaded {stream.tell() / 1e9:.2f} GB", flush=True)
                last = time.monotonic()
    with partial.open("rb") as stream:
        signature = stream.read(6)
    if signature != b"7z\xbc\xaf\x27\x1c":
        raise ValueError("Downloaded content is not a 7z archive")
    h = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    partial.replace(output)
    output.chmod(0o400)
    (destination / "download_provenance.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "drive_file_id": FILE_ID,
                "sha256": h.hexdigest(),
                "bytes": output.stat().st_size,
                "download_seconds": time.monotonic() - started,
                "terms": "Noncommercial research/education only; no dataset redistribution.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Archive verified: {output}; sha256={h.hexdigest()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/breastvid"))
    download(parser.parse_args().output_dir)
