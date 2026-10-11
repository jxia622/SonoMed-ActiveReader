from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (8, 6), color=128).save(buffer, format="PNG")
    return buffer.getvalue()
