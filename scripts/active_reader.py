#!/usr/bin/env python3
"""Separate active-reader application; existing SFT commands remain intact."""

import _bootstrap  # noqa: F401

from sonomed_vlm.active_reader.cli import main

if __name__ == "__main__":
    main()
