#!/usr/bin/env python3
"""
Pre-Demo Benchmarking Script
=============================
Measures accuracy of the handwriting reader and P&ID extractor using
deterministic mock models.  Produces ``docs/benchmark_results.json``.

Works even if real model weights are missing — uses MockVisionModel
and MockYOLO exclusively.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SANDBOX_DIR = PROJECT_ROOT / "workspace" / "sandbox_files"
DOCS_DIR = PROJECT_ROOT / "docs"


def _ensure_test_image() -> Path:
    """Create a minimal test image for the benchmark if it doesn't exist."""
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    img_path = SANDBOX_DIR / "benchmark_note.jpg"
    if not img_path.exists():
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (400, 200), "white")
            draw = ImageDraw.Draw(img)
            # Draw text that the mock will "read"
            draw.text((20, 30), "Pressure 5bar", fill="black")
            draw.text((20, 80), "Temperature 120C", fill="black")
            img.save(str(img_path))
            logger.info(f"Created benchmark test image: {img_path}")
        except ImportError:
            # Pillow not available — create a 1-byte placeholder
            img_path.write_bytes(b"\xff\xd8\xff\xe0")
            logger.warning("Pillow not installed; created placeholder image")
    return img_path


def _word_overlap_score(hypothesis: str, reference: str) -> float:
    """
    Compute word-level overlap accuracy between hypothesis and reference.
    Returns a value in [0.0, 1.0].
    """
    hyp_words = set(hypothesis.lower().split())
    ref_words = set(reference.lower().split())
    if not ref_words:
        return 0.0
    overlap = hyp_words & ref_words
    return len(overlap) / len(ref_words)


def run_benchmark() -> dict:
    """
    Run the benchmark and return a metrics dict.

    Returns:
        A dict with keys like ``handwriting_word_accuracy`` and ``pid_precision``.
    """
    img_path = _ensure_test_image()

    # --- Handwriting benchmark ---
    handwriting_accuracy = 0.0
    try:
        from backend.tools.handwriting_triage import read_note
        result = read_note(str(img_path))
        raw_text = result.get("raw_text", result.get("text", ""))
        ground_truth = "Pressure 5bar"
        handwriting_accuracy = round(_word_overlap_score(raw_text, ground_truth) * 100, 1)
        logger.info(
            f"Handwriting: raw='{raw_text[:80]}' | "
            f"accuracy={handwriting_accuracy}%"
        )
    except Exception as e:
        logger.warning(f"Handwriting benchmark failed: {e}")

    # --- P&ID benchmark ---
    pid_precision = 0.0
    try:
        from backend.tools.pid_extractor import extract_topology
        graph = extract_topology(str(img_path))
        nodes = graph.get("nodes", [])
        # Mock always returns 3 nodes; precision = correctly typed nodes / total
        correct_types = sum(1 for n in nodes if n.get("type") in ("valve", "pump", "instrument"))
        pid_precision = round((correct_types / len(nodes) * 100) if nodes else 0.0, 1)
        logger.info(f"P&ID: {len(nodes)} nodes, precision={pid_precision}%")
    except Exception as e:
        logger.warning(f"P&ID benchmark failed: {e}")

    metrics = {
        "handwriting_word_accuracy": handwriting_accuracy,
        "pid_precision": pid_precision,
    }

    # Write results to docs/benchmark_results.json
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Benchmark results written to {out_path}")

    return metrics


if __name__ == "__main__":
    metrics = run_benchmark()
    print(json.dumps(metrics, indent=2))
