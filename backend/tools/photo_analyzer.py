"""
Field Photo Analyzer: Extracts equipment nameplate data from field photos
using a Vision Model (or MockVisionModel fallback).
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_vision_model():
    """Return a MockVisionModel instance."""
    from backend.core.model_manager import MockVisionModel
    return MockVisionModel()


def _parse_nameplate_output(raw_text: str) -> dict:
    """
    Parse the raw Vision Model output into structured nameplate fields.
    Handles various formats: "Model: X-200", "Serial: 12345", etc.
    """
    result = {
        "equipment_type": "unknown",
        "model": "unknown",
        "serial": "unknown",
        "manufacturer": "unknown",
        "raw_text": raw_text,
    }

    if not raw_text:
        return result

    # Try to extract structured fields
    patterns = {
        "model": r"(?:model|mfg|type)[:\s]+([A-Za-z0-9\-\.]+)",
        "serial": r"(?:serial|s/?n|serial\s*no?)[:\s]+([A-Za-z0-9\-\.]+)",
        "manufacturer": r"(?:manufacturer|mfr|made\s+by)[:\s]+([A-Za-z0-9\s\-\.]+)",
        "equipment_type": r"(?:equipment|type|desc(?:ription)?)[:\s]+([A-Za-z0-9\s\-\.]+)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            result[field] = match.group(1).strip()

    # If no structured fields found, try the MockVisionModel format
    if result["model"] == "unknown" and "Model:" in raw_text:
        model_match = re.search(r"Model:\s*(\S+)", raw_text)
        if model_match:
            result["model"] = model_match.group(1)

    if result["serial"] == "unknown" and "Serial:" in raw_text:
        serial_match = re.search(r"Serial:\s*(\S+)", raw_text)
        if serial_match:
            result["serial"] = serial_match.group(1)

    return result


def analyze_nameplate(image_path: str) -> dict:
    """
    Analyze a field photo to extract equipment nameplate data.

    Steps:
    1. Call Vision Model with nameplate extraction prompt.
    2. Parse the raw output into structured fields.
    3. Return a dictionary with equipment details.

    Args:
        image_path: Path to the field photo.

    Returns:
        A dict with equipment_type, model, serial, manufacturer, raw_text.
    """
    image_path = str(Path(image_path).resolve())

    # Step 1: Call Vision Model
    vision = _get_vision_model()
    prompt = "Extract equipment nameplate data from this photo. Read model number, serial number, manufacturer, and equipment type."
    raw_text = vision.analyze_image(image_path, prompt)

    # Step 2: Parse output
    result = _parse_nameplate_output(raw_text)

    # Add source metadata
    result["source"] = Path(image_path).name
    result["analysis_method"] = "vision_model"

    logger.info(
        f"Nameplate analyzed: model={result['model']}, "
        f"serial={result['serial']}, type={result['equipment_type']}"
    )
    return result
