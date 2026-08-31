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
    # Path containment: reject paths that escape the sandbox directory
    from backend.tools.path_safety import safe_resolve_input_path
    _sandbox_dir = Path(__file__).resolve().parent.parent.parent / "workspace" / "sandbox_files"
    try:
        resolved = safe_resolve_input_path(image_path, _sandbox_dir)
    except ValueError as e:
        raise ValueError(f"Photo analyzer rejected path: {e}")
    image_path = str(resolved)

    # Step 1: Call Vision Model
    vision = _get_vision_model()
    prompt = "Extract equipment nameplate data from this photo. Read model number, serial number, manufacturer, and equipment type."
    raw_text = vision.analyze_image(image_path, prompt)

    # Step 2: Parse output
    result = _parse_nameplate_output(raw_text)

    # Step 3: Confidence score and warning
    from backend.tools.confidence_helpers import safe_confidence, apply_confidence_warning
    try:
        confidence = safe_confidence(vision.get_mock_confidence(image_path))
    except Exception:
        confidence = 0.0

    # Apply low-confidence warning to the raw_text field
    result["raw_text"] = apply_confidence_warning(raw_text, confidence, tool_name="PhotoAnalyzer")
    result["confidence"] = round(confidence, 3)

    # Add source metadata
    result["source"] = Path(image_path).name
    result["analysis_method"] = "vision_model"

    logger.info(
        f"Nameplate analyzed: model={result['model']}, "
        f"serial={result['serial']}, type={result['equipment_type']}, "
        f"confidence={confidence:.3f}"
    )
    return result
