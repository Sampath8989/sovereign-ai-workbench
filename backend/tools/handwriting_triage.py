"""
Handwriting Triage Reader: Transcribes handwritten text from field notes
using a Vision Model (or MockVisionModel fallback).
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_vision_model():
    """Return a MockVisionModel instance."""
    from backend.core.model_manager import MockVisionModel
    return MockVisionModel()


def _preprocess_image(image_path: str) -> str:
    """
    Preprocess image for OCR. Uses Pillow to resize and convert to grayscale
    if available. Returns the path to the preprocessed image.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        # Convert to grayscale and resize for better OCR
        img = img.convert("L")
        # Resize to a reasonable width for OCR (keep aspect ratio)
        target_width = 800
        if img.width > target_width:
            ratio = target_width / img.width
            img = img.resize((target_width, int(img.height * ratio)), Image.LANCZOS)

        preprocessed_path = str(Path(image_path).parent / f"_preprocessed_{Path(image_path).stem}.png")
        img.save(preprocessed_path)
        return preprocessed_path
    except Exception as e:
        logger.warning(f"Preprocessing failed ({e}), using original image")
        return image_path


def read_note(image_path: str) -> dict:
    """
    Read handwritten text from a field note image.

    Steps:
    1. Preprocess the image (grayscale, resize).
    2. Call Vision Model to transcribe handwritten text.
    3. Generate a confidence score.
    4. Return structured result.

    Args:
        image_path: Path to the handwritten note image.

    Returns:
        A dict with "text", "confidence", and "source" keys.
    """
    # Path containment: reject paths that escape the sandbox directory
    from backend.tools.path_safety import safe_resolve_input_path, resolve_existing_casefold
    _sandbox_dir = Path(__file__).resolve().parent.parent.parent / "workspace" / "sandbox_files"
    try:
        resolved = safe_resolve_input_path(image_path, _sandbox_dir)
    except ValueError as e:
        raise ValueError(f"Handwriting triage rejected path: {e}")

    # Resolve to the real on-disk file when the referenced name differs only
    # by letter case from the actual uploaded filename.
    real_resolved = resolve_existing_casefold(resolved)
    if real_resolved is not None:
        resolved = real_resolved
    image_path = str(resolved)

    # Explicit no-match: never fabricate transcription or a confidence score
    # for an image that does not exist.
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        logger.warning(f"Handwriting triage: no file found at {image_path}")
        return {
            "status": "no_match_found",
            "message": f"No image file found at {Path(image_path).name}. "
                       "Upload the note via the attach button first.",
            "text": "No file found",
            "raw_text": "",
            "confidence": None,
            "source": Path(image_path).name,
        }

    # Step 1: Preprocess
    preprocessed = _preprocess_image(image_path)

    # Step 2: Call Vision Model
    vision = _get_vision_model()
    prompt = "Transcribe the handwritten text in this image. Include all text you can read."
    raw_text = vision.analyze_image(preprocessed, prompt)

    # Step 3: Clean up and generate confidence
    text = raw_text.strip() if raw_text else "No text detected"

    # Mock confidence: derived from image file bytes (not mock text length)
    # so different images produce different, deterministic confidence values.
    # This is MOCK/DEMO behavior only — not calibrated model confidence.
    # Fail-safe: if confidence is None or missing, default to 0.0 (triggers warning).
    if text and text != "No text detected":
        try:
            confidence = vision.get_mock_confidence(image_path)
        except Exception:
            confidence = 0.0
    else:
        confidence = 0.0

    from backend.tools.confidence_helpers import safe_confidence, apply_confidence_warning
    confidence = safe_confidence(confidence)

    # Step 4: Apply low-confidence warning if needed
    display_text = apply_confidence_warning(text, confidence, tool_name="HandwritingTriage")

    result = {
        "text": display_text,
        "raw_text": text,
        "confidence": round(confidence, 3),
        "source": Path(image_path).name,
    }

    logger.info(f"Handwriting read: {len(text)} chars, confidence={confidence:.3f}")
    return result
