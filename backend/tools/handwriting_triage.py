"""
Handwriting Triage Reader: Transcribes handwritten text from field notes
using a Vision Model (or MockVisionModel fallback).
"""

import logging
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
    from backend.tools.path_safety import safe_resolve_input_path
    _sandbox_dir = Path(__file__).resolve().parent.parent.parent / "workspace" / "sandbox_files"
    try:
        resolved = safe_resolve_input_path(image_path, _sandbox_dir)
    except ValueError as e:
        raise ValueError(f"Handwriting triage rejected path: {e}")
    image_path = str(resolved)

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
    if text and text != "No text detected":
        confidence = vision.get_mock_confidence(image_path)
    else:
        confidence = 0.0

    # Low confidence warning: if below 0.6, prepend a human-review notice
    display_text = text
    if confidence < 0.6 and text and text != "No text detected":
        display_text = f"⚠️ LOW CONFIDENCE - HUMAN REVIEW REQUIRED: {text}"
        logger.warning(f"Handwriting confidence {confidence:.3f} below 0.6 threshold")

    result = {
        "text": display_text,
        "raw_text": text,
        "confidence": round(confidence, 3),
        "source": Path(image_path).name,
    }

    logger.info(f"Handwriting read: {len(text)} chars, confidence={confidence:.3f}")
    return result
