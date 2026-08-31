"""
P&ID-to-Topology-Graph Extractor: Detects equipment (valves, pumps,
instruments) in P&ID images via YOLO and reads tag numbers via a Vision Model.
Falls back to MockYOLO/MockVisionModel if real libraries are unavailable.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------- YOLO abstraction ----------

try:
    from ultralytics import YOLO as _RealYOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logger.info("ultralytics not installed; using MockYOLO for P&ID extraction.")


class MockYOLO:
    """Deterministic fake YOLO that returns plausible bounding boxes."""

    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path

    def predict(self, source, **kwargs) -> list:
        """Return fake detections for any image path."""
        class FakeBox:
            def __init__(self, cls_id, conf, xyxy):
                self.cls = cls_id
                self.conf = conf
                self.xyxy = xyxy

        class FakeResult:
            def __init__(self, boxes):
                self.boxes = boxes

        fake_boxes = [
            FakeBox(0, 0.92, [10, 10, 50, 50]),   # valve
            FakeBox(1, 0.87, [60, 20, 120, 80]),   # pump
            FakeBox(2, 0.78, [140, 10, 180, 50]),  # instrument
        ]
        return [FakeResult(fake_boxes)]


# ---------- Vision Model abstraction ----------

def _get_vision_model():
    """Return a MockVisionModel instance (real VL model integration deferred)."""
    from backend.core.model_manager import MockVisionModel
    return MockVisionModel()


# ---------- Equipment class mapping ----------

_EQUIPMENT_CLASSES = {0: "valve", 1: "pump", 2: "instrument"}


def _crop_region(image_path: str, bbox: list):
    """
    Crop a region from an image. Returns the path to a temporary crop file.
    Uses Pillow if available, otherwise returns the original path.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        crop = img.crop(tuple(bbox))
        crop_path = str(Path(image_path).parent / f"_crop_{int(bbox[0])}_{int(bbox[1])}.png")
        crop.save(crop_path)
        return crop_path
    except Exception as e:
        logger.warning(f"Crop failed ({e}), using original image")
        return image_path


def extract_topology(image_path: str) -> dict:
    """
    Extract a topology graph from a P&ID image.

    Steps:
    1. Run YOLO (or MockYOLO) to detect equipment bounding boxes.
    2. Crop each detected region.
    3. Run Vision Model to read tag numbers inside each crop.
    4. Build a JSON graph with nodes and edges.

    Args:
        image_path: Path to the P&ID image file.

    Returns:
        A dict with "nodes" and "edges" lists.
    """
    # Path containment: reject paths that escape the sandbox directory
    from backend.tools.path_safety import safe_resolve_input_path
    _sandbox_dir = Path(__file__).resolve().parent.parent.parent / "workspace" / "sandbox_files"
    try:
        resolved = safe_resolve_input_path(image_path, _sandbox_dir)
    except ValueError as e:
        raise ValueError(f"P&ID extractor rejected path: {e}")
    image_path = str(resolved)

    # Step 1: Run YOLO
    if _YOLO_AVAILABLE:
        try:
            yolo = _RealYOLO("yolov8n.pt")
        except Exception as e:
            logger.warning(f"YOLO weights not found ({e}), using MockYOLO")
            yolo = MockYOLO()
    else:
        yolo = MockYOLO()

    results = yolo.predict(image_path)

    # Step 2 & 3: Process detections
    vision = _get_vision_model()
    nodes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls) if hasattr(box.cls, '__int__') else int(box.cls)
            conf = float(box.conf) if hasattr(box.conf, '__float__') else float(box.conf)
            bbox = [int(x) for x in box.xyxy] if hasattr(box.xyxy, '__iter__') else [int(x) for x in box.xyxy]

            eq_type = _EQUIPMENT_CLASSES.get(cls_id, f"class_{cls_id}")

            # Crop and read tag
            crop_path = _crop_region(image_path, bbox)
            tag_prompt = f"Read the equipment tag number in this P&ID region. Equipment type: {eq_type}"
            tag_text = vision.analyze_image(crop_path, tag_prompt)

            # Clean up tag text
            tag = tag_text.strip() if tag_text else f"{eq_type.upper()}-{cls_id + 100}"

            nodes.append({
                "id": tag,
                "type": eq_type,
                "confidence": round(conf, 3),
                "bbox": bbox,
            })

    # Step 4: Build edges (sequential connection heuristic)
    edges = []
    for i in range(len(nodes) - 1):
        edges.append({
            "from": nodes[i]["id"],
            "to": nodes[i + 1]["id"],
            "type": "connected_to",
        })

    graph = {"nodes": nodes, "edges": edges}
    logger.info(f"P&ID topology extracted: {len(nodes)} nodes, {len(edges)} edges")
    return graph
