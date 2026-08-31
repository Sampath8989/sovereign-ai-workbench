"""
Spreadsheet Generator Tool: Creates Excel (.xlsx) files using openpyxl.
"""

import logging
from pathlib import Path
from typing import List

from openpyxl import Workbook

from backend.tools.path_safety import safe_resolve_output_path

logger = logging.getLogger(__name__)

# Output directory for generated spreadsheets
OUTPUT_DIR = Path(__file__).parent.parent.parent / "workspace" / "outputs"

# Types openpyxl can natively serialize
_SAFE_CELL_TYPES = (str, int, float, bool, type(None))


def _coerce_cell_value(value):
    """Coerce a cell value to an openpyxl-safe type."""
    if isinstance(value, _SAFE_CELL_TYPES):
        return value
    # Convert anything else to string representation
    logger.warning(f"Coercing non-serializable cell value to string: {type(value).__name__}")
    return str(value)


def generate_sheet(filename: str, data: List[List[str]]) -> str:
    """
    Generate an Excel spreadsheet from a 2D list of data.

    Args:
        filename: Name of the output file (e.g. "data.xlsx").
        data: 2D list of strings to write starting at cell A1.

    Returns:
        Absolute path to the created file.
    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve path with containment check
    output_path = safe_resolve_output_path(filename, OUTPUT_DIR)

    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"

        for row_idx, row in enumerate(data, start=1):
            for col_idx, value in enumerate(row, start=1):
                ws.cell(row=row_idx, column=col_idx, value=_coerce_cell_value(value))

        wb.save(str(output_path))

        logger.info(f"Spreadsheet generated: {output_path} ({len(data)} rows)")
        return str(output_path)

    except Exception as e:
        logger.error(f"Spreadsheet generation failed: {e}")
        raise
