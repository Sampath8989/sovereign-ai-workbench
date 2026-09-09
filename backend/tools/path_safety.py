"""
Path Safety Utilities: Containment checks for file generators and analyzers.
Ensures all file operations stay within their intended directory boundaries.
"""

from pathlib import Path
from typing import Optional

# Maximum allowed filename length (common filesystem limit)
MAX_FILENAME_LENGTH = 200


def resolve_existing_casefold(path: Path) -> Optional[Path]:
    """
    Resolve a candidate path to the REAL file on disk.

    1. If the candidate exists exactly as written, it is returned unchanged.
    2. Otherwise, if the candidate's parent directory contains exactly one
       file whose name differs only by letter case (e.g. the model referenced
       ``screenshot.png`` but the uploaded file is ``Screenshot.png``), the
       real on-disk file is returned. This mirrors case-insensitive filesystems
       without ever lowercasing or rewriting the real filename.
    3. Ambiguous (zero or multiple) matches return ``None`` so a lookup never
       silently picks the wrong file.

    Never performs path traversal: only the candidate's own parent directory
    is listed, and the candidate is expected to already be containment-checked.
    """
    candidate = Path(path)
    try:
        if candidate.exists():
            return candidate
    except OSError:
        return None

    parent = candidate.parent
    if not parent.is_dir():
        return None

    name_lower = candidate.name.lower()
    matches = []
    try:
        for child in parent.iterdir():
            if child.is_file() and child.name.lower() == name_lower:
                matches.append(child)
    except OSError:
        return None

    if len(matches) == 1:
        return matches[0]
    return None


def safe_resolve_output_path(filename: str, base_dir: Path) -> Path:
    """
    Resolve a filename against a base directory, enforcing:
    1. No path traversal (../ , absolute paths, etc.)
    2. Reasonable filename length
    3. No null bytes

    Args:
        filename: User-supplied filename (e.g., "report.docx").
        base_dir: The directory that must contain the resolved path.

    Returns:
        Resolved Path inside base_dir.

    Raises:
        ValueError: If the path escapes the base directory, filename is too
                    long, or contains null bytes.
    """
    # Pre-checks on the raw filename
    if "\x00" in filename:
        raise ValueError("Filename contains null bytes.")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(
            f"Filename too long ({len(filename)} chars, max {MAX_FILENAME_LENGTH})."
        )

    # Resolve the full path
    output_path = (base_dir / filename).resolve()

    # Containment check: resolved path must be inside base_dir
    base_resolved = base_dir.resolve()
    if not _is_within_dir(output_path, base_resolved):
        raise ValueError(
            f"Path traversal detected: '{filename}' resolves outside the allowed directory."
        )

    return output_path


def safe_resolve_input_path(image_path: str, base_dir: Path) -> Path:
    """
    Resolve an input file path against a base directory, enforcing containment.
    Used by vision tools (pid_extractor, handwriting_triage, photo_analyzer) to
    prevent path-traversal attacks that could read arbitrary files from disk.

    Args:
        image_path: User-supplied path (e.g., "workspace/sandbox_files/note.jpg").
        base_dir: The directory that must contain the resolved path.

    Returns:
        Resolved Path inside base_dir.

    Raises:
        ValueError: If the path escapes the base directory, contains null bytes,
                    or the filename is too long.
    """
    if "\x00" in image_path:
        raise ValueError("Path contains null bytes.")

    if len(image_path) > MAX_FILENAME_LENGTH:
        raise ValueError(
            f"Path too long ({len(image_path)} chars, max {MAX_FILENAME_LENGTH})."
        )

    resolved = Path(image_path).resolve()
    base_resolved = base_dir.resolve()

    if not _is_within_dir(resolved, base_resolved):
        raise ValueError(
            f"Path traversal detected: '{image_path}' resolves outside the allowed sandbox directory."
        )

    return resolved


def _is_within_dir(path: Path, directory: Path) -> bool:
    """Check if path is inside directory (or is the directory itself)."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
