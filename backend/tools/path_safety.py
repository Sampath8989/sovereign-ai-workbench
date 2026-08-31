"""
Path Safety Utilities: Containment checks for file generators and analyzers.
Ensures all file operations stay within their intended directory boundaries.
"""

from pathlib import Path

# Maximum allowed filename length (common filesystem limit)
MAX_FILENAME_LENGTH = 200


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


def _is_within_dir(path: Path, directory: Path) -> bool:
    """Check if path is inside directory (or is the directory itself)."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
