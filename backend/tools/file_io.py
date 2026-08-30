"""
File I/O Tool: Path-scoped read/write for agent file operations.
All operations are sandboxed to workspace/sandbox_files/.
Directory traversal is blocked to prevent escape.
"""

import os
from pathlib import Path

# Base directory for all agent file operations
BASE_DIR = Path(__file__).parent.parent.parent / "workspace" / "sandbox_files"
BASE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_resolve(filename: str) -> Path:
    """
    Resolve filename against BASE_DIR with traversal protection.
    Raises ValueError if the resolved path escapes BASE_DIR.
    """
    # Normalize the path and resolve any .. or symlinks
    target = (BASE_DIR / filename).resolve()

    # Ensure the resolved path is within BASE_DIR
    if not str(target).startswith(str(BASE_DIR.resolve())):
        raise ValueError(
            f"Path traversal detected: '{filename}' resolves outside the sandbox."
        )

    return target


def read_file(filename: str) -> str:
    """
    Read a file from the sandboxed directory.

    Args:
        filename: Relative path within the sandbox (no directory traversal allowed).

    Returns:
        File contents as a string, or an error message.
    """
    try:
        path = _safe_resolve(filename)
    except ValueError as e:
        return f"Error: {e}"

    if not path.exists():
        return f"Error: File not found: {filename}"

    if not path.is_file():
        return f"Error: Path is not a file: {filename}"

    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: Could not read file: {e}"


def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the sandboxed directory.

    Args:
        filename: Relative path within the sandbox (no directory traversal allowed).
        content: String content to write.

    Returns:
        Success or error message.
    """
    try:
        path = _safe_resolve(filename)
    except ValueError as e:
        return f"Error: {e}"

    # Ensure parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"Error: Could not create directory: {e}"

    try:
        path.write_text(content, encoding="utf-8")
        return f"Success: File written to {filename}"
    except Exception as e:
        return f"Error: Could not write file: {e}"
