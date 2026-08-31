"""
Minimal 2-Role RBAC: engineer and manager.

The engineer role is denied access to restricted collections (e.g.,
financials_restricted).  The manager role has access to everything.

Usage as a FastAPI dependency::

    @app.post("/chat")
    async def chat(req: ChatRequest, role: str = Depends(get_role)):
        ...
"""

import logging
from typing import List

from fastapi import Query

logger = logging.getLogger(__name__)

# Valid roles
VALID_ROLES = {"engineer", "manager"}

# Collections that engineers are NOT allowed to see
RESTRICTED_COLLECTIONS: List[str] = ["financials_restricted"]


def get_role(role: str = Query("engineer", description="User role: 'engineer' or 'manager'")) -> str:
    """
    FastAPI dependency that extracts and validates the role query parameter.

    Args:
        role: Role string from query parameter (default: "engineer").

    Returns:
        The validated role string.

    Raises:
        HTTPException 400 if the role is not recognised.
    """
    from fastapi import HTTPException
    # Strip null bytes and whitespace — treat as opaque string
    role = (role or "engineer").replace("\x00", "").lower().strip()
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{role}'. Must be one of: {sorted(VALID_ROLES)}",
        )
    return role


def is_restricted(collection_name: str, role: str) -> bool:
    """
    Check whether a collection is restricted for the given role.

    Args:
        collection_name: The metadata collection name to check.
        role: The user's role.

    Returns:
        True if the collection is restricted AND the role is not manager.
    """
    if role == "manager":
        return False
    return collection_name in RESTRICTED_COLLECTIONS
