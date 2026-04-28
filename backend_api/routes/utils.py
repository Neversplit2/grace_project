"""
Utility helper functions and endpoints
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/api/utils", tags=["Utils"])

@router.get("/timestamp")
def get_timestamp():
    """Get current server timestamp"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "unix": datetime.utcnow().timestamp()
    }
