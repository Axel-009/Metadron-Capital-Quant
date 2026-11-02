import hashlib
import logging
import secrets
from typing import Any, Dict, List

logger = logging.getLogger("metadron.api.api_keys")

try:
    from fastapi import APIRouter, Depends, HTTPException
    from sqlalchemy.orm import Session
except ImportError:
    raise RuntimeError("FastAPI and SQLAlchemy are required.")

from engine.db.database import get_db
from engine.db.tables import ApiKey
from engine.db.schemas import ApiKeyCreate

router = APIRouter()


def _hash_key(raw: str) -> str:
    """Return a SHA-256 hex digest of the raw API key."""
    return hashlib.sha256(raw.encode()).hexdigest()


@router.post("", status_code=201)
def create_api_key(payload: ApiKeyCreate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Generate a new API key and return the raw value once.

    The raw key is shown only in this response; only its hash is stored.
    """
    raw_key = secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)

    record = ApiKey(key_hash=key_hash, name=payload.name)
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info("API key '%s' created (id=%s)", payload.name, record.id)
    return {
        "id": record.id,
        "name": record.name,
        "key": raw_key,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


@router.get("")
def list_api_keys(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Return metadata for all API keys (hashes are never exposed)."""
    keys = db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
    return [
        {
            "id": k.id,
            "name": k.name,
            "is_active": k.is_active,
            "created_at": k.created_at.isoformat() if k.created_at else None,
        }
        for k in keys
    ]


@router.delete("/{key_id}", status_code=204)
def revoke_api_key(key_id: int, db: Session = Depends(get_db)):
    """Soft-delete an API key by marking it inactive."""
    record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="API key not found")
    record.is_active = False
    db.commit()
    logger.info("API key id=%s revoked", key_id)
    return None
