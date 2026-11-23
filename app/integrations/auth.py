"""
API Authentication & Key Management Module for AMINOID Speech Integrations.
Generates cryptographically secure API keys, verifies SHA-256 hashes,
checks granular permissions, and provides FastAPI dependencies.
"""

import secrets
import hashlib
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from fastapi import Request, HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from .db import (
    create_api_key_record,
    get_api_key_by_hash,
    update_api_key_last_used,
    list_api_keys,
    toggle_api_key,
    revoke_api_key,
    get_api_key_record
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_AUTH = HTTPBearer(auto_error=False)

SUPPORTED_PERMISSIONS = [
    "stt:transcribe",
    "tts:generate",
    "voices:read",
    "audio:read",
    "webhooks:manage"
]

def hash_api_key(raw_key: str) -> str:
    """Computes SHA-256 hash of an API key."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()

def generate_api_key(
    name: str,
    permissions: Optional[List[str]] = None,
    expires_in_days: Optional[int] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Generates a cryptographically secure API key.
    Format: aminoid_live_<32 hex characters>
    Returns: (raw_key, record_dict)
    The raw_key must be displayed to the user immediately and is never stored in plaintext.
    """
    random_hex = secrets.token_hex(16)
    raw_key = f"aminoid_live_{random_hex}"
    key_prefix = f"aminoid_live_{random_hex[:6]}..."
    key_hash = hash_api_key(raw_key)
    key_id = f"key_{uuid.uuid4().hex[:10]}"

    perms = permissions if permissions else SUPPORTED_PERMISSIONS
    # Validate permissions
    valid_perms = [p for p in perms if p in SUPPORTED_PERMISSIONS]

    expires_at = None
    if expires_in_days and expires_in_days > 0:
        exp_time = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        expires_at = exp_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    record = create_api_key_record(
        key_id=key_id,
        name=name,
        key_prefix=key_prefix,
        key_hash=key_hash,
        permissions=valid_perms,
        expires_at=expires_at
    )
    return raw_key, record

def extract_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
    bearer_auth: Optional[HTTPAuthorizationCredentials] = Security(BEARER_AUTH)
) -> Optional[str]:
    """Extracts raw API key from either X-API-Key or Authorization Bearer header."""
    if api_key_header and api_key_header.strip():
        return api_key_header.strip()
    if bearer_auth and bearer_auth.credentials:
        return bearer_auth.credentials.strip()
    # Also check query param fallback for audio streaming links
    query_key = request.query_params.get("api_key")
    if query_key and query_key.strip():
        return query_key.strip()
    return None

class RequireApiKey:
    """FastAPI Dependency class for enforcing API Key authentication and permissions."""

    def __init__(self, required_permission: Optional[str] = None):
        self.required_permission = required_permission

    def __call__(
        self,
        request: Request,
        raw_key: Optional[str] = Depends(extract_api_key)
    ) -> Dict[str, Any]:
        if not raw_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key missing. Provide via 'X-API-Key' header or 'Authorization: Bearer <key>'.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        key_hash = hash_api_key(raw_key)
        record = get_api_key_by_hash(key_hash)

        if not record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API Key.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if not record.get("is_active", 1):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This API Key is deactivated."
            )

        # Check expiration if present
        if record.get("expires_at"):
            try:
                exp_dt = datetime.fromisoformat(record["expires_at"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > exp_dt:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="This API Key has expired."
                    )
            except Exception:
                pass

        # Check permission scope
        if self.required_permission:
            user_perms = record.get("permissions", [])
            if self.required_permission not in user_perms and "*" not in user_perms:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API Key does not have the required permission: '{self.required_permission}'."
                )

        # Update last_used_at in background
        update_api_key_last_used(record["id"])

        # Attach key info to request state
        request.state.api_key_id = record["id"]
        request.state.api_key_name = record["name"]
        return record

class OptionalApiKey:
    """Allows endpoints to accept either authenticated external clients or local UI calls."""

    def __call__(
        self,
        request: Request,
        raw_key: Optional[str] = Depends(extract_api_key)
    ) -> Optional[Dict[str, Any]]:
        if not raw_key:
            return None
        key_hash = hash_api_key(raw_key)
        record = get_api_key_by_hash(key_hash)
        if record and record.get("is_active", 1):
            update_api_key_last_used(record["id"])
            request.state.api_key_id = record["id"]
            return record
        return None
