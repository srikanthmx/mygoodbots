"""JWT + API-key authentication for the FastAPI gateway."""
from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

JWT_ALGORITHM = "HS256"
API_KEY_HEADER_NAME = "X-API-Key"

security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
    api_key: str = Security(api_key_header),
) -> dict:
    """Verify JWT bearer token or API key.

    Tries JWT first, then falls back to API key.
    Raises HTTP 401 if neither is valid.
    Returns the decoded JWT payload (or a minimal dict for API-key auth).
    """
    jwt_secret = os.environ.get("JWT_SECRET", "")

    # --- Try JWT ---
    if credentials is not None:
        token = credentials.credentials
        if jwt_secret:
            try:
                payload = jwt.decode(token, jwt_secret, algorithms=[JWT_ALGORITHM])
                return payload
            except JWTError:
                pass  # fall through to API key check

    # --- Try API key ---
    expected_api_key = os.environ.get("API_KEY", "")
    if expected_api_key and api_key and api_key == expected_api_key:
        return {"sub": "api-key-user", "auth_method": "api_key"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
