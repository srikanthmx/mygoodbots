"""Input sanitization and validation for gateway messages."""
from __future__ import annotations

import re

from fastapi import HTTPException, status

MAX_MESSAGE_LENGTH = 4000

# Matches <script ...>...</script> blocks (case-insensitive, dotall)
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
# Matches any remaining HTML tag
_TAG_RE = re.compile(r"<[^>]+>")


def sanitize(message: str) -> str:
    """Strip script blocks and HTML tags from *message*."""
    cleaned = _SCRIPT_RE.sub("", message)
    cleaned = _TAG_RE.sub("", cleaned)
    return cleaned


def validate_length(message: str) -> None:
    """Raise HTTP 400 if *message* exceeds MAX_MESSAGE_LENGTH characters."""
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters",
        )
