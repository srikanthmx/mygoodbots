"""SessionStore: Redis-backed session context storage with in-memory fallback."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from backend.models import Message, SessionContext

logger = logging.getLogger(__name__)


def _serialize_context(context: SessionContext) -> str:
    """Serialize SessionContext to a JSON string."""
    data: dict[str, Any] = {
        "session_id": context.session_id,
        "bot_id": context.bot_id,
        "history": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in context.history
        ],
        "metadata": context.metadata,
        "created_at": context.created_at.isoformat(),
        "updated_at": context.updated_at.isoformat(),
    }
    return json.dumps(data)


def _deserialize_context(raw: str) -> SessionContext:
    """Deserialize a JSON string back into a SessionContext."""
    data = json.loads(raw)
    history = [
        Message(
            role=msg["role"],
            content=msg["content"],
            timestamp=datetime.fromisoformat(msg["timestamp"]),
        )
        for msg in data.get("history", [])
    ]
    return SessionContext(
        session_id=data["session_id"],
        bot_id=data["bot_id"],
        history=history,
        metadata=data.get("metadata", {}),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
    )


class SessionStore:
    """Async session store backed by Redis, with in-memory fallback."""

    def __init__(self, redis_url: str, ttl: int = 3600) -> None:
        self._redis_url = redis_url
        self._default_ttl = ttl
        self._redis: Any = None
        self._fallback: dict[str, str] = {}
        self._use_fallback = False
        self._connect()

    def _connect(self) -> None:
        try:
            import redis.asyncio as aioredis  # type: ignore[import]
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Redis unavailable, using in-memory fallback: %s", exc)
            self._use_fallback = True

    async def _ensure_connected(self) -> bool:
        """Return True if Redis is usable, False if we should use fallback."""
        if self._use_fallback:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception as exc:
            logger.warning("Redis ping failed, switching to in-memory fallback: %s", exc)
            self._use_fallback = True
            return False

    async def get(self, session_id: str) -> SessionContext | None:
        """Return the SessionContext for session_id, or None if not found."""
        if await self._ensure_connected():
            try:
                raw = await self._redis.get(session_id)
                if raw is None:
                    return None
                return _deserialize_context(raw)
            except Exception as exc:
                logger.warning("Redis get failed, falling back to memory: %s", exc)
                self._use_fallback = True

        raw = self._fallback.get(session_id)
        return _deserialize_context(raw) if raw is not None else None

    async def set(
        self,
        session_id: str,
        context: SessionContext,
        ttl: int | None = None,
    ) -> None:
        """Serialize and store context with TTL."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        raw = _serialize_context(context)

        if await self._ensure_connected():
            try:
                await self._redis.set(session_id, raw, ex=effective_ttl)
                return
            except Exception as exc:
                logger.warning("Redis set failed, falling back to memory: %s", exc)
                self._use_fallback = True

        self._fallback[session_id] = raw

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None and not self._use_fallback:
            try:
                await self._redis.aclose()
            except Exception:
                pass
