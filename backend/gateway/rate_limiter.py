"""Sliding window rate limiter backed by Redis."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Per-session sliding window rate limiter using Redis INCR + EXPIRE."""

    def __init__(
        self,
        redis_url: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._redis = None
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url, decode_responses=True)
        except Exception:
            logger.warning("Redis unavailable — rate limiter will allow all requests")

    async def check(self, session_id: str) -> tuple[bool, int]:
        """Check whether the session is within the rate limit.

        Returns:
            (allowed, retry_after_seconds) — retry_after is 0 when allowed.
        """
        if self._redis is None:
            return True, 0

        key = f"rate:{session_id}"
        try:
            count = await self._redis.incr(key)
            if count == 1:
                # First request in this window — set the expiry
                await self._redis.expire(key, self._window_seconds)

            if count > self._max_requests:
                ttl = await self._redis.ttl(key)
                retry_after = max(ttl, 1)
                return False, retry_after

            return True, 0
        except Exception as exc:
            logger.warning("Rate limiter Redis error (allowing request): %s", exc)
            return True, 0
