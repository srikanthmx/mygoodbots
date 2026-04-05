from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

from backend.models import ApprovalError, ApprovalRequest, ApprovalStatus

logger = logging.getLogger(__name__)


def _serialize(request: ApprovalRequest) -> str:
    """Serialize ApprovalRequest to JSON string for Redis storage."""
    return json.dumps(
        {
            "approval_id": request.approval_id,
            "diff": request.diff,
            "summary": request.summary,
            "status": request.status.value,
            "rejection_reason": request.rejection_reason,
            "created_at": request.created_at.isoformat(),
            "expires_at": request.expires_at.isoformat() if request.expires_at else None,
        }
    )


def _deserialize(data: str) -> ApprovalRequest:
    """Deserialize JSON string back to ApprovalRequest."""
    obj = json.loads(data)
    return ApprovalRequest(
        approval_id=obj["approval_id"],
        diff=obj["diff"],
        summary=obj["summary"],
        status=ApprovalStatus(obj["status"]),
        rejection_reason=obj.get("rejection_reason"),
        created_at=datetime.fromisoformat(obj["created_at"]),
        expires_at=datetime.fromisoformat(obj["expires_at"]) if obj.get("expires_at") else None,
    )


class ApprovalGate:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        notifier: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._notifier = notifier
        # Callbacks are never serialized — kept in memory only
        self._callbacks: dict[str, Callable[[bool, str], Awaitable[None]]] = {}
        # Try to connect to Redis; fall back to in-memory dict
        self._redis = None
        self._memory_store: dict[str, str] = {}
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url, decode_responses=True)
        except Exception:
            logger.warning("Redis unavailable — using in-memory store for ApprovalGate")

    # ------------------------------------------------------------------
    # Internal store helpers
    # ------------------------------------------------------------------

    async def _store_get(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                return await self._redis.get(key)
            except Exception:
                logger.warning("Redis read failed, falling back to memory")
        return self._memory_store.get(key)

    async def _store_set(self, key: str, value: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=self._ttl_seconds)
                return
            except Exception:
                logger.warning("Redis write failed, falling back to memory")
        self._memory_store[key] = value

    async def _store_keys(self) -> list[str]:
        if self._redis is not None:
            try:
                return [k async for k in self._redis.scan_iter("approval:*")]
            except Exception:
                logger.warning("Redis scan failed, falling back to memory")
        return [k for k in self._memory_store if k.startswith("approval:")]

    def _key(self, approval_id: str) -> str:
        return f"approval:{approval_id}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_proposal(
        self,
        approval_id: str,
        diff: str,
        summary: str,
        callback: Callable[[bool, str], Awaitable[None]],
    ) -> None:
        """Store a new PENDING proposal and notify the human approver."""
        if not approval_id:
            raise ApprovalError("approval_id must be non-empty")
        if not diff:
            raise ApprovalError("diff must be non-empty")

        existing = await self._store_get(self._key(approval_id))
        if existing is not None:
            raise ApprovalError(f"Proposal already exists: {approval_id}")

        now = datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(
            now.timestamp() + self._ttl_seconds, tz=timezone.utc
        )
        if expires_at <= now:
            raise ApprovalError("expires_at must be after created_at")

        request = ApprovalRequest(
            approval_id=approval_id,
            diff=diff,
            summary=summary,
            status=ApprovalStatus.PENDING,
            created_at=now,
            expires_at=expires_at,
        )

        self._callbacks[approval_id] = callback
        await self._store_set(self._key(approval_id), _serialize(request))

        message = f"New proposal [{approval_id}]: {summary}\n\nDiff:\n{diff}"
        await self.notify_human(approval_id, message)

    async def approve(self, approval_id: str) -> None:
        """Approve a PENDING proposal and invoke its callback."""
        raw = await self._store_get(self._key(approval_id))
        if raw is None:
            raise ApprovalError(f"No approval found for ID: {approval_id}")

        request = _deserialize(raw)
        if request.status != ApprovalStatus.PENDING:
            raise ApprovalError(
                f"Approval {approval_id} is not PENDING (status={request.status.value})"
            )

        request.status = ApprovalStatus.APPROVED
        await self._store_set(self._key(approval_id), _serialize(request))

        callback = self._callbacks.get(approval_id)
        if callback:
            await callback(True, "")

    async def reject(self, approval_id: str, reason: str) -> None:
        """Reject a PENDING proposal and invoke its callback."""
        raw = await self._store_get(self._key(approval_id))
        if raw is None:
            raise ApprovalError(f"No approval found for ID: {approval_id}")

        request = _deserialize(raw)
        if request.status != ApprovalStatus.PENDING:
            raise ApprovalError(
                f"Approval {approval_id} is not PENDING (status={request.status.value})"
            )

        request.status = ApprovalStatus.REJECTED
        request.rejection_reason = reason
        await self._store_set(self._key(approval_id), _serialize(request))

        callback = self._callbacks.get(approval_id)
        if callback:
            await callback(False, reason)

    async def get_pending(self) -> list[ApprovalRequest]:
        """Return all ApprovalRequests currently in PENDING status."""
        keys = await self._store_keys()
        pending = []
        for key in keys:
            raw = await self._store_get(key)
            if raw is None:
                continue
            request = _deserialize(raw)
            if request.status == ApprovalStatus.PENDING:
                pending.append(request)
        return pending

    async def expire_stale(self, ttl_seconds: int | None = None) -> None:
        """Expire any PENDING proposals whose TTL has elapsed."""
        keys = await self._store_keys()
        now = datetime.now(timezone.utc)
        for key in keys:
            raw = await self._store_get(key)
            if raw is None:
                continue
            request = _deserialize(raw)
            if request.status != ApprovalStatus.PENDING:
                continue
            expires_at = request.expires_at
            if expires_at is None:
                continue
            # Make expires_at timezone-aware if needed
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if now > expires_at:
                request.status = ApprovalStatus.EXPIRED
                await self._store_set(key, _serialize(request))
                callback = self._callbacks.get(request.approval_id)
                if callback:
                    await callback(False, "expired")

    async def notify_human(self, approval_id: str, message: str) -> None:
        """Send a notification via the configured notifier, or log if none."""
        if self._notifier is not None:
            await self._notifier(approval_id, message)
        else:
            logger.info("ApprovalGate notification [%s]: %s", approval_id, message)
