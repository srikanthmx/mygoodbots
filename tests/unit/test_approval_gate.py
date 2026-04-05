"""Unit tests for ApprovalGate (in-memory fallback, no Redis required)."""
from __future__ import annotations

import pytest

from backend.approval.gate import ApprovalGate
from backend.models import ApprovalError, ApprovalRequest, ApprovalStatus
from datetime import datetime, timezone, timedelta


def make_gate(**kwargs) -> ApprovalGate:
    """Create an ApprovalGate that always uses the in-memory store."""
    gate = ApprovalGate(redis_url="redis://localhost:6379", **kwargs)
    gate._redis = None  # force in-memory fallback
    return gate


async def noop_callback(approved: bool, reason: str) -> None:
    pass


# ---------------------------------------------------------------------------
# submit_proposal
# ---------------------------------------------------------------------------


async def test_submit_proposal_stores_pending_request():
    gate = make_gate()
    await gate.submit_proposal("id-1", "diff content", "summary text", noop_callback)

    pending = await gate.get_pending()
    assert len(pending) == 1
    req = pending[0]
    assert req.approval_id == "id-1"
    assert req.status == ApprovalStatus.PENDING
    assert req.diff == "diff content"
    assert req.summary == "summary text"


async def test_submit_proposal_raises_on_empty_approval_id():
    gate = make_gate()
    with pytest.raises(ApprovalError):
        await gate.submit_proposal("", "diff", "summary", noop_callback)


async def test_submit_proposal_raises_on_empty_diff():
    gate = make_gate()
    with pytest.raises(ApprovalError):
        await gate.submit_proposal("id-1", "", "summary", noop_callback)


async def test_submit_proposal_raises_on_duplicate_id():
    gate = make_gate()
    await gate.submit_proposal("id-dup", "diff", "summary", noop_callback)
    with pytest.raises(ApprovalError):
        await gate.submit_proposal("id-dup", "diff2", "summary2", noop_callback)


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------


async def test_approve_triggers_callback_with_true():
    gate = make_gate()
    results: list[tuple[bool, str]] = []

    async def cb(approved: bool, reason: str) -> None:
        results.append((approved, reason))

    await gate.submit_proposal("id-approve", "diff", "summary", cb)
    await gate.approve("id-approve")

    assert results == [(True, "")]


async def test_approve_non_existent_raises_approval_error():
    gate = make_gate()
    with pytest.raises(ApprovalError):
        await gate.approve("does-not-exist")


async def test_approve_already_approved_raises_approval_error():
    gate = make_gate()
    await gate.submit_proposal("id-double", "diff", "summary", noop_callback)
    await gate.approve("id-double")
    with pytest.raises(ApprovalError):
        await gate.approve("id-double")


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


async def test_reject_triggers_callback_with_false_and_reason():
    gate = make_gate()
    results: list[tuple[bool, str]] = []

    async def cb(approved: bool, reason: str) -> None:
        results.append((approved, reason))

    await gate.submit_proposal("id-reject", "diff", "summary", cb)
    await gate.reject("id-reject", "not good enough")

    assert results == [(False, "not good enough")]


async def test_reject_non_existent_raises_approval_error():
    gate = make_gate()
    with pytest.raises(ApprovalError):
        await gate.reject("does-not-exist", "reason")


async def test_reject_already_rejected_raises_approval_error():
    gate = make_gate()
    await gate.submit_proposal("id-rej2", "diff", "summary", noop_callback)
    await gate.reject("id-rej2", "first rejection")
    with pytest.raises(ApprovalError):
        await gate.reject("id-rej2", "second rejection")


# ---------------------------------------------------------------------------
# expire_stale
# ---------------------------------------------------------------------------


async def test_expire_stale_sets_expired_and_triggers_callback():
    gate = make_gate()
    results: list[tuple[bool, str]] = []

    async def cb(approved: bool, reason: str) -> None:
        results.append((approved, reason))

    await gate.submit_proposal("id-expire", "diff", "summary", cb)

    # Manually backdate expires_at so it appears stale
    key = gate._key("id-expire")
    from backend.approval.gate import _deserialize, _serialize

    raw = gate._memory_store[key]
    req = _deserialize(raw)
    req.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    gate._memory_store[key] = _serialize(req)

    await gate.expire_stale()

    assert results == [(False, "expired")]

    # Verify status updated in store
    raw2 = gate._memory_store[key]
    req2 = _deserialize(raw2)
    assert req2.status == ApprovalStatus.EXPIRED


# ---------------------------------------------------------------------------
# get_pending
# ---------------------------------------------------------------------------


async def test_get_pending_returns_only_pending_requests():
    gate = make_gate()

    await gate.submit_proposal("p1", "diff1", "s1", noop_callback)
    await gate.submit_proposal("p2", "diff2", "s2", noop_callback)
    await gate.submit_proposal("p3", "diff3", "s3", noop_callback)

    await gate.approve("p2")
    await gate.reject("p3", "no")

    pending = await gate.get_pending()
    assert len(pending) == 1
    assert pending[0].approval_id == "p1"
