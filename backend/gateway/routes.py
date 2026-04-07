"""FastAPI routes for the multi-bot AI gateway."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.gateway.auth import verify_token
from backend.gateway.sanitizer import sanitize, validate_length
from backend.models import ApprovalError, LLMError

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level state — injected at app startup via configure_router()
# ---------------------------------------------------------------------------
_dispatcher: Any = None
_approval_gate: Any = None
_rate_limiter: Any = None
_telegram_handler: Any = None
_redis_client: Any = None


def configure_router(
    dispatcher: Any,
    approval_gate: Any,
    rate_limiter: Any,
    telegram_handler: Any = None,
    redis_client: Any = None,
) -> None:
    """Wire dependencies into the router at application startup."""
    global _dispatcher, _approval_gate, _rate_limiter, _telegram_handler, _redis_client
    _dispatcher = dispatcher
    _approval_gate = approval_gate
    _rate_limiter = rate_limiter
    _telegram_handler = telegram_handler
    _redis_client = redis_client


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str
    bot_id: str
    message: str
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    bot_id: str
    reply: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


class RejectRequest(BaseModel):
    reason: str = "Rejected by user"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict:
    """Liveness probe — no auth required."""
    return {"status": "ok"}


@router.get("/api/v1/bots")
async def list_bots(payload: dict = Depends(verify_token)) -> list[dict]:
    """Return all registered bots."""
    bots = _dispatcher.list_bots()
    return [
        {
            "bot_id": b.bot_id,
            "description": b.description,
            "model": b.model,
            "available": b.available,
        }
        for b in bots
    ]


async def _sse_generator(session_id: str, bot_id: str, message: str) -> AsyncIterator[str]:
    """Yield SSE-formatted chunks from the dispatcher."""
    try:
        response = await _dispatcher.dispatch(session_id, bot_id, message)
        if response.error:
            data = json.dumps({"error": response.error})
            yield f"data: {data}\n\n"
        else:
            # Stream reply word-by-word as a simple simulation;
            # real streaming would come from LLMClient stream=True
            data = json.dumps(
                {
                    "session_id": session_id,
                    "bot_id": bot_id,
                    "reply": response.reply,
                    "tokens_used": response.tokens_used,
                    "latency_ms": response.latency_ms,
                }
            )
            yield f"data: {data}\n\n"
    except LLMError as exc:
        data = json.dumps({"error": str(exc)})
        yield f"data: {data}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@router.post("/api/v1/chat")
async def chat(
    request: ChatRequest,
    payload: dict = Depends(verify_token),
) -> Any:
    """Dispatch a chat message to the appropriate bot."""
    # Rate limit
    allowed, retry_after = await _rate_limiter.check(request.session_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )

    # Validate and sanitize
    validate_length(request.message)
    clean_message = sanitize(request.message)

    # SSE streaming
    if request.stream:
        return StreamingResponse(
            _sse_generator(request.session_id, request.bot_id, clean_message),
            media_type="text/event-stream",
        )

    # Standard JSON response
    try:
        response = await _dispatcher.dispatch(
            request.session_id, request.bot_id, clean_message
        )
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "30"},
        ) from exc

    return ChatResponse(
        session_id=request.session_id,
        bot_id=request.bot_id,
        reply=response.reply,
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
        error=response.error,
    )


@router.get("/api/v1/approvals/pending")
async def get_pending_approvals(payload: dict = Depends(verify_token)) -> list[dict]:
    """Return all pending approval requests."""
    pending = await _approval_gate.get_pending()
    return [
        {
            "approval_id": r.approval_id,
            "diff": r.diff,
            "summary": r.summary,
            "status": r.status.value,
            "created_at": r.created_at.isoformat(),
            "expires_at": r.expires_at.isoformat() if r.expires_at else None,
        }
        for r in pending
    ]


@router.post("/api/v1/approvals/{approval_id}/approve")
async def approve_proposal(
    approval_id: str,
    payload: dict = Depends(verify_token),
) -> dict:
    """Approve a pending code-change proposal."""
    try:
        await _approval_gate.approve(approval_id)
    except ApprovalError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "approved", "approval_id": approval_id}


@router.post("/api/v1/approvals/{approval_id}/reject")
async def reject_proposal(
    approval_id: str,
    body: RejectRequest,
    payload: dict = Depends(verify_token),
) -> dict:
    """Reject a pending code-change proposal."""
    try:
        await _approval_gate.reject(approval_id, body.reason)
    except ApprovalError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "rejected", "approval_id": approval_id}


# ---------------------------------------------------------------------------
# Telegram Webhook
# ---------------------------------------------------------------------------

_TELEGRAM_WEBHOOK_SECRET_ENV = "TELEGRAM_WEBHOOK_SECRET"
_TELEGRAM_UPDATE_TTL = 86400  # 24 hours


@router.post("/webhook/telegram")
async def telegram_webhook(request: Request) -> dict:
    """Receive and process Telegram webhook updates."""
    # 1. Validate secret token header
    expected_secret = os.environ.get(_TELEGRAM_WEBHOOK_SECRET_ENV, "")
    incoming_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not expected_secret or incoming_secret != expected_secret:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    # 2. Parse JSON body
    body = await request.json()

    # 3. Deduplicate by update_id using Redis
    update_id = body.get("update_id")
    if update_id is not None and _redis_client is not None:
        redis_key = f"tg_update:{update_id}"
        try:
            already_seen = await _redis_client.get(redis_key)
            if already_seen:
                return {"ok": True}
            await _redis_client.set(redis_key, "1", ex=_TELEGRAM_UPDATE_TTL)
        except Exception as exc:
            logger.warning("Redis deduplication failed (non-fatal): %s", exc)

    # 4. Handle callback_query (approve/reject button taps)
    if "callback_query" in body:
        await _handle_callback_query(body["callback_query"])
        return {"ok": True}

    # 5. Parse regular message update
    if _telegram_handler is None:
        logger.warning("telegram_handler not configured; skipping update")
        return {"ok": True}

    try:
        tg_message = _telegram_handler.parse_update(body)
    except (KeyError, TypeError) as exc:
        logger.warning("Failed to parse Telegram update: %s", exc)
        return {"ok": True}

    # 6. Dispatch to the appropriate bot
    session_id = f"tg:{tg_message.chat_id}"
    # Store sender_id in Redis so CronBot can verify ownership
    if _redis_client and tg_message.sender_id:
        try:
            await _redis_client.set(f"sender:{session_id}", str(tg_message.sender_id), ex=300)
        except Exception:
            pass
    try:
        response = await _dispatcher.dispatch(
            session_id=session_id,
            bot_id=tg_message.bot_id,
            message=tg_message.text,
            sender_id=tg_message.sender_id,
        )
    except Exception as exc:
        logger.error("Dispatch error for Telegram update: %s", exc)
        return {"ok": True}

    # 7. Send reply
    reply_text = response.reply if not response.error else f"Error: {response.error}"
    try:
        await _telegram_handler.send_message(tg_message.chat_id, reply_text)
    except Exception as exc:
        logger.error("Failed to send Telegram reply: %s", exc)

    return {"ok": True}


async def _handle_callback_query(callback_query: dict) -> None:
    """Process approve/reject button taps from Telegram inline keyboards."""
    if _approval_gate is None or _telegram_handler is None:
        return

    data: str = callback_query.get("data", "")
    chat_id: int = callback_query.get("message", {}).get("chat", {}).get("id", 0)
    sender_id: int = callback_query.get("from", {}).get("id", 0)

    try:
        if data.startswith("approve:"):
            approval_id = data.split(":", 1)[1]
            await _approval_gate.approve(approval_id)
            await _telegram_handler.send_message(chat_id, f"✅ Approved `{approval_id}` — deploying...")
        elif data.startswith("reject:"):
            approval_id = data.split(":", 1)[1]
            await _approval_gate.reject(approval_id, "Rejected via Telegram")
            await _telegram_handler.send_message(chat_id, f"❌ Rejected `{approval_id}`.")
    except Exception as exc:
        logger.error("Error handling callback_query %r: %s", data, exc)
        try:
            await _telegram_handler.send_message(chat_id, f"Error processing decision: {exc}")
        except Exception:
            pass
