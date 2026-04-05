"""Unit tests for Dispatcher and SessionStore."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.bots.base import BaseBot
from backend.dispatcher.dispatcher import Dispatcher
from backend.dispatcher.session_store import SessionStore, _deserialize_context, _serialize_context
from backend.models import BotInfo, BotResponse, Message, SessionContext


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class StubBot(BaseBot):
    bot_id = "stub"
    description = "Stub bot"
    model = "llama3"

    def build_system_prompt(self) -> str:
        return "stub"

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        return BotResponse(reply=f"echo: {message}", bot_id=self.bot_id)


class AnotherBot(BaseBot):
    bot_id = "another"
    description = "Another bot"
    model = "mistral"

    def build_system_prompt(self) -> str:
        return "another"

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        return BotResponse(reply="another reply", bot_id=self.bot_id)


def make_stub_bot() -> StubBot:
    return StubBot(llm=AsyncMock())


def make_another_bot() -> AnotherBot:
    return AnotherBot(llm=AsyncMock())


def make_session_store(use_fallback: bool = True) -> SessionStore:
    """Create a SessionStore that always uses the in-memory fallback."""
    with patch("redis.asyncio.from_url", side_effect=Exception("no redis")):
        store = SessionStore(redis_url="redis://localhost:6379")
    store._use_fallback = use_fallback
    return store


def make_dispatcher(
    store: SessionStore | None = None,
    db: AsyncMock | None = None,
    max_history: int = 50,
) -> Dispatcher:
    if store is None:
        store = make_session_store()
    if db is None:
        db = AsyncMock()
        db.save_turn = AsyncMock()
    return Dispatcher(session_store=store, db=db, max_history=max_history)


# ---------------------------------------------------------------------------
# Dispatcher.register / get / list_bots
# ---------------------------------------------------------------------------

def test_register_adds_bot():
    d = make_dispatcher()
    bot = make_stub_bot()
    d.register("stub", bot)
    assert d.get("stub") is bot


def test_register_overwrites_existing_bot():
    d = make_dispatcher()
    bot1 = make_stub_bot()
    bot2 = make_stub_bot()
    d.register("stub", bot1)
    d.register("stub", bot2)
    assert d.get("stub") is bot2


def test_register_does_not_affect_other_entries():
    d = make_dispatcher()
    bot1 = make_stub_bot()
    bot2 = make_another_bot()
    d.register("stub", bot1)
    d.register("another", bot2)
    d.register("stub", make_stub_bot())  # overwrite stub
    assert d.get("another") is bot2


def test_list_bots_returns_correct_bot_info():
    d = make_dispatcher()
    d.register("stub", make_stub_bot())
    d.register("another", make_another_bot())

    infos = d.list_bots()
    assert len(infos) == 2

    by_id = {info.bot_id: info for info in infos}
    assert by_id["stub"] == BotInfo(bot_id="stub", description="Stub bot", model="llama3", available=True)
    assert by_id["another"] == BotInfo(bot_id="another", description="Another bot", model="mistral", available=True)


def test_list_bots_empty_registry():
    d = make_dispatcher()
    assert d.list_bots() == []


# ---------------------------------------------------------------------------
# Dispatcher.dispatch — unknown bot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_unknown_bot_returns_error_response():
    d = make_dispatcher()
    response = await d.dispatch("sess-1", "nonexistent", "hello")

    assert isinstance(response, BotResponse)
    assert response.error == "Unknown bot: nonexistent"
    assert response.reply == ""
    assert response.bot_id == "nonexistent"


@pytest.mark.asyncio
async def test_dispatch_unknown_bot_never_raises():
    d = make_dispatcher()
    # Should not raise under any circumstances
    result = await d.dispatch("sess-x", "ghost_bot", "test")
    assert isinstance(result, BotResponse)


# ---------------------------------------------------------------------------
# Dispatcher.dispatch — known bot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_known_bot_returns_bot_response():
    d = make_dispatcher()
    d.register("stub", make_stub_bot())

    response = await d.dispatch("sess-1", "stub", "hello")

    assert isinstance(response, BotResponse)
    assert response.reply == "echo: hello"
    assert response.bot_id == "stub"
    assert response.error is None


@pytest.mark.asyncio
async def test_dispatch_sets_latency_ms():
    d = make_dispatcher()
    d.register("stub", make_stub_bot())

    response = await d.dispatch("sess-1", "stub", "hi")
    assert response.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_dispatch_bot_exception_returns_error_response():
    class BrokenBot(BaseBot):
        bot_id = "broken"
        description = "Broken"
        model = "llama3"

        def build_system_prompt(self) -> str:
            return ""

        async def handle(self, context: SessionContext, message: str) -> BotResponse:
            raise RuntimeError("something went wrong")

    d = make_dispatcher()
    d.register("broken", BrokenBot(llm=AsyncMock()))

    response = await d.dispatch("sess-1", "broken", "hi")
    assert response.error == "something went wrong"
    assert response.reply == ""


@pytest.mark.asyncio
async def test_dispatch_never_raises_on_bot_exception():
    class BrokenBot(BaseBot):
        bot_id = "broken"
        description = "Broken"
        model = "llama3"

        def build_system_prompt(self) -> str:
            return ""

        async def handle(self, context: SessionContext, message: str) -> BotResponse:
            raise ValueError("boom")

    d = make_dispatcher()
    d.register("broken", BrokenBot(llm=AsyncMock()))

    # Must not raise
    result = await d.dispatch("sess-1", "broken", "test")
    assert isinstance(result, BotResponse)


# ---------------------------------------------------------------------------
# Dispatcher.dispatch — history window
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_history_window_enforced():
    """After many dispatches, history must never exceed max_history."""
    max_history = 5
    d = make_dispatcher(max_history=max_history)
    d.register("stub", make_stub_bot())

    for i in range(20):
        await d.dispatch("sess-win", "stub", f"message {i}")

    context = await d._session_store.get("sess-win")
    assert context is not None
    assert len(context.history) <= max_history


@pytest.mark.asyncio
async def test_dispatch_history_window_keeps_most_recent():
    """The sliding window should keep the most recent messages."""
    max_history = 4
    d = make_dispatcher(max_history=max_history)
    d.register("stub", make_stub_bot())

    for i in range(6):
        await d.dispatch("sess-slide", "stub", f"msg{i}")

    context = await d._session_store.get("sess-slide")
    assert context is not None
    # The last user message in history should be the most recent one
    user_msgs = [m for m in context.history if m.role == "user"]
    assert user_msgs[-1].content == "msg5"


# ---------------------------------------------------------------------------
# Dispatcher.dispatch — DB save_turn best-effort
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_db_failure_does_not_fail_request():
    db = AsyncMock()
    db.save_turn = AsyncMock(side_effect=Exception("DB down"))

    d = make_dispatcher(db=db)
    d.register("stub", make_stub_bot())

    # Should not raise even when DB fails
    response = await d.dispatch("sess-1", "stub", "hello")
    assert isinstance(response, BotResponse)
    assert response.error is None


# ---------------------------------------------------------------------------
# SessionStore — in-memory fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_store_fallback_get_returns_none_for_missing():
    store = make_session_store()
    result = await store.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_session_store_fallback_set_and_get_roundtrip():
    store = make_session_store()
    ctx = SessionContext(session_id="s1", bot_id="stub")
    ctx.history.append(Message(role="user", content="hello"))

    await store.set("s1", ctx)
    retrieved = await store.get("s1")

    assert retrieved is not None
    assert retrieved.session_id == "s1"
    assert retrieved.bot_id == "stub"
    assert len(retrieved.history) == 1
    assert retrieved.history[0].content == "hello"


@pytest.mark.asyncio
async def test_session_store_redis_unavailable_falls_back_to_memory():
    """When Redis is unavailable, SessionStore uses in-memory dict."""
    with patch("redis.asyncio.from_url", side_effect=Exception("connection refused")):
        store = SessionStore(redis_url="redis://localhost:9999")

    assert store._use_fallback is True

    ctx = SessionContext(session_id="fallback-sess", bot_id="stub")
    await store.set("fallback-sess", ctx)
    result = await store.get("fallback-sess")

    assert result is not None
    assert result.session_id == "fallback-sess"


@pytest.mark.asyncio
async def test_session_store_fallback_preserves_datetime_fields():
    store = make_session_store()
    ctx = SessionContext(session_id="s2", bot_id="stub")
    original_created = ctx.created_at

    await store.set("s2", ctx)
    retrieved = await store.get("s2")

    assert retrieved is not None
    assert retrieved.created_at.isoformat() == original_created.isoformat()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def test_serialize_deserialize_roundtrip():
    ctx = SessionContext(session_id="s3", bot_id="stub")
    ctx.history.append(Message(role="user", content="hi"))
    ctx.history.append(Message(role="assistant", content="hello"))

    raw = _serialize_context(ctx)
    restored = _deserialize_context(raw)

    assert restored.session_id == ctx.session_id
    assert restored.bot_id == ctx.bot_id
    assert len(restored.history) == 2
    assert restored.history[0].role == "user"
    assert restored.history[1].content == "hello"


def test_serialize_deserialize_empty_history():
    ctx = SessionContext(session_id="s4", bot_id="stub")
    raw = _serialize_context(ctx)
    restored = _deserialize_context(raw)
    assert restored.history == []
