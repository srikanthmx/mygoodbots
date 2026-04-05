"""Unit tests for BaseBot."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.bots.base import BaseBot
from backend.llm.client import LLMClient
from backend.models import BotResponse, Message, SessionContext


# ---------------------------------------------------------------------------
# Concrete stub for testing the abstract class
# ---------------------------------------------------------------------------

class StubBot(BaseBot):
    bot_id = "stub"
    description = "A stub bot for testing"
    model = "llama3"
    temperature = 0.7

    def build_system_prompt(self) -> str:
        return "You are a stub bot."

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        return await super().handle(context, message)


def make_bot(llm: LLMClient | None = None) -> StubBot:
    if llm is None:
        llm = AsyncMock(spec=LLMClient)
    return StubBot(llm=llm)


def make_context(history: list[Message] | None = None) -> SessionContext:
    return SessionContext(
        session_id="sess-test",
        bot_id="stub",
        history=history or [],
    )


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def test_preprocess_strips_leading_whitespace():
    bot = make_bot()
    assert bot.preprocess("  hello") == "hello"


def test_preprocess_strips_trailing_whitespace():
    bot = make_bot()
    assert bot.preprocess("hello   ") == "hello"


def test_preprocess_strips_both_ends():
    bot = make_bot()
    assert bot.preprocess("  hello world  ") == "hello world"


def test_preprocess_empty_string():
    bot = make_bot()
    assert bot.preprocess("") == ""


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------

def test_postprocess_strips_leading_whitespace():
    bot = make_bot()
    assert bot.postprocess("  reply") == "reply"


def test_postprocess_strips_trailing_whitespace():
    bot = make_bot()
    assert bot.postprocess("reply\n\n") == "reply"


def test_postprocess_strips_both_ends():
    bot = make_bot()
    assert bot.postprocess("\n  reply text  \n") == "reply text"


def test_postprocess_empty_string():
    bot = make_bot()
    assert bot.postprocess("") == ""


# ---------------------------------------------------------------------------
# handle: returns BotResponse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_returns_bot_response():
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="  LLM reply  ")
    bot = make_bot(llm)

    context = make_context()
    response = await bot.handle(context, "hello")

    assert isinstance(response, BotResponse)
    assert response.bot_id == "stub"
    assert response.reply == "LLM reply"  # postprocessed (stripped)


@pytest.mark.asyncio
async def test_handle_does_not_mutate_context():
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="answer")
    bot = make_bot(llm)

    history_before = [Message(role="user", content="prior message")]
    context = make_context(history=list(history_before))
    original_len = len(context.history)

    await bot.handle(context, "new message")

    # context must not have been mutated
    assert len(context.history) == original_len
    assert context.history[0].content == "prior message"


# ---------------------------------------------------------------------------
# handle: builds correct message list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_builds_messages_with_system_prompt():
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="ok")
    bot = make_bot(llm)

    context = make_context()
    await bot.handle(context, "hi")

    call_args = llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[1]

    assert messages[0] == {"role": "system", "content": "You are a stub bot."}


@pytest.mark.asyncio
async def test_handle_excludes_last_history_entry_and_appends_current_message():
    """
    Dispatcher appends the user message to history before calling handle.
    handle should exclude that last entry and append the current message itself.
    """
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="ok")
    bot = make_bot(llm)

    # Simulate dispatcher having appended the user message already
    history = [
        Message(role="user", content="first turn"),
        Message(role="assistant", content="first reply"),
        Message(role="user", content="current message"),  # last entry — should be excluded
    ]
    context = make_context(history=history)

    await bot.handle(context, "current message")

    call_args = llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[1]

    # system + first turn + first reply + current message (re-appended)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "first turn"}
    assert messages[2] == {"role": "assistant", "content": "first reply"}
    assert messages[3] == {"role": "user", "content": "current message"}


@pytest.mark.asyncio
async def test_handle_empty_history_sends_only_system_and_user():
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="response")
    bot = make_bot(llm)

    context = make_context(history=[])
    await bot.handle(context, "hello")

    call_args = llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[1]

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_handle_passes_model_and_temperature_to_llm():
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value="ok")
    bot = make_bot(llm)
    bot.model = "mistral"
    bot.temperature = 0.3

    context = make_context()
    await bot.handle(context, "test")

    call_args = llm.complete.call_args
    assert (call_args.kwargs.get("model") or call_args.args[0]) == "mistral"
    assert call_args.kwargs.get("temperature") == 0.3
