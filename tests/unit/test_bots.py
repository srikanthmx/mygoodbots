"""Unit tests for GenAIBot, ResearchBot, TradingBot, and CodingBot."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.bots.coding_bot import CodingBot
from backend.bots.genai_bot import GenAIBot
from backend.bots.research_bot import ResearchBot
from backend.bots.trading_bot import RISK_DISCLAIMER, TradingBot
from backend.llm.client import LLMClient
from backend.models import BotResponse, SessionContext


def make_context() -> SessionContext:
    return SessionContext(session_id="test-session", bot_id="test", history=[])


def make_llm(reply: str = "some response") -> LLMClient:
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value=reply)
    return llm


# ---------------------------------------------------------------------------
# GenAIBot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_genai_bot_returns_bot_response():
    bot = GenAIBot(llm=make_llm("A vivid creative reply"))
    response = await bot.handle(make_context(), "write a poem")
    assert isinstance(response, BotResponse)
    assert response.bot_id == "genai"
    assert response.reply == "A vivid creative reply"


# ---------------------------------------------------------------------------
# ResearchBot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_research_bot_returns_bot_response():
    bot = ResearchBot(llm=make_llm("Research findings here"))
    response = await bot.handle(make_context(), "research quantum computing")
    assert isinstance(response, BotResponse)
    assert response.bot_id == "research"
    assert response.reply == "Research findings here"


# ---------------------------------------------------------------------------
# TradingBot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trading_bot_reply_always_contains_risk_disclaimer():
    bot = TradingBot(llm=make_llm("BTC looks bullish today."))
    response = await bot.handle(make_context(), "analyse BTC")
    assert RISK_DISCLAIMER in response.reply


@pytest.mark.asyncio
async def test_trading_bot_does_not_duplicate_disclaimer_if_already_present():
    bot = TradingBot(llm=make_llm("Analysis." + RISK_DISCLAIMER))
    response = await bot.handle(make_context(), "analyse ETH")
    assert response.reply.count(RISK_DISCLAIMER) == 1


def test_trading_bot_inject_news_context_stores_insights():
    bot = TradingBot(llm=make_llm())
    bot.inject_news_context(["Fed raises rates", "Oil prices drop"])
    assert "Fed raises rates" in bot._news_context
    assert "Oil prices drop" in bot._news_context


def test_trading_bot_inject_news_context_appends_multiple_calls():
    bot = TradingBot(llm=make_llm())
    bot.inject_news_context(["insight A"])
    bot.inject_news_context(["insight B"])
    assert bot._news_context == ["insight A", "insight B"]


def test_trading_bot_system_prompt_includes_news_when_present():
    bot = TradingBot(llm=make_llm())
    bot.inject_news_context(["Gold surges"])
    prompt = bot.build_system_prompt()
    assert "Gold surges" in prompt


def test_trading_bot_system_prompt_excludes_news_section_when_empty():
    bot = TradingBot(llm=make_llm())
    prompt = bot.build_system_prompt()
    assert "Recent market news context" not in prompt


# ---------------------------------------------------------------------------
# CodingBot
# ---------------------------------------------------------------------------

def make_approval_gate() -> MagicMock:
    gate = AsyncMock()
    gate.submit_proposal = AsyncMock()
    gate.notify_human = AsyncMock()
    return gate


@pytest.mark.asyncio
async def test_coding_bot_returns_no_changes_when_diff_is_empty():
    # LLM returns a response with no DIFF/SUMMARY markers → diff will be empty
    llm = make_llm("No changes are required for this request.")
    bot = CodingBot(llm=llm, approval_gate=make_approval_gate())
    response = await bot.handle(make_context(), "review my code")
    assert response.reply == "No changes needed."
    assert response.bot_id == "coding"


@pytest.mark.asyncio
async def test_coding_bot_submits_to_approval_gate_when_diff_non_empty():
    llm = make_llm("DIFF:\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n\nSUMMARY:\nRename variable.")
    gate = make_approval_gate()
    bot = CodingBot(llm=llm, approval_gate=gate)
    response = await bot.handle(make_context(), "rename variable x to y")
    gate.submit_proposal.assert_awaited_once()
    assert "submitted for approval" in response.reply
    assert response.bot_id == "coding"


@pytest.mark.asyncio
async def test_coding_bot_never_calls_deployer_before_approval():
    llm = make_llm("DIFF:\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n\nSUMMARY:\nFix typo.")
    gate = make_approval_gate()
    deployer = AsyncMock()
    bot = CodingBot(llm=llm, approval_gate=gate)
    bot._deployer = deployer

    await bot.handle(make_context(), "fix typo")

    # deployer.apply must NOT have been called — approval hasn't happened yet
    deployer.apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_coding_bot_calls_deployer_after_approval():
    llm = make_llm("DIFF:\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n\nSUMMARY:\nFix typo.")
    gate = make_approval_gate()
    deployer = AsyncMock()

    captured_callback = None

    async def capture_submit(approval_id, diff, summary, callback):
        nonlocal captured_callback
        captured_callback = callback

    gate.submit_proposal.side_effect = capture_submit

    bot = CodingBot(llm=llm, approval_gate=gate)
    bot._deployer = deployer

    await bot.handle(make_context(), "fix typo")

    assert captured_callback is not None
    await captured_callback(True, "")
    deployer.apply.assert_awaited_once()
