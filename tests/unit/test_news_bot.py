"""Unit tests for NewsAggregatorBot."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.bots.news_bot import NewsAggregatorBot, _classify_category, _is_valid_url
from backend.llm.client import LLMClient
from backend.models import BotResponse, NewsItem, NewsSummary, ResearchReport, SessionContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm(reply: str = '{"digest":"test digest","market_impact":"neutral","key_insights":[],"topics_for_research":[]}') -> LLMClient:
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(return_value=reply)
    return llm


def make_bot(llm=None, telegram_handler=None) -> NewsAggregatorBot:
    return NewsAggregatorBot(llm=llm or make_llm(), telegram_handler=telegram_handler)


def make_feed_entry(title: str, link: str, summary: str = "") -> SimpleNamespace:
    entry = SimpleNamespace()
    entry.title = title
    entry.link = link
    entry.summary = summary
    entry.published_parsed = None
    return entry


def make_feed(entries: list) -> SimpleNamespace:
    feed = SimpleNamespace()
    feed.entries = entries
    return feed


def make_news_item(title="Market rises", url="https://example.com/1") -> NewsItem:
    return NewsItem(
        title=title,
        source="https://example.com",
        url=url,
        published_at=datetime.utcnow(),
        category="market",
        raw_text=title,
    )


# ---------------------------------------------------------------------------
# fetch_headlines — deduplication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_headlines_deduplicates_by_url():
    bot = make_bot()
    bot._sources = ["https://source1.com", "https://source2.com"]

    entry = make_feed_entry("Stocks rally", "https://example.com/story1")
    feed = make_feed([entry, entry])  # same entry twice in same feed

    with patch("backend.bots.news_bot.feedparser.parse", return_value=feed):
        items = await bot.fetch_headlines()

    urls = [item.url for item in items]
    assert len(urls) == len(set(urls)), "Duplicate URLs found"


@pytest.mark.asyncio
async def test_fetch_headlines_deduplicates_across_sources():
    bot = make_bot()
    bot._sources = ["https://source1.com", "https://source2.com"]

    shared_entry = make_feed_entry("Shared story", "https://example.com/shared")
    feed = make_feed([shared_entry])

    with patch("backend.bots.news_bot.feedparser.parse", return_value=feed):
        items = await bot.fetch_headlines()

    urls = [item.url for item in items]
    assert urls.count("https://example.com/shared") == 1


# ---------------------------------------------------------------------------
# fetch_headlines — source failure handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_headlines_handles_partial_source_failure():
    bot = make_bot()
    bot._sources = ["https://good.com", "https://bad.com"]

    good_entry = make_feed_entry("Good news", "https://good.com/story")
    good_feed = make_feed([good_entry])

    def side_effect(url):
        if "bad" in url:
            raise ConnectionError("Network error")
        return good_feed

    with patch("backend.bots.news_bot.feedparser.parse", side_effect=side_effect):
        items = await bot.fetch_headlines()

    assert len(items) == 1
    assert items[0].url == "https://good.com/story"


@pytest.mark.asyncio
async def test_fetch_headlines_returns_empty_when_all_sources_fail():
    bot = make_bot()
    bot._sources = ["https://bad1.com", "https://bad2.com"]

    with patch("backend.bots.news_bot.feedparser.parse", side_effect=ConnectionError("fail")):
        items = await bot.fetch_headlines()

    assert items == []


@pytest.mark.asyncio
async def test_fetch_headlines_skips_entries_with_empty_title():
    bot = make_bot()
    bot._sources = ["https://source.com"]

    entries = [
        make_feed_entry("", "https://example.com/no-title"),
        make_feed_entry("Valid title", "https://example.com/valid"),
    ]
    with patch("backend.bots.news_bot.feedparser.parse", return_value=make_feed(entries)):
        items = await bot.fetch_headlines()

    assert len(items) == 1
    assert items[0].title == "Valid title"


@pytest.mark.asyncio
async def test_fetch_headlines_skips_entries_with_invalid_url():
    bot = make_bot()
    bot._sources = ["https://source.com"]

    entries = [
        make_feed_entry("Bad URL story", "not-a-url"),
        make_feed_entry("Good URL story", "https://example.com/good"),
    ]
    with patch("backend.bots.news_bot.feedparser.parse", return_value=make_feed(entries)):
        items = await bot.fetch_headlines()

    assert len(items) == 1
    assert items[0].url == "https://example.com/good"


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarise_returns_news_summary_for_valid_items():
    llm_reply = '{"digest":"Markets up","market_impact":"bullish","key_insights":["Tech leads"],"topics_for_research":["AI regulation"]}'
    bot = make_bot(llm=make_llm(llm_reply))
    items = [make_news_item()]

    result = await bot.summarise(items)

    assert isinstance(result, NewsSummary)
    assert result.digest == "Markets up"
    assert result.market_impact == "bullish"
    assert result.key_insights == ["Tech leads"]
    assert result.topics_for_research == ["AI regulation"]


@pytest.mark.asyncio
async def test_summarise_returns_news_summary_for_empty_items():
    bot = make_bot()
    result = await bot.summarise([])

    assert isinstance(result, NewsSummary)
    assert result.market_impact == "neutral"


@pytest.mark.asyncio
async def test_summarise_falls_back_on_invalid_json():
    bot = make_bot(llm=make_llm("This is not JSON at all"))
    items = [make_news_item()]

    result = await bot.summarise(items)

    assert isinstance(result, NewsSummary)
    assert result.market_impact == "neutral"


@pytest.mark.asyncio
async def test_summarise_never_raises():
    llm = AsyncMock(spec=LLMClient)
    llm.complete = AsyncMock(side_effect=Exception("LLM exploded"))
    bot = make_bot(llm=llm)
    items = [make_news_item()]

    result = await bot.summarise(items)

    assert isinstance(result, NewsSummary)


# ---------------------------------------------------------------------------
# run_scheduled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_scheduled_posts_to_telegram_and_injects_trading_context():
    telegram = AsyncMock()
    telegram.send_message = AsyncMock()
    bot = make_bot(
        llm=make_llm('{"digest":"digest","market_impact":"neutral","key_insights":["insight1"],"topics_for_research":[]}'),
        telegram_handler=telegram,
    )

    trading_bot = MagicMock()
    trading_bot.inject_news_context = MagicMock()
    bot.set_trading_bot(trading_bot)

    items = [make_news_item()]
    with patch.object(bot, "fetch_headlines", AsyncMock(return_value=items)):
        await bot.run_scheduled()

    telegram.send_message.assert_awaited_once()
    trading_bot.inject_news_context.assert_called_once_with(["insight1"])


@pytest.mark.asyncio
async def test_run_scheduled_does_nothing_when_no_headlines():
    telegram = AsyncMock()
    bot = make_bot(telegram_handler=telegram)

    with patch.object(bot, "fetch_headlines", AsyncMock(return_value=[])):
        await bot.run_scheduled()

    telegram.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_scheduled_requests_deep_research_for_topics():
    bot = make_bot(
        llm=make_llm('{"digest":"d","market_impact":"neutral","key_insights":[],"topics_for_research":["AI regulation"]}'),
    )
    research_bot = AsyncMock()
    research_bot.handle = AsyncMock(return_value=BotResponse(reply="Research report", bot_id="research"))
    bot.set_research_bot(research_bot)

    items = [make_news_item()]
    with patch.object(bot, "fetch_headlines", AsyncMock(return_value=items)):
        with patch.object(bot, "post_to_telegram", AsyncMock()) as mock_post:
            await bot.run_scheduled()

    research_bot.handle.assert_awaited_once()
    # post_to_telegram called twice: once for main summary, once for research report
    assert mock_post.await_count == 2


# ---------------------------------------------------------------------------
# handle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_returns_bot_response():
    bot = make_bot()
    context = SessionContext(session_id="s1", bot_id="news")

    items = [make_news_item()]
    with patch.object(bot, "fetch_headlines", AsyncMock(return_value=items)):
        response = await bot.handle(context, "latest news")

    assert isinstance(response, BotResponse)
    assert response.bot_id == "news"
    assert response.reply


@pytest.mark.asyncio
async def test_handle_returns_bot_response_when_no_headlines():
    bot = make_bot()
    context = SessionContext(session_id="s1", bot_id="news")

    with patch.object(bot, "fetch_headlines", AsyncMock(return_value=[])):
        response = await bot.handle(context, "latest news")

    assert isinstance(response, BotResponse)
    assert response.bot_id == "news"
    assert "No news" in response.reply


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_is_valid_url_accepts_http():
    assert _is_valid_url("http://example.com/path") is True


def test_is_valid_url_accepts_https():
    assert _is_valid_url("https://example.com/path") is True


def test_is_valid_url_rejects_plain_string():
    assert _is_valid_url("not-a-url") is False


def test_is_valid_url_rejects_ftp():
    assert _is_valid_url("ftp://example.com") is False


def test_classify_category_market():
    assert _classify_category("S&P 500 hits record high") == "market"


def test_classify_category_financial():
    assert _classify_category("Company reports record earnings") == "financial"


def test_classify_category_geopolitical():
    assert _classify_category("New trade war sanctions imposed") == "geopolitical"


def test_classify_category_macro():
    assert _classify_category("Fed raises interest rate by 25bps") == "macro"


def test_classify_category_defaults_to_financial():
    assert _classify_category("Something completely unrelated") == "financial"
