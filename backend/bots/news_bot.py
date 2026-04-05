"""NewsAggregatorBot — financial & market news aggregator."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from urllib.parse import urlparse

import feedparser

from backend.bots.base import BaseBot
from backend.models import (
    BotResponse,
    NewsItem,
    NewsSummary,
    ResearchReport,
    SessionContext,
)

logger = logging.getLogger(__name__)

NEWS_SOURCES = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bloomberg.com/markets/news.rss",
]

VALID_CATEGORIES = {"financial", "market", "geopolitical", "macro"}

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "financial": ["stock", "shares", "earnings", "profit", "revenue", "ipo", "dividend", "bond", "debt", "finance", "financial"],
    "market": ["market", "trading", "index", "nasdaq", "s&p", "dow", "futures", "commodity", "crypto", "bitcoin", "forex"],
    "geopolitical": ["war", "sanction", "election", "government", "policy", "geopolit", "conflict", "treaty", "tariff", "trade war"],
    "macro": ["inflation", "gdp", "interest rate", "fed", "central bank", "recession", "unemployment", "cpi", "ppi", "macro"],
}


def _classify_category(title: str) -> str:
    lower = title.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "financial"  # default


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _parse_published(entry) -> datetime:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6])
        except Exception:
            pass
    return datetime.utcnow()


class NewsAggregatorBot(BaseBot):
    bot_id = "news"
    description = "Financial & market news aggregator — scheduled and on-demand"
    model = "llama-3.1-8b-instant"
    temperature = 0.3

    def __init__(self, llm, telegram_handler=None) -> None:
        super().__init__(llm)
        self._telegram_handler = telegram_handler
        self._trading_bot = None
        self._research_bot = None
        self._sources: list[str] = list(NEWS_SOURCES)

    def set_trading_bot(self, bot) -> None:
        self._trading_bot = bot

    def set_research_bot(self, bot) -> None:
        self._research_bot = bot

    def build_system_prompt(self) -> str:
        return (
            "You are a financial news analyst. Summarise market news concisely, "
            "identify key market impacts (bullish/bearish/neutral), extract actionable insights, "
            "and flag topics that need deeper research."
        )

    async def fetch_headlines(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        seen_urls: set[str] = set()

        for source_url in self._sources:
            try:
                feed = feedparser.parse(source_url)
                for entry in feed.entries:
                    url = getattr(entry, "link", "") or ""
                    title = getattr(entry, "title", "") or ""

                    # Validate
                    if not title.strip():
                        continue
                    if not _is_valid_url(url):
                        continue

                    # Deduplicate by URL
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    category = _classify_category(title)
                    raw_text = getattr(entry, "summary", "") or title

                    items.append(NewsItem(
                        title=title.strip(),
                        source=source_url,
                        url=url,
                        published_at=_parse_published(entry),
                        category=category,
                        raw_text=raw_text,
                    ))
            except Exception as exc:
                logger.warning("Failed to fetch news from %s: %s", source_url, exc)

        return items

    async def summarise(self, items: list[NewsItem]) -> NewsSummary:
        if not items:
            return NewsSummary(
                digest="No news items to summarise.",
                market_impact="neutral",
                key_insights=[],
                topics_for_research=[],
            )

        headlines_text = "\n".join(
            f"- [{item.category.upper()}] {item.title}" for item in items
        )
        prompt = (
            f"Analyse the following financial news headlines and return a JSON object with keys:\n"
            f"  digest (string summary), market_impact (bullish|bearish|neutral),\n"
            f"  key_insights (list of strings), topics_for_research (list of strings).\n\n"
            f"Headlines:\n{headlines_text}\n\n"
            f"Return only valid JSON."
        )

        try:
            raw = await self._llm.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            # Extract JSON from response (handle markdown code blocks)
            text = raw.strip()
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                text = text[start:end] if start != -1 and end > 0 else text
            data = json.loads(text)
            return NewsSummary(
                digest=str(data.get("digest", raw)),
                market_impact=str(data.get("market_impact", "neutral")).lower(),
                key_insights=list(data.get("key_insights", [])),
                topics_for_research=list(data.get("topics_for_research", [])),
            )
        except Exception as exc:
            logger.warning("Failed to parse LLM JSON response for summarise: %s", exc)
            # Fallback: use raw LLM output as digest
            digest = raw if "raw" in dir() else f"News digest for {len(items)} items."
            return NewsSummary(
                digest=digest,
                market_impact="neutral",
                key_insights=[],
                topics_for_research=[],
            )

    async def post_to_telegram(self, summary: NewsSummary) -> None:
        message = (
            f"📰 *News Digest* ({summary.market_impact.upper()})\n\n"
            f"{summary.digest}"
        )
        if self._telegram_handler is not None:
            try:
                await self._telegram_handler.send_message(message)
            except Exception as exc:
                logger.warning("Failed to post to Telegram: %s", exc)
        else:
            logger.info("News summary (no Telegram handler): %s", message)

    async def inject_trading_context(self, insights: list[str]) -> None:
        if self._trading_bot is not None:
            self._trading_bot.inject_news_context(insights)

    async def request_deep_research(self, topic: str) -> ResearchReport:
        if self._research_bot is not None:
            context = SessionContext(
                session_id="news-research",
                bot_id="research",
            )
            response = await self._research_bot.handle(context, topic)
            return ResearchReport(
                topic=topic,
                content=response.reply,
                sources=[],
            )
        return ResearchReport(topic=topic, content=f"No research bot available for: {topic}")

    async def run_scheduled(self) -> None:
        items = await self.fetch_headlines()
        if not items:
            return

        summary = await self.summarise(items)
        await self.post_to_telegram(summary)
        await self.inject_trading_context(summary.key_insights)

        for topic in summary.topics_for_research:
            report = await self.request_deep_research(topic)
            await self.post_to_telegram(NewsSummary(
                digest=report.content,
                market_impact="neutral",
                key_insights=[],
                topics_for_research=[],
            ))

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        await self.run_scheduled()
        # Fetch a fresh summary to return as the reply
        items = await self.fetch_headlines()
        if items:
            summary = await self.summarise(items)
            reply = summary.digest
        else:
            reply = "No news headlines available at this time."
        return BotResponse(reply=reply, bot_id=self.bot_id)
