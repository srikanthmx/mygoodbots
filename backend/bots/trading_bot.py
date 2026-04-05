"""TradingBot — market analysis with live price data from the continuous price feed."""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone

import httpx

from backend.bots.base import BaseBot
from backend.models import BotResponse, Message as Msg, SessionContext

logger = logging.getLogger(__name__)

RISK_DISCLAIMER = (
    "\n\n⚠️ DISCLAIMER: This is not financial advice. "
    "All trading analysis is for informational purposes only. "
    "Always consult a qualified financial advisor before making investment decisions."
)

_BINANCE_PAIRS = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT",
    "sol": "SOLUSDT", "solana": "SOLUSDT",
    "bnb": "BNBUSDT",
    "xrp": "XRPUSDT",
    "ada": "ADAUSDT",
    "doge": "DOGEUSDT",
    "avax": "AVAXUSDT",
    "dot": "DOTUSDT",
    "matic": "MATICUSDT", "polygon": "MATICUSDT",
    "link": "LINKUSDT",
    "ltc": "LTCUSDT",
    "uni": "UNIUSDT",
    "atom": "ATOMUSDT",
}


def _extract_symbol(message: str) -> str | None:
    for word in message.lower().split():
        word = word.strip(".,!?/")
        if word in _BINANCE_PAIRS:
            return word
    return None


def _get_live_data(symbol: str) -> str:
    """Get price from the in-memory feed (always fresh, updated every 10s)."""
    try:
        from backend.market import price_feed
        pair = _BINANCE_PAIRS.get(symbol)
        if pair:
            block = price_feed.format_price_block(pair)
            if "not yet available" not in block:
                return block
    except Exception:
        pass
    # Fallback: one-shot fetch if feed not warmed up yet
    return f"[Fetching live data for {symbol.upper()}...]"


class TradingBot(BaseBot):
    bot_id = "trading"
    description = "Market analysis, technical indicators, and trading commentary"
    model = "llama-3.1-8b-instant"
    temperature = 0.4

    def __init__(self, llm) -> None:
        super().__init__(llm)
        self._news_context: list[str] = []

    def build_system_prompt(self) -> str:
        base = (
            "You are a trading analysis assistant with access to live market data. "
            "When live data is provided in the user message, use it as the ground truth — "
            "never contradict or ignore it. Provide market commentary and technical insights "
            "based on the real numbers given. Never fabricate prices or dates. "
            "Always include a risk disclaimer."
        )
        if self._news_context:
            news_section = "\n\nRecent market news context:\n" + "\n".join(
                f"- {insight}" for insight in self._news_context[-10:]
            )
            return base + news_section
        return base

    def inject_news_context(self, insights: list[str]) -> None:
        self._news_context.extend(insights)

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        # Get live price from in-memory feed (updated every 10s in background)
        symbol = _extract_symbol(message)
        enriched_message = message
        if symbol:
            live_data = _get_live_data(symbol)
            enriched_message = f"{live_data}\n\nUser request: {message}"

        # Replace last user message in context with enriched version
        enriched_context = copy.copy(context)
        enriched_context.history = list(context.history)
        if enriched_context.history and enriched_context.history[-1].role == "user":
            enriched_context.history = enriched_context.history[:-1] + [
                Msg(role="user", content=enriched_message)
            ]

        response = await super().handle(enriched_context, enriched_message)
        if RISK_DISCLAIMER not in response.reply:
            response.reply = response.reply + RISK_DISCLAIMER
        return response
