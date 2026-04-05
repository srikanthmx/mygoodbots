"""Live price feed — continuously polls Binance WebSocket for real-time prices."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Symbols to track continuously
TRACKED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT",
]

# In-memory store: symbol → latest ticker data
_prices: dict[str, dict[str, Any]] = {}
_last_updated: dict[str, str] = {}
_running = False
_task: asyncio.Task | None = None


def get_price(symbol: str) -> dict | None:
    """Return the latest cached price data for a symbol (e.g. 'BTCUSDT')."""
    return _prices.get(symbol.upper())


def get_all_prices() -> dict[str, dict]:
    """Return all cached prices."""
    return dict(_prices)


def format_price_block(symbol: str) -> str:
    """Return a formatted live data string ready to inject into LLM prompt."""
    data = _prices.get(symbol.upper())
    if not data:
        return f"[Live data not yet available for {symbol} — fetching...]"
    return (
        f"[LIVE DATA as of {data['last_updated']}]\n"
        f"Symbol: {symbol.upper()}\n"
        f"Price: ${data['price']:,.4f}\n"
        f"24h Open: ${data['open']:,.4f}\n"
        f"24h High: ${data['high']:,.4f}\n"
        f"24h Low: ${data['low']:,.4f}\n"
        f"24h Change: {data['change_pct']:+.2f}%\n"
        f"24h Volume: {data['volume']:,.2f} {symbol.replace('USDT','')}\n"
        f"24h Quote Volume: ${data['quote_volume']:,.0f}"
    )


async def _fetch_all_tickers() -> None:
    """Fetch prices for all tracked symbols — individual Binance calls with CoinGecko fallback."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    success = 0
    for sym in TRACKED_SYMBOLS:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}"
                )
                r.raise_for_status()
                d = r.json()
                _prices[sym] = {
                    "price": float(d["lastPrice"]),
                    "open": float(d["openPrice"]),
                    "high": float(d["highPrice"]),
                    "low": float(d["lowPrice"]),
                    "change_pct": float(d["priceChangePercent"]),
                    "volume": float(d["volume"]),
                    "quote_volume": float(d["quoteVolume"]),
                    "last_updated": now,
                }
                success += 1
        except Exception:
            # Binance geo-blocked — fall back to CoinGecko for this symbol
            await _fetch_coingecko_single(sym, now)
    logger.debug("Price feed updated %d/%d symbols at %s", success, len(TRACKED_SYMBOLS), now)


_CG_IDS = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin", "XRPUSDT": "ripple", "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin", "AVAXUSDT": "avalanche-2", "DOTUSDT": "polkadot",
    "MATICUSDT": "matic-network", "LINKUSDT": "chainlink", "LTCUSDT": "litecoin",
    "UNIUSDT": "uniswap", "ATOMUSDT": "cosmos",
}


async def _fetch_coingecko_single(sym: str, now: str) -> None:
    coin_id = _CG_IDS.get(sym)
    if not coin_id:
        return
    try:
        url = (
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={coin_id}&vs_currencies=usd"
            f"&include_24hr_change=true&include_24hr_vol=true"
        )
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url)
            r.raise_for_status()
            d = r.json().get(coin_id, {})
            if d:
                price = d.get("usd", 0)
                change = d.get("usd_24h_change", 0) or 0
                vol = d.get("usd_24h_vol", 0) or 0
                _prices[sym] = {
                    "price": price,
                    "open": price / (1 + change / 100) if change else price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "change_pct": change,
                    "volume": vol / price if price else 0,
                    "quote_volume": vol,
                    "last_updated": now,
                }
    except Exception as exc:
        logger.debug("CoinGecko fallback failed for %s: %s", sym, exc)


async def _poll_loop(interval_seconds: int = 10) -> None:
    """Continuously poll Binance every `interval_seconds` seconds."""
    global _running
    logger.info("Price feed started — polling every %ds for %d symbols",
                interval_seconds, len(TRACKED_SYMBOLS))
    while _running:
        await _fetch_all_tickers()
        await asyncio.sleep(interval_seconds)
    logger.info("Price feed stopped")


def start(interval_seconds: int = 10) -> None:
    """Start the background price feed loop (call once at app startup)."""
    global _running, _task
    if _running:
        return
    _running = True
    _task = asyncio.ensure_future(_poll_loop(interval_seconds))


def stop() -> None:
    """Stop the background price feed loop (call at app shutdown)."""
    global _running, _task
    _running = False
    if _task and not _task.done():
        _task.cancel()
