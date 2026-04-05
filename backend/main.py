"""Application bootstrap for the multi-bot AI system.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key, default)
    if value is None:
        logger.warning("Environment variable %s is not set", key)
    return value


def _env_required(key: str, default: str) -> str:
    value = os.getenv(key)
    if not value:
        logger.warning(
            "Environment variable %s is not set — using default: %s", key, default
        )
        return default
    return value


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer for %s=%r — using default: %d", key, raw, default
        )
        return default


# ---------------------------------------------------------------------------
# Component instantiation
# ---------------------------------------------------------------------------

from backend.approval.gate import ApprovalGate
from backend.bots.coding_bot import CodingBot
from backend.bots.cron_bot import CronBot
from backend.bots.genai_bot import GenAIBot
from backend.bots.meta_bot import MetaBot
from backend.bots.news_bot import NewsAggregatorBot
from backend.bots.research_bot import ResearchBot
from backend.bots.trading_bot import TradingBot
from backend.db.database import Database
from backend.dispatcher.dispatcher import Dispatcher
from backend.dispatcher.session_store import SessionStore
from backend.gateway.app import create_app
from backend.gateway.rate_limiter import RateLimiter
from backend.gateway.routes import configure_router
from backend.integrations.telegram_handler import TelegramHandler
from backend.llm.client import LLMClient

# LLM client
llm = LLMClient(
    base_url=_env_required("OLLAMA_BASE_URL", "http://localhost:11434"),
    openai_api_key=_env_required("OPENAI_API_KEY", ""),
    openai_base_url=_env_required("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# Storage
database = Database(dsn=_env_required("DATABASE_URL", "postgresql://localhost/multibotai"))

# Redis is optional — all consumers fall back to in-memory when not set
_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
session_store = SessionStore(redis_url=_redis_url)

# Approval gate
approval_gate = ApprovalGate(
    redis_url=_redis_url,
    ttl_seconds=_env_int("APPROVAL_TTL_SECONDS", 3600),
)

# Rate limiter
rate_limiter = RateLimiter(redis_url=_redis_url)

# Telegram handler (optional — gracefully skip if token is missing)
_telegram_token = _env("TELEGRAM_TOKEN")
_telegram_group_chat_id_raw = _env("TELEGRAM_GROUP_CHAT_ID")
_telegram_group_chat_id: int | None = None
if _telegram_group_chat_id_raw:
    try:
        _telegram_group_chat_id = int(_telegram_group_chat_id_raw)
    except ValueError:
        logger.warning(
            "Invalid TELEGRAM_GROUP_CHAT_ID=%r — group messages will be disabled",
            _telegram_group_chat_id_raw,
        )

telegram_handler: TelegramHandler | None = None
if _telegram_token:
    try:
        telegram_handler = TelegramHandler(
            token=_telegram_token,
            group_chat_id=_telegram_group_chat_id,
        )
    except Exception as exc:
        logger.warning("Failed to initialise TelegramHandler: %s", exc)

# Git deployer — applies approved diffs, commits, and pushes to trigger auto-deploy
from backend.deployer import GitDeployer
_git_deployer = GitDeployer()

# Bots
coding_bot = CodingBot(llm=llm, approval_gate=approval_gate)
coding_bot._deployer = _git_deployer
research_bot = ResearchBot(llm=llm)
trading_bot = TradingBot(llm=llm)
genai_bot = GenAIBot(llm=llm)
news_bot = NewsAggregatorBot(llm=llm, telegram_handler=telegram_handler)

# Scheduler (shared — CronBot and NewsBot both use it)
from apscheduler.schedulers.asyncio import AsyncIOScheduler as _Scheduler
_shared_scheduler = _Scheduler()

# CronBot — wired after dispatcher is created (see below)
_cron_bot_placeholder: CronBot | None = None  # filled after dispatcher init

# Dispatcher
dispatcher = Dispatcher(session_store=session_store, db=database)
dispatcher.register("coding", coding_bot)
dispatcher.register("research", research_bot)
dispatcher.register("trading", trading_bot)
dispatcher.register("genai", genai_bot)
dispatcher.register("news", news_bot)

# Inter-agent wiring
news_bot.set_trading_bot(trading_bot)
news_bot.set_research_bot(research_bot)
news_bot._telegram_handler = telegram_handler

_redis_client = None
try:
    import redis.asyncio as aioredis  # type: ignore[import]
    if os.getenv("REDIS_URL"):
        _redis_client = aioredis.from_url(
            os.getenv("REDIS_URL"),
            decode_responses=True,
        )
except Exception as exc:
    logger.warning("Could not create shared Redis client: %s", exc)

# CronBot — needs dispatcher and redis_client, registered after both exist
cron_bot = CronBot(
    llm=llm,
    scheduler=_shared_scheduler,
    dispatcher=dispatcher,
    telegram_handler=telegram_handler,
    group_chat_id=_telegram_group_chat_id,
    redis_client=_redis_client,
)
dispatcher.register("cron", cron_bot)

# MetaBot — /bots, /start, /help
meta_bot = MetaBot(llm=llm, redis_client=_redis_client)
dispatcher.register("meta", meta_bot)

# Wire router dependencies
configure_router(
    dispatcher=dispatcher,
    approval_gate=approval_gate,
    rate_limiter=rate_limiter,
    telegram_handler=telegram_handler,
    redis_client=_redis_client,
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app) -> AsyncIterator[None]:  # type: ignore[type-arg]
    """FastAPI lifespan: startup → yield → shutdown."""
    # --- Startup ---
    try:
        await database.connect()
        logger.info("Database connected")
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)

    # Start live price feed (polls Binance every 10s)
    try:
        from backend.market import price_feed
        price_feed.start(interval_seconds=10)
        logger.info("Live price feed started")
    except Exception as exc:
        logger.warning("Price feed could not be started: %s", exc)

    scheduler = None
    try:
        _shared_scheduler.add_job(news_bot.run_scheduled, "interval", minutes=30,
                                   id="news_scheduled", replace_existing=True)
        _shared_scheduler.start()
        scheduler = _shared_scheduler
        logger.info("APScheduler started — news_bot scheduled every 30 minutes")
    except Exception as exc:
        logger.warning("APScheduler could not be started: %s", exc)

    yield

    # --- Shutdown ---
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
            logger.info("APScheduler stopped")
        except Exception as exc:
            logger.warning("Error stopping scheduler: %s", exc)

    # Stop live price feed
    try:
        from backend.market import price_feed
        price_feed.stop()
        logger.info("Price feed stopped")
    except Exception:
        pass

    try:
        await database.disconnect()
        logger.info("Database disconnected")
    except Exception as exc:
        logger.warning("Error disconnecting database: %s", exc)

    try:
        await session_store.close()
        logger.info("Session store closed")
    except Exception as exc:
        logger.warning("Error closing session store: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = create_app()
app.router.lifespan_context = lifespan
