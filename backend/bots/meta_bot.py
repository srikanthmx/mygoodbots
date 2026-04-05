"""MetaBot — handles /bots, /start, /help commands for the group."""
from __future__ import annotations

import os
from typing import Any

from backend.bots.base import BaseBot
from backend.models import BotResponse, SessionContext

OWNER_TELEGRAM_ID = int(os.getenv("OWNER_TELEGRAM_ID") or "0")

BOT_CATALOG = [
    {
        "command": "/code",
        "bot_id": "coding",
        "emoji": "💻",
        "name": "Coding Bot",
        "desc": "Code generation, review & smart refactoring",
        "example": "/code write a Python function to sort a dict by value",
        "owner_only": False,
    },
    {
        "command": "/research",
        "bot_id": "research",
        "emoji": "🔬",
        "name": "Research Bot",
        "desc": "Web research, summaries with citations",
        "example": "/research latest breakthroughs in quantum computing",
        "owner_only": False,
    },
    {
        "command": "/trade",
        "bot_id": "trading",
        "emoji": "📈",
        "name": "Trading Bot",
        "desc": "Market analysis & technical commentary",
        "example": "/trade BTC technical analysis for today",
        "owner_only": False,
    },
    {
        "command": "/gen",
        "bot_id": "genai",
        "emoji": "🎨",
        "name": "Generative AI Bot",
        "desc": "Creative writing, image prompts, style transfer",
        "example": "/gen write a haiku about the stock market",
        "owner_only": False,
    },
    {
        "command": "/news",
        "bot_id": "news",
        "emoji": "📰",
        "name": "News Aggregator Bot",
        "desc": "Financial & market news digest (also runs every 30 min automatically)",
        "example": "/news",
        "owner_only": False,
    },
    {
        "command": "/cron",
        "bot_id": "cron",
        "emoji": "⏰",
        "name": "Cron Bot",
        "desc": "Schedule recurring tasks (owner only)",
        "example": "/cron add every 1h trade BTC market update",
        "owner_only": True,
    },
]

CRON_HELP = """⏰ *Cron Job Management* (owner only)

*Commands:*
`/cron list` — show all scheduled jobs
`/cron add <schedule> <bot> <message>` — create a job
`/cron remove <job_id>` — delete a job
`/cron help` — show cron-specific help

*Schedule formats:*
• `every 30m` — every 30 minutes
• `every 2h` — every 2 hours
• `every 1d` — every day
• `at 09:00` — daily at 09:00
• `at 09:00 mon-fri` — weekdays at 09:00
• `cron 0 9 * * 1-5` — raw cron expression

*Examples:*
`/cron add every 1h trade BTC market update`
`/cron add at 08:00 news morning market briefing`
`/cron add at 09:00 mon-fri research AI news summary`
`/cron add cron 0 */4 * * * trade ETH 4-hour analysis`
"""


def _build_bots_message(is_owner: bool) -> str:
    lines = ["🤖 *Available Bots*\n"]
    for bot in BOT_CATALOG:
        if bot["owner_only"] and not is_owner:
            continue
        lines.append(
            f"{bot['emoji']} *{bot['name']}* — `{bot['command']}`\n"
            f"  _{bot['desc']}_\n"
            f"  Example: `{bot['example']}`\n"
        )
    lines.append("─────────────────")
    lines.append("Just type a command followed by your question.")
    lines.append("Plain text (no command) goes to 🎨 GenAI Bot by default.")
    return "\n".join(lines)


def _build_start_message() -> str:
    return (
        "👋 *Welcome to Multi-Bot AI!*\n\n"
        "I'm a multi-agent AI system with specialized bots for different tasks.\n\n"
        "Type `/bots` to see all available bots and how to use them.\n"
        "Type `/help` for a quick reference.\n\n"
        "📰 News digest is posted automatically every 30 minutes."
    )


def _build_help_message(is_owner: bool) -> str:
    lines = [
        "📖 *Quick Reference*\n",
        "`/code <request>` — 💻 coding help",
        "`/research <topic>` — 🔬 research & citations",
        "`/trade <query>` — 📈 market analysis",
        "`/gen <prompt>` — 🎨 creative generation",
        "`/news` — 📰 latest market news",
    ]
    if is_owner:
        lines.append("`/cron <subcommand>` — ⏰ schedule tasks")
    lines += [
        "",
        "`/bots` — full bot catalog",
        "`/help` — this message",
        "",
        "_Plain text → GenAI Bot_",
    ]
    return "\n".join(lines)


class MetaBot(BaseBot):
    bot_id = "meta"
    description = "Bot catalog, help, and system info"
    model = "llama-3.1-8b-instant"

    def __init__(self, llm: Any, redis_client: Any = None) -> None:
        super().__init__(llm)
        self._redis = redis_client

    def build_system_prompt(self) -> str:
        return "You are a helpful assistant that explains how to use the multi-bot AI system."

    async def _is_owner(self, context: SessionContext) -> bool:
        if OWNER_TELEGRAM_ID == 0:
            return True
        if self._redis:
            try:
                stored = await self._redis.get(f"sender:{context.session_id}")
                if stored and int(stored) == OWNER_TELEGRAM_ID:
                    return True
            except Exception:
                pass
        return context.session_id == f"tg:{OWNER_TELEGRAM_ID}"

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        is_owner = await self._is_owner(context)
        text = message.strip().lower()

        # strip command prefix
        for prefix in ("/bots", "/start", "/help"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        # determine which command triggered this
        original = message.strip().lower().split()[0] if message.strip() else ""

        if original in ("/start",):
            return BotResponse(reply=_build_start_message(), bot_id=self.bot_id)

        if original in ("/help",):
            return BotResponse(reply=_build_help_message(is_owner), bot_id=self.bot_id)

        # /bots — full catalog
        return BotResponse(reply=_build_bots_message(is_owner), bot_id=self.bot_id)
