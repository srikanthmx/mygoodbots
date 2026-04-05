"""CronBot — schedule recurring tasks via Telegram group messages (owner only)."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.bots.base import BaseBot
from backend.models import BotResponse, SessionContext

logger = logging.getLogger(__name__)

# Owner check — only this Telegram user ID can manage cron jobs
OWNER_TELEGRAM_ID = int(os.getenv("OWNER_TELEGRAM_ID") or "0")

HELP_TEXT = """⏰ *CronBot — Schedule Manager*

*Commands (owner only):*

`/cron list` — show all scheduled jobs
`/cron add <schedule> <bot> <message>` — add a new job
`/cron remove <job_id>` — remove a job
`/cron help` — show this help

*Schedule formats:*
• `every 30m` — every 30 minutes
• `every 2h` — every 2 hours
• `every 1d` — every day
• `at 09:00` — daily at 09:00
• `at 09:00 mon-fri` — weekdays at 09:00
• `cron 0 9 * * 1-5` — raw cron expression

*Examples:*
`/cron add every 1h trade btc market update`
`/cron add at 08:00 news morning market briefing`
`/cron add cron 0 */4 * * * research latest AI news`

_Use the command name (trade, code, research, gen, news) or the full bot_id (trading, coding, etc.)_
"""


@dataclass
class CronJob:
    job_id: str
    schedule_desc: str
    bot_id: str
    message: str
    created_at: datetime = field(default_factory=datetime.utcnow)


class CronBot(BaseBot):
    bot_id = "cron"
    description = "Schedule recurring tasks via Telegram (owner only)"
    model = "llama-3.1-8b-instant"
    temperature = 0.1

    def __init__(self, llm: Any, scheduler: AsyncIOScheduler, dispatcher: Any,
                 telegram_handler: Any = None, group_chat_id: int | None = None,
                 redis_client: Any = None) -> None:
        super().__init__(llm)
        self._scheduler = scheduler
        self._dispatcher = dispatcher
        self._telegram_handler = telegram_handler
        self._group_chat_id = group_chat_id
        self._redis = redis_client
        self._jobs: dict[str, CronJob] = {}  # job_id → CronJob metadata

    def build_system_prompt(self) -> str:
        return "You are a scheduling assistant. Help users set up and manage cron jobs."

    async def _is_owner(self, context: SessionContext) -> bool:
        """Check if the request comes from the owner's Telegram session."""
        if OWNER_TELEGRAM_ID == 0:
            logger.warning("OWNER_TELEGRAM_ID not set — CronBot is open to everyone")
            return True
        # Check sender_id stored in session metadata by the dispatcher
        sender_id = context.metadata.get("sender_id")
        if sender_id is not None and int(sender_id) == OWNER_TELEGRAM_ID:
            return True
        # Fallback: Redis lookup (if available)
        if self._redis:
            try:
                stored = await self._redis.get(f"sender:{context.session_id}")
                if stored and int(stored) == OWNER_TELEGRAM_ID:
                    return True
            except Exception:
                pass
        # Last resort: owner's private chat session
        return context.session_id == f"tg:{OWNER_TELEGRAM_ID}"

    def _parse_schedule(self, schedule_str: str) -> tuple[Any, str]:
        """
        Parse schedule string into an APScheduler trigger.
        Returns (trigger, human_readable_description).
        """
        s = schedule_str.strip().lower()

        # every Xm / Xh / Xd
        m = re.match(r"every\s+(\d+)(m|h|d)$", s)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            kwargs = {"minutes": n} if unit == "m" else {"hours": n} if unit == "h" else {"days": n}
            return IntervalTrigger(**kwargs), f"every {n}{'min' if unit=='m' else 'h' if unit=='h' else 'd'}"

        # at HH:MM [day_of_week]
        m = re.match(r"at\s+(\d{1,2}):(\d{2})(?:\s+([\w\-,]+))?$", s)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            dow = m.group(3) or "mon-sun"
            return CronTrigger(hour=hour, minute=minute, day_of_week=dow), \
                   f"daily at {hour:02d}:{minute:02d} ({dow})"

        # raw cron expression: cron <expr>
        m = re.match(r"cron\s+(.+)$", s)
        if m:
            parts = m.group(1).split()
            if len(parts) == 5:
                minute, hour, day, month, dow = parts
                return CronTrigger(minute=minute, hour=hour, day=day,
                                   month=month, day_of_week=dow), \
                       f"cron({m.group(1)})"

        raise ValueError(f"Unrecognised schedule format: '{schedule_str}'")

    async def _run_job(self, job_id: str) -> None:
        """Execute a scheduled job — dispatch to the target bot and post result to group."""
        job = self._jobs.get(job_id)
        if not job:
            return
        logger.info("CronBot running job %s: [%s] %s", job_id, job.bot_id, job.message)
        try:
            response = await self._dispatcher.dispatch(
                session_id=f"cron:{job_id}",
                bot_id=job.bot_id,
                message=job.message,
            )
            reply = response.reply if not response.error else f"⚠️ Error: {response.error}"
        except Exception as exc:
            reply = f"⚠️ CronJob {job_id} failed: {exc}"

        # Post result to Telegram group
        text = f"⏰ *Scheduled: {job.schedule_desc}*\n_{job.bot_id}: {job.message}_\n\n{reply}"
        if self._telegram_handler and self._group_chat_id:
            try:
                await self._telegram_handler.send_message(self._group_chat_id, text)
            except Exception as exc:
                logger.error("CronBot failed to post job result to Telegram: %s", exc)
        else:
            logger.info("CronBot job result (no Telegram): %s", text)

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        """Parse /cron commands — only the owner can manage jobs."""
        if not await self._is_owner(context):
            return BotResponse(
                reply="🚫 Only the group owner can manage cron jobs.",
                bot_id=self.bot_id,
            )

        text = message.strip()

        # Strip leading /cron prefix if present
        if text.lower().startswith("/cron"):
            text = text[5:].strip()

        parts = text.split(None, 1)
        sub = parts[0].lower() if parts else "help"
        args = parts[1] if len(parts) > 1 else ""

        if sub == "help" or not sub:
            return BotResponse(reply=HELP_TEXT, bot_id=self.bot_id)

        if sub == "list":
            return self._cmd_list()

        if sub == "add":
            return await self._cmd_add(args)

        if sub == "remove":
            return self._cmd_remove(args.strip())

        return BotResponse(
            reply=f"Unknown subcommand `{sub}`. Try `/cron help`.",
            bot_id=self.bot_id,
        )

    def _cmd_list(self) -> BotResponse:
        if not self._jobs:
            return BotResponse(reply="No scheduled jobs. Use `/cron add` to create one.", bot_id=self.bot_id)
        lines = ["⏰ *Scheduled Jobs:*\n"]
        for jid, job in self._jobs.items():
            lines.append(f"• `{jid}` — {job.schedule_desc} → `/{job.bot_id}` _{job.message}_")
        return BotResponse(reply="\n".join(lines), bot_id=self.bot_id)

    async def _cmd_add(self, args: str) -> BotResponse:
        """
        Syntax: <schedule_tokens> <bot_id> <message>
        e.g.:   every 1h trade BTC market update
                at 09:00 news morning briefing
                cron 0 9 * * 1-5 research AI news
        """
        if not args:
            return BotResponse(reply="Usage: `/cron add <schedule> <bot> <message>`", bot_id=self.bot_id)

        # Try to extract schedule + bot_id + message
        # Schedule can be 2-6 tokens; bot_id is one token; rest is message
        tokens = args.split()
        schedule_str, bot_id, message = None, None, None

        # Try each possible schedule length (2 to 6 tokens)
        for schedule_len in [6, 3, 2]:
            if len(tokens) < schedule_len + 2:
                continue
            candidate_schedule = " ".join(tokens[:schedule_len])
            candidate_bot = tokens[schedule_len]
            candidate_msg = " ".join(tokens[schedule_len + 1:])
            try:
                trigger, desc = self._parse_schedule(candidate_schedule)
                schedule_str = candidate_schedule
                bot_id = candidate_bot
                message = candidate_msg
                break
            except ValueError:
                continue

        if not schedule_str:
            return BotResponse(
                reply="❌ Could not parse schedule. Try:\n`/cron add every 1h trade BTC update`",
                bot_id=self.bot_id,
            )

        if not message:
            return BotResponse(reply="❌ Message cannot be empty.", bot_id=self.bot_id)

        # Resolve command aliases: trade→trading, code→coding, research→research, gen→genai, news→news
        _ALIASES = {
            "trade": "trading",
            "code": "coding",
            "gen": "genai",
            "research": "research",
            "news": "news",
            "cron": "cron",
        }
        bot_id = _ALIASES.get(bot_id, bot_id)

        # Validate bot_id exists in dispatcher
        try:
            self._dispatcher.get(bot_id)
        except KeyError:
            registered = [b.bot_id for b in self._dispatcher.list_bots()]
            return BotResponse(
                reply=f"❌ Unknown bot `{bot_id}`. Available: {', '.join(registered)}",
                bot_id=self.bot_id,
            )

        trigger, desc = self._parse_schedule(schedule_str)
        job_id = f"cron_{len(self._jobs) + 1:03d}"

        self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            args=[job_id],
            id=job_id,
            replace_existing=True,
        )

        self._jobs[job_id] = CronJob(
            job_id=job_id,
            schedule_desc=desc,
            bot_id=bot_id,
            message=message,
        )

        return BotResponse(
            reply=f"✅ Job `{job_id}` scheduled: *{desc}*\n→ `/{bot_id}` _{message}_",
            bot_id=self.bot_id,
        )

    def _cmd_remove(self, job_id: str) -> BotResponse:
        if not job_id:
            return BotResponse(reply="Usage: `/cron remove <job_id>`", bot_id=self.bot_id)
        if job_id not in self._jobs:
            return BotResponse(reply=f"❌ Job `{job_id}` not found.", bot_id=self.bot_id)
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass
        del self._jobs[job_id]
        return BotResponse(reply=f"🗑️ Job `{job_id}` removed.", bot_id=self.bot_id)
