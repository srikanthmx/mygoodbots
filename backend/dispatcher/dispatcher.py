"""Dispatcher: routes messages to registered bots and manages session context."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from backend.bots.base import BaseBot
from backend.dispatcher.session_store import SessionStore
from backend.models import BotInfo, BotResponse, Message, SessionContext

logger = logging.getLogger(__name__)


class Dispatcher:
    """Central dispatcher that routes messages to registered bots."""

    def __init__(
        self,
        session_store: SessionStore,
        db: Any,
        max_history: int = 50,
    ) -> None:
        self._session_store = session_store
        self._db = db
        self._max_history = max_history
        self._registry: dict[str, BaseBot] = {}

    def register(self, bot_id: str, bot: BaseBot) -> None:
        """Add or overwrite a bot in the registry."""
        self._registry[bot_id] = bot

    def get(self, bot_id: str) -> BaseBot:
        """Return the bot registered under bot_id."""
        return self._registry[bot_id]

    def list_bots(self) -> list[BotInfo]:
        """Return BotInfo for each registered bot."""
        return [
            BotInfo(
                bot_id=bot_id,
                description=bot.description,
                model=bot.model,
                available=True,
            )
            for bot_id, bot in self._registry.items()
        ]

    async def dispatch(
        self,
        session_id: str,
        bot_id: str,
        message: str,
        sender_id: int | None = None,
    ) -> BotResponse:
        """
        Full dispatch algorithm — never raises.

        1. Unknown bot → error BotResponse
        2. Load or create SessionContext
        3. Append user Message
        4. Enforce MAX_HISTORY sliding window
        5. Delegate to bot.handle()
        6. Record latency
        7. Append assistant Message on success
        8. Persist context
        9. Persist turn to DB (best-effort)
        10. Return BotResponse
        """
        # 1. Validate bot exists
        if bot_id not in self._registry:
            return BotResponse(reply="", bot_id=bot_id, error=f"Unknown bot: {bot_id}")

        # 2. Load or create session context
        context = await self._session_store.get(session_id) or SessionContext(
            session_id=session_id,
            bot_id=bot_id,
        )

        # Store sender_id in metadata so bots can verify ownership without Redis
        if sender_id is not None:
            context.metadata["sender_id"] = sender_id

        # 3. Append user turn
        context.history.append(Message(role="user", content=message))

        # 4. Enforce history window
        if len(context.history) > self._max_history:
            context.history = context.history[-self._max_history:]

        # 5 & 6. Delegate to bot, measure latency
        bot = self._registry[bot_id]
        start = time.monotonic()
        try:
            response = await bot.handle(context, message)
        except Exception as exc:
            response = BotResponse(reply="", bot_id=bot_id, error=str(exc))

        response.latency_ms = (time.monotonic() - start) * 1000

        # 7. Append assistant turn on success
        if not response.error:
            context.history.append(Message(role="assistant", content=response.reply))
            # Re-enforce window after assistant append
            if len(context.history) > self._max_history:
                context.history = context.history[-self._max_history:]

        # 8. Update timestamp and persist context
        context.updated_at = datetime.utcnow()
        await self._session_store.set(session_id, context)

        # 9. Persist turn to DB (best-effort — never fail the request)
        try:
            await self._db.save_turn(session_id, bot_id, message, response.reply)
        except Exception as exc:
            logger.warning("DB save_turn failed (non-fatal): %s", exc)

        # 10. Return
        return response
