"""Telegram integration handler for the multi-bot AI system."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)

COMMAND_MAP: dict[str, str] = {
    "/code": "coding",
    "/research": "research",
    "/trade": "trading",
    "/gen": "genai",
    "/news": "news",
    "/cron": "cron",
    "/bots": "meta",
    "/start": "meta",
    "/help": "meta",
}


@dataclass
class TelegramMessage:
    chat_id: int
    text: str
    update_id: int
    bot_id: str
    is_command: bool
    sender_id: int = 0


class TelegramHandler:
    """Parses Telegram Update objects and sends messages via python-telegram-bot."""

    def __init__(self, token: str, group_chat_id: int | None = None) -> None:
        self._bot = Bot(token=token)
        self._group_chat_id = group_chat_id

    def parse_update(self, update: dict) -> TelegramMessage:
        """Extract chat_id, text, update_id from a Telegram Update dict."""
        update_id: int = update["update_id"]
        message = update.get("message", {})
        chat_id: int = message["chat"]["id"]
        text: str = message.get("text", "")
        sender_id: int = message.get("from", {}).get("id", 0)

        is_command = text.startswith("/")
        if is_command:
            command = text.split()[0].split("@")[0]
            bot_id = self.command_to_bot_id(command)
        else:
            bot_id = "genai"

        return TelegramMessage(
            chat_id=chat_id,
            text=text,
            update_id=update_id,
            bot_id=bot_id,
            is_command=is_command,
            sender_id=sender_id,
        )

    def command_to_bot_id(self, command: str) -> str:
        """Map a Telegram command to a bot_id. Returns 'genai' for unknown commands."""
        return COMMAND_MAP.get(command, "genai")

    async def send_message(self, chat_id: int, text: str) -> None:
        """Send a text message to the given chat_id."""
        await self._bot.send_message(chat_id=chat_id, text=text)

    async def send_group_message(self, text: str) -> None:
        """Send a message to the configured group chat, if set."""
        if self._group_chat_id is not None:
            await self._bot.send_message(chat_id=self._group_chat_id, text=text)
        else:
            logger.warning("send_group_message called but group_chat_id is not configured")

    async def send_approval_notification(
        self,
        chat_id: int,
        diff: str,
        summary: str,
        approval_id: str,
    ) -> None:
        """Send a diff + summary with inline approve/reject buttons."""
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✅ Approve", callback_data=f"approve:{approval_id}"
                    ),
                    InlineKeyboardButton(
                        "❌ Reject", callback_data=f"reject:{approval_id}"
                    ),
                ]
            ]
        )
        text = f"*Code Change Proposal*\n\n*Summary:* {summary}\n\n```\n{diff}\n```"
        await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
