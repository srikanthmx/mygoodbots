"""Unit tests for TelegramHandler."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.integrations.telegram_handler import (
    COMMAND_MAP,
    TelegramHandler,
    TelegramMessage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_handler() -> TelegramHandler:
    """Create a TelegramHandler with a mocked Bot."""
    with patch("backend.integrations.telegram_handler.Bot") as mock_bot_cls:
        mock_bot_cls.return_value = MagicMock()
        handler = TelegramHandler(token="fake-token", group_chat_id=None)
    return handler


def make_update(
    update_id: int = 1,
    chat_id: int = 12345,
    text: str = "hello",
) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": chat_id},
            "text": text,
        },
    }


# ---------------------------------------------------------------------------
# command_to_bot_id — known commands
# ---------------------------------------------------------------------------

class TestCommandToBotId:
    def setup_method(self) -> None:
        self.handler = make_handler()

    def test_code_maps_to_coding(self) -> None:
        assert self.handler.command_to_bot_id("/code") == "coding"

    def test_research_maps_to_research(self) -> None:
        assert self.handler.command_to_bot_id("/research") == "research"

    def test_trade_maps_to_trading(self) -> None:
        assert self.handler.command_to_bot_id("/trade") == "trading"

    def test_gen_maps_to_genai(self) -> None:
        assert self.handler.command_to_bot_id("/gen") == "genai"

    def test_news_maps_to_news(self) -> None:
        assert self.handler.command_to_bot_id("/news") == "news"

    def test_all_command_map_entries_covered(self) -> None:
        """Every entry in COMMAND_MAP is correctly mapped."""
        for cmd, expected_bot_id in COMMAND_MAP.items():
            assert self.handler.command_to_bot_id(cmd) == expected_bot_id

    # Unknown commands → "genai"
    def test_unknown_command_returns_genai(self) -> None:
        assert self.handler.command_to_bot_id("/unknown") == "genai"

    def test_empty_slash_returns_genai(self) -> None:
        assert self.handler.command_to_bot_id("/") == "genai"

    def test_arbitrary_unknown_command_returns_genai(self) -> None:
        assert self.handler.command_to_bot_id("/foobar") == "genai"

    def test_mixed_case_unknown_returns_genai(self) -> None:
        # Commands are case-sensitive; /Code is not /code
        assert self.handler.command_to_bot_id("/Code") == "genai"


# ---------------------------------------------------------------------------
# parse_update — field extraction
# ---------------------------------------------------------------------------

class TestParseUpdate:
    def setup_method(self) -> None:
        self.handler = make_handler()

    def test_extracts_chat_id(self) -> None:
        update = make_update(chat_id=99999)
        msg = self.handler.parse_update(update)
        assert msg.chat_id == 99999

    def test_extracts_text(self) -> None:
        update = make_update(text="hello world")
        msg = self.handler.parse_update(update)
        assert msg.text == "hello world"

    def test_extracts_update_id(self) -> None:
        update = make_update(update_id=42)
        msg = self.handler.parse_update(update)
        assert msg.update_id == 42

    def test_returns_telegram_message_instance(self) -> None:
        update = make_update()
        msg = self.handler.parse_update(update)
        assert isinstance(msg, TelegramMessage)

    # Command detection
    def test_plain_text_is_not_command(self) -> None:
        update = make_update(text="just a normal message")
        msg = self.handler.parse_update(update)
        assert msg.is_command is False

    def test_slash_prefix_is_command(self) -> None:
        update = make_update(text="/code write a function")
        msg = self.handler.parse_update(update)
        assert msg.is_command is True

    def test_command_sets_correct_bot_id(self) -> None:
        update = make_update(text="/research find papers on LLMs")
        msg = self.handler.parse_update(update)
        assert msg.bot_id == "research"

    def test_unknown_command_sets_genai_bot_id(self) -> None:
        update = make_update(text="/unknown do something")
        msg = self.handler.parse_update(update)
        assert msg.bot_id == "genai"

    def test_plain_text_defaults_to_genai_bot_id(self) -> None:
        update = make_update(text="tell me a joke")
        msg = self.handler.parse_update(update)
        assert msg.bot_id == "genai"

    def test_command_with_bot_name_suffix_parsed_correctly(self) -> None:
        """Commands like /code@MyBot should still map correctly."""
        update = make_update(text="/code@MyBot write tests")
        msg = self.handler.parse_update(update)
        assert msg.bot_id == "coding"
        assert msg.is_command is True

    def test_trade_command_maps_to_trading(self) -> None:
        update = make_update(text="/trade BTC analysis")
        msg = self.handler.parse_update(update)
        assert msg.bot_id == "trading"

    def test_news_command_maps_to_news(self) -> None:
        update = make_update(text="/news")
        msg = self.handler.parse_update(update)
        assert msg.bot_id == "news"
