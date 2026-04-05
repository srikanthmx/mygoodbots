"""GenAIBot — creative generation, image-prompt crafting, and style transfer."""
from __future__ import annotations

from backend.bots.base import BaseBot
from backend.models import BotResponse, SessionContext


class GenAIBot(BaseBot):
    bot_id = "genai"
    description = "Creative generation, image-prompt crafting, and style transfer"
    model = "llama-3.1-8b-instant"
    temperature = 0.9

    def build_system_prompt(self) -> str:
        return (
            "You are a creative AI assistant specializing in generative tasks. "
            "You excel at creative writing, image prompt crafting, style transfer descriptions, "
            "and imaginative content generation. Be vivid, creative, and expressive."
        )

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        return await super().handle(context, message)
