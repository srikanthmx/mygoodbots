"""ResearchBot — web research, summarisation, and citation formatting."""
from __future__ import annotations

from backend.bots.base import BaseBot
from backend.models import BotResponse, SessionContext


class ResearchBot(BaseBot):
    bot_id = "research"
    description = "Web research, summarisation, and citation formatting"
    model = "llama-3.1-8b-instant"
    temperature = 0.3

    def build_system_prompt(self) -> str:
        return (
            "You are a research assistant. You perform thorough research, "
            "summarise findings clearly, and always cite your sources. "
            "Format citations as [Source: URL or publication name]. "
            "Be factual, precise, and comprehensive."
        )

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        # Delegates to base LLM-based implementation
        # In production, this would integrate web search tools
        return await super().handle(context, message)
