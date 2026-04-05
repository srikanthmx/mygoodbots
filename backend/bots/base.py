"""BaseBot abstract base class for all bot agents."""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.llm.client import LLMClient
from backend.models import BotResponse, SessionContext


class BaseBot(ABC):
    bot_id: str
    description: str
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.7

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @abstractmethod
    def build_system_prompt(self) -> str: ...

    @abstractmethod
    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        """
        Build the message list from context history and call the LLM.

        Preconditions:
          - context.history is a valid list of Message objects
          - message is the current user message (already appended to history by dispatcher)

        Postconditions:
          - Returns BotResponse with reply populated
          - Does NOT mutate context — read-only access only
        """
        # 1. Build message list: system prompt + history (excluding last user message) + current message
        messages: list[dict] = [{"role": "system", "content": self.build_system_prompt()}]
        for turn in context.history[:-1]:  # exclude the just-appended user message
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": self.preprocess(message)})

        # 2. Call LLM
        raw = await self._llm.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        # 3. Return postprocessed response
        return BotResponse(reply=self.postprocess(raw), bot_id=self.bot_id)

    def preprocess(self, message: str) -> str:
        return message.strip()

    def postprocess(self, raw: str) -> str:
        return raw.strip()
