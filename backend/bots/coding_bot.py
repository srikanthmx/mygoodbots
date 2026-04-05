"""CodingBot — code generation, review, debugging, and smart refactoring with approval gate."""
from __future__ import annotations

import uuid

from backend.bots.base import BaseBot
from backend.models import BotResponse, SessionContext


class CodingBot(BaseBot):
    bot_id = "coding"
    description = "Code generation, review, debugging, and smart refactoring with approval gate"
    model = "llama-3.1-8b-instant"
    temperature = 0.2

    def __init__(self, llm, approval_gate) -> None:
        super().__init__(llm)
        self._approval_gate = approval_gate
        self._deployer = None  # injectable for testing

    def build_system_prompt(self) -> str:
        return (
            "You are an expert software engineer. You write clean, well-tested, "
            "and maintainable code. When asked to make changes, you analyse the "
            "codebase carefully and produce a clear diff with explanation. "
            "Always follow best practices and consider edge cases."
        )

    async def _analyse_and_generate_diff(self, context: SessionContext, message: str) -> tuple[str, str]:
        """Use LLM to analyse the request and generate a proposed diff and summary."""
        analysis_prompt = (
            f"Analyse this code change request and produce:\n"
            f"1. A unified diff of the proposed changes (or empty string if no changes needed)\n"
            f"2. A plain-English summary of what changes are proposed and why\n\n"
            f"Request: {message}\n\n"
            f"Format your response as:\n"
            f"DIFF:\n<unified diff or empty>\n\nSUMMARY:\n<explanation>"
        )
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": analysis_prompt},
        ]
        raw = await self._llm.complete(model=self.model, messages=messages, temperature=self.temperature)

        diff = ""
        summary = raw
        if "DIFF:" in raw and "SUMMARY:" in raw:
            parts = raw.split("SUMMARY:", 1)
            diff_part = parts[0].replace("DIFF:", "").strip()
            summary = parts[1].strip()
            diff = diff_part

        return diff, summary

    async def _apply_changes(self, diff: str) -> None:
        """Apply the approved diff. In production, this writes files."""
        if self._deployer:
            await self._deployer.apply(diff)

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        diff, summary = await self._analyse_and_generate_diff(context, message)

        if not diff:
            return BotResponse(reply="No changes needed.", bot_id=self.bot_id)

        approval_id = str(uuid.uuid4())

        async def on_decision(approved: bool, reason: str) -> None:
            if approved:
                await self._apply_changes(diff)
                await self._approval_gate.notify_human(
                    approval_id, "✅ Changes deployed successfully."
                )
            else:
                await self._approval_gate.notify_human(
                    approval_id, f"❌ Changes rejected: {reason}"
                )

        await self._approval_gate.submit_proposal(
            approval_id=approval_id,
            diff=diff,
            summary=summary,
            callback=on_decision,
        )

        return BotResponse(
            reply=(
                f"🔍 Proposed changes submitted for approval (ID: `{approval_id}`).\n\n"
                f"**Summary:** {summary}\n\n"
                "Please review the diff in Telegram or the Web UI and approve/reject."
            ),
            bot_id=self.bot_id,
        )
