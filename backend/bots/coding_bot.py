"""CodingBot — analyses Telegram requests, edits code, validates, and pushes to git."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import uuid
from pathlib import Path

from backend.bots.base import BaseBot
from backend.models import BotResponse, SessionContext

logger = logging.getLogger(__name__)

REPO_ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[2]))

# File extensions we'll read for context
_CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml", ".toml", ".md"}
# Directories to skip when gathering context
_SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "dist", "build", ".egg-info"}
# Max chars of repo context to send to LLM (keep prompt manageable)
_MAX_CONTEXT_CHARS = 40_000


def _gather_repo_context(root: Path, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Walk the repo and return a concatenated snapshot of source files."""
    parts: list[str] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        # Skip unwanted dirs
        if any(skip in path.parts for skip in _SKIP_DIRS):
            continue
        if path.suffix not in _CODE_EXTS:
            continue
        try:
            content = path.read_text(errors="replace")
        except Exception:
            continue
        rel = path.relative_to(root)
        entry = f"### {rel}\n{content}\n"
        if total + len(entry) > max_chars:
            parts.append(f"### {rel}\n[truncated — file too large]\n")
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


def _run_sync(*args: str, cwd: Path) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined output)."""
    result = subprocess.run(
        list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


class CodingBot(BaseBot):
    bot_id = "coding"
    description = "Code generation, review, debugging, and smart refactoring with approval gate"
    model = "llama-3.1-8b-instant"
    temperature = 0.2

    def __init__(self, llm, approval_gate) -> None:
        super().__init__(llm)
        self._approval_gate = approval_gate
        self._deployer = None  # injectable — set in main.py

    def build_system_prompt(self) -> str:
        return (
            "You are an expert software engineer. You write clean, well-tested, "
            "and maintainable code. When asked to make changes, you analyse the "
            "provided codebase carefully and produce a valid unified diff. "
            "Always follow best practices and consider edge cases. "
            "Output ONLY valid `git diff`-style unified diffs — no prose outside "
            "the DIFF/SUMMARY markers."
        )

    # ------------------------------------------------------------------
    # Step 1 — Analyse request and generate a diff with full repo context
    # ------------------------------------------------------------------

    async def _analyse_and_generate_diff(
        self, message: str, repo_context: str
    ) -> tuple[str, str]:
        """Ask the LLM to produce a unified diff given the full repo snapshot."""
        prompt = (
            "You have been given the full source of a repository below.\n"
            "Analyse the change request and produce:\n"
            "1. A valid unified diff (git diff format) of ALL proposed changes.\n"
            "   Use real file paths relative to the repo root.\n"
            "   If no changes are needed, leave DIFF empty.\n"
            "2. A plain-English summary of what changes are proposed and why.\n\n"
            f"Change request: {message}\n\n"
            "=== REPOSITORY SOURCE ===\n"
            f"{repo_context}\n"
            "=== END REPOSITORY SOURCE ===\n\n"
            "Respond in EXACTLY this format:\n"
            "DIFF:\n<unified diff or empty>\n\nSUMMARY:\n<explanation>"
        )
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        raw = await self._llm.complete(
            model=self.model, messages=messages, temperature=self.temperature
        )

        diff, summary = "", raw
        if "DIFF:" in raw and "SUMMARY:" in raw:
            parts = raw.split("SUMMARY:", 1)
            diff = parts[0].replace("DIFF:", "").strip()
            summary = parts[1].strip()
        return diff, summary

    # ------------------------------------------------------------------
    # Step 2 — Validate the diff (dry-run git apply)
    # ------------------------------------------------------------------

    def _validate_diff(self, diff: str) -> tuple[bool, str]:
        """Run `git apply --check` to verify the patch applies cleanly."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(diff)
            patch_path = f.name
        try:
            code, out = _run_sync("git", "apply", "--check", patch_path, cwd=REPO_ROOT)
            return code == 0, out
        finally:
            Path(patch_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Step 3 — Run tests / lint after applying (in a temp branch)
    # ------------------------------------------------------------------

    async def _run_validation(self, diff: str) -> tuple[bool, str]:
        """Apply diff to a temp branch, run tests, then reset."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_validation_sync, diff
        )

    def _run_validation_sync(self, diff: str) -> tuple[bool, str]:
        import tempfile

        # Stash current state so we can restore on failure
        _run_sync("git", "stash", "--include-untracked", cwd=REPO_ROOT)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(diff)
            patch_path = f.name

        try:
            # Apply the patch
            code, out = _run_sync("git", "apply", patch_path, cwd=REPO_ROOT)
            if code != 0:
                return False, f"git apply failed:\n{out}"

            # Run tests (pytest) — non-fatal if pytest not installed
            test_code, test_out = _run_sync(
                "python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short",
                cwd=REPO_ROOT,
            )
            if test_code != 0:
                return False, f"Tests failed:\n{test_out}"

            return True, test_out

        except Exception as exc:
            return False, str(exc)

        finally:
            Path(patch_path).unlink(missing_ok=True)
            # Always restore original state — the deployer will re-apply on approval
            _run_sync("git", "checkout", ".", cwd=REPO_ROOT)
            _run_sync("git", "stash", "pop", cwd=REPO_ROOT)

    # ------------------------------------------------------------------
    # Step 4 — Apply + push (called after human approval)
    # ------------------------------------------------------------------

    async def _apply_and_deploy(self, diff: str) -> None:
        if self._deployer:
            await self._deployer.apply(diff)
        else:
            logger.warning("CodingBot: no deployer configured — skipping git push")

    # ------------------------------------------------------------------
    # Main handle
    # ------------------------------------------------------------------

    async def handle(self, context: SessionContext, message: str) -> BotResponse:
        # 1. Gather repo context
        repo_context = await asyncio.get_event_loop().run_in_executor(
            None, _gather_repo_context, REPO_ROOT
        )

        # 2. Generate diff with full context
        diff, summary = await self._analyse_and_generate_diff(message, repo_context)

        if not diff:
            return BotResponse(reply="No code changes needed for that request.", bot_id=self.bot_id)

        # 3. Validate diff applies cleanly
        valid, validate_msg = self._validate_diff(diff)
        if not valid:
            return BotResponse(
                reply=(
                    f"⚠️ Generated diff does not apply cleanly — cannot proceed.\n\n"
                    f"Error: {validate_msg}\n\n"
                    f"Summary of intended changes: {summary}\n\n"
                    "Please rephrase your request with more detail."
                ),
                bot_id=self.bot_id,
            )

        # 4. Run tests against the proposed changes
        tests_pass, test_output = await self._run_validation(diff)
        if not tests_pass:
            return BotResponse(
                reply=(
                    f"⚠️ Proposed changes fail validation — not submitting for approval.\n\n"
                    f"Test output:\n```\n{test_output[:2000]}\n```\n\n"
                    f"Summary of intended changes: {summary}"
                ),
                bot_id=self.bot_id,
            )

        # 5. Submit for human approval
        approval_id = str(uuid.uuid4())

        async def on_decision(approved: bool, reason: str) -> None:
            if approved:
                try:
                    await self._apply_and_deploy(diff)
                    await self._approval_gate.notify_human(
                        approval_id, "✅ Changes applied and pushed — deployment triggered."
                    )
                except Exception as exc:
                    await self._approval_gate.notify_human(
                        approval_id, f"❌ Deploy failed: {exc}"
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
                f"✅ Tests passed. Proposed changes submitted for approval.\n\n"
                f"ID: `{approval_id}`\n\n"
                f"Summary: {summary}\n\n"
                "Approve or reject via the Telegram buttons or Web UI."
            ),
            bot_id=self.bot_id,
        )
