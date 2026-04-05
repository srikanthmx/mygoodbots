"""GitDeployer — applies a unified diff, commits, and pushes to trigger auto-deploy."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[2]))
GIT_REMOTE = os.getenv("GIT_REMOTE", "origin")
GIT_BRANCH = os.getenv("GIT_BRANCH", "main")
DEPLOY_HOOK_URL = os.getenv("DEPLOY_HOOK_URL", "")  # optional Railway/Render deploy hook


class GitDeployer:
    """
    Applies a unified diff to the repo, commits, and pushes.
    Railway/Render/Fly auto-deploy on push — no extra step needed unless
    DEPLOY_HOOK_URL is set (e.g. Render manual deploy hook).
    """

    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        remote: str = GIT_REMOTE,
        branch: str = GIT_BRANCH,
        deploy_hook_url: str = DEPLOY_HOOK_URL,
    ) -> None:
        self.repo_root = repo_root
        self.remote = remote
        self.branch = branch
        self.deploy_hook_url = deploy_hook_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def apply(self, diff: str) -> None:
        """Apply diff, commit, push. Runs git commands in a thread pool."""
        await asyncio.get_event_loop().run_in_executor(None, self._apply_sync, diff)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            list(args),
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def _apply_sync(self, diff: str) -> None:
        if not diff.strip():
            logger.warning("GitDeployer.apply called with empty diff — skipping")
            return

        # 1. Write diff to a temp file and apply with `git apply`
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(diff)
            patch_path = f.name

        try:
            self._run("git", "apply", "--check", patch_path)  # dry-run first
            self._run("git", "apply", patch_path)
            logger.info("Patch applied successfully")
        except subprocess.CalledProcessError as exc:
            logger.error("git apply failed: %s\n%s", exc.stderr, exc.stdout)
            raise RuntimeError(f"Failed to apply patch: {exc.stderr}") from exc
        finally:
            Path(patch_path).unlink(missing_ok=True)

        # 2. Stage all changes
        self._run("git", "add", "-A")

        # 3. Commit
        commit_msg = "bot: apply AI-generated code changes [skip ci]"
        try:
            self._run("git", "commit", "-m", commit_msg)
            logger.info("Committed: %s", commit_msg)
        except subprocess.CalledProcessError:
            logger.info("Nothing to commit after patch — already clean")
            return

        # 4. Push → triggers Railway/Render/Fly auto-deploy
        self._run("git", "push", self.remote, self.branch)
        logger.info("Pushed to %s/%s", self.remote, self.branch)

        # 5. Optional: hit a deploy webhook (Render manual deploy hook, etc.)
        if self.deploy_hook_url:
            import urllib.request
            urllib.request.urlopen(self.deploy_hook_url, timeout=10)  # noqa: S310
            logger.info("Deploy hook triggered: %s", self.deploy_hook_url)
