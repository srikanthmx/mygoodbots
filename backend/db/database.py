from __future__ import annotations

import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Create asyncpg connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
            )
            logger.info("Database connection pool created")
        except Exception as exc:
            logger.error("Failed to create database connection pool: %s", exc)
            raise

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            try:
                await self._pool.close()
                logger.info("Database connection pool closed")
            except Exception as exc:
                logger.error("Error closing database connection pool: %s", exc)
                raise
            finally:
                self._pool = None

    async def save_turn(
        self,
        session_id: str,
        bot_id: str,
        user_message: str,
        bot_reply: str,
    ) -> None:
        """Persist a conversation turn to the database."""
        if self._pool is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        try:
            await self._pool.execute(
                """
                INSERT INTO conversation_turns (session_id, bot_id, user_message, bot_reply)
                VALUES ($1, $2, $3, $4)
                """,
                session_id,
                bot_id,
                user_message,
                bot_reply,
            )
        except Exception as exc:
            logger.error(
                "Failed to save conversation turn for session=%s bot=%s: %s",
                session_id,
                bot_id,
                exc,
            )
            raise
