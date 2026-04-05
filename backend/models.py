from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class LLMError(Exception):
    pass


class ApprovalError(Exception):
    pass


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionContext:
    session_id: str
    bot_id: str
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BotResponse:
    reply: str
    bot_id: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class BotInfo:
    bot_id: str
    description: str
    model: str
    available: bool


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published_at: datetime
    category: str  # "financial" | "market" | "geopolitical" | "macro"
    raw_text: str


@dataclass
class NewsSummary:
    digest: str
    market_impact: str  # "bullish" | "bearish" | "neutral"
    key_insights: list[str]
    topics_for_research: list[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResearchReport:
    topic: str
    content: str
    sources: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ApprovalRequest:
    approval_id: str
    diff: str
    summary: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    rejection_reason: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
