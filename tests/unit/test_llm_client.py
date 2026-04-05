"""Unit tests for LLMClient."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.llm.client import LLMClient
from backend.models import LLMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client(**kwargs) -> LLMClient:
    defaults = dict(
        base_url="http://ollama:11434",
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="test-key",
        ollama_models={"llama3", "mistral"},
        max_retries=2,
        backoff_seconds=0.0,  # no real sleeping in tests
    )
    defaults.update(kwargs)
    return LLMClient(**defaults)


# ---------------------------------------------------------------------------
# _resolve_provider
# ---------------------------------------------------------------------------

def test_resolve_provider_ollama():
    client = make_client()
    assert client._resolve_provider("llama3") == "ollama"
    assert client._resolve_provider("mistral") == "ollama"


def test_resolve_provider_openai():
    client = make_client()
    assert client._resolve_provider("gpt-4") == "openai"
    assert client._resolve_provider("gpt-3.5-turbo") == "openai"


# ---------------------------------------------------------------------------
# temperature validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_invalid_temperature_below():
    client = make_client()
    with pytest.raises(ValueError, match="temperature"):
        await client.complete("llama3", [{"role": "user", "content": "hi"}], temperature=-0.1)


@pytest.mark.asyncio
async def test_complete_invalid_temperature_above():
    client = make_client()
    with pytest.raises(ValueError, match="temperature"):
        await client.complete("llama3", [{"role": "user", "content": "hi"}], temperature=2.1)


@pytest.mark.asyncio
async def test_complete_boundary_temperatures_accepted():
    """0.0 and 2.0 are valid boundary values."""
    client = make_client()

    ollama_response = {"message": {"content": "hello"}}

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ollama_response
        mock_http.post = AsyncMock(return_value=mock_resp)

        result = await client.complete("llama3", [{"role": "user", "content": "hi"}], temperature=0.0)
        assert result == "hello"

        result = await client.complete("llama3", [{"role": "user", "content": "hi"}], temperature=2.0)
        assert result == "hello"


# ---------------------------------------------------------------------------
# Non-streaming: Ollama
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_complete_success():
    client = make_client()
    ollama_response = {"message": {"content": "I am llama"}}

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ollama_response
        mock_http.post = AsyncMock(return_value=mock_resp)

        result = await client.complete("llama3", [{"role": "user", "content": "hello"}])

    assert result == "I am llama"


# ---------------------------------------------------------------------------
# Non-streaming: OpenAI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_complete_success():
    client = make_client()
    openai_response = {
        "choices": [{"message": {"content": "I am GPT"}}]
    }

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = openai_response
        mock_http.post = AsyncMock(return_value=mock_resp)

        result = await client.complete("gpt-4", [{"role": "user", "content": "hello"}])

    assert result == "I am GPT"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retries_on_5xx_then_succeeds():
    client = make_client(max_retries=2, backoff_seconds=0.0)
    ollama_response = {"message": {"content": "ok"}}

    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            mock_err_resp = MagicMock()
            mock_err_resp.status_code = 503
            raise httpx.HTTPStatusError("server error", request=MagicMock(), response=mock_err_resp)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ollama_response
        return mock_resp

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = fake_post

        result = await client.complete("llama3", [{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_raises_llm_error_after_max_retries():
    client = make_client(max_retries=2, backoff_seconds=0.0)

    async def always_fail(*args, **kwargs):
        mock_err_resp = MagicMock()
        mock_err_resp.status_code = 500
        raise httpx.HTTPStatusError("server error", request=MagicMock(), response=mock_err_resp)

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = always_fail

        with pytest.raises(LLMError, match="Max retries"):
            await client.complete("llama3", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_raises_llm_error_on_4xx_immediately():
    """4xx errors should NOT be retried — raise LLMError immediately."""
    client = make_client(max_retries=3, backoff_seconds=0.0)
    call_count = 0

    async def client_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_err_resp = MagicMock()
        mock_err_resp.status_code = 400
        raise httpx.HTTPStatusError("bad request", request=MagicMock(), response=mock_err_resp)

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = client_error

        with pytest.raises(LLMError):
            await client.complete("llama3", [{"role": "user", "content": "hi"}])

    assert call_count == 1  # no retries for 4xx


@pytest.mark.asyncio
async def test_retries_on_timeout():
    client = make_client(max_retries=2, backoff_seconds=0.0)
    call_count = 0

    async def timeout_then_ok(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.TimeoutException("timed out")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "recovered"}}
        return mock_resp

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = timeout_then_ok

        result = await client.complete("llama3", [{"role": "user", "content": "hi"}])

    assert result == "recovered"


# ---------------------------------------------------------------------------
# Streaming: returns AsyncIterator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_stream_true_returns_async_iterator():
    client = make_client()

    async def fake_aiter_lines():
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": True}),
        ]
        for line in lines:
            yield line

    mock_stream_resp = MagicMock()
    mock_stream_resp.raise_for_status = MagicMock()
    mock_stream_resp.aiter_lines = fake_aiter_lines

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = MagicMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_http.stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_http.stream.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await client.complete("llama3", [{"role": "user", "content": "hi"}], stream=True)

        assert hasattr(result, "__aiter__"), "stream=True should return an async iterator"

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_openai_stream_returns_async_iterator():
    client = make_client()

    async def fake_aiter_lines():
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "Hi"}}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"content": " there"}}]}),
            'data: [DONE]',
        ]
        for line in lines:
            yield line

    mock_stream_resp = MagicMock()
    mock_stream_resp.raise_for_status = MagicMock()
    mock_stream_resp.aiter_lines = fake_aiter_lines

    with patch("backend.llm.client.httpx.AsyncClient") as mock_cls:
        mock_http = MagicMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_http.stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_http.stream.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await client.complete("gpt-4", [{"role": "user", "content": "hi"}], stream=True)

        assert hasattr(result, "__aiter__")

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

    assert chunks == ["Hi", " there"]
