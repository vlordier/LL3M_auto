"""LLM client abstraction for OpenAI and local models."""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import structlog
from openai import AsyncOpenAI

from .config import get_settings

logger = structlog.get_logger(__name__)

# HTTP status constants
HTTP_OK = 200


class LMStudioError(Exception):
    """Custom exception for LM Studio API errors."""
    
    def __init__(self, message: str, status_code: int | None = None):
        """Initialize LM Studio error."""
        super().__init__(message)
        self.status_code = status_code


def _raise_lm_studio_error(error_text: str, status_code: int | None = None) -> None:
    """Helper function to raise LM Studio API errors."""
    raise LMStudioError(f"LM Studio API error: {error_text}", status_code)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a chat completion."""

    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Create a streaming chat completion."""


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self):
        """Initialize OpenAI client."""
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai.api_key)
        self.default_model = settings.openai.model
        self.default_temperature = settings.openai.temperature
        self.default_max_tokens = settings.openai.max_tokens

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a chat completion."""
        try:
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                **kwargs,
            )
            return response.model_dump()
        except Exception as e:
            logger.exception("OpenAI chat completion failed", error=str(e))
            raise

    async def stream_chat_completion(  # type: ignore[override,misc]
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Create a streaming chat completion."""
        try:
            stream = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:  # type: ignore[union-attr]
                yield chunk.model_dump()
        except Exception as e:
            logger.exception("OpenAI streaming chat completion failed", error=str(e))
            raise


class LMStudioClient(LLMClient):
    """LM Studio local LLM client."""

    def __init__(self):
        """Initialize LM Studio client."""
        settings = get_settings()
        self.base_url = settings.lmstudio.base_url
        self.api_key = settings.lmstudio.api_key
        self.timeout = settings.lmstudio.timeout
        self.default_model = settings.lmstudio.model
        self.default_temperature = settings.lmstudio.temperature
        self.default_max_tokens = settings.lmstudio.max_tokens

    async def _get_available_models(self) -> list[str]:
        """Get available models from LM Studio."""
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == HTTP_OK:
                    data = await response.json()
                    models = [model["id"] for model in data.get("data", [])]
                    if models:
                        return models
        except Exception as e:
            logger.warning("Failed to get models from LM Studio", error=str(e))

        return ["local-model"]  # Fallback

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a chat completion."""
        # Auto-detect model if default is used
        if model is None or model == "local-model":
            available_models = await self._get_available_models()
            model = available_models[0] if available_models else "local-model"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
            **kwargs,
        }

        try:
            async with aiohttp.ClientSession() as session, session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != HTTP_OK:
                    error_text = await response.text()
                    _raise_lm_studio_error(error_text)

                data = await response.json()
                return dict(data)  # Ensure dict[str, Any] return type

        except Exception as e:
            logger.exception("LM Studio chat completion failed", error=str(e))
            raise

    async def stream_chat_completion(  # type: ignore[override,misc]
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Create a streaming chat completion."""
        # Auto-detect model if default is used
        if model is None or model == "local-model":
            available_models = await self._get_available_models()
            model = available_models[0] if available_models else "local-model"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
            "stream": True,
            **kwargs,
        }

        try:
            async with aiohttp.ClientSession() as session, session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != HTTP_OK:
                    error_text = await response.text()
                    _raise_lm_studio_error(error_text)

                async for line_bytes in response.content:
                    line = line_bytes.decode("utf-8").strip()
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            yield data
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.exception("LM Studio streaming chat completion failed", error=str(e))
            raise


def get_llm_client() -> LLMClient:
    """Get the appropriate LLM client based on configuration."""
    settings = get_settings()

    if settings.app.use_local_llm:
        logger.info("Using LM Studio local LLM")
        return LMStudioClient()
    else:
        logger.info("Using OpenAI API")
        return OpenAIClient()
