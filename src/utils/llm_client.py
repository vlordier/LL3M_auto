"""LLM client abstraction for OpenAI and local models."""

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

# LangChain imports (optional, fallback to manual retry if not available)
try:
    from langchain.output_parsers import RetryOutputParser
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .config import get_settings

logger = structlog.get_logger(__name__)

# HTTP status constants
HTTP_OK = 200


def _safe_extract_content(content: str | list[dict[str, Any]] | None) -> str:
    """Safely extract text content from message content field.

    Handles both string format and list format (with content parts).

    Args:
        content: Message content - can be string, list of content parts, or None

    Returns:
        Extracted text content as string
    """
    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Extract text from content parts
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts)

    # Fallback - convert to string
    return str(content)


def _safe_append_to_message_content(message: dict[str, Any], additional_text: str) -> None:
    """Safely append text to a message content field.

    Handles both string format and list format (with content parts).

    Args:
        message: Message dictionary to modify
        additional_text: Text to append to the message content
    """
    content = message.get("content")

    if isinstance(content, list):
        # Append as a new text part for list format
        content.append({"type": "text", "text": additional_text})
    elif isinstance(content, str):
        # String format - concatenate directly
        message["content"] = content + additional_text
    elif content is None:
        # Initialize with text if None
        message["content"] = additional_text
    else:
        # Fallback - convert to string and append
        message["content"] = str(content) + additional_text


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
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: str | int | float | bool,
    ) -> dict[str, str | int | float | list[dict[str, str]]]:
        """Create a chat completion."""

    @abstractmethod
    def stream_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: str | int | float | bool,
    ) -> AsyncGenerator[dict[str, str | int | float | list[dict[str, str]]], None]:
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
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a chat completion."""
        try:
            # Use the appropriate token parameter based on model
            model_name = model or self.default_model
            if "gpt-5" in model_name:
                # GPT-5 uses max_completion_tokens and supports enhanced parameters
                params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature or self.default_temperature,
                    "max_completion_tokens": max_tokens or self.default_max_tokens,
                }

                # Add GPT-5 specific parameters if provided
                if "reasoning_effort" in kwargs:
                    params["reasoning_effort"] = kwargs.pop("reasoning_effort")
                if "verbosity" in kwargs:
                    params["verbosity"] = kwargs.pop("verbosity")

                # Add remaining kwargs
                params.update(kwargs)

                # Type safety: ensure params match OpenAI API
                safe_params = {
                    "model": params["model"],
                    "messages": params["messages"],
                    "temperature": params.get("temperature"),
                    "max_completion_tokens": params.get("max_completion_tokens"),
                }
                if "response_format" in params:
                    safe_params["response_format"] = params["response_format"]
                response = await self.client.chat.completions.create(**safe_params)
            else:
                # GPT-4 and earlier use max_tokens
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    **kwargs,
                )
            return response.model_dump()
        except Exception as e:
            logger.exception("OpenAI chat completion failed", error=str(e))
            raise

    def _supports_structured_output(self, model_name: str) -> bool:
        """Check if a model supports structured output."""
        # Models that support structured output as of 2025
        supported_models = [
            # GPT-4o family
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            # GPT-5 family (confirmed to support structured output)
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-chat",
            # Note: gpt-5-chat-latest doesn't support structured output yet
            # Additional current models
            "chatgpt-4o-latest",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
        ]

        # Check if model is in supported list (handles versioned models)
        return any(model_name.startswith(supported) for supported in supported_models)

    async def structured_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[BaseModel],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Create a structured chat completion with guaranteed JSON schema compliance."""
        model_name = model or self.default_model

        # Check if model supports structured output
        if not self._supports_structured_output(model_name):
            logger.warning(
                f"Model {model_name} doesn't support structured output, falling back to JSON parsing",
                supported_models=[
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-5-nano",
                    "gpt-5-chat",
                ],
            )
            return await self._fallback_structured_completion(
                messages, response_format, model_name, temperature, max_tokens, **kwargs
            )

        # GPT-5 models need special handling due to known issues
        max_retries = 3 if model_name.startswith("gpt-5") else 1

        for attempt in range(max_retries):
            try:
                return await self._attempt_structured_completion(
                    messages,
                    response_format,
                    model_name,
                    temperature,
                    max_tokens,
                    attempt,
                    **kwargs,
                )
            except Exception as e:
                error_msg = str(e).lower()
                is_gpt5_known_issue = any(
                    keyword in error_msg
                    for keyword in [
                        "malformed",
                        "length limit",
                        "thinking",
                        "reasoning",
                        "could not parse response",
                    ]
                )

                if is_gpt5_known_issue and attempt < max_retries - 1:
                    logger.warning(f"GPT-5 structured output attempt {attempt + 1} failed: {e}. Retrying...")
                    continue

                # Final attempt failed or non-retryable error
                logger.warning(
                    f"Structured output failed for {model_name} after {attempt + 1} attempts, using fallback: {e}"
                )
                return await self._fallback_structured_completion(
                    messages,
                    response_format,
                    model_name,
                    temperature,
                    max_tokens,
                    **kwargs,
                )

    async def _attempt_structured_completion(  # noqa: PLR0912
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[BaseModel],
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        attempt: int = 0,
        **kwargs: Any,
    ) -> BaseModel:
        """Single attempt at structured completion with model-specific optimizations."""
        # Prepare parameters for structured output
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
        }

        # Convert Pydantic model to proper JSON schema format for GPT-5
        if model_name.startswith("gpt-5"):
            # GPT-5 requires explicit JSON schema format
            schema = response_format.model_json_schema()
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            # GPT-4o family can use the Pydantic model directly
            params["response_format"] = response_format

        # Use appropriate token parameter based on model
        if model_name.startswith("gpt-5"):
            # For GPT-5, use more conservative token limits on retries to avoid length issues
            max_completion_tokens = max_tokens or self.default_max_tokens
            if attempt > 0:
                max_completion_tokens = min(max_completion_tokens, 8000)  # More conservative on retry
            params["max_completion_tokens"] = max_completion_tokens

            # Add GPT-5 enhanced parameters if provided, but be more conservative on retries
            if "reasoning_effort" in kwargs:
                effort = kwargs.pop("reasoning_effort")
                # Reduce reasoning effort on retries to avoid length issues
                if attempt > 0 and effort in ["high", "max"]:
                    effort = "medium"
                params["reasoning_effort"] = effort

            if "verbosity" in kwargs:
                verbosity = kwargs.pop("verbosity")
                # Reduce verbosity on retries
                if attempt > 0 and verbosity > 0:
                    verbosity = max(0, verbosity - 1)
                params["verbosity"] = verbosity
        else:
            params["max_tokens"] = max_tokens or self.default_max_tokens

        # Add any remaining kwargs
        params.update(kwargs)

        # Use appropriate API method based on model and response format
        if model_name.startswith("gpt-5") and isinstance(params["response_format"], dict):
            # GPT-5 with manual JSON schema uses create method
            response = await self.client.chat.completions.create(**params)

            # Manual parsing for GPT-5 JSON schema response
            content = response.choices[0].message.content
            if not content:
                if hasattr(response.choices[0].message, "refusal") and response.choices[0].message.refusal:
                    raise ValueError(f"Model refused to respond: {response.choices[0].message.refusal}")
                else:
                    raise ValueError("No content in response")

            try:
                # Parse JSON and create Pydantic model
                json_data = json.loads(content)
                return response_format.model_validate(json_data)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content}") from e
        else:
            # GPT-4o family uses parse method
            response = await self.client.chat.completions.parse(**params)

            if response.choices[0].message.parsed is None:
                # Fall back to refusal or error handling
                if response.choices[0].message.refusal:
                    raise ValueError(f"Model refused to respond: {response.choices[0].message.refusal}")
                else:
                    raise ValueError("Failed to parse structured response - parsed field is None")

            return response.choices[0].message.parsed

    async def _fallback_structured_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[BaseModel],
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Fallback to RetryOutputParser (if available) or manual parsing for unsupported models."""
        # Try RetryOutputParser if LangChain is available
        if LANGCHAIN_AVAILABLE:
            try:
                logger.info("Attempting RetryOutputParser fallback")
                return await self._retry_parser_fallback(
                    messages,
                    response_format,
                    model_name,
                    temperature,
                    max_tokens,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"RetryOutputParser fallback failed: {e}, using manual fallback")
        else:
            logger.info("LangChain not available, using manual fallback")

        # Final fallback to our manual parsing
        return await self._manual_json_fallback(
            messages, response_format, model_name, temperature, max_tokens, **kwargs
        )

    async def _retry_parser_fallback(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[BaseModel],
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Use RetryOutputParser for robust parsing with retries."""
        # Create LangChain ChatOpenAI instance for RetryOutputParser
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            openai_api_key=self.client.api_key,
            **kwargs,
        )

        # Create parsers
        base_parser = PydanticOutputParser(pydantic_object=response_format)
        retry_parser = RetryOutputParser.from_llm(parser=base_parser, llm=llm, max_retries=2)

        # Format messages for LangChain
        if not messages:
            raise ValueError("No messages provided for structured completion")

        # Combine system and user messages
        system_content = ""
        user_content = ""

        for msg in messages:
            content_str = _safe_extract_content(msg["content"])

            if msg["role"] == "system":
                system_content += content_str + "\n"
            elif msg["role"] == "user":
                user_content += content_str + "\n"

        # Add format instructions to user message
        format_instructions = base_parser.get_format_instructions()
        full_prompt = f"{system_content.strip()}\n\n{user_content.strip()}\n\n{format_instructions}"

        # Use RetryOutputParser to get structured response
        # First try to get a completion, then parse with prompt for retry capability
        response_data = await self.chat_completion(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Extract content
        bad_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Use parse_with_prompt as documented - this allows RetryOutputParser to fix bad responses
        # RetryOutputParser expects a PromptValue object, so create one from our formatted prompt
        from langchain_core.prompt_values import StringPromptValue

        prompt_value = StringPromptValue(text=full_prompt)
        result = retry_parser.parse_with_prompt(bad_response, prompt_value)

        logger.info("RetryOutputParser successfully parsed structured response")
        return result

    async def _manual_json_fallback(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[BaseModel],
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Robust manual fallback with retry logic for JSON parsing."""
        logger.info("Using manual JSON fallback with retries")

        # Generate schema for the model
        schema = response_format.model_json_schema()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add JSON format instruction to the last user message
                enhanced_messages = list(messages)
                if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                    retry_note = f" (Retry {attempt + 1}/{max_retries})" if attempt > 0 else ""
                    last_msg = enhanced_messages[-1]

                    # Use safe helper function to append instructions
                    additional_instruction = (
                        f"\n\nIMPORTANT{retry_note}: Respond with valid JSON matching this exact schema:\n"
                        f"{json.dumps(schema, indent=2)}\n\n"
                        "Requirements:\n"
                        "- Response must contain ONLY the JSON object\n"
                        "- No additional text, explanations, or markdown\n"
                        "- All required fields must be present\n"
                        "- Use exact field names and types as shown"
                    )

                    _safe_append_to_message_content(last_msg, additional_instruction)

                # Make regular completion
                response_data = await self.chat_completion(
                    messages=enhanced_messages,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                # Extract content
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Parse the response
                parsed_result = self._extract_and_parse_json(content, response_format)

                logger.info(f"Manual JSON fallback successful on attempt {attempt + 1}")
                return parsed_result

            except Exception as e:
                logger.warning(f"Manual fallback attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.exception("All manual fallback attempts failed")
                    raise ValueError(f"All fallback methods failed after {max_retries} attempts: {e}") from e

                # Brief delay before retry
                await asyncio.sleep(0.5 * (attempt + 1))

        raise ValueError("Fallback retry loop exited unexpectedly")

    def _extract_and_parse_json(self, content: str, response_format: type[BaseModel]) -> BaseModel:
        """Extract and parse JSON from LLM response content."""
        # Remove markdown code blocks
        cleaned_response = content.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        # Find JSON boundaries using brace counting (our improved method)
        cleaned_response = cleaned_response.strip()
        start_idx = cleaned_response.find("{")
        if start_idx == -1:
            raise ValueError(f"No JSON object found in response: {content[:200]}...")

        # Find the end of the JSON object by counting braces
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(cleaned_response[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        json_str = cleaned_response[start_idx:end_idx]

        # Parse and validate
        json_data = json.loads(json_str)
        return response_format(**json_data)

    async def stream_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, str | int | float | list[dict[str, str]]], None]:
        """Create a streaming chat completion."""
        try:
            # Use the appropriate token parameter based on model
            model_name = model or self.default_model
            if "gpt-5" in model_name:
                # GPT-5 uses max_completion_tokens instead of max_tokens
                stream_response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature or self.default_temperature,
                    max_completion_tokens=max_tokens or self.default_max_tokens,
                    stream=True,
                    **kwargs,
                )
            else:
                # GPT-4 and earlier use max_tokens
                stream_response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    stream=True,
                    **kwargs,
                )
            # Stream response is AsyncStream[ChatCompletionChunk]
            async for chunk in stream_response:
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
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response,
            ):
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
        messages: list[ChatCompletionMessageParam],
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
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response,
            ):
                if response.status != HTTP_OK:
                    error_text = await response.text()
                    _raise_lm_studio_error(error_text)

                data = await response.json()
                return dict(data)  # Ensure dict[str, Any] return type

        except Exception as e:
            logger.exception("LM Studio chat completion failed", error=str(e))
            raise

    async def stream_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, str | int | float | list[dict[str, str]]], None]:
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
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response,
            ):
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
