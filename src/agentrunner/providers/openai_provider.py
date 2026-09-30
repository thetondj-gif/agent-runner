"""OpenAI provider implementation for Agent Runner."""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import tiktoken
from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ConfigurationError, ModelResponseError
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition
from agentrunner.providers.base import (
    BaseLLMProvider,
    ModelInfo,
    ProviderConfig,
    ProviderResponse,
    StreamChunk,
)
from agentrunner.providers.registry import ModelRegistry


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation.

    Supports GPT-4, GPT-4o, GPT-5 Codex, GPT-3.5-turbo and compatible models.
    Handles OpenAI-specific tool call format and streaming.
    """

    def __init__(
        self,
        api_key: str,
        config: ProviderConfig,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            config: Provider configuration
            base_url: Optional custom API base URL (for OpenAI-compatible APIs)
            max_retries: Maximum number of retries for rate limits
        """
        super().__init__(api_key, config)
        self.max_retries = max_retries

        # Initialize sync and async clients
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _requires_responses_api(self, model: str) -> bool:
        """Check if model requires Responses API (vs Chat Completions).

        O1 and Codex models ONLY support Responses API (no temperature).
        Other models use Chat Completions API (faster, more compatible).

        Args:
            model: Model identifier

        Returns:
            True if model requires Responses API
        """
        responses_only_models = ["o1", "o1-preview", "o1-mini", "gpt-5-codex", "gpt-5.1"]
        return any(m in model.lower() for m in responses_only_models)

    def _get_reasoning_effort(self, model: str) -> str | None:
        """Extract reasoning effort level from model ID.

        For GPT-5.1 models with reasoning suffixes, extract the reasoning level.

        Args:
            model: Model identifier (e.g., "gpt-5.1-2025-11-13-low")

        Returns:
            Reasoning effort level ("low", "medium", "high") or None
        """
        if "gpt-5.1" not in model.lower():
            return None

        # Check for reasoning level suffix
        if model.endswith("-low"):
            return "low"
        elif model.endswith("-medium"):
            return "medium"
        elif model.endswith("-high"):
            return "high"
        else:
            # Base model without suffix means no reasoning
            return None

    def _normalize_model_id(self, model: str) -> str:
        """Normalize model ID by removing reasoning effort suffix.

        Args:
            model: Model identifier with potential reasoning suffix

        Returns:
            Base model identifier without reasoning suffix
        """
        # Remove reasoning effort suffixes for API calls
        for suffix in ["-low", "-medium", "-high"]:
            if model.endswith(suffix):
                return model[: -len(suffix)]
        return model

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> ProviderResponse:
        """Send messages to OpenAI using appropriate API.

        Uses Responses API for O1/Codex (required, slower).
        Uses Chat Completions API for other models (faster).

        Args:
            messages: Conversation history
            tools: Available tools for the LLM to call
            config: Agent configuration (deprecated, not used)

        Returns:
            ProviderResponse with assistant's reply and usage info

        Raises:
            ModelResponseError: If OpenAI returns an error
            TokenLimitExceededError: If input exceeds context window
        """
        # Route to appropriate API
        if self._requires_responses_api(self.config.model):
            return await self._chat_responses_api(messages, tools)
        else:
            return await self._chat_completions_api(messages, tools)

    async def _chat_completions_api(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> ProviderResponse:
        """Chat using Chat Completions API (for gpt-4o, gpt-4-turbo, etc).

        Args:
            messages: Conversation history
            tools: Available tools

        Returns:
            ProviderResponse
        """
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages_to_openai(messages)

        # Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_openai(tools) if tools else None

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
        }

        if openai_tools:
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"

        if self.config.max_tokens:
            api_params["max_tokens"] = self.config.max_tokens

        # Call OpenAI API with retries
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(**api_params)

                # Parse response
                assistant_message = self._parse_response_message(response)
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                return ProviderResponse(messages=[assistant_message], usage=usage)

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise ModelResponseError(
                    message=f"Rate limit exceeded after {self.max_retries} retries",
                    provider="openai",
                    status_code=429,
                ) from e

            except APITimeoutError as e:
                raise ModelResponseError(
                    message="Request timeout",
                    provider="openai",
                    error_code="E_TIMEOUT",
                ) from e

            except APIError as e:
                status_code = getattr(e, "status_code", None)
                raise ModelResponseError(
                    message=str(e),
                    provider="openai",
                    status_code=status_code,
                ) from e

        # Should not reach here
        raise ModelResponseError(
            message="Max retries exceeded",
            provider="openai",
        )

    async def _chat_responses_api(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> ProviderResponse:
        """Chat using Responses API (for gpt-5-codex, o1 models).

        Args:
            messages: Conversation history
            tools: Available tools

        Returns:
            ProviderResponse
        """
        # Convert messages to Responses API format
        instructions, input_items = self._convert_messages_to_responses_format(messages)

        # Convert tools to Responses API format (flat structure)
        openai_tools = self._convert_tools_to_responses_format(tools) if tools else None

        # Normalize model ID (remove reasoning suffix) and extract reasoning effort
        normalized_model = self._normalize_model_id(self.config.model)
        reasoning_effort = self._get_reasoning_effort(self.config.model)

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": normalized_model,
            "input": input_items,
        }

        # O1/Codex models don't support temperature
        # (Already handled by _requires_responses_api check)

        if instructions:
            api_params["instructions"] = instructions

        if openai_tools:
            api_params["tools"] = openai_tools

        if self.config.max_tokens:
            api_params["max_output_tokens"] = self.config.max_tokens

        # Add reasoning effort if specified
        if reasoning_effort:
            api_params["reasoning"] = {"effort": reasoning_effort}

        # Call Responses API with retries
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.responses.create(**api_params)

                # Parse response
                assistant_message = self._parse_responses_message(response)
                usage = {
                    "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                    "completion_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                return ProviderResponse(messages=[assistant_message], usage=usage)

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise ModelResponseError(
                    message=f"Rate limit exceeded after {self.max_retries} retries",
                    provider="openai",
                    status_code=429,
                ) from e

            except APITimeoutError as e:
                raise ModelResponseError(
                    message="Request timeout",
                    provider="openai",
                    error_code="E_TIMEOUT",
                ) from e

            except APIError as e:
                status_code = getattr(e, "status_code", None)
                raise ModelResponseError(
                    message=str(e),
                    provider="openai",
                    status_code=status_code,
                ) from e

        # Should not reach here
        raise ModelResponseError(
            message="Max retries exceeded",
            provider="openai",
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream messages from OpenAI using appropriate API.

        Uses Responses API for O1/Codex (required, slower).
        Uses Chat Completions API for other models (faster).

        Args:
            messages: Conversation history
            tools: Available tools for the LLM to call
            config: Agent configuration (deprecated, not used)

        Yields:
            StreamChunk objects with incremental response data

        Raises:
            ModelResponseError: If OpenAI returns an error
        """
        # Route to appropriate API
        if self._requires_responses_api(self.config.model):
            async for chunk in self._chat_stream_responses_api(messages, tools):
                yield chunk
        else:
            async for chunk in self._chat_stream_completions_api(messages, tools):
                yield chunk

    async def _chat_stream_completions_api(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream using Chat Completions API (for gpt-4o, etc).

        Args:
            messages: Conversation history
            tools: Available tools

        Yields:
            StreamChunk objects
        """
        # Convert messages and tools
        openai_messages = self._convert_messages_to_openai(messages)
        openai_tools = self._convert_tools_to_openai(tools) if tools else None

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "stream": True,
        }

        if openai_tools:
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"

        if self.config.max_tokens:
            api_params["max_tokens"] = self.config.max_tokens

        try:
            stream = await self.async_client.chat.completions.create(**api_params)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle text content
                if delta.content:
                    yield StreamChunk(
                        type="token",
                        payload={"content": delta.content},
                    )

                # Handle tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        # OpenAI streams tool calls incrementally
                        # We accumulate and emit complete tool calls
                        if tool_call.function:
                            yield StreamChunk(
                                type="tool_call",
                                payload={
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            )

                # Handle finish reason
                if chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        type="status",
                        payload={"finish_reason": chunk.choices[0].finish_reason},
                    )

        except RateLimitError as e:
            yield StreamChunk(
                type="error",
                payload={"error": "rate_limit", "message": str(e)},
            )

        except APITimeoutError as e:
            yield StreamChunk(
                type="error",
                payload={"error": "timeout", "message": str(e)},
            )

        except APIError as e:
            yield StreamChunk(
                type="error",
                payload={"error": "api_error", "message": str(e)},
            )

    def get_model_info(self) -> ModelInfo:
        """Get information about the model from registry.

        Returns:
            ModelInfo with context window size and pricing
        """
        try:
            model_spec = ModelRegistry.get_model_spec(self.config.model)
            return model_spec.to_model_info()
        except ConfigurationError:
            # Unknown model IDs are permitted only when the caller explicitly
            # configured an OpenAI-compatible endpoint. Local models have no
            # external token cost; context size is supplied by configuration.
            if not self.config.provider_extensions.get("openai_compatible"):
                raise
            context_window = int(
                self.config.provider_extensions.get("context_window")
                or os.getenv("AGENTRUNNER_CONTEXT_WINDOW", "32768")
            )
            return ModelInfo(
                name=self.config.model,
                context_window=context_window,
                pricing={"input_per_1k": 0.0, "output_per_1k": 0.0},
            )

    async def _chat_stream_responses_api(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream using Responses API (for gpt-5-codex, o1 models).

        Args:
            messages: Conversation history
            tools: Available tools

        Yields:
            StreamChunk objects
        """
        # Convert messages to Responses API format
        instructions, input_items = self._convert_messages_to_responses_format(messages)
        # Convert tools to Responses API format (flat structure)
        openai_tools = self._convert_tools_to_responses_format(tools) if tools else None

        # Normalize model ID (remove reasoning suffix) and extract reasoning effort
        normalized_model = self._normalize_model_id(self.config.model)
        reasoning_effort = self._get_reasoning_effort(self.config.model)

        # Prepare API parameters
        api_params: dict[str, Any] = {
            "model": normalized_model,
            "input": input_items,
            "stream": True,
        }

        # O1/Codex models don't support temperature
        # (Already handled by _requires_responses_api check)

        if instructions:
            api_params["instructions"] = instructions

        if openai_tools:
            api_params["tools"] = openai_tools

        if self.config.max_tokens:
            api_params["max_output_tokens"] = self.config.max_tokens

        # Add reasoning effort if specified
        if reasoning_effort:
            api_params["reasoning"] = {"effort": reasoning_effort}

        try:
            stream = await self.async_client.responses.create(**api_params)

            async for event in stream:
                # Handle text deltas
                if event.type == "response.output_text.delta":
                    yield StreamChunk(
                        type="token",
                        payload={"content": event.delta},
                    )

                # Handle tool call events
                elif event.type == "response.function_call_arguments.done":
                    yield StreamChunk(
                        type="tool_call",
                        payload={
                            "id": getattr(event, "call_id", None),
                            "name": getattr(event, "name", None),
                            "arguments": getattr(event, "arguments", None),
                        },
                    )

        except APIError as e:
            status_code = getattr(e, "status_code", None)
            raise ModelResponseError(
                message=str(e),
                provider="openai",
                status_code=status_code,
            ) from e

    def get_system_prompt(
        self, workspace_root: str, tools: list["ToolDefinition"] | None = None
    ) -> str:
        """Build OpenAI-optimized system prompt.

        Uses default prompt sections with OpenAI-specific optimizations.

        Args:
            workspace_root: Path to workspace directory
            tools: Optional tool definitions for dynamic tool section

        Returns:
            Formatted system prompt string
        """
        from agentrunner.core.prompts.utils import build_prompt

        return build_prompt(
            workspace_root=workspace_root,
            model_name=self.config.model,
            tool_definitions=tools,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using OpenAI's tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def _convert_messages_to_openai(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert AgentRunner messages to OpenAI format.

        Args:
            messages: AgentRunner messages

        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []

        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Add tool_call_id and name for tool messages
            if msg.role == "tool":
                if msg.tool_call_id:
                    openai_msg["tool_call_id"] = msg.tool_call_id
                # OpenAI requires 'name' field in tool messages
                if msg.meta.get("tool_name"):
                    openai_msg["name"] = msg.meta["tool_name"]

            # Add tool_calls for assistant messages
            if msg.role == "assistant" and msg.tool_calls:
                # Ensure tool_calls are in proper OpenAI format
                formatted_calls = []
                for tc in msg.tool_calls:
                    # Extract function info
                    if "function" in tc:
                        func = tc["function"]
                        func_name = func.get("name", "")
                        func_args = func.get("arguments", "{}")
                    else:
                        func_name = tc.get("name", "")
                        func_args = tc.get("arguments", {})

                    # Ensure arguments is a JSON string
                    if not isinstance(func_args, str):
                        func_args = json.dumps(func_args)

                    formatted_calls.append(
                        {
                            "id": tc.get("id", f"call_{uuid.uuid4()}"),
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": func_args,
                            },
                        }
                    )

                if formatted_calls:
                    openai_msg["tool_calls"] = formatted_calls

            openai_messages.append(openai_msg)

        return openai_messages

    def _convert_messages_to_responses_format(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert AgentRunner messages to Responses API format.

        Extracts system message as instructions and converts rest to input items.

        Args:
            messages: AgentRunner messages

        Returns:
            Tuple of (instructions, input_items)
        """
        instructions = None
        input_items = []

        for msg in messages:
            # System message becomes instructions
            if msg.role == "system":
                instructions = msg.content
                continue

            # User messages
            elif msg.role == "user":
                input_items.append(
                    {
                        "role": "user",
                        "content": msg.content,
                    }
                )

            # Assistant messages (may include tool calls)
            elif msg.role == "assistant":
                # If assistant made tool calls, include them as separate function_call items
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            }
                        )
                # If assistant had text content, include it as a message
                elif msg.content:
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": msg.content,
                        }
                    )

            # Tool messages → function_call_output items
            elif msg.role == "tool":
                tool_item = {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": msg.content or "",
                }
                input_items.append(tool_item)

        return instructions, input_items

    def _parse_responses_message(self, response: Any) -> Message:  # noqa: ANN401
        """Parse Responses API response into AgentRunner Message.

        Args:
            response: OpenAI responses.create() response

        Returns:
            AgentRunner Message object
        """
        # Extract text content and tool calls
        content = ""
        tool_calls = None

        # Response.output is a list of output items
        # Items can be: message (with text), function_call, etc.
        if response.output:
            for item in response.output:
                item_type = getattr(item, "type", None)

                # Handle message output (text content)
                if item_type == "message":
                    if hasattr(item, "content") and item.content is not None:
                        for content_item in item.content:
                            content_type = getattr(content_item, "type", None)
                            # Text output
                            if content_type == "output_text" and hasattr(content_item, "text"):
                                content += content_item.text

                # Handle function call output (tool calls)
                elif item_type == "function_call":
                    if tool_calls is None:
                        tool_calls = []

                    # Responses API uses call_id instead of id
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", None)

                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        }
                    )

        # Fallback to output_text if available
        if not content and hasattr(response, "output_text"):
            content = response.output_text or ""

        return Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            meta={"ts": datetime.now(UTC).isoformat()},
        )

    def _convert_tools_to_openai(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert AgentRunner tool definitions to OpenAI Chat Completions format.

        Args:
            tools: AgentRunner tool definitions

        Returns:
            List of tools in OpenAI format (nested structure)
        """
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def _convert_tools_to_responses_format(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]]:
        """Convert AgentRunner tool definitions to Responses API format.

        Args:
            tools: AgentRunner tool definitions

        Returns:
            List of tools in Responses API format (flat structure)
        """
        responses_tools = []

        for tool in tools:
            responses_tool = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            responses_tools.append(responses_tool)

        return responses_tools

    def _parse_response_message(self, response: Any) -> Message:  # noqa: ANN401
        """Parse OpenAI response into AgentRunner Message.

        Args:
            response: OpenAI chat completion response

        Returns:
            AgentRunner Message object
        """
        choice = response.choices[0]
        message = choice.message

        # Extract content
        content = message.content or ""

        # Extract tool calls if present - keep OpenAI format
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                # Parse arguments - OpenAI returns them as JSON string
                if isinstance(tc.function.arguments, str):
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        raise ModelResponseError(
                            f"Invalid JSON in tool call arguments for {tc.function.name}: {e}"
                        ) from e
                else:
                    arguments = tc.function.arguments

                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": arguments,
                        },
                    }
                )

        return Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            meta={"ts": datetime.now(UTC).isoformat()},
        )

    def parse_tool_calls(self, assistant_message: Message) -> list[ToolCall]:
        """Parse tool calls from assistant message.

        Args:
            assistant_message: Assistant message with tool_calls

        Returns:
            List of normalized ToolCall objects
        """
        if not assistant_message.tool_calls:
            return []

        tool_calls = []
        for tc in assistant_message.tool_calls:
            # Parse arguments JSON
            try:
                if isinstance(tc["function"]["arguments"], str):
                    arguments = json.loads(tc["function"]["arguments"])
                else:
                    arguments = tc["function"]["arguments"]
            except json.JSONDecodeError:
                # If arguments aren't valid JSON, use empty dict
                arguments = {}

            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=arguments,
                )
            )

        return tool_calls
