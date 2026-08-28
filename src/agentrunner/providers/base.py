"""Base provider interface for LLM integrations.

Defines the abstract interface that all LLM providers must implement.
See INTERFACES/PROVIDERS.md for full specification.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentrunner.core.compaction.base import CompactionContext, CompactionResult
from agentrunner.core.config import AgentConfig
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition

if TYPE_CHECKING:
    from agentrunner.tools.base import BaseTool


@dataclass
class ProviderConfig:
    """Configuration for LLM provider behavior.

    This contains all provider-specific settings (model selection, temperature, etc.)
    separate from agent orchestration settings (max_rounds, timeout, etc.).

    Attributes:
        model: Model identifier (e.g., "gpt-4-turbo", "claude-3-5-sonnet")
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
        max_tokens: Maximum tokens in response (None = use model default)
        top_p: Nucleus sampling threshold (None = use model default)
        stop_sequences: Custom stop sequences (None = use model default)
        frequency_penalty: Penalize repeated tokens (OpenAI-specific, -2.0 to 2.0)
        presence_penalty: Penalize new topics (OpenAI-specific, -2.0 to 2.0)
        compaction: Context compaction settings (when/how to compact)
        provider_extensions: Provider-specific extensions (e.g., Anthropic skills, OpenAI plugins)
    """

    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    compaction: CompactionContext = field(
        default_factory=lambda: CompactionContext(
            current_tokens=0,  # Will be set at runtime
            target_tokens=0,  # Will be set at runtime
        )
    )
    provider_extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate provider config."""
        if not self.model:
            raise ValueError("model must not be empty")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be 0.0-1.0, got {self.top_p}")


@dataclass
class ModelInfo:
    """Information about a model's capabilities and limits."""

    name: str
    context_window: int  # Maximum context window size
    pricing: dict[str, float] = field(default_factory=dict)

    # Alias for compatibility
    @property
    def max_tokens(self) -> int:
        """Alias for context_window."""
        return self.context_window

    def __post_init__(self) -> None:
        """Validate model info."""
        if self.context_window <= 0:
            raise ValueError(f"context_window must be positive, got {self.context_window}")


@dataclass
class ProviderResponse:
    """Normalized response from any LLM provider."""

    messages: list[Message]
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )

    def __post_init__(self) -> None:
        """Validate provider response."""
        if not self.messages:
            raise ValueError("ProviderResponse must contain at least one message")
        if self.messages[-1].role != "assistant":
            raise ValueError("Last message in ProviderResponse must be from assistant")


@dataclass
class StreamChunk:
    """A chunk of streaming data from provider."""

    type: str  # token|tool_call|status|error
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate stream chunk."""
        valid_types = {"token", "tool_call", "status", "error"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid chunk type: {self.type}. Must be one of {valid_types}")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers.

    To add a new provider (Cohere, Mistral, AI21, local models):
    1. Subclass BaseLLMProvider
    2. Implement all abstract methods
    3. Handle provider-specific tool call format → normalize to ToolCall
    4. Handle provider-specific errors → raise AgentRunnerException subclasses
    5. Convert ToolDefinition → provider's tool schema format

    Provider-Owned Responsibilities:
    - Tool call parsing (vendor format → ToolCall)
    - Tool definition conversion (ToolDefinition → vendor format)
    - Error normalization (vendor errors → AgentRunnerException)
    - Rate limiting and retries
    - Streaming event emission

    For Providers Without Native Tool Calling:
    - Override supports_native_tool_calling to return False
    - Use NonNativeToolFormat.to_prompt() to add tool instructions to system prompt
    - Use NonNativeToolFormat.to_tools() to extract tool calls from text response
    - Or create custom ToolFormat subclass for provider-specific format
    """

    requires_api_key = True

    def __init__(self, api_key: str | None, config: ProviderConfig) -> None:
        """Initialize provider with API key and configuration.

        All providers must accept api_key and config as the first two parameters.
        Subclasses can extend __init__ with additional provider-specific parameters.

        Args:
            api_key: API key for authentication, or None for providers that
                authenticate outside Agent Runner (such as Codex CLI)
            config: Provider configuration (model, temperature, etc.)
        """
        self.config = config

    @property
    def supports_native_tool_calling(self) -> bool:
        """Check if provider supports native tool calling.

        Override this to False for providers without native tool calling
        (e.g., local models, completion-only APIs).

        Returns:
            True if provider has native tool calling (default)
            False if tools must be serialized to prompt
        """
        return True

    def get_tool_classes(self) -> list[type["BaseTool"]]:
        """Return list of tool classes this provider wants to use.

        Provider declares WHAT tools it wants (just the classes).
        Factory handles HOW to instantiate them with runtime context.

        Default implementation returns all available agentrunner tools.
        Override in subclasses to customize tool selection.

        Returns:
            List of tool class types (not instances)

        Example:
            def get_tool_classes(self) -> list[type[BaseTool]]:
                from agentrunner.tools.file_io import WriteFileTool
                from agentrunner.tools.read_file import ReadFileTool
                from agentrunner.tools.bash import BashTool
                return [ReadFileTool, WriteFileTool, BashTool]
        """
        from agentrunner.tools.bash import BashTool
        from agentrunner.tools.batch import BatchCreateFilesTool
        from agentrunner.tools.clean_directory import CleanWorkspaceTool
        from agentrunner.tools.edit import EditFileTool, InsertLinesTool, MultiEditTool
        from agentrunner.tools.file_io import (
            DeleteFileTool,
            WriteFileTool,
        )
        from agentrunner.tools.image import (
            ImageFetchTool,
            ImageGenerationTool,
            VideoFetchTool,
            VideoGenerationTool,
        )
        from agentrunner.tools.read_file import ReadFileTool
        from agentrunner.tools.scaffold import ScaffoldProjectTool
        from agentrunner.tools.screenshot import ScreenshotTool
        from agentrunner.tools.search import GrepSearchTool
        from agentrunner.tools.vercel_deploy import VercelDeployTool

        return [
            ReadFileTool,
            WriteFileTool,
            DeleteFileTool,
            EditFileTool,
            MultiEditTool,
            InsertLinesTool,
            GrepSearchTool,
            BatchCreateFilesTool,
            BashTool,
            CleanWorkspaceTool,
            ScaffoldProjectTool,
            ScreenshotTool,
            VercelDeployTool,
            ImageFetchTool,
            ImageGenerationTool,
            VideoFetchTool,
            VideoGenerationTool,
        ]

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> ProviderResponse:
        """Execute chat completion.

        Args:
            messages: Conversation history
            tools: Available tools (None if no tools)
            config: Agent configuration (deprecated, providers use self.config instead)

        Returns:
            ProviderResponse with assistant message(s) and usage stats

        Raises:
            ModelResponseError: On API errors
            TokenLimitExceededError: If request exceeds context window
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Execute streaming chat completion.

        Args:
            messages: Conversation history
            tools: Available tools (None if no tools)
            config: Agent configuration

        Yields:
            StreamChunk objects with incremental data

        Raises:
            ModelResponseError: On API errors
            TokenLimitExceededError: If request exceeds context window
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get model information.

        Returns:
            ModelInfo with context window and pricing
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Token count
        """
        ...

    @abstractmethod
    def get_system_prompt(
        self, workspace_root: str, tools: list[ToolDefinition] | None = None
    ) -> str:
        """Build provider-optimized system prompt.

        Args:
            workspace_root: Path to workspace root directory
            tools: Available tool definitions (None if no tools)

        Returns:
            System prompt string optimized for this provider
        """
        ...

    async def compact(
        self,
        messages: list[Message],
        target_tokens: int,
        context: CompactionContext,
    ) -> CompactionResult:
        """Compact message history for this provider's model.

        Default implementation uses the pluggable strategy system based on
        config.compaction.strategy. Providers can override this method to
        implement model-specific compaction logic.

        Args:
            messages: Message history to compact
            target_tokens: Target token count after compaction
            context: Compaction context with settings and constraints

        Returns:
            CompactionResult with compacted messages and metadata

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If compaction fails

        Example:
            Override for model-specific compaction:

            ```python
            class AnthropicProvider(BaseLLMProvider):
                async def compact(self, messages, target_tokens, context):
                    # Claude-specific compaction logic
                    # E.g., leverage context caching, preserve tool patterns
                    ...
                    return CompactionResult(...)
            ```
        """
        # Import here to avoid circular dependency
        from agentrunner.core.compaction import get_compactor

        # Use strategy from provider config
        strategy = self.config.compaction.strategy
        compactor = get_compactor(strategy)

        return await compactor.compact(
            messages=messages,
            target_tokens=target_tokens,
            context=context,
        )
