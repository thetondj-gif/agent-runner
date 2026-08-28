"""Agent factory for creating configured AgentRunnerAgent instances.

Central factory function that can be used by CLI, backend, or any other interface.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from agentrunner.core.agent import AgentRunnerAgent
from agentrunner.core.config import AgentConfig, load_config
from agentrunner.core.events import EventBus
from agentrunner.core.exceptions import ConfigurationError
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tokens import ContextManager, TokenCounter
from agentrunner.core.workspace import Workspace
from agentrunner.providers.anthropic_provider import AnthropicProvider
from agentrunner.providers.base import BaseLLMProvider, ProviderConfig
from agentrunner.providers.codex_cli_provider import CodexCLIProvider
from agentrunner.providers.gemini_provider import GeminiProvider
from agentrunner.providers.kimi_provider import KimiProvider
from agentrunner.providers.mistral_provider import MistralProvider
from agentrunner.providers.openai_provider import OpenAIProvider
from agentrunner.providers.registry import ModelRegistry
from agentrunner.providers.xai_provider import XAIProvider
from agentrunner.providers.zai_provider import ZAIProvider
from agentrunner.security.command_validator import CommandValidator
from agentrunner.security.confirmation import ConfirmationService
from agentrunner.tools.base import ToolContext, ToolRegistry

if TYPE_CHECKING:
    pass

# Register provider classes
ModelRegistry.register_provider("openai", OpenAIProvider)
ModelRegistry.register_provider("codex-cli", CodexCLIProvider)
ModelRegistry.register_provider("anthropic", AnthropicProvider)
ModelRegistry.register_provider("google", GeminiProvider)
ModelRegistry.register_provider("kimi", KimiProvider)
ModelRegistry.register_provider("mistral", MistralProvider)
ModelRegistry.register_provider("xai", XAIProvider)
ModelRegistry.register_provider("zai", ZAIProvider)


def create_provider(provider_config: ProviderConfig) -> "BaseLLMProvider":
    """Create LLM provider using model registry.

    Args:
        provider_config: Provider configuration (model, temperature, etc.)

    Returns:
        Configured provider instance

    Raises:
        ConfigurationError: If model not registered or API key missing
    """
    # Get model spec from registry
    model_spec = ModelRegistry.get_model_spec(provider_config.model)

    # Get provider class and instantiate
    provider_class = ModelRegistry.get_provider_class(model_spec.provider_name)
    api_key: str | None = None
    if provider_class.requires_api_key:
        if not model_spec.api_key_env:
            raise ConfigurationError(
                f"Provider {model_spec.provider_name} requires an API key but has no key setting"
            )
        api_key = os.getenv(model_spec.api_key_env)
        if not api_key:
            raise ConfigurationError(f"API key required (set {model_spec.api_key_env})")

    return provider_class(api_key=api_key, config=provider_config)


def create_tool_registry(
    provider: BaseLLMProvider,
    workspace: Workspace,
    logger: AgentRunnerLogger,
    config: AgentConfig,
    event_bus: EventBus | None = None,
    session_uid: int | None = None,
    session_gid: int | None = None,
) -> ToolRegistry:
    """Create tool registry from provider's tool class list.

    Provider declares WHAT tools it wants (via get_tool_classes()).
    Factory handles HOW to instantiate them with runtime context.

    Args:
        provider: LLM provider that specifies which tools to use
        workspace: Workspace instance
        logger: Logger instance
        config: Agent configuration
        event_bus: Event bus for streaming (optional)
        session_uid: Unix UID for process isolation (optional)
        session_gid: Unix GID for process isolation (optional)

    Returns:
        Configured tool registry with provider's tools
    """
    ctx = ToolContext(
        workspace=workspace,
        logger=logger,
        model_id=provider.config.model,
        event_bus=event_bus,
        config=config.to_dict(),
        session_uid=session_uid,
        session_gid=session_gid,
    )

    registry = ToolRegistry(context=ctx)

    # Get tool classes from provider
    tool_classes = provider.get_tool_classes()

    logger.debug("Creating tool registry from provider", tool_count=len(tool_classes))

    # Instantiate each tool class - tools handle their own defaults
    for tool_class in tool_classes:
        logger.debug("Instantiating tool", tool_class=tool_class.__name__)
        tool = tool_class()
        registry.register(tool)

    return registry


def create_agent(
    workspace_path: str,
    provider_config: ProviderConfig,
    agent_config: AgentConfig | None = None,
    profile: str = "default",
    event_bus: EventBus | None = None,
    session_uid: int | None = None,
    session_gid: int | None = None,
    strict_commands: bool = True,
    require_confirmation: bool = True,
) -> AgentRunnerAgent:
    """Create configured AgentRunnerAgent instance.

    Core factory function for creating agents. Can be used by CLI, backend,
    or any other interface.

    Args:
        workspace_path: Path to workspace directory
        provider_config: Provider configuration (model, temperature, etc.)
        agent_config: Agent configuration (orchestration settings). If None, loads from profile.
        profile: Configuration profile name (default: "default")
        event_bus: Optional EventBus to use (for multi-agent scenarios)
        session_uid: Unix UID for process isolation (optional)
        session_gid: Unix GID for process isolation (optional)
        strict_commands: If True, only allow whitelisted bash commands (default: True)
        require_confirmation: If True, require user confirmation for dangerous operations (default: True)

    Returns:
        Configured AgentRunnerAgent instance with EventBus

    Raises:
        ConfigurationError: If configuration invalid
        AgentRunnerException: If agent creation fails

    Example:
        ```python
        from agentrunner.providers.base import ProviderConfig
        from agentrunner.core.config import AgentConfig

        provider_cfg = ProviderConfig(model="gpt-4-turbo", temperature=0.7)
        agent_cfg = AgentConfig(max_rounds=50)

        agent = create_agent(
            workspace_path="/tmp/workspace",
            provider_config=provider_cfg,
            agent_config=agent_cfg,
            strict_commands=True,  # Enable command validation
            require_confirmation=True,  # Require user confirmation
        )
        ```
    """
    # Convert to absolute path
    abs_workspace = str(Path(workspace_path).resolve())
    workspace = Workspace(abs_workspace)
    logger = AgentRunnerLogger()

    # Load or use provided config (orchestration settings only)
    if agent_config is None:
        agent_config = load_config(profile, Path(workspace_path))

    # Create provider
    provider = create_provider(provider_config)
    counter = TokenCounter()

    model_info = provider.get_model_info()
    context_manager = ContextManager(counter, model_info)

    # Use provided EventBus or create new one
    if event_bus is None:
        event_bus = EventBus()

    # Create tools from provider's tool class list
    tools = create_tool_registry(
        provider=provider,
        workspace=workspace,
        logger=logger,
        config=agent_config,
        event_bus=event_bus,
        session_uid=session_uid,
        session_gid=session_gid,
    )
    validator = CommandValidator(allow_unlisted=not strict_commands)
    confirmation = ConfirmationService(auto_approve=not require_confirmation)

    agent = AgentRunnerAgent(
        provider=provider,
        workspace=workspace,
        config=agent_config,
        context_manager=context_manager,
        logger=logger,
        tools=tools,
        confirmation=confirmation,
        command_validator=validator,
        event_bus=event_bus,
    )

    return agent
