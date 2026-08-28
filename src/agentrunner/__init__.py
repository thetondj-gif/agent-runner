"""
AgentRunner

A powerful, model-agnostic, framework agnostic AI agent framework with multi-provider support
and advanced tooling capabilities. Configure any model as a coding agent.
"""

__version__ = "0.2.0"
__author__ = "Design Arena Contributors"
__license__ = "MIT"

# Core imports
from agentrunner.core.agent import AgentRunnerAgent
from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import (
    AgentRunnerException,
    ConfigurationError,
    ModelResponseError,
    TokenLimitExceededError,
    ToolExecutionError,
    WorkspaceSecurityError,
)
from agentrunner.core.session import SessionManager
from agentrunner.providers.anthropic_provider import AnthropicProvider

# Provider imports
from agentrunner.providers.base import BaseLLMProvider
from agentrunner.providers.codex_cli_provider import CodexCLIProvider
from agentrunner.providers.gemini_provider import GeminiProvider
from agentrunner.providers.kimi_provider import KimiProvider
from agentrunner.providers.mistral_provider import MistralProvider
from agentrunner.providers.openai_provider import OpenAIProvider
from agentrunner.providers.xai_provider import XAIProvider
from agentrunner.providers.zai_provider import ZAIProvider

# Convenience aliases
Agent = AgentRunnerAgent
Session = SessionManager

__all__ = [
    # Version
    "__version__",
    # Core
    "AgentRunnerAgent",
    "Agent",  # Alias for AgentRunnerAgent
    "AgentConfig",
    "SessionManager",
    "Session",  # Alias for SessionManager
    # Exceptions
    "AgentRunnerException",
    "ToolExecutionError",
    "WorkspaceSecurityError",
    "TokenLimitExceededError",
    "ModelResponseError",
    "ConfigurationError",
    # Providers
    "BaseLLMProvider",
    "CodexCLIProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "KimiProvider",
    "MistralProvider",
    "XAIProvider",
    "ZAIProvider",
]


def hello() -> str:
    """Legacy hello function for backwards compatibility."""
    return f"Hello from agentrunner v{__version__}!"
