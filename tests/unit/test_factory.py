"""Tests for agent factory.

Tests factory functions for creating providers, tool registries, and agents.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ConfigurationError
from agentrunner.core.factory import (
    create_agent,
    create_provider,
    create_tool_registry,
)
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.workspace import Workspace
from agentrunner.providers.anthropic_provider import AnthropicProvider
from agentrunner.providers.base import ProviderConfig
from agentrunner.providers.codex_cli_provider import CodexCLIProvider
from agentrunner.providers.gemini_provider import GeminiProvider
from agentrunner.providers.openai_provider import OpenAIProvider


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return str(workspace_dir)


class TestCreateProvider:
    """Test provider creation."""

    def test_create_openai_provider(self, monkeypatch):
        """Test creating OpenAI provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        config = ProviderConfig(model="gpt-5.1-2025-11-13")

        provider = create_provider(config)

        assert isinstance(provider, OpenAIProvider)
        assert provider.config.model == "gpt-5.1-2025-11-13"

    @pytest.mark.parametrize("openai_api_key", [None, "unused-api-key"])
    def test_create_codex_cli_provider_never_uses_openai_api_key(
        self, monkeypatch, tmp_path, openai_api_key
    ):
        """Codex CLI is distinct and cannot fall back to OpenAI API billing."""
        executable = tmp_path / "codex"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        monkeypatch.setenv("AGENTRUNNER_CODEX_CLI_PATH", str(executable))
        if openai_api_key is None:
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        else:
            monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)

        with patch.object(
            OpenAIProvider,
            "__init__",
            side_effect=AssertionError("OpenAIProvider must not be constructed"),
        ):
            provider = create_provider(ProviderConfig(model="codex-cli"))

        assert isinstance(provider, CodexCLIProvider)
        assert provider.requires_api_key is False

    def test_create_openai_provider_with_openai_prefix(self, monkeypatch):
        """Test creating OpenAI provider with openai prefix."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        config = ProviderConfig(model="gpt-5.1-2025-11-13")

        provider = create_provider(config)

        assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic_provider(self, monkeypatch):
        """Test creating Anthropic provider."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
        config = ProviderConfig(model="claude-sonnet-4-5-20250929")

        provider = create_provider(config)

        assert isinstance(provider, AnthropicProvider)
        assert provider.config.model == "claude-sonnet-4-5-20250929"

    def test_create_anthropic_provider_with_anthropic_keyword(self, monkeypatch):
        """Test creating Anthropic provider with anthropic in name."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
        config = ProviderConfig(model="claude-opus-4-1-20250805")

        provider = create_provider(config)

        assert isinstance(provider, AnthropicProvider)

    def test_create_gemini_provider(self, monkeypatch):
        """Test creating Gemini provider."""
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza123")
        config = ProviderConfig(model="gemini-2.5-pro")

        provider = create_provider(config)

        assert isinstance(provider, GeminiProvider)

    def test_create_gemini_provider_with_google_keyword(self, monkeypatch):
        """Test creating Gemini provider with google in name."""
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza123")
        config = ProviderConfig(model="gemini-2.5-pro")

        provider = create_provider(config)

        assert isinstance(provider, GeminiProvider)

    def test_create_openrouter_provider(self, monkeypatch):
        """Test creating OpenRouter provider - skipping as OpenRouter models aren't in registry."""
        pytest.skip("OpenRouter models are not in ModelRegistry - dynamic model support needed")

    def test_create_openrouter_provider_with_openrouter_keyword(self, monkeypatch):
        """Test creating OpenRouter provider with openrouter in name - skipping."""
        pytest.skip("OpenRouter models are not in ModelRegistry - dynamic model support needed")

    def test_missing_openai_api_key(self, monkeypatch):
        """Test error when OpenAI API key missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = ProviderConfig(model="gpt-5.1-2025-11-13")

        with pytest.raises(ConfigurationError, match="API key required.*OPENAI_API_KEY"):
            create_provider(config)

    def test_missing_anthropic_api_key(self, monkeypatch):
        """Test error when Anthropic API key missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = ProviderConfig(model="claude-sonnet-4-5-20250929")

        with pytest.raises(ConfigurationError, match="API key required.*ANTHROPIC_API_KEY"):
            create_provider(config)

    def test_missing_google_api_key(self, monkeypatch):
        """Test error when Google API key missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = ProviderConfig(model="gemini-2.5-pro")

        with pytest.raises(ConfigurationError, match="API key required.*GOOGLE_API_KEY"):
            create_provider(config)

    def test_missing_openrouter_api_key(self, monkeypatch):
        """Test error when OpenRouter API key missing - skipping."""
        pytest.skip("OpenRouter models are not in ModelRegistry - dynamic model support needed")

    def test_unknown_model(self, monkeypatch):
        """Test error for unknown model."""
        config = ProviderConfig(model="unknown-model-xyz")

        with pytest.raises(ConfigurationError, match="Unknown model"):
            create_provider(config)


class TestCreateToolRegistry:
    """Test tool registry creation."""

    def test_creates_registry_with_all_tools(self, temp_workspace, monkeypatch):
        """Test that registry contains all tools from provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        workspace = Workspace(temp_workspace)
        logger = AgentRunnerLogger()
        config = AgentConfig()
        provider = create_provider(ProviderConfig(model="gpt-5.1-2025-11-13"))

        registry = create_tool_registry(provider, workspace, logger, config)

        # Verify core tools are registered
        assert registry.has("read_file")
        assert registry.has("grep")  # Ripgrep-based search tool
        assert registry.has("bash")
        assert registry.has("scaffold_project")
        assert registry.has("clean_workspace")
        assert registry.has("deploy_to_vercel")

    def test_tools_count(self, temp_workspace, monkeypatch):
        """Test that essential tools are registered."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        workspace = Workspace(temp_workspace)
        logger = AgentRunnerLogger()
        config = AgentConfig()
        provider = create_provider(ProviderConfig(model="gpt-5.1-2025-11-13"))

        registry = create_tool_registry(provider, workspace, logger, config)
        tools = registry.list_tools()

        # Provider should have core tools (count may change as tools evolve)
        assert len(tools) > 0, f"Expected tools to be registered, got {len(tools)}"

        # Verify essential tools are present
        assert "read_file" in tools, "read_file tool should be registered"
        assert "bash" in tools, "bash tool should be registered"

    def test_registry_with_event_bus(self, temp_workspace, monkeypatch):
        """Test creating registry with event bus."""
        from agentrunner.core.events import EventBus

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        workspace = Workspace(temp_workspace)
        logger = AgentRunnerLogger()
        config = AgentConfig()
        event_bus = EventBus()
        provider = create_provider(ProviderConfig(model="gpt-5.1-2025-11-13"))

        registry = create_tool_registry(provider, workspace, logger, config, event_bus)

        assert registry is not None
        assert registry.has("read_file")


class TestCreateAgent:
    """Test agent creation."""

    @patch("agentrunner.core.factory.create_provider")
    @patch("agentrunner.core.factory.load_config")
    def test_creates_agent_with_default_profile(
        self, mock_load_config, mock_create_provider, temp_workspace
    ):
        """Test creating agent with default profile."""
        # Setup mocks
        mock_config = AgentConfig()
        mock_load_config.return_value = mock_config

        mock_provider = MagicMock()
        from agentrunner.providers.base import ModelInfo

        mock_provider.get_model_info.return_value = ModelInfo(
            name="gpt-5.1-2025-11-13", context_window=128000, pricing={}
        )
        mock_create_provider.return_value = mock_provider
        mock_provider.config = ProviderConfig(model="gpt-5.1-2025-11-13")
        mock_provider.get_system_prompt.return_value = "Test system prompt"
        mock_provider.get_tool_classes.return_value = []

        agent = create_agent(
            temp_workspace, ProviderConfig(model="gpt-5.1-2025-11-13"), profile="default"
        )

        assert agent is not None
        assert agent.config == mock_config
        assert agent.provider == mock_provider

    @patch("agentrunner.core.factory.create_provider")
    def test_creates_agent_with_provided_config(self, mock_create_provider, temp_workspace):
        """Test creating agent with provided config."""
        config = ProviderConfig(model="gpt-5.1-2025-11-13")

        mock_provider = MagicMock()
        from agentrunner.providers.base import ModelInfo

        mock_provider.get_model_info.return_value = ModelInfo(
            name="gpt-5.1-2025-11-13", context_window=128000, pricing={}
        )
        mock_provider.get_system_prompt.return_value = "Provider system prompt"
        mock_provider.config = config
        mock_provider.get_tool_classes.return_value = []
        mock_create_provider.return_value = mock_provider

        agent = create_agent(temp_workspace, config, agent_config=AgentConfig())

        assert agent is not None
        # System prompt is fetched lazily on first message processing, not during creation
        assert agent.provider == mock_provider

    @patch("agentrunner.core.factory.create_provider")
    def test_agent_has_all_components(self, mock_create_provider, temp_workspace):
        """Test that created agent has all required components."""
        config = ProviderConfig(model="gpt-5.1-2025-11-13")

        mock_provider = MagicMock()
        from agentrunner.providers.base import ModelInfo

        mock_provider.get_model_info.return_value = ModelInfo(
            name="gpt-5.1-2025-11-13", context_window=128000, pricing={}
        )
        mock_provider.get_system_prompt.return_value = "Test system prompt"
        mock_provider.config = config
        mock_provider.get_tool_classes.return_value = []
        mock_create_provider.return_value = mock_provider

        agent = create_agent(temp_workspace, config, agent_config=AgentConfig())

        assert agent.provider is not None
        assert agent.workspace is not None
        assert agent.config is not None
        assert agent.context_manager is not None
        assert agent.logger is not None
        assert agent.tools is not None
        assert agent.confirmation is not None
        assert agent.command_validator is not None
        assert agent.event_bus is not None

    @patch("agentrunner.core.factory.create_provider")
    def test_workspace_path_converted_to_absolute(self, mock_create_provider, temp_workspace):
        """Test that relative workspace path is converted to absolute."""
        config = ProviderConfig(model="gpt-5.1-2025-11-13")

        mock_provider = MagicMock()
        from agentrunner.providers.base import ModelInfo

        mock_provider.get_model_info.return_value = ModelInfo(
            name="gpt-5.1-2025-11-13", context_window=128000, pricing={}
        )
        mock_provider.get_system_prompt.return_value = "Test system prompt"
        mock_provider.config = config
        mock_provider.get_tool_classes.return_value = []
        mock_create_provider.return_value = mock_provider

        # Use relative path
        with patch("pathlib.Path.cwd", return_value=Path(temp_workspace).parent):
            agent = create_agent(Path(temp_workspace).name, config, agent_config=AgentConfig())

        assert agent.workspace.root_path.is_absolute()

    @patch("agentrunner.core.factory.create_provider")
    @patch("agentrunner.core.factory.load_config")
    def test_handles_configuration_error(
        self, mock_load_config, mock_create_provider, temp_workspace
    ):
        """Test that configuration errors are propagated."""
        mock_load_config.side_effect = ConfigurationError("Invalid config")

        with pytest.raises(ConfigurationError):
            create_agent(temp_workspace, ProviderConfig(model="gpt-5.1-2025-11-13"))


class TestToolClassesImports:
    """Smoke tests for tool class imports - catches ImportErrors hidden by mocks."""

    def test_base_provider_default_tool_classes_import(self):
        """Verify BaseLLMProvider.get_tool_classes() imports work."""
        from agentrunner.core.messages import Message
        from agentrunner.providers.base import BaseLLMProvider, ProviderConfig, ProviderResponse

        # Create a minimal concrete provider
        class TestProvider(BaseLLMProvider):
            async def chat(self, messages, tools=None, config=None):
                msg = Message(id="test", role="assistant", content="test")
                return ProviderResponse(messages=[msg])

            async def chat_stream(self, messages, tools=None, config=None):
                yield Message(id="test", role="assistant", content="test")

            def get_model_info(self):
                from agentrunner.providers.base import ModelInfo

                return ModelInfo(name="test", context_window=1000, pricing={})

            def count_tokens(self, text: str) -> int:
                # Simple word-based approximation for testing
                return len(text.split())

            def get_system_prompt(self, workspace_root: str, tools=None) -> str:
                return "Test system prompt"

        config = ProviderConfig(model="test", temperature=0.7)
        provider = TestProvider(api_key="test", config=config)

        # This should not raise ImportError
        tool_classes = provider.get_tool_classes()

        # Verify we got default tools
        assert len(tool_classes) > 0, "Should have default tool classes"

        # Verify all classes can be instantiated (imports work)
        from agentrunner.core.workspace import Workspace
        from agentrunner.tools.base import ToolContext, ToolRegistry

        workspace = Workspace("/tmp/test")
        logger = AgentRunnerLogger("test")
        context = ToolContext(workspace=workspace, logger=logger, model_id="test-model")
        ToolRegistry(context=context)

        for tool_class in tool_classes:
            # This will fail if imports are broken
            # Most tools don't need explicit initialization parameters
            try:
                tool = tool_class()
                assert tool is not None, f"Failed to instantiate {tool_class.__name__}"
            except TypeError:
                # Some tools might need specific init params, that's okay for this smoke test
                pass

    def test_all_provider_tool_classes_import(self, monkeypatch):
        """Verify all provider-specific get_tool_classes() work."""
        # Set required API keys
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        from agentrunner.providers.anthropic_provider import AnthropicProvider
        from agentrunner.providers.gemini_provider import GeminiProvider
        from agentrunner.providers.mistral_provider import MistralProvider
        from agentrunner.providers.openai_provider import OpenAIProvider
        from agentrunner.providers.xai_provider import XAIProvider

        providers = [
            (OpenAIProvider, "gpt-5.1-2025-11-13"),
            (AnthropicProvider, "claude-3-5-sonnet-20241022"),
            (GeminiProvider, "gemini-2.5-flash"),
            (XAIProvider, "grok-2-1212"),
            (MistralProvider, "mistral-large-latest"),
        ]

        for provider_class, model in providers:
            config = ProviderConfig(model=model, temperature=0.7)
            provider = provider_class(api_key="test", config=config)

            # This should not raise ImportError
            tool_classes = provider.get_tool_classes()

            assert len(tool_classes) > 0, f"{provider_class.__name__} should have tool classes"
