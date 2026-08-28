"""Model registry for mapping model IDs to providers.

Provides explicit registry instead of hacky substring matching.
Single source of truth for all model metadata.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentrunner.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from agentrunner.providers.base import BaseLLMProvider, ModelInfo


@dataclass
class ModelSpec:
    """Complete specification for a supported model.

    This replaces the duplicate MODEL_INFO dicts in each provider.
    Single source of truth for model metadata.
    """

    model_id: str  # Unique ID: "gpt-4-turbo", "claude-3-5-sonnet-20241022"
    provider_name: str  # "openai", "anthropic", "google", "openrouter"
    display_name: str  # "GPT-4 Turbo", "Claude 3.5 Sonnet"
    context_window: int  # Maximum context window (tokens)
    input_cost_per_1k: float  # Cost per 1K input tokens (USD)
    output_cost_per_1k: float  # Cost per 1K output tokens (USD)
    api_key_env: str | None  # None for providers with external/local authentication

    def to_model_info(self) -> "ModelInfo":
        """Convert to ModelInfo for runtime use.

        Returns:
            ModelInfo instance with pricing dict
        """
        from agentrunner.providers.base import ModelInfo

        return ModelInfo(
            name=self.model_id,
            context_window=self.context_window,
            pricing={
                "input_per_1k": self.input_cost_per_1k,
                "output_per_1k": self.output_cost_per_1k,
            },
        )


class ModelRegistry:
    """Registry of all available models and their providers.

    Instead of substring matching ("gpt" in model → OpenAI),
    we have explicit model registration.
    """

    _models: dict[str, ModelSpec] = {}
    _providers: dict[str, type["BaseLLMProvider"]] = {}

    @classmethod
    def register_model(cls, spec: ModelSpec) -> None:
        """Register a model specification."""
        cls._models[spec.model_id] = spec

    @classmethod
    def register_provider(cls, name: str, provider_class: type["BaseLLMProvider"]) -> None:
        """Register a provider class."""
        cls._providers[name] = provider_class

    @classmethod
    def get_model_spec(cls, model_id: str) -> ModelSpec:
        """Get model specification by ID.

        Args:
            model_id: Model identifier (e.g., "gpt-4-turbo")

        Returns:
            ModelSpec for the model

        Raises:
            ConfigurationError: If model not registered
        """
        if model_id not in cls._models:
            available = ", ".join(sorted(cls._models.keys()))
            raise ConfigurationError(
                f"Unknown model: {model_id}\n" f"Available models: {available}"
            )
        return cls._models[model_id]

    @classmethod
    def get_provider_class(cls, provider_name: str) -> type["BaseLLMProvider"]:
        """Get provider class by name.

        Args:
            provider_name: Provider name (e.g., "openai")

        Returns:
            Provider class

        Raises:
            ConfigurationError: If provider not registered
        """
        if provider_name not in cls._providers:
            raise ConfigurationError(f"Unknown provider: {provider_name}")
        return cls._providers[provider_name]

    @classmethod
    def list_models(cls) -> list[ModelSpec]:
        """List all registered models."""
        return list(cls._models.values())

    @classmethod
    def list_models_by_provider(cls, provider_name: str) -> list[ModelSpec]:
        """List all models for a specific provider."""
        return [spec for spec in cls._models.values() if spec.provider_name == provider_name]


# Register OpenAI models

# GPT-5 Codex
ModelRegistry.register_model(
    ModelSpec(
        model_id="gpt-5-codex",
        provider_name="openai",
        display_name="GPT-5 Codex",
        context_window=256000,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        api_key_env="OPENAI_API_KEY",
    )
)

# Register the locally authenticated Codex CLI bridge. This is intentionally a
# distinct provider/model from the OpenAI API-backed models above.
ModelRegistry.register_model(
    ModelSpec(
        model_id="codex-cli",
        provider_name="codex-cli",
        display_name="Codex CLI (local authenticated session)",
        context_window=200000,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        api_key_env=None,
    )
)

# GPT-5.1
ModelRegistry.register_model(
    ModelSpec(
        model_id="gpt-5.1-2025-11-13",
        provider_name="openai",
        display_name="GPT-5.1 (No Reasoning)",
        context_window=400000,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.01,
        api_key_env="OPENAI_API_KEY",
    )
)

ModelRegistry.register_model(
    ModelSpec(
        model_id="gpt-5.1-2025-11-13-low",
        provider_name="openai",
        display_name="GPT-5.1 (Low Reasoning)",
        context_window=400000,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.01,
        api_key_env="OPENAI_API_KEY",
    )
)

ModelRegistry.register_model(
    ModelSpec(
        model_id="gpt-5.1-2025-11-13-medium",
        provider_name="openai",
        display_name="GPT-5.1 (Medium Reasoning)",
        context_window=400000,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.01,
        api_key_env="OPENAI_API_KEY",
    )
)

ModelRegistry.register_model(
    ModelSpec(
        model_id="gpt-5.1-2025-11-13-high",
        provider_name="openai",
        display_name="GPT-5.1 (High Reasoning)",
        context_window=400000,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.01,
        api_key_env="OPENAI_API_KEY",
    )
)

ModelRegistry.register_model(
    ModelSpec(
        model_id="gpt-5.1-codex",
        provider_name="openai",
        display_name="GPT-5.1 Codex",
        context_window=400000,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.01,
        api_key_env="OPENAI_API_KEY",
    )
)

# Register Anthropic models

# Claude Sonnet 4.5
ModelRegistry.register_model(
    ModelSpec(
        model_id="claude-sonnet-4-5-20250929",
        provider_name="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200000,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        api_key_env="ANTHROPIC_API_KEY",
    )
)

# Claude Opus 4.1
ModelRegistry.register_model(
    ModelSpec(
        model_id="claude-opus-4-1-20250805",
        provider_name="anthropic",
        display_name="Claude Opus 4.1",
        context_window=200000,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        api_key_env="ANTHROPIC_API_KEY",
    )
)

# Register Google models

# Gemini 2.5 models
ModelRegistry.register_model(
    ModelSpec(
        model_id="gemini-2.5-flash",
        provider_name="google",
        display_name="Gemini 2.5 Flash",
        context_window=1048576,
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
        api_key_env="GOOGLE_API_KEY",
    )
)

ModelRegistry.register_model(
    ModelSpec(
        model_id="gemini-2.5-pro",
        provider_name="google",
        display_name="Gemini 2.5 Pro",
        context_window=2097152,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
        api_key_env="GOOGLE_API_KEY",
    )
)

# Gemini 3 Pro
ModelRegistry.register_model(
    ModelSpec(
        model_id="gemini-3-pro-preview",
        provider_name="google",
        display_name="Gemini 3 Pro",
        context_window=2097152,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
        api_key_env="GOOGLE_API_KEY",
    )
)

# Register Kimi (Moonshot AI) models
ModelRegistry.register_model(
    ModelSpec(
        model_id="kimi-k2-turbo-preview",
        provider_name="kimi",
        display_name="Kimi K2 Turbo Preview",
        context_window=128000,
        input_cost_per_1k=0.0017,
        output_cost_per_1k=0.0017,
        api_key_env="MOONSHOT_API_KEY",
    )
)

ModelRegistry.register_model(
    ModelSpec(
        model_id="kimi-k2-0905-preview",
        provider_name="kimi",
        display_name="Kimi K2",
        context_window=128000,
        input_cost_per_1k=0.0017,
        output_cost_per_1k=0.0017,
        api_key_env="MOONSHOT_API_KEY",
    )
)

ModelRegistry.register_model(
    ModelSpec(
        model_id="kimi-k2-thinking",
        provider_name="kimi",
        display_name="Kimi K2 Thinking",
        context_window=128000,
        input_cost_per_1k=0.0017,
        output_cost_per_1k=0.0017,
        api_key_env="MOONSHOT_API_KEY",
    )
)

# Register ZAI (Z.AI) models
ModelRegistry.register_model(
    ModelSpec(
        model_id="glm-4.6",
        provider_name="zai",
        display_name="GLM 4.6",
        context_window=128000,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.001,
        api_key_env="ZAI_API_KEY",
    )
)

# Register xAI models
# NOTE: Only grok-code-fast-1 is enabled for now
# Uncomment other models as needed

ModelRegistry.register_model(
    ModelSpec(
        model_id="grok-code-fast-1",
        provider_name="xai",
        display_name="Grok Code Fast 1",
        context_window=256000,
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0015,
        api_key_env="XAI_API_KEY",
    )
)

# Register Mistral AI models

# Mistral Large 2.1
ModelRegistry.register_model(
    ModelSpec(
        model_id="mistral-large-latest",
        provider_name="mistral",
        display_name="Mistral Large Latest",
        context_window=128000,
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.006,
        api_key_env="MISTRAL_API_KEY",
    )
)

# Mistral Medium 3.1
ModelRegistry.register_model(
    ModelSpec(
        model_id="mistral-medium-latest",
        provider_name="mistral",
        display_name="Mistral Medium Latest",
        context_window=128000,
        input_cost_per_1k=0.0015,
        output_cost_per_1k=0.0045,
        api_key_env="MISTRAL_API_KEY",
    )
)
