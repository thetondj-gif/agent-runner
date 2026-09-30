from __future__ import annotations

from agentrunner.core.factory import create_provider
from agentrunner.providers.base import ProviderConfig
from agentrunner.providers.openai_provider import OpenAIProvider


def test_unknown_model_uses_explicit_openai_compatible_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("AGENTRUNNER_OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("AGENTRUNNER_CONTEXT_WINDOW", "65536")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTRUNNER_OPENAI_API_KEY", raising=False)

    provider = create_provider(
        ProviderConfig(
            model="local-test-model",
            provider_extensions={
                "openai_compatible": True,
                "context_window": 65536,
            },
        )
    )

    assert isinstance(provider, OpenAIProvider)
    info = provider.get_model_info()
    assert info.name == "local-test-model"
    assert info.context_window == 65536
    assert info.pricing["input_per_1k"] == 0.0
