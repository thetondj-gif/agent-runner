"""Tests for the authenticated local Codex CLI provider."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ConfigurationError, ModelResponseError
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import ProviderConfig
from agentrunner.providers.codex_cli_provider import CodexCLIProvider


@pytest.fixture
def fake_codex(tmp_path: Path) -> Path:
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


@pytest.fixture
def provider(fake_codex: Path, tmp_path: Path) -> CodexCLIProvider:
    instance = CodexCLIProvider(
        api_key=None,
        config=ProviderConfig(
            model="codex-cli",
            provider_extensions={"executable": str(fake_codex), "timeout_s": 2},
        ),
    )
    instance.get_system_prompt(str(tmp_path), tools=[])
    return instance


@pytest.fixture
def write_tool() -> ToolDefinition:
    return ToolDefinition(
        name="write_file",
        description="Write a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )


def jsonl_result(
    result: dict[str, object], input_tokens: int = 11, output_tokens: int = 7
) -> str:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": json.dumps(result)},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    ]
    return "\n".join(json.dumps(event) for event in events)


class TestConfiguration:
    def test_missing_executable_is_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AGENTRUNNER_CODEX_CLI_PATH", raising=False)
        with patch("shutil.which", return_value=None):
            with pytest.raises(ConfigurationError, match="executable not found"):
                CodexCLIProvider(api_key=None, config=ProviderConfig(model="codex-cli"))

    def test_rejects_non_executable_path(self, tmp_path: Path) -> None:
        path = tmp_path / "codex"
        path.write_text("not executable", encoding="utf-8")
        config = ProviderConfig(
            model="codex-cli", provider_extensions={"executable": str(path)}
        )
        with pytest.raises(ConfigurationError, match="not an executable file"):
            CodexCLIProvider(api_key=None, config=config)

    def test_rejects_relative_executable_path(self) -> None:
        config = ProviderConfig(
            model="codex-cli", provider_extensions={"executable": "bin/codex"}
        )
        with pytest.raises(ConfigurationError, match="absolute path"):
            CodexCLIProvider(api_key=None, config=config)

    def test_rejects_invalid_timeout(self, fake_codex: Path) -> None:
        config = ProviderConfig(
            model="codex-cli",
            provider_extensions={"executable": str(fake_codex), "timeout_s": 0},
        )
        with pytest.raises(ConfigurationError, match="greater than zero"):
            CodexCLIProvider(api_key=None, config=config)


class TestCommandSafety:
    def test_output_schema_uses_strict_json_encoded_arguments(self) -> None:
        schema = CodexCLIProvider._output_schema()
        item_schema = schema["properties"]["tool_calls"]["items"]

        assert item_schema["additionalProperties"] is False
        assert item_schema["properties"]["arguments"]["type"] == "string"

    def test_command_is_read_only_ephemeral_and_workspace_pinned(
        self, provider: CodexCLIProvider, tmp_path: Path
    ) -> None:
        command = provider._build_command(tmp_path / "schema.json")

        assert command[:3] == [str(provider.executable), "exec", "--json"]
        assert "--ephemeral" in command
        assert "--ignore-user-config" in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert command[command.index("--cd") + 1] == str(provider.workspace_root)
        assert "--add-dir" not in command
        assert "--dangerously-bypass-approvals-and-sandbox" not in command
        assert command[-1] == "-"

    @pytest.mark.asyncio
    async def test_openai_api_key_is_removed_from_child_environment(
        self, provider: CodexCLIProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0

            async def communicate(self, input_data: bytes | None = None):
                captured["input"] = input_data
                return b"", b""

        async def fake_create_subprocess_exec(*args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs["env"]
            captured["cwd"] = kwargs["cwd"]
            return FakeProcess()

        monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-codex")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

        await provider._run([str(provider.executable), "exec"], "prompt")

        assert "OPENAI_API_KEY" not in captured["env"]
        assert captured["cwd"] == str(provider.workspace_root)
        assert os.environ["OPENAI_API_KEY"] == "must-not-reach-codex"


class TestChat:
    @pytest.mark.asyncio
    async def test_parses_structured_content_tool_calls_and_usage(
        self, provider: CodexCLIProvider, write_tool: ToolDefinition
    ) -> None:
        stdout = jsonl_result(
            {
                "content": "I will create the requested file.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "write_file",
                        "arguments": {"path": "hello.txt", "content": "hello\n"},
                    }
                ],
            }
        )
        provider._run = AsyncMock(return_value=(stdout, "", 0))

        response = await provider.chat(
            [Message(id="user-1", role="user", content="Create hello.txt")], [write_tool]
        )

        message = response.messages[-1]
        assert message.content == "I will create the requested file."
        assert message.tool_calls == [
            {
                "id": "call_1",
                "name": "write_file",
                "arguments": {"path": "hello.txt", "content": "hello\n"},
            }
        ]
        assert response.usage == {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }

    @pytest.mark.asyncio
    async def test_json_encoded_tool_arguments_are_normalized(
        self, provider: CodexCLIProvider, write_tool: ToolDefinition
    ) -> None:
        stdout = jsonl_result(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "write_file",
                        "arguments": json.dumps({"path": "hello.txt", "content": "hello\n"}),
                    }
                ],
            }
        )
        provider._run = AsyncMock(return_value=(stdout, "", 0))

        response = await provider.chat(
            [Message(id="user-1", role="user", content="Create hello.txt")], [write_tool]
        )

        assert response.messages[-1].tool_calls == [
            {
                "id": "call_1",
                "name": "write_file",
                "arguments": {"path": "hello.txt", "content": "hello\n"},
            }
        ]

    @pytest.mark.asyncio
    async def test_completion_has_no_tool_calls(self, provider: CodexCLIProvider) -> None:
        provider._run = AsyncMock(
            return_value=(jsonl_result({"content": "Done.", "tool_calls": []}), "", 0)
        )

        response = await provider.chat(
            [Message(id="user-1", role="user", content="Report status")], []
        )

        assert response.messages[-1].content == "Done."
        assert response.messages[-1].tool_calls is None

    @pytest.mark.asyncio
    async def test_rejects_unknown_tool_call(
        self, provider: CodexCLIProvider, write_tool: ToolDefinition
    ) -> None:
        stdout = jsonl_result(
            {
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "shell_escape", "arguments": {}}],
            }
        )
        provider._run = AsyncMock(return_value=(stdout, "", 0))

        with pytest.raises(ModelResponseError, match="unknown tool"):
            await provider.chat(
                [Message(id="user-1", role="user", content="Do it")], [write_tool]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("stdout", "message"),
        [
            ("not-json", "Malformed Codex CLI JSONL"),
            (json.dumps({"type": "turn.completed"}), "completed agent message"),
            (
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {"type": "agent_message", "text": "not-json"},
                            }
                        ),
                        json.dumps({"type": "turn.completed"}),
                    ]
                ),
                "not valid structured JSON",
            ),
        ],
    )
    async def test_malformed_output_is_explicit(
        self, provider: CodexCLIProvider, stdout: str, message: str
    ) -> None:
        provider._run = AsyncMock(return_value=(stdout, "", 0))

        with pytest.raises(ModelResponseError, match=message):
            await provider.chat([Message(id="user-1", role="user", content="Hello")], [])

    @pytest.mark.asyncio
    async def test_authentication_failure_is_explicit(self, provider: CodexCLIProvider) -> None:
        provider._run = AsyncMock(return_value=("", "Not logged in; please run codex login", 1))

        with pytest.raises(ModelResponseError, match="authentication failed") as exc_info:
            await provider.chat([Message(id="user-1", role="user", content="Hello")], [])

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_nonzero_exit_is_explicit(self, provider: CodexCLIProvider) -> None:
        provider._run = AsyncMock(return_value=("", "sandbox setup failed", 23))

        with pytest.raises(ModelResponseError, match="exited with status 23") as exc_info:
            await provider.chat([Message(id="user-1", role="user", content="Hello")], [])

        assert exc_info.value.status_code == 23

    @pytest.mark.asyncio
    async def test_turn_failure_event_is_explicit(self, provider: CodexCLIProvider) -> None:
        stdout = json.dumps(
            {"type": "turn.failed", "error": {"message": "network unavailable"}}
        )
        provider._run = AsyncMock(return_value=(stdout, "", 0))

        with pytest.raises(ModelResponseError, match="turn failed.*network unavailable"):
            await provider.chat([Message(id="user-1", role="user", content="Hello")], [])

    @pytest.mark.asyncio
    async def test_timeout_kills_process(
        self, fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = CodexCLIProvider(
            api_key=None,
            config=ProviderConfig(
                model="codex-cli",
                provider_extensions={"executable": str(fake_codex), "timeout_s": 0.001},
            ),
        )
        provider.get_system_prompt(str(tmp_path), tools=[])

        class SlowProcess:
            returncode = None
            killed = False
            calls = 0

            async def communicate(self, input_data: bytes | None = None):
                self.calls += 1
                if self.calls == 1:
                    await asyncio.sleep(60)
                self.returncode = -9
                return b"", b""

            def kill(self) -> None:
                self.killed = True

        process = SlowProcess()

        async def fake_create_subprocess_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

        with pytest.raises(ModelResponseError, match="timed out"):
            await provider._run([str(fake_codex), "exec"], "prompt")

        assert process.killed is True


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_normalizes_response(
        self, provider: CodexCLIProvider, write_tool: ToolDefinition
    ) -> None:
        provider.chat = AsyncMock(
            return_value=_provider_response_with_tool_call(write_tool.name)
        )

        chunks = [
            chunk
            async for chunk in provider.chat_stream(
                [Message(id="user-1", role="user", content="Create it")],
                [write_tool],
                AgentConfig(),
            )
        ]

        assert [chunk.type for chunk in chunks] == ["token", "tool_call", "status"]


def _provider_response_with_tool_call(tool_name: str):
    from agentrunner.providers.base import ProviderResponse

    return ProviderResponse(
        messages=[
            Message(
                id="assistant-1",
                role="assistant",
                content="Working.",
                tool_calls=[{"id": "call_1", "name": tool_name, "arguments": {}}],
            )
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
