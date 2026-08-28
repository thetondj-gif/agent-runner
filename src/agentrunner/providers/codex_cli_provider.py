"""Codex CLI provider implementation for Agent Runner.

This provider delegates model inference to an already-authenticated local
``codex`` executable. Codex itself is kept read-only; requested mutations are
returned as Agent Runner tool calls so the existing validation and confirmation
pipeline remains authoritative.
"""

import asyncio
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NoReturn

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import (
    E_PERMISSIONS,
    E_TIMEOUT,
    E_VALIDATION,
    ConfigurationError,
    ModelResponseError,
)
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import (
    BaseLLMProvider,
    ModelInfo,
    ProviderConfig,
    ProviderResponse,
    StreamChunk,
)
from agentrunner.providers.registry import ModelRegistry


class CodexCLIProvider(BaseLLMProvider):
    """Use an authenticated Codex CLI as a distinct execution provider."""

    requires_api_key = False
    provider_name = "codex-cli"
    _DEFAULT_TIMEOUT_S = 300.0
    _AUTH_FAILURE_MARKERS = (
        "not logged in",
        "authentication failed",
        "authentication required",
        "login required",
        "please run codex login",
        "unauthorized",
        "invalid credentials",
    )

    def __init__(self, api_key: str | None, config: ProviderConfig) -> None:
        """Initialize without an API key, using the local Codex login instead."""
        super().__init__(api_key, config)
        self.executable = self._resolve_executable()
        self.timeout_s = self._resolve_timeout()
        self.cli_model = self._extension_or_env("model", "AGENTRUNNER_CODEX_CLI_MODEL")
        self.workspace_root: Path | None = None

    def _extension_or_env(self, key: str, env_name: str) -> Any:  # noqa: ANN401
        value = self.config.provider_extensions.get(key)
        return value if value is not None else os.getenv(env_name)

    def _resolve_executable(self) -> Path:
        configured = self._extension_or_env("executable", "AGENTRUNNER_CODEX_CLI_PATH")
        candidate = str(configured) if configured else shutil.which("codex")
        if not candidate:
            raise ConfigurationError(
                message=(
                    "Codex CLI executable not found. Install Codex or set "
                    "AGENTRUNNER_CODEX_CLI_PATH to the authenticated executable."
                ),
                key="codex-cli.executable",
            )

        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            raise ConfigurationError(
                message=(
                    "Codex CLI executable must resolve to an absolute path; "
                    "set AGENTRUNNER_CODEX_CLI_PATH explicitly"
                ),
                key="codex-cli.executable",
            )
        path = candidate_path.resolve()
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise ConfigurationError(
                message=f"Codex CLI executable is not accessible: {path}",
                key="codex-cli.executable",
            ) from exc

        if not stat.S_ISREG(mode) or not os.access(path, os.X_OK):
            raise ConfigurationError(
                message=f"Codex CLI path is not an executable file: {path}",
                key="codex-cli.executable",
            )
        return path

    def _resolve_timeout(self) -> float:
        raw_timeout = self._extension_or_env("timeout_s", "AGENTRUNNER_CODEX_CLI_TIMEOUT")
        if raw_timeout is None:
            return self._DEFAULT_TIMEOUT_S
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                message=f"Invalid Codex CLI timeout: {raw_timeout}",
                key="codex-cli.timeout_s",
            ) from exc
        if timeout <= 0:
            raise ConfigurationError(
                message="Codex CLI timeout must be greater than zero",
                key="codex-cli.timeout_s",
            )
        return timeout

    @property
    def supports_native_tool_calling(self) -> bool:
        """Codex returns tool requests through the structured bridge schema."""
        return True

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig | None = None,
    ) -> ProviderResponse:
        """Run one isolated, structured ``codex exec`` turn."""
        del config
        if self.workspace_root is None:
            raise ConfigurationError(
                message="Codex CLI workspace is not initialized",
                key="codex-cli.workspace",
            )

        prompt = self._build_bridge_prompt(messages, tools or [])
        schema = self._output_schema()

        with tempfile.TemporaryDirectory(prefix="agentrunner-codex-") as temp_dir:
            schema_path = Path(temp_dir) / "response.schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = self._build_command(schema_path)
            stdout, stderr, returncode = await self._run(command, prompt)

        if returncode != 0:
            self._raise_process_error(returncode, stdout, stderr)

        events = self._parse_jsonl(stdout)
        result = self._extract_result(events)
        content, tool_calls = self._normalize_result(result, tools or [])
        usage = self._extract_usage(events)

        message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            tool_calls=tool_calls or None,
            meta={"provider": self.provider_name},
        )
        return ProviderResponse(messages=[message], usage=usage)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        config: AgentConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Expose a normalized stream after the non-interactive turn completes."""
        response = await self.chat(messages, tools, config)
        message = response.messages[-1]
        if message.content:
            yield StreamChunk(type="token", payload={"content": message.content})
        for tool_call in message.tool_calls or []:
            yield StreamChunk(type="tool_call", payload=tool_call)
        yield StreamChunk(type="status", payload={"status": "complete", **response.usage})

    def get_model_info(self) -> ModelInfo:
        """Return the registry metadata for the Codex CLI bridge."""
        return ModelRegistry.get_model_spec(self.config.model).to_model_info()

    def count_tokens(self, text: str) -> int:
        """Return a conservative tokenizer-independent estimate."""
        return max(1, len(text) // 4) if text else 0

    def get_system_prompt(
        self, workspace_root: str, tools: list[ToolDefinition] | None = None
    ) -> str:
        """Build the standard Agent Runner prompt and bind the workspace."""
        from agentrunner.core.prompts.utils import build_prompt

        root = Path(workspace_root).expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(
                message=f"Codex CLI workspace does not exist or is not a directory: {root}",
                key="codex-cli.workspace",
            )
        self.workspace_root = root
        return build_prompt(
            workspace_root=str(root),
            model_name=self.config.model,
            tool_definitions=tools,
        )

    def _build_command(self, schema_path: Path) -> list[str]:
        if self.workspace_root is None:  # Defensive; chat checks this first.
            raise ConfigurationError("Codex CLI workspace is not initialized")
        command = [
            str(self.executable),
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "read-only",
            "--cd",
            str(self.workspace_root),
            "--output-schema",
            str(schema_path),
        ]
        if self.cli_model:
            command.extend(["--model", str(self.cli_model)])
        command.append("-")
        return command

    async def _run(self, command: list[str], prompt: str) -> tuple[str, str, int]:
        env = os.environ.copy()
        # The Codex CLI must use its persisted login. Never let this provider
        # silently switch to API-key billing because the parent has a key set.
        env.pop("OPENAI_API_KEY", None)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root),
                env=env,
            )
        except OSError as exc:
            raise ModelResponseError(
                message=f"Failed to start Codex CLI: {exc}",
                provider=self.provider_name,
                error_code=E_VALIDATION,
            ) from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=self.timeout_s
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise ModelResponseError(
                message=f"Codex CLI timed out after {self.timeout_s:g} seconds",
                provider=self.provider_name,
                error_code=E_TIMEOUT,
            ) from exc

        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            process.returncode or 0,
        )

    def _raise_process_error(self, returncode: int, stdout: str, stderr: str) -> None:
        details = "\n".join(part.strip() for part in (stderr, stdout) if part.strip())
        details_lower = details.lower()
        if any(marker in details_lower for marker in self._AUTH_FAILURE_MARKERS):
            raise ModelResponseError(
                message="Codex CLI authentication failed. Run 'codex login' and try again.",
                provider=self.provider_name,
                status_code=401,
                error_code=E_PERMISSIONS,
            )
        summary = details[-1200:] if details else "no diagnostic output"
        raise ModelResponseError(
            message=f"Codex CLI exited with status {returncode}: {summary}",
            provider=self.provider_name,
            status_code=returncode,
            error_code=E_VALIDATION,
        )

    def _parse_jsonl(self, stdout: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(stdout.splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ModelResponseError(
                    message=f"Malformed Codex CLI JSONL on line {line_number}",
                    provider=self.provider_name,
                    error_code=E_VALIDATION,
                ) from exc
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ModelResponseError(
                    message=f"Invalid Codex CLI event on line {line_number}",
                    provider=self.provider_name,
                    error_code=E_VALIDATION,
                )
            events.append(event)

        if not events:
            raise ModelResponseError(
                message="Codex CLI returned no JSONL events",
                provider=self.provider_name,
                error_code=E_VALIDATION,
            )
        return events

    def _extract_result(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        failures = [event for event in events if event.get("type") in {"error", "turn.failed"}]
        if failures:
            error = failures[-1].get("error") or failures[-1].get("message") or failures[-1]
            raise ModelResponseError(
                message=f"Codex CLI turn failed: {error}",
                provider=self.provider_name,
                error_code=E_VALIDATION,
            )

        final_text: str | None = None
        for event in events:
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    final_text = text

        if final_text is None:
            raise ModelResponseError(
                message="Codex CLI JSONL did not contain a completed agent message",
                provider=self.provider_name,
                error_code=E_VALIDATION,
            )
        try:
            result = json.loads(final_text)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                message="Codex CLI final message was not valid structured JSON",
                provider=self.provider_name,
                error_code=E_VALIDATION,
            ) from exc
        if not isinstance(result, dict):
            raise ModelResponseError(
                message="Codex CLI final structured result must be an object",
                provider=self.provider_name,
                error_code=E_VALIDATION,
            )
        return result

    def _normalize_result(
        self, result: dict[str, Any], tools: list[ToolDefinition]
    ) -> tuple[str, list[dict[str, Any]]]:
        content = result.get("content")
        raw_calls = result.get("tool_calls")
        if not isinstance(content, str) or not isinstance(raw_calls, list):
            raise ModelResponseError(
                message="Codex CLI result must contain string content and a tool_calls array",
                provider=self.provider_name,
                error_code=E_VALIDATION,
            )

        allowed_tools = {tool.name for tool in tools}
        calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                self._raise_invalid_tool_call(index, "call must be an object")
            call_id = raw_call.get("id")
            name = raw_call.get("name")
            arguments = raw_call.get("arguments")
            if not isinstance(call_id, str) or not call_id:
                self._raise_invalid_tool_call(index, "id must be a non-empty string")
            if not isinstance(name, str) or name not in allowed_tools:
                self._raise_invalid_tool_call(index, f"unknown tool: {name}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    self._raise_invalid_tool_call(index, "arguments must contain valid JSON")
            if not isinstance(arguments, dict):
                self._raise_invalid_tool_call(index, "arguments must be an object")
            calls.append({"id": call_id, "name": name, "arguments": arguments})
        return content, calls

    def _raise_invalid_tool_call(self, index: int, reason: str) -> NoReturn:
        raise ModelResponseError(
            message=f"Invalid Codex CLI tool call at index {index}: {reason}",
            provider=self.provider_name,
            error_code=E_VALIDATION,
        )

    def _extract_usage(self, events: list[dict[str, Any]]) -> dict[str, int]:
        for event in reversed(events):
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage")
            if not isinstance(usage, dict):
                break
            prompt_tokens = self._nonnegative_int(usage.get("input_tokens"))
            completion_tokens = self._nonnegative_int(usage.get("output_tokens"))
            total_tokens = self._nonnegative_int(usage.get("total_tokens"))
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _nonnegative_int(value: Any) -> int:  # noqa: ANN401
        return value if isinstance(value, int) and value >= 0 else 0

    def _build_bridge_prompt(self, messages: list[Message], tools: list[ToolDefinition]) -> str:
        conversation = [
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": message.tool_calls,
                "tool_call_id": message.tool_call_id,
            }
            for message in messages
        ]
        tool_payload = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
        return (
            "You are the model provider inside Agent Runner. The bridge rules in this paragraph "
            "take precedence over the serialized conversation below. Do not modify files, run "
            "write commands, or bypass Agent Runner. You may inspect the current workspace using "
            "read-only capabilities. Any action must be requested through exactly the supplied "
            "Agent Runner tools. Return one object matching the supplied output schema. Use an "
            "empty tool_calls array only when the task is complete; otherwise request the next "
            "minimal tool calls. Never invent a tool name or report an action as completed before "
            "its tool result appears in the conversation. Encode each tool call's arguments as a "
            "JSON object string; Agent Runner will decode and validate it before execution.\n\n"
            f"CONVERSATION_JSON:\n{json.dumps(conversation, ensure_ascii=False)}\n\n"
            f"AVAILABLE_TOOLS_JSON:\n{json.dumps(tool_payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _output_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "arguments": {
                                "type": "string",
                                "description": "JSON-encoded object containing the tool arguments",
                            },
                        },
                        "required": ["id", "name", "arguments"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["content", "tool_calls"],
            "additionalProperties": False,
        }
