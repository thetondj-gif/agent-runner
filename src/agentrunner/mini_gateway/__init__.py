"""Mac Mini specialist-agent gateway for Agent Runner.

This package exposes locally installed coding agents through a controlled
execution layer that can be mounted as an MCP server for Devin or other MCP
clients.
"""

from .executor import agent_status, run_agent

__all__ = ["agent_status", "run_agent"]
