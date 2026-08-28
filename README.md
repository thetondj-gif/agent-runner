<div align="center">
  <img src="https://www.designarena.ai/DesignArenaAssets/DesignArenaTitleLogo.svg" alt="Design Arena" width="200" />
</div>

# Design Arena Agent Runner

[![PyPI version](https://badge.fury.io/py/agent-runner.svg)](https://badge.fury.io/py/agent-runner)
[![Python Versions](https://img.shields.io/pypi/pyversions/agent-runner.svg)](https://pypi.org/project/agent-runner/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A model-agnostic, framework-agnostic agent harness that enables autonomous AI agents across any LLM provider. 

> **Beta Version** - Currently in active development.

## Features

- **Turn Any Model Into an Agent**: Configure any model as a coding agents
- **Flexible Configuration**: Set up agent behavior, tools, and constraints however you need
- **CLI or Python API**: Use as a command-line tool or integrate into your applications
- **Powerful Tooling**: File operations, code editing, search, bash execution
- **Smart Context Management**: Automatic context window handling and compaction
- **Workspace Isolation**: Secure file operations with command validation

## Installation

### Prerequisites

**Python:**
- Python 3.11 or higher

### Install Agent Runner

**Option 1: From PyPI (recommended)**
```bash
pip install agent-runner
```

**Option 2: From source**
```bash
git clone https://github.com/Design-Arena/agent-runner.git
cd agent-runner
pip install -e .
```

**With optional dependencies:**
```bash
# Development tools (testing, linting)
pip install agent-runner[dev]

# Optional tools (screenshot, patch tools)
pip install agent-runner[tools]

# Additional providers (Mistral)
pip install agent-runner[providers]

# Everything
pip install agent-runner[all]
```

**For screenshot tool (optional):**
```bash
pip install agent-runner[tools]
playwright install chromium
```

**For search functionality (recommended):**

Install `ripgrep` for fast code search:
```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
sudo apt-get install ripgrep
```

### Configure API Keys

Set up your API keys as environment variables. You only need keys for the providers you'll use.

The `codex-cli` provider is the exception: it uses the existing local Codex CLI
login and does not read or require `OPENAI_API_KEY`:

```bash
codex login
codex login status
agentrunner run "Create hello.py" --model codex-cli
```

Agent Runner invokes `codex exec --json` in a read-only sandbox rooted at the
selected workspace. Codex returns structured Agent Runner tool requests; file
writes and commands still pass through Agent Runner's existing validation and
confirmation flow. Authentication or CLI failures are reported explicitly and
never fall back to the OpenAI API provider.

**Create a `.env` file in your project directory:**

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
XAI_API_KEY=...
KIMI_API_KEY=...
ZAI_API_KEY=...
```

**Or set environment variables directly:**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# etc.
```

## Quick Start

### Basic Usage

**Run a single task:**
```bash
agentrunner run "Create a Python calculator with add and multiply functions"
agentrunner run "Build a FastAPI todo app" --verbose  # See live events
```

**Interactive chat:**
```bash
agentrunner chat
```

**Review code (read-only, no modifications):**
```bash
agentrunner review .
agentrunner review src/main.py
```

**Model management:**
```bash
agentrunner models  # List all available models
agentrunner run "Create a small Python module" --model codex-cli
agentrunner run "Build a REST API" --model claude-sonnet-4-5-20250929
agentrunner run "Analyze this code" --model gemini-2.5-pro
```

**Session management:**
```bash
agentrunner sessions list              # List all sessions
agentrunner sessions show <id>         # View session details
agentrunner sessions show <id> --events  # View with full event log
agentrunner sessions delete <id>       # Delete a session
```

**Note:** There are two types of sessions:
- `sessions` (plural) - CLI session history with full event traces
- `session` (singular) - Agent conversation sessions that can be resumed

**Configuration:**
```bash
agentrunner config list              # List profiles
agentrunner config show              # Show current profile
agentrunner config set-default --model claude-sonnet-4-5-20250929  # Set default model
```

### Usage from Python

**Basic example:**

```python
import asyncio
import os
from agentrunner.core.config import AgentConfig
from agentrunner.core.factory import create_agent
from agentrunner.providers.base import ProviderConfig

async def main():
    # Set your API key
    os.environ['OPENAI_API_KEY'] = "your-api-key"
    
    # Configure agent behavior (orchestration settings)
    agent_config = AgentConfig(
        max_rounds=50,           # Maximum number of agent turns
        tool_timeout_s=120       # Timeout for tool execution
    )
    
    # Configure the LLM provider (model settings)
    provider_config = ProviderConfig(
        model="gpt-5-codex",     # Model to use
        temperature=0.7,         # Sampling temperature
        max_tokens=4096          # Max tokens in response
    )
    
    # Create the agent
    agent = create_agent(
        workspace_path=".",              # Working directory
        provider_config=provider_config,
        agent_config=agent_config,
        profile="default"                # Optional profile name
    )
    
    # Run a task
    result = await agent.process_message(
        "Create a REST API for user management with FastAPI"
    )
    
    print(f"Result: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Using different providers:**

```python
# Using an already authenticated Codex CLI (no OPENAI_API_KEY)
provider_config = ProviderConfig(model="codex-cli")

# Using Claude Sonnet 4.5
os.environ['ANTHROPIC_API_KEY'] = "your-api-key"
provider_config = ProviderConfig(
    model="claude-sonnet-4-5-20250929",
    temperature=0.7
)
```

## Available Tools

Agent Runner provides a comprehensive set of tools that work consistently across all LLM providers:

### File Operations
- `read_file` - Read file contents
- `write_file` - Write content to files
- `create_file` - Create new files
- `delete_file` - Delete files
- `edit_file` - Edit files with search/replace
- `multi_edit` - Multiple edits in one operation
- `insert_lines` - Insert lines at specific positions
- `batch_create_files` - Create multiple files at once

### Search
- `grep` - Fast code search using ripgrep

### Execution & Project Management
- `bash` - Execute bash commands
- `scaffold_project` - Generate project templates
- `clean_workspace` - Clean up workspace directory

### Media & Generation
- `take_screenshot` - Capture screenshots (requires Playwright)
- `fetch_image` - Fetch images from URLs
- `generate_image` - Generate AI images
- `fetch_video` - Fetch videos from URLs
- `generate_video` - Generate AI videos

### Deployment
- `deploy_to_vercel` - Deploy to Vercel

**Note:** All file operations are restricted to the workspace directory. Bash commands are validated against a whitelist and require user confirmation for dangerous operations, but exercise caution.

## Supported Providers

Agent Runner provides a unified interface for multiple LLM providers:

**Codex CLI** • **OpenAI** • **Anthropic** • **Google** • **xAI** • **Mistral** • **Moonshot AI (Kimi)** • **Z.AI**

See [`src/agentrunner/providers/registry.py`](src/agentrunner/providers/registry.py) for the complete list of available models.


## Security Note

Agent Runner executes code and commands within your specified workspace directory. While file operations are restricted to the workspace, the agent can modify/delete files within it and execute bash commands. Always use a dedicated directory for agent work, exercise caution when pointing it at sensitive directories, and use version control to track changes.

## Configuration

### Environment Variables

**Setting defaults via CLI:**
```bash
# Set default model
agentrunner config set-default --model claude-sonnet-4-5-20250929

# Set default temperature
agentrunner config set-default --temperature 0.8

# Set multiple defaults at once
agentrunner config set-default --model gpt-5-codex --temperature 0.7
```

This creates/updates a `.env` file in your current directory.

**Or set them manually in `.env` file:**

**Provider defaults (model, temperature, etc.):**
```bash
# Default model (if not specified with --model)
export AGENTRUNNER_MODEL="claude-sonnet-4-5-20250929"

# Default temperature (if not specified with --temperature)
export AGENTRUNNER_TEMPERATURE="0.7"

# Max tokens for responses (optional)
export AGENTRUNNER_MAX_TOKENS="4096"
```

**Codex CLI provider (optional):**
```bash
# Override executable discovery; otherwise Agent Runner safely resolves `codex` from PATH
export AGENTRUNNER_CODEX_CLI_PATH="/absolute/path/to/codex"

# Provider subprocess timeout in seconds (default: 300)
export AGENTRUNNER_CODEX_CLI_TIMEOUT="300"

# Override the model selected by the user's Codex configuration (optional)
export AGENTRUNNER_CODEX_CLI_MODEL="your-codex-model"
```

The equivalent Python configuration uses `ProviderConfig.provider_extensions`
with the keys `executable`, `timeout_s`, and `model`.

**Agent behavior:**
```bash
# Maximum agentic loop iterations
export AGENTRUNNER_MAX_ROUNDS=100

# Tool execution timeout in seconds
export AGENTRUNNER_TOOL_TIMEOUT=180
```

### Profile Configuration

Profiles control agent orchestration behavior. Create JSON files in `~/.agentrunner/profiles/`:

**Example: `~/.agentrunner/profiles/default.json`**
```json
{
  "max_rounds": 50,
  "tool_timeout_s": 120,
  "response_buffer_tokens": 1000,
  "allow_streaming": true
}
```

**Example: `~/.agentrunner/profiles/thorough.json`** (for complex tasks)
```json
{
  "max_rounds": 100,
  "tool_timeout_s": 300
}
```

Use with: `agentrunner run "task" --profile thorough`

**Note:** Model selection (`--model`), temperature (`--temperature`), and token limits (`--max-tokens`) are specified via CLI flags, not in profiles.

### Project Configuration

Create `.agentrunner/config.json` in your project to override profile settings per-project:

```json
{
  "max_rounds": 100,
  "tool_timeout_s": 180
}
```

## Usage Examples

### Create a New Project

```bash
mkdir ~/my-portfolio && cd ~/my-portfolio
agentrunner run "Create a cyberpunk-themed portfolio website with neon accents, animated background, project showcase grid, and contact form using Next.js, shadcn/ui, and Tailwind"
```

### Code Review & Refactoring

```bash
cd ~/my-project
agentrunner run "Review src/api.py for security issues and refactor to use async/await"
agentrunner review src/api.py
```

### Interactive Development

```bash
agentrunner chat
> Add error handling to all API endpoints
> Write unit tests for the new error handlers
> Run the tests with pytest
> quit
```

### Debugging

```bash
agentrunner run "The tests in test_api.py are failing. Investigate and fix the bugs"
```

### Generate Images & Media

```bash
agentrunner run "Generate a futuristic cityscape at sunset with neon lights and flying cars, then set it as the hero section background image on my landing page"
```

## Architecture

```
agentrunner/
├── core/           # Agent logic, session management, config
├── providers/      # LLM provider integrations
├── tools/          # Tool implementations
├── security/       # Command validation, sandboxing
└── cli/            # Command-line interface
```

**Key Components:**
- **Agent**: Orchestrates LLM interactions and tool execution
- **Providers**: Unified interface for LLM APIs
- **Tools**: Extensible tool system
- **Session**: Conversation history persistence
- **Context Strategy**: Context window management

## Troubleshooting

### Command Not Found

If `agentrunner` command is not found:

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Reinstall if needed
pip install -e .
```

### API Key Issues

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Source .env file if using one
cd /path/to/agentrunner
set -a && source .env && set +a
```

For `--model codex-cli`, do not add an API key. Verify the persisted local login
instead:

```bash
codex login status
```

### Session Management

**CLI Session History:**
```bash
# List all CLI sessions (with event traces)
agentrunner sessions list

# View session details and messages
agentrunner sessions show <session-id>

# View with full event log (tool calls, file changes, etc.)
agentrunner sessions show <session-id> --events

# Delete a CLI session
agentrunner sessions delete <session-id>
```

**Agent Conversation Sessions:**
```bash
# List agent sessions (resumable conversations)
agentrunner session list

# Load and resume a conversation
agentrunner session load <session-id>

# Delete an agent session
agentrunner session delete <session-id>
```

### Permission Errors

Agent Runner restricts file operations to the workspace directory for safety. Make sure you're running from the correct directory and have write permissions. **Always use a dedicated workspace directory, not your home or system directories.**

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=agentrunner --cov-report=html

# Specific test
pytest tests/unit/test_agent.py
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
