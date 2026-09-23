# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an RL (Reinforcement Learning) web agent project that enables automated browser interactions through a WebArena-like environment. The project uses Playwright for browser automation, Hydra for configuration management, and includes a proxy system for host rewriting. Development primarily happens through Jupyter notebooks.

## Architecture

### Core Components
- **WebAgentEnv** (`rl_web_agent/env.py`): Main environment class that manages browser sessions, page interactions, and provides a step-based interface for RL agents
- **Configuration System**: Single YAML config file (`rl_web_agent/conf/config.yaml`) managed by Hydra with override support
- **JavaScript Layer**: Browser-side scripts for DOM parsing (`parser.js`) and event detection (`initscript.js`)
- **Proxy System**: HTTP proxy client (`proxy/proxy_client_aiohttp.py`) for host rewriting and API gateway communication

### Key Design Patterns
- **Async/Await**: All browser operations are asynchronous using Playwright's async API
- **Semantic IDs**: DOM elements are tagged with unique `data-semantic-id` attributes for reliable interaction
- **Action-Observation Loop**: Environment provides JSON-based action interface and structured observations
- **Shared Playwright Instance**: ClassVar pattern for efficient resource management across multiple environment instances

## Development Workflow

### Running the Agent
```bash
# Basic execution with default config
python -m rl_web_agent.main

# Override specific configuration values
python -m rl_web_agent.main environment.browser.headless=true environment.proxy.enabled=false

# Development with Jupyter notebooks (primary workflow)
jupyter notebook  # Use notebooks/ directory for experimentation
```

### Dependency Management (UV)
```bash
# Install dependencies
uv sync

# Add new dependencies
uv add package_name

# Install with GPU support (includes transformers, torch, etc.)
uv sync --group gpu

# Install WebArena support
uv sync --extra webarena
```

### Code Quality
```bash
# Linting (uses ruff configuration in pyproject.toml)
ruff check .
ruff format .
```

## Configuration System

### Single Config File Pattern
All configuration is centralized in `rl_web_agent/conf/config.yaml`. The project uses Hydra's single config approach rather than component configs.

### Key Configuration Sections
- `environment.browser`: Playwright launch and context options
- `environment.proxy`: Proxy server settings for host rewriting
- `environment.sites`: Mapping of site names to hostnames for WebArena environments
- `hydra.run.dir`: Output directory for execution logs

### Override Examples
```bash
# Disable headless mode for debugging
python -m rl_web_agent.main environment.browser.launch_options.headless=false

# Change proxy settings
python -m rl_web_agent.main environment.proxy.server=http://localhost:9090 environment.proxy.enabled=true

# Add new site mapping
python -m rl_web_agent.main environment.sites.custom_site=example.com:8080
```

## WebAgentEnv API

### Action Interface
Actions are JSON strings with specific formats:
```python
# Click actions
await env.step('{"action": "click", "target": "login_button"}')

# Text input with optional enter
await env.step('{"action": "type", "target": "username", "text": "john_doe", "enter": true}')

# Navigation
await env.step('{"action": "goto_url", "url": "https://example.com"}')

# Tab management
await env.step('{"action": "new_tab", "url": "https://example.com"}')
await env.step('{"action": "switch_tab", "tab_id": 1}')
```

### Observation Structure
```python
observation = await env.observation()
# Returns:
# {
#   "html": "...",  # Processed DOM with semantic IDs
#   "clickable_elements": ["button1", "link2", ...],
#   "input_elements": [{"id": "username", "type": "text", "value": "", ...}],
#   "tabs": [{"id": 0, "title": "Page Title", "url": "...", "is_active": true}]
# }
```

## Browser Automation Details

### DOM Processing Pipeline
1. **Initialization Script** (`initscript.js`): Detects hover events and marks hoverable elements
2. **Parser Script** (`parser.js`): Strips DOM to essential interactive elements, assigns semantic IDs, preserves form state
3. **Semantic ID Generation**: Creates hierarchical, unique identifiers for reliable element targeting

### Element Interaction Patterns
- All interactions use semantic IDs rather than CSS selectors
- Elements are automatically scrolled into view before interaction
- Form state is captured and preserved in observations
- Empty interactive elements (inputs, selects) are preserved even if visually empty

## Third-Party Integration

### WebArena Integration
- Located in `thirdparty/webarena/` as editable dependency
- Provides web-based RL environments for agent training
- Task configurations define start URLs, evaluation criteria, and required actions

### VERL Integration
- Located in `thirdparty/verl/` for reinforcement learning workflows
- GPU dependency group includes all necessary ML packages
- Supports distributed training with Ray

## Important Conventions

### Code Style
- Use async/await for all browser operations
- Follow PEP 8 with 4-space indentation
- Use `pathlib.Path` instead of `os.path`
- **NEVER use dict.get()** - Always use direct key access (dict["key"]) to fail fast
- **No graceful error handling for missing keys** - Let KeyErrors happen to catch bugs early
- **Fail fast approach** - Don't hide missing keys or data structure issues

### Error Handling
- **FAIL FAST** - No try-catch blocks unless absolutely necessary (e.g., auto retry for LLM API calls to avoid rate limits)
- **NEVER use getattr() or dict.get()** - Always use direct key access to fail fast for missing keys
- **No graceful error handling for missing keys** - Let KeyErrors happen to catch bugs early
- Wrap browser operations in try/finally blocks only when needed
- Use structured logging with semantic context
- Return error information in step() observations

### Logging and Debugging Rules
- **NEVER truncate conversation history or message content** - Always log full content without [:200] or similar truncation
- **NEVER limit message history** - Log all messages, not just recent ones (avoid [-4:] or similar slicing)
- **NEVER assume content length limits** - Show complete debugging information
- **ALWAYS show full conversation history** - No shortcuts or "last N messages" approaches

### Prompt Management Rules
- **NEVER hardcode prompts in code** - Always load prompts from .txt files in rl_web_agent/prompts/
- **Use load_prompt() function** - Import from rl_web_agent.prompts and use load_prompt("prompt_name")
- **Create descriptive prompt files** - Name files clearly (e.g., fuzzy_match_evaluator.txt, system_prompt.txt)
- **Keep prompts maintainable** - Use format strings with placeholders like {objective}, {question}, {reference}

## Development Notes

### Testing Approach
- Primary development through Jupyter notebooks in `notebooks/` directory
- No formal test suite - notebooks serve as interactive testing environment
- Use `notebooks/playwright-test.ipynb` for browser automation experiments

### Proxy System
- Enables host rewriting for WebArena environments
- Supports AWS SigV4 authentication for API gateway communication
- Configure target host rewrites in proxy client for local development

### Performance Considerations
- Shared Playwright instance across multiple environments
- Image blocking enabled by default to speed up page loads
- Browser args optimized for automation (disable extensions, autofill, etc.)

## IDE Integration Test
This is a test message added to verify IDE integration is working correctly.
