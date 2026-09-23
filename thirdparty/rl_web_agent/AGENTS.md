# AGENTS.md

Repo-specific notes for agents. Also read `.cursorrules` (authoritative coding conventions); `CLAUDE.md` is older and partly stale (see "Known stale doc claims" below).

## Layout at a glance

- `rl_web_agent/` — main package. Real entrypoints live in `rl_web_agent/entrypoints/`, not in `rl_web_agent/main.py`.
  - `env.py` `WebAgentEnv` (Playwright-based, async, shared `Playwright` via ClassVar).
  - `agent.py` chain-of-thought agent; `tool_agent.py` function-calling agent.
  - `llm.py` LiteLLM-backed multi-provider client (openai / azure_openai / bedrock).
  - `conf/base.yaml` canonical defaults (only file in `conf/`).
  - `javascript/` browser-side DOM parser + init script injected by `WebAgentEnv`.
  - `prompts/*.txt` loaded via `from rl_web_agent.prompts import load_prompt`.
- Repo-root `config.yaml` is the Hydra composition root: it pulls `rl_web_agent/conf/base` and an optional repo-root `local_config.yaml` (`@_global_`). Put local overrides there, not in `base.yaml`.
- `proxy/client/` + `proxy/server/` are **Rust** (`cargo`), not Python. The Playwright browser points at the client proxy (default `http://localhost:8080`) which does host rewriting / SigV4. You must run the proxy out-of-band for `environment.proxy.enabled=true` (the default).
- `incus_server.py` (Quart) — separate HTTP service at `environment.incus_server_url` (default `http://127.0.0.1:8001`) that launches/kills per-task Incus containers. `WebAgentEnv.setup()` will call it; must be running for non-trivial tasks.
- `thirdparty/webarena/` — editable install, source of the upstream WebArena evaluators. **The task configs under `thirdparty/webarena/config_files/` are NOT the live ones — do not edit or grep there.** Use `dataset/` instead (see below).
- `dataset/test_webarena_lite/<task_id>.json` and `dataset/train_webarena/<task_id>.json` — the **live task configs** the agent actually loads. These are per-task JSON files (intent, eval, reference answers, start URLs). When fixing ground truth, eval references, or start URLs, edit these files, not `thirdparty/webarena/config_files/`.
- Benchmark URLs in live configs use `metis.lti.cs.cmu.edu:<port>` (the real host), not the upstream WebArena `localhost` canonicalization. When comparing strings the agent sees against reference answers, expect `metis.lti.cs.cmu.edu` in whatever the browser renders.
- `sft_training/` — self-contained SFT training project with its own `pyproject.toml` / `uv.lock` / Dockerfile. Do not mix with the root env.
- `dataset/`, `results/`, `workshop_exp/`, `outputs/` — large data dirs; never `grep`/read them blindly (except the per-task JSON files in `dataset/test_webarena_lite/` and `dataset/train_webarena/`, which are small and targeted).
- `results/deprecate/` — archived result runs from older training steps. Active runs live directly under `results/webarena_lite/<run_name>/`.

## Running things

Use Hydra overrides; do not edit `base.yaml` for one-off changes.

- Single task: `python -m rl_web_agent.entrypoints.agent task_config=thirdparty/webarena/config_files/1.json`
- Tool agent: `python -m rl_web_agent.entrypoints.tool_agent ...`
- Batch: `python -m rl_web_agent.entrypoints.batch_agent --task_ids 1,2,3 --max_concurrent 3 --agent_type tool`
- Replay a trace: `python -m rl_web_agent.entrypoints.replay --trace_file results/task_506/trace.json`
- Human REPL: `python -m rl_web_agent.entrypoints.repl task_config=...`
- Override examples: `environment.browser.launch_options.headless=false environment.proxy.enabled=false llm.provider=openai`

`rl_web_agent/main.py` is a stale demo — it points at `config_path="conf", config_name="config"` but `conf/` only has `base.yaml`, so it will not start as-is. Prefer the `entrypoints/` modules.

Prerequisites before a task can actually run end-to-end: the Rust proxy client on `:8080`, the Incus server on `:8001`, Playwright browsers installed (`playwright install`), and credentials for the selected `llm.provider` in env vars (LiteLLM picks them up). Evaluator LLM is configured separately under `environment.evaluator_llm`.

## Dev workflow

- **Always use the project venv at `.venv/`.** The system `python` / `/root/miniforge3/bin/python` is missing project deps (ray, sglang, litellm, playwright, etc.). Never invoke a bare `python ...` — always `.venv/bin/python ...`, or `uv run --no-sync python ...`, or activate with `source .venv/bin/activate`. Same rule for `pip` / `ruff` / `pytest` / any project tool; use `.venv/bin/<tool>` or `uv run --no-sync <tool>`.
- Install: `uv sync` (add `--extra webarena` to pull WebArena deps). Python ≥3.10.
- Lint/format: `ruff check .` / `ruff format .`. Pre-commit is configured (`.pre-commit-config.yaml`); `thirdparty/` is excluded from hooks — don't lint it.
- `line-length = 300` in `pyproject.toml`; do not re-wrap to 88/120.
- No test suite. Experimentation lives in `notebooks/`. Do **not** add `tests/`, example scripts, or `*.md` docs — `.cursorrules` explicitly forbids creating test, documentation, or example files.

## Hard conventions (enforced by `.cursorrules`)

- **Never** use `dict.get(...)` or `getattr(obj, name, default)`. Always `dict["key"]` / `obj.attr`. Missing keys must raise.
- No try/except for "graceful" handling — `try/finally` for cleanup only; `try/except` only for things like LLM retries.
- Never truncate logs, conversation history, or debug output (no `[:200]`, `[-4:]` slicing for "last N messages", etc.).
- Prompts live in `rl_web_agent/prompts/*.txt` and are loaded with `load_prompt("name")`. Do not hardcode prompt strings.
- Config is a single file — put new settings in `rl_web_agent/conf/base.yaml` (with YAML comments for docs). No component config subdirs.
- All browser interaction uses `data-semantic-id` selectors produced by `javascript/parser.js`, not CSS/XPath.

## Known stale doc claims

When the docs and code disagree, trust the code.

- `CLAUDE.md` and `.cursorrules` say the proxy client is `proxy/proxy_client_aiohttp.py`. It is not — the proxy is the Rust crates under `proxy/client/` and `proxy/server/`.
- `CLAUDE.md` instructs running `python -m rl_web_agent.main`. That module is stale; use `rl_web_agent.entrypoints.*`.
- `CLAUDE.md` references `rl_web_agent/conf/config.yaml` and a `gpu` dependency group. Neither exists — the config file is `conf/base.yaml` composed via repo-root `config.yaml`, and `pyproject.toml` defines a `webarena` optional-extra but no `gpu` group.
- `thirdparty/verl/` is referenced in docs but is not checked into this tree; only `thirdparty/webarena/` is present.
