#!/usr/bin/env python3
"""
Run WebArena evaluations across multiple sglang-served model checkpoints using Ray.

Each checkpoint gets a Ray actor that:
  1. Launches sglang server (acquires GPUs via Ray)
  2. Waits for health check
  3. Runs batch_agent evaluation as a subprocess
  4. Shuts down sglang
  5. Returns results

Checkpoints run sequentially (one sglang server at a time) to avoid GPU contention,
but the evaluation itself runs many tasks in parallel.

Usage:
    # Submit as a Ray job
    ray job submit --address=auto --working-dir . -- python run_sglang_eval.py

    # Or run directly (Ray must be initialized)
    python run_sglang_eval.py

    # Dry run
    python run_sglang_eval.py --dry-run

    # Specific task IDs
    python run_sglang_eval.py --task-ids 104,117,118
"""

import argparse
import multiprocessing
import socket
import subprocess
import sys
import time

import ray
import requests
from sglang.srt.server_args import ServerArgs

# ---------------------------------------------------------------------------
# Model definitions: list of (name, checkpoint_path)
# ---------------------------------------------------------------------------
# Current run: llama-3.1 SFT checkpoints (two LR sweeps, 3 iters each).
MODELS = [
    # --- Llama-3.1 SFT 128k (default LR) ---
    (
        "llama31-sft-128k-step22",
        "/data/checkpoints/rl_web_agent/web_agent_sft_128k_llama-3.1-hf/iter_0000022_hf/",
    ),
    (
        "llama31-sft-128k-step44",
        "/data/checkpoints/rl_web_agent/web_agent_sft_128k_llama-3.1-hf/iter_0000044_hf/",
    ),
    (
        "llama31-sft-128k-step66",
        "/data/checkpoints/rl_web_agent/web_agent_sft_128k_llama-3.1-hf/iter_0000066_hf/",
    ),
    # --- Llama-3.1 SFT 128k (lr=1e-5) ---
    (
        "llama31-sft-128k-lr1e5-step22",
        "/data/checkpoints/rl_web_agent/web_agent_sft_128k_llama-3.1-lr1e5-hf/iter_0000022_hf/",
    ),
    (
        "llama31-sft-128k-lr1e5-step44",
        "/data/checkpoints/rl_web_agent/web_agent_sft_128k_llama-3.1-lr1e5-hf/iter_0000044_hf/",
    ),
    (
        "llama31-sft-128k-lr1e5-step66",
        "/data/checkpoints/rl_web_agent/web_agent_sft_128k_llama-3.1-lr1e5-hf/iter_0000066_hf/",
    ),
    # --- Previous runs (kept for reference) ---
    # (
    #     "qwen3-30b-a3b-rl-step19",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000019/",
    # ),
    # (
    #     "qwen3-30b-a3b-rl-step39",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000039/",
    # ),
    # (
    #     "qwen3-30b-a3b-rl-step59",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000059/",
    # ),
    # (
    #     "qwen3-30b-a3b-rl-step79",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000079/",
    # ),
    # (
    #     "qwen3-30b-a3b-rl-step99",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000099/",
    # ),
    # (
    #     "qwen3-30b-a3b-rl-step119",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000119/",
    # ),
    # (
    #     "qwen3-30b-a3b-rl-step139",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000139/",
    # ),
    # (
    #     "qwen3-30b-a3b-rl-step159",
    #     "/data/checkpoints/rl_web_agent/30b-train/qwen3-30B-A3B-30b-train-hf/iter_0000159/",
    # ),
    # (
    #     "qwen3-4b-rl-step19",
    #     "/data/checkpoints/rl_web_agent/4b-train/qwen3-4B-Instruct-2507-3-site-sft-3-epoch-hf/iter_0000019/",
    # ),
    # (
    #     "qwen3-4b-rl-step39",
    #     "/data/checkpoints/rl_web_agent/4b-train/qwen3-4B-Instruct-2507-3-site-sft-3-epoch-hf/iter_0000039/",
    # ),
    # (
    #     "qwen3-4b-rl-step59",
    #     "/data/checkpoints/rl_web_agent/4b-train/qwen3-4B-Instruct-2507-3-site-sft-3-epoch-hf/iter_0000059/",
    # ),
    # (
    #     "qwen3-4b-rl-step79",
    #     "/data/checkpoints/rl_web_agent/4b-train/qwen3-4B-Instruct-2507-3-site-sft-3-epoch-hf/iter_0000079/",
    # ),
    # (
    #     "qwen3-30b-a3b-sft-128k-step22",
    #     "/data/checkpoints/rl_web_agent/web_agent_sft_128k_30b_hf/iter_0000022_hf/",
    # ),
    # (
    #     "qwen3-30b-a3b-sft-128k-step44",
    #     "/data/checkpoints/rl_web_agent/web_agent_sft_128k_30b_hf/iter_0000044_hf/",
    # ),
    # (
    #     "qwen3-30b-a3b-sft-128k-step66",
    #     "/data/checkpoints/rl_web_agent/web_agent_sft_128k_30b_hf/iter_0000066_hf/",
    # ),
    # (
    #     "qwen3-4b-sft-128k-step22",
    #     "/data/checkpoints/rl_web_agent/web_agent_sft_128k_4b_hf/iter_0000022_hf/",
    # ),
    # (
    #     "qwen3-4b-sft-128k-step44",
    #     "/data/checkpoints/rl_web_agent/web_agent_sft_128k_4b_hf/iter_0000044_hf/",
    # ),
    # (
    #     "qwen3-4b-sft-128k-step66",
    #     "/data/checkpoints/rl_web_agent/web_agent_sft_128k_4b_hf/iter_0000066_hf/",
    # ),
]

# ---------------------------------------------------------------------------
# sglang server defaults
# ---------------------------------------------------------------------------
SGLANG_DEFAULT_CONTEXT_LENGTH = 262144
SGLANG_DEFAULT_REASONING_PARSER = "deepseek-r1"
SGLANG_DEFAULT_TOOL_CALL_PARSER = "qwen"


# ---------------------------------------------------------------------------
# Per-model family overrides for sglang server args.
# Keys are substrings matched against the lowercase model name.
# ---------------------------------------------------------------------------
_MODEL_SGLANG_OVERRIDES: dict[str, dict] = {
    "llama": {
        "context_length": 131072,  # llama-3.1 max_position_embeddings
        "tool_call_parser": "llama3",  # native llama-3.1 tool-call format: {"name":..., "parameters":...}
    },
    "qwen": {
        "context_length": 262144,
        "tool_call_parser": "qwen",
    },
}


def _get_sglang_args(model_name: str) -> dict:
    """Return sglang server args for a model, applying per-family overrides."""
    name_lower = model_name.lower()
    overrides = {}
    for key, vals in _MODEL_SGLANG_OVERRIDES.items():
        if key in name_lower:
            overrides = vals
            break
    return {
        "context_length": overrides.get("context_length", SGLANG_DEFAULT_CONTEXT_LENGTH),
        "reasoning_parser": overrides.get("reasoning_parser", SGLANG_DEFAULT_REASONING_PARSER),
        "tool_call_parser": overrides.get("tool_call_parser", SGLANG_DEFAULT_TOOL_CALL_PARSER),
    }


def _detect_tp_size(model_name: str) -> int:
    """Auto-detect tp_size (and num_gpus) from model name."""
    name_lower = model_name.lower()
    if "llama" in name_lower:
        # llama-3.1-8B dense: use full TP to maximize KV budget for 128k context
        return 8
    if "30b" in name_lower:
        # qwen3-30b-a3b MoE: TP across all 8 GPUs
        return 8
    if "4b" in name_lower:
        # qwen3-4B dense: 2-way TP is plenty
        return 2
    return 8


# ---------------------------------------------------------------------------
# Evaluation defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "results/webarena_lite"
DEFAULT_TASKS_DIR = "dataset/test_webarena_lite"
DEFAULT_SITES = "shopping,shopping_admin,gitlab"
DEFAULT_MAX_CONCURRENT = 200
DEFAULT_MAX_TOKENS = 8192
DEFAULT_AGENT_TYPE = "tool"

# Health check settings
HEALTH_CHECK_INTERVAL = 10  # seconds between retries
HEALTH_CHECK_TIMEOUT = 1200  # total seconds to wait


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_random_free_port() -> int:
    """Get a random free port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_current_node_ip() -> str:
    """Get the IP address of the current Ray node."""
    address = ray._private.services.get_node_ip_address()
    return address.strip("[]")


def _wait_server_healthy(base_url: str, timeout: int, is_process_alive) -> bool:
    """Poll sglang health endpoint until it responds 200 or timeout is reached."""
    start = time.time()
    with requests.Session() as session:
        while time.time() - start < timeout:
            try:
                r = session.get(f"{base_url}/health_generate", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass

            if not is_process_alive():
                raise RuntimeError("SGLang server process terminated unexpectedly.")

            time.sleep(HEALTH_CHECK_INTERVAL)
    return False


def _launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    """Start SGLang server in a spawned process."""
    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")

    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()
    return p


# ---------------------------------------------------------------------------
# Ray Actor: runs sglang + evaluation for a single checkpoint
# ---------------------------------------------------------------------------
# The actor itself owns the GPUs for its sglang server. batch_agent is spawned
# as a subprocess inside the same actor, so the browser-env workers and the
# sglang server are guaranteed to live on the same host (subprocess inherits
# the actor's node placement). Ray resource requirements (num_gpus, num_cpus)
# are set per-model via `.options(...)` at submission time in main(), since
# GPU count depends on the model size.
@ray.remote
class EvalCheckpointActor:
    """Ray actor that runs sglang and evaluates a single checkpoint on one host."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        eval_args: dict,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.eval_args = eval_args
        self.proc: multiprocessing.Process | None = None
        self.server_url: str | None = None

    async def run(self) -> dict:
        """Run full evaluation: launch sglang in-process, run eval subprocess, shutdown."""
        tp_size = _detect_tp_size(self.model_name)
        try:
            server_url = self._start_sglang(tp_size)
            print(f"[{self.model_name}] sglang ready at {server_url}")
            return self._run_eval(server_url)

        except Exception as e:
            print(f"[{self.model_name}] FAILED: {e}")
            return {
                "model_name": self.model_name,
                "success": False,
                "error": str(e),
            }

        finally:
            self._shutdown_sglang()

    def _start_sglang(self, tp_size: int) -> str:
        """Launch sglang server process on this actor's host and wait for health."""
        model_args = _get_sglang_args(self.model_name)
        sglang_cli_args = [
            "--model-path",
            self.model_path,
            "--context-length",
            str(model_args["context_length"]),
            "--reasoning-parser",
            model_args["reasoning_parser"],
            "--tool-call-parser",
            model_args["tool_call_parser"],
            "--tp-size",
            str(tp_size),
        ]

        sglang_parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(sglang_parser)
        sglang_ns = sglang_parser.parse_args(sglang_cli_args)
        server_args = ServerArgs.from_cli_args(sglang_ns)

        node_ip = get_current_node_ip()
        port = get_random_free_port()

        server_args.host = "0.0.0.0"
        server_args.port = port

        print(f"[{self.model_name}] Launching sglang on {node_ip}:{port} (tp_size={tp_size}, ctx={model_args['context_length']}, tool_parser={model_args['tool_call_parser']})")
        self.proc = _launch_server_process(server_args)
        self.server_url = f"http://{node_ip}:{port}"

        healthy = _wait_server_healthy(
            base_url=self.server_url,
            timeout=HEALTH_CHECK_TIMEOUT,
            is_process_alive=lambda: self.proc.is_alive(),
        )
        if not healthy:
            raise RuntimeError(f"SGLang server did not become healthy within {HEALTH_CHECK_TIMEOUT}s")

        print(f"[{self.model_name}] sglang healthy at {self.server_url}")
        return self.server_url

    def _shutdown_sglang(self) -> None:
        """Terminate the sglang server process if it is still alive."""
        if self.proc is None:
            return
        if self.proc.is_alive():
            print(f"[{self.model_name}] Shutting down sglang...")
            self.proc.terminate()
            self.proc.join(timeout=30)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=10)
            print(f"[{self.model_name}] sglang stopped.")
        self.proc = None

    def _run_eval(self, server_url: str) -> dict:
        """Build and run the batch_agent command as a subprocess on this host."""
        args = argparse.Namespace(**self.eval_args)
        output_dir = f"{args.output_dir}/{self.model_name}_default"

        cmd = [
            sys.executable,
            "-m",
            "rl_web_agent.entrypoints.batch_agent",
            "--max_concurrent",
            str(args.max_concurrent),
            "--max_concurrent_launch",
            "5",
            "--output_dir",
            output_dir,
            "--tasks_dir",
            args.tasks_dir,
            "--agent_type",
            args.agent_type,
        ]

        if args.only_failed:
            cmd.append("--only-failed")

        cmd.extend(
            [
                "llm.provider=openai",
                "llm.generation.temperature=null",
                "llm.generation.top_p=null",
                "llm.max_concurrent=100",
                f"llm.generation.max_tokens={args.max_tokens}",
                f"llm.openai.base_url={server_url}/v1",
                "llm.openai.api_key=none",
                "llm.openai.timeout=360",
                f"llm.openai.model={args.openai_model_name}",
            ]
        )

        if args.task_ids:
            cmd.extend(["--task_ids", args.task_ids])
        else:
            cmd.extend(["--sites", args.sites])

        print(f"[{self.model_name}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            return {
                "model_name": self.model_name,
                "success": False,
                "error": f"batch_agent exit code {result.returncode}",
                "output_dir": output_dir,
            }

        return {
            "model_name": self.model_name,
            "success": True,
            "output_dir": output_dir,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run sglang-served model evaluations via Ray.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models (defined in script):
{chr(10).join(f"  {name}: {path}" for name, path in MODELS)}

Examples:
  # Dry run
  python run_sglang_eval.py --dry-run

  # Run specific task IDs
  python run_sglang_eval.py --task-ids 104,117,118

  # Override output directory
  python run_sglang_eval.py --output-dir results/custom_run
""",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print configuration without executing")
    parser.add_argument("--sites", type=str, default=DEFAULT_SITES, help=f"Sites to test (default: {DEFAULT_SITES})")
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated task IDs (overrides --sites)")
    parser.add_argument("--tasks-dir", type=str, default=DEFAULT_TASKS_DIR, help="Tasks directory")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT, help="Max concurrent tasks")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for generation")
    parser.add_argument("--agent-type", type=str, default=DEFAULT_AGENT_TYPE, choices=["regular", "tool"], help="Agent type")
    parser.add_argument("--only-failed", action="store_true", help="Only re-run failed tasks from previous results")
    parser.add_argument(
        "--openai-model-name",
        type=str,
        default="default",
        help="Model name to pass as llm.openai.model (default: 'default')",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("sglang Ray Evaluation Runner")
    print("=" * 60)
    print(f"Models: {len(MODELS)}")
    for name, path in MODELS:
        ma = _get_sglang_args(name)
        print(f"  - {name}: {path} (tp_size={_detect_tp_size(name)}, ctx={ma['context_length']}, tool_parser={ma['tool_call_parser']})")
    if args.task_ids:
        print(f"Task IDs: {args.task_ids}")
    else:
        print(f"Sites: {args.sites}")
    print(f"Agent type: {args.agent_type}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    print()

    if args.dry_run:
        for model_name, model_path in MODELS:
            tp_size = _detect_tp_size(model_name)
            ma = _get_sglang_args(model_name)
            print(f"[{model_name}]")
            print(f"  sglang: --model-path {model_path} --context-length {ma['context_length']} --reasoning-parser {ma['reasoning_parser']} --tool-call-parser {ma['tool_call_parser']} --tp-size {tp_size}")
            print(f"  output: {args.output_dir}/{model_name}_default")
            print()
        sys.exit(0)

    # Initialize Ray
    ray.init(address="auto", ignore_reinit_error=True)

    # Serialize args for actors
    eval_args = vars(args)

    # Launch ALL actors in parallel — Ray schedules them based on GPU availability.
    # Each EvalCheckpointActor owns its sglang GPUs *and* the batch_agent
    # subprocess, so the browser env workers and the sglang server are
    # guaranteed to run on the same host. If there aren't enough GPUs for all
    # models at once, Ray queues the remaining actors.
    actors = []
    refs = []
    for model_name, model_path in MODELS:
        tp_size = _detect_tp_size(model_name)
        print(f"[{model_name}] Submitting ({model_path}, num_gpus={tp_size})")
        actor = EvalCheckpointActor.options(
            num_gpus=tp_size,
            num_cpus=1,
        ).remote(
            model_name=model_name,
            model_path=model_path,
            eval_args=eval_args,
        )
        actors.append((model_name, actor))
        refs.append(actor.run.remote())

    # Collect results as they complete
    all_results = []
    for i, (model_name, actor) in enumerate(actors):
        try:
            result = ray.get(refs[i])
            all_results.append(result)
            print(f"[{model_name}] Completed: {'SUCCESS' if result['success'] else 'FAILED'}")
        except Exception as e:
            print(f"[{model_name}] FAILED with exception: {e}")
            all_results.append(
                {
                    "model_name": model_name,
                    "success": False,
                    "error": str(e),
                }
            )
        finally:
            ray.kill(actor)

    # Summary
    failed_runs = [r for r in all_results if not r["success"]]
    successful_runs = [r for r in all_results if r["success"]]

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if successful_runs:
        print(f"Successful ({len(successful_runs)}):")
        for r in successful_runs:
            print(f"  - {r['model_name']} -> {r.get('output_dir', '?')}")

    if failed_runs:
        print(f"Failed ({len(failed_runs)}):")
        for r in failed_runs:
            print(f"  - {r['model_name']}: {r.get('error', 'unknown')}")

    print("=" * 60)

    sys.exit(1 if failed_runs else 0)


if __name__ == "__main__":
    main()
