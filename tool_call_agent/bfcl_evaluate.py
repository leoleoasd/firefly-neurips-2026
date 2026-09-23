#!/usr/bin/env python3
"""
BFCL Evaluator - Run Berkeley Function Calling Leaderboard on model checkpoints.

Launches a sglang server for each checkpoint, then runs BFCL generate + evaluate
against it. Each checkpoint gets its own isolated result/score directory.
Multiple checkpoints are evaluated in parallel via Ray actors (each queues for
GPUs independently, same pattern as tool_call_agent/evaluate.py).

Usage Examples:
    # Evaluate a single checkpoint on all categories
    python tool_call_agent/bfcl_evaluate.py \\
        --checkpoint /tmp/instance_storage/my_model_hf/iter_0000059/ \\
        --num-gpus 1

    # Evaluate multiple checkpoints in parallel (auto-discovered from iter_* dirs)
    python tool_call_agent/bfcl_evaluate.py \\
        --checkpoint /tmp/instance_storage/my_model_hf/ \\
        --num-gpus 1

    # Specify test categories
    python tool_call_agent/bfcl_evaluate.py \\
        --checkpoint /tmp/instance_storage/my_model_hf/iter_0000059/ \\
        --num-gpus 1 \\
        --test-category simple,multiple,parallel

    # Use a specific BFCL model name (must exist in MODEL_CONFIG_MAPPING)
    python tool_call_agent/bfcl_evaluate.py \\
        --checkpoint /tmp/instance_storage/my_model_hf/iter_0000059/ \\
        --bfcl-model "Qwen/Qwen3-30B-A3B-Instruct-2507-FC" \\
        --num-gpus 1
"""

import argparse
import contextlib
import csv
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

import ray
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint resolution (same pattern as tool_call_agent/evaluate.py)
# ---------------------------------------------------------------------------


def resolve_checkpoints(checkpoint_path: str) -> list[str]:
    """Resolve checkpoint path to a list of HF checkpoint directories.

    - If path contains config.json -> single HF checkpoint
    - If path contains iter_* subdirs with config.json -> multiple checkpoints
    - Otherwise -> error
    """
    p = Path(checkpoint_path)
    if (p / "config.json").exists():
        return [str(p)]

    iter_dirs = sorted(p.glob("iter_*"))
    iter_dirs = [d for d in iter_dirs if d.is_dir() and (d / "config.json").exists()]
    if iter_dirs:
        return [str(d) for d in iter_dirs]

    raise FileNotFoundError(
        f"No config.json found in {checkpoint_path} and no iter_*/config.json subdirs found. "
        f"Expected an HF checkpoint or a directory containing iter_* checkpoints."
    )


def _make_checkpoint_name(checkpoint: str) -> str:
    """Create a short name from a checkpoint path.

    e.g. /tmp/.../batch_hf/iter_0000059/ -> 'batch_hf__iter_0000059'
    """
    parts = Path(checkpoint).resolve().parts
    return "__".join(parts[-2:]).rstrip("_")


# ---------------------------------------------------------------------------
# BFCL subprocess helpers (used inside the Ray actor)
# ---------------------------------------------------------------------------


def _make_bfcl_env(
    checkpoint_output_dir: str,
    server_url: str | None = None,
) -> dict[str, str]:
    """Build environment dict for BFCL subprocesses.

    Sets BFCL_PROJECT_ROOT so that result/ and score/ land under
    checkpoint_output_dir.  If server_url is provided, parses it and sets
    LOCAL_SERVER_ENDPOINT / LOCAL_SERVER_PORT so BFCL talks to our sglang
    server via the local endpoint path (avoids the REMOTE_OPENAI_* code
    path which has different tokenizer-loading behaviour).
    """
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(checkpoint_output_dir)
    # Make sure REMOTE_OPENAI_* are NOT set so BFCL uses the local path
    env.pop("REMOTE_OPENAI_BASE_URL", None)
    env.pop("REMOTE_OPENAI_API_KEY", None)
    env.pop("REMOTE_OPENAI_TOKENIZER_PATH", None)
    if server_url:
        from urllib.parse import urlparse

        parsed = urlparse(server_url)
        env["LOCAL_SERVER_ENDPOINT"] = parsed.hostname or "localhost"
        env["LOCAL_SERVER_PORT"] = str(parsed.port or 1053)
    return env


def _run_bfcl_generate(
    bfcl_model: str,
    checkpoint: str,
    server_url: str,
    test_categories: list[str],
    checkpoint_output_dir: str,
    temperature: float = 0.001,
    num_threads: int = 100,
    allow_overwrite: bool = True,
) -> None:
    """Run ``bfcl generate`` against an already-running sglang endpoint.

    Sets BFCL_PROJECT_ROOT to *checkpoint_output_dir* so BFCL writes to
    ``<checkpoint_output_dir>/result/``.  Uses ``LOCAL_SERVER_ENDPOINT``
    and ``LOCAL_SERVER_PORT`` to point to our sglang server and
    ``--skip-server-setup`` so BFCL doesn't try to launch its own server.
    ``--local-model-path`` is set to the checkpoint so the tokenizer is
    loaded from there and model= in the completions API matches the served
    model.
    """
    env = _make_bfcl_env(checkpoint_output_dir, server_url)

    cmd = [
        "bfcl",
        "generate",
        "--model",
        bfcl_model,
        "--test-category",
        ",".join(test_categories),
        "--skip-server-setup",
        "--local-model-path",
        checkpoint,
        "--temperature",
        str(temperature),
        "--num-threads",
        str(num_threads),
        "--backend",
        "sglang",
    ]
    if allow_overwrite:
        cmd.append("--allow-overwrite")

    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"bfcl generate failed with return code {proc.returncode}")


def _run_bfcl_evaluate(
    bfcl_model: str,
    test_categories: list[str],
    checkpoint_output_dir: str,
) -> None:
    """Run ``bfcl evaluate`` to score the generated results."""
    env = _make_bfcl_env(checkpoint_output_dir)

    cmd = [
        "bfcl",
        "evaluate",
        "--model",
        bfcl_model,
        "--test-category",
        ",".join(test_categories),
    ]

    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"bfcl evaluate failed with return code {proc.returncode}")


def _read_overall_scores(score_dir: Path) -> dict[str, str] | None:
    """Read the first data row from ``data_overall.csv`` and return as dict."""
    overall_csv = score_dir / "data_overall.csv"
    if not overall_csv.exists():
        return None
    with open(overall_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return None


# ---------------------------------------------------------------------------
# Ray actor: one per checkpoint, runs the full pipeline
# ---------------------------------------------------------------------------


@ray.remote
class BFCLCheckpointActor:
    """Ray actor that evaluates a single checkpoint end-to-end.

    Each actor runs in its own process.  It requests ``num_gpus`` GPUs so
    Ray will queue actors if the cluster is fully occupied — exactly the
    same pattern as ``EvalCheckpointActor`` in ``tool_call_agent/evaluate.py``.
    """

    def __init__(
        self,
        checkpoint: str,
        args_dict: dict,
    ):
        self.checkpoint = checkpoint
        self.args_dict = args_dict
        self.checkpoint_name = _make_checkpoint_name(checkpoint)

    def run(self) -> dict:
        """Launch sglang, run BFCL generate + evaluate, shutdown, return summary."""
        # Re-import heavy deps inside the worker process
        from slime.utils.misc import get_current_node_ip

        from scripts.sglang_job import RewardSGLangActor

        args = argparse.Namespace(**self.args_dict)
        ckpt_name = self.checkpoint_name
        checkpoint = self.checkpoint

        # Per-checkpoint output directory
        checkpoint_output_dir = Path(args.output) / ckpt_name
        result_dir = checkpoint_output_dir / "result"
        score_dir = checkpoint_output_dir / "score"
        result_dir.mkdir(parents=True, exist_ok=True)
        score_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"[{ckpt_name}] Starting evaluation")
        print(f"[{ckpt_name}] Checkpoint: {checkpoint}")
        print(f"{'=' * 70}")

        # ---- 1. Launch sglang server ----
        print(f"[{ckpt_name}] Launching sglang server...")
        node_ip = get_current_node_ip()

        sglang_cli_args = [
            "--model-path",
            checkpoint,
            "--tool-call-parser",
            args.tool_call_parser,
            "--tp",
            str(int(args.num_gpus)),
            "--log-level",
            "warning",
            "--reasoning-parser",
            "deepseek-r1",
        ]
        sglang_parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(sglang_parser)
        sglang_ns = sglang_parser.parse_args(sglang_cli_args)

        actor_options: dict = dict(num_gpus=args.num_gpus, num_cpus=1)
        if not args.no_pin_node:
            actor_options["resources"] = {f"node:{node_ip}": 0.001}

        sglang_actor = RewardSGLangActor.options(**actor_options).remote(
            args=sglang_ns,
            registry_name="sglang_registry",
        )

        try:
            server_url: str = ray.get(sglang_actor.start.remote())
            print(f"[{ckpt_name}] Server ready: {server_url}")

            # ---- 2. BFCL generate ----
            print(f"[{ckpt_name}] Running BFCL generate...")
            _run_bfcl_generate(
                bfcl_model=args.bfcl_model,
                checkpoint=checkpoint,
                server_url=server_url,
                test_categories=args.test_categories,
                checkpoint_output_dir=str(checkpoint_output_dir),
                temperature=args.temperature,
                num_threads=args.num_threads,
                allow_overwrite=args.allow_overwrite,
            )
            print(f"[{ckpt_name}] Generate complete.")

            # ---- 3. BFCL evaluate ----
            print(f"[{ckpt_name}] Running BFCL evaluate...")
            _run_bfcl_evaluate(
                bfcl_model=args.bfcl_model,
                test_categories=args.test_categories,
                checkpoint_output_dir=str(checkpoint_output_dir),
            )
            print(f"[{ckpt_name}] Evaluate complete.")

        finally:
            # ---- 4. Shutdown sglang ----
            print(f"[{ckpt_name}] Shutting down sglang server...")
            with contextlib.suppress(Exception):
                ray.kill(sglang_actor)
            print(f"[{ckpt_name}] Server stopped.")

        # ---- 5. Save metadata ----
        metadata = {
            "checkpoint": checkpoint,
            "checkpoint_name": ckpt_name,
            "bfcl_model": args.bfcl_model,
            "test_categories": args.test_categories,
            "num_gpus": args.num_gpus,
            "tool_call_parser": args.tool_call_parser,
            "temperature": args.temperature,
            "num_threads": args.num_threads,
            "timestamp": datetime.now().isoformat(),
            "result_dir": str(result_dir),
            "score_dir": str(score_dir),
        }
        metadata_file = checkpoint_output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[{ckpt_name}] Metadata saved to {metadata_file}")

        # ---- 6. Collect scores ----
        scores = _read_overall_scores(score_dir)
        if scores is None:
            print(f"[{ckpt_name}] Warning: data_overall.csv not found after evaluation.")

        return {
            "checkpoint": checkpoint,
            "checkpoint_name": ckpt_name,
            "scores": scores,
        }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

SUMMARY_KEYS = [
    "Overall Acc",
    "Non-Live AST Acc",
    "Live Acc",
    "Multi Turn Acc",
    "Memory Acc",
    "Web Search Acc",
]


def _print_checkpoint_summary(result: dict) -> None:
    name = result["checkpoint_name"]
    if "error" in result:
        print(f"\n  {name}: ERROR - {result['error']}")
        return
    scores = result.get("scores")
    if not scores:
        print(f"\n  {name}: No scores available")
        return
    print(f"\n  {name}:")
    for key in SUMMARY_KEYS:
        val = scores.get(key, "N/A")
        if val != "N/A":
            print(f"    {key}: {val}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run BFCL evaluation on one or more model checkpoints (parallel)")

    # Checkpoint (the model(s) to evaluate)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to HF checkpoint directory, or a directory containing iter_* subdirs. "
            "e.g. /tmp/instance_storage/my_model_hf/iter_0000059/ for a single checkpoint, "
            "or /tmp/instance_storage/my_model_hf/ for all iter_* checkpoints."
        ),
    )
    parser.add_argument(
        "--bfcl-model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FC",
        help=(
            "BFCL model registry name (must exist in MODEL_CONFIG_MAPPING). "
            "This determines the handler, template, and tokenizer behavior. "
            "Default: Qwen/Qwen3-30B-A3B-Instruct-2507-FC"
        ),
    )

    # SGLang server config
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=1,
        help="Number of GPUs for the sglang server (tensor parallel degree)",
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=os.environ.get("TOOL_CALL_PARSER", "qwen"),
        help="Tool call parser for sglang (default: $TOOL_CALL_PARSER or 'qwen')",
    )
    parser.add_argument(
        "--no-pin-node",
        action="store_true",
        help="Don't pin sglang server to the current node",
    )

    # BFCL config
    parser.add_argument(
        "--test-category",
        type=str,
        default="all",
        help=(
            "Comma-separated BFCL test categories to run. "
            "e.g. 'simple,multiple,parallel,live_simple,multi_turn' or 'all'. "
            "See BFCL TEST_CATEGORIES.md for full list."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Temperature for generation (default: 0.001)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=800,
        help="Number of concurrent inference threads for BFCL generate (default: 100)",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        default=True,
        help="Allow overwriting existing BFCL result files (default: True)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/instance_storage/bfcl_eval_results",
        help=("Root output directory. Each checkpoint gets a subdirectory with result/, score/, and metadata.json."),
    )

    args = parser.parse_args()

    # Parse comma-separated test categories
    args.test_categories = [c.strip() for c in args.test_category.split(",") if c.strip()]

    # Suppress noisy logs
    logging.getLogger("sglang").setLevel(logging.WARNING)

    print("=" * 70)
    print("BFCL Evaluator (parallel)")
    print("=" * 70)

    # Initialize Ray
    ray.init(address="auto", namespace="sglang", ignore_reinit_error=True)

    # Resolve checkpoints
    checkpoints = resolve_checkpoints(args.checkpoint)
    print(f"BFCL model:          {args.bfcl_model}")
    print(f"Test categories:     {args.test_categories}")
    print(f"Checkpoints:         {len(checkpoints)}")
    for ckpt in checkpoints:
        print(f"  {ckpt}")
    print(f"Output directory:    {args.output}")
    print()

    # Serialize args for Ray (Namespace isn't picklable by default)
    args_dict = vars(args)

    # Launch one BFCLCheckpointActor per checkpoint.
    # Each actor requests num_gpus GPUs, so Ray will queue actors if the
    # cluster doesn't have enough GPUs for all of them simultaneously.
    actors = []
    for ckpt in checkpoints:
        actor = BFCLCheckpointActor.options(num_cpus=1).remote(
            checkpoint=ckpt,
            args_dict=args_dict,
        )
        actors.append(actor)

    # Kick off all actors in parallel and collect results
    result_refs = [actor.run.remote() for actor in actors]
    all_results = ray.get(result_refs)

    # Print summary and save combined results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for result in all_results:
        _print_checkpoint_summary(result)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "bfcl_model": args.bfcl_model,
                "test_categories": args.test_categories,
                "checkpoints": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nCombined summary saved to {summary_file}")


if __name__ == "__main__":
    main()
