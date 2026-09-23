"""
ray job submit --address=auto \
  --working-dir . \
  -- python run_on_each_node.py \
     bash ./download_convert_model.sh \
     --model Qwen/Qwen3-4B \
     --config qwen3-4B


ray job submit --address=auto \
  --working-dir . \
  -- python run_on_each_node.py \
     bash ./download_convert_model.sh \
     --model Qwen/Qwen3-30B-A3B \
     --config qwen3-30B-A3B



ray job submit --address=auto \
  --working-dir . \
  -- python run_on_each_node.py \
     bash ./download_convert_model.sh \
     --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
     --config qwen3-30B-A3B
"""

import argparse
import os
import subprocess
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote
class PerNodeCommandRunner:
    def run(self, cmd, extra_env):
        env = os.environ.copy()
        env.update(extra_env)
        print(f"[PerNode] Running: {' '.join(cmd)}", flush=True)
        if extra_env:
            print(f"[PerNode] Extra env: {extra_env}", flush=True)
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
        ret = proc.wait()
        if ret != 0:
            print(f"Command failed with exit code {ret}: {cmd}")
            return f"Command failed with exit code {ret}: {cmd}"
        return "ok"


def main():
    parser = argparse.ArgumentParser(description="Run a command once on each Ray node and wait for completion.")
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="If set, only request 1 CPU. Otherwise request 8 GPUs (default).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to pass to the command (can be repeated, e.g. --env FOO=bar --env BAZ=qux).",
    )
    args, cmd = parser.parse_known_args()

    # Parse --env KEY=VALUE pairs into a dict
    extra_env = {}
    for item in args.env:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"Invalid --env format: {item!r}. Expected KEY=VALUE.")
        extra_env[key] = value

    if not cmd:
        raise SystemExit(
            "No command provided.\nUsage: python run_on_each_node.py [--no-gpu] [--env KEY=VALUE ...] <command...>"
        )

    ray.init(address="auto")

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise SystemExit("No alive Ray nodes found.")

    print(f"[PerNode] Found {len(nodes)} nodes", flush=True)

    # Set resource requirements based on --no-gpu flag
    if args.no_gpu:
        num_cpus = 1
        num_gpus = 0
        print("[PerNode] Running with CPU only (no GPU)", flush=True)
    else:
        num_cpus = 1
        num_gpus = 8
        print("[PerNode] Running with 8 GPUs per node", flush=True)

    actors = []
    for n in nodes:
        actors.append(
            PerNodeCommandRunner.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=n["NodeID"],
                    soft=False,
                ),
            ).remote()
        )

    ray.get([a.run.remote(cmd, extra_env) for a in actors])
    print("[PerNode] All nodes finished successfully.", flush=True)


if __name__ == "__main__":
    main()
