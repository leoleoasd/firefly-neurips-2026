#!/usr/bin/env python3
"""
All-gather files across all GPU nodes via NCCL.

Each node contributes files from its local src_dir. At the end, every node
has all files in dst_dir.

Usage:
    # All-gather in-place (src_dir == dst_dir)
    ray job submit --address=auto --working-dir . -- \
        python -m scripts.all_gather_files /tmp/instance_storage/merged_output/

    # All-gather to a different destination
    ray job submit --address=auto --working-dir . -- \
        python -m scripts.all_gather_files /tmp/instance_storage/per_node_output/ \
            --dst /tmp/instance_storage/all_output/

    # Tune chunk size (default 256 MB)
    ray job submit --address=auto --working-dir . -- \
        python -m scripts.all_gather_files /tmp/instance_storage/shards/ \
            --chunk-size 1024
"""

import argparse
import logging
import time

import ray

from shared.fileops import all_gather_files

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="All-gather files across all GPU nodes via NCCL")
    parser.add_argument(
        "src_dir",
        type=str,
        help="Source directory containing this node's files",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Destination directory on all nodes (defaults to same as src_dir)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size in MB for NCCL transfers (default: 256)",
    )
    parser.add_argument(
        "--num-buffers",
        type=int,
        default=10,
        help="Buffer pool size -- controls overlap depth (default: 10)",
    )
    parser.add_argument(
        "--bench-mode",
        type=str,
        default=None,
        choices=["sender", "receiver"],
        help="Benchmark mode: 'sender' skips disk reads, 'receiver' skips D2H+writes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ray.init(address="auto", ignore_reinit_error=True)

    num_nodes = len([n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0])
    chunk_bytes = args.chunk_size * 1024 * 1024

    print(f"All-gather:   {args.src_dir}")
    print(f"Destination:  {args.dst or args.src_dir}")
    print(f"GPU nodes:    {num_nodes}")
    print(f"Chunk size:   {args.chunk_size} MB")
    print(f"Buffers:      {args.num_buffers}")
    print(f"Bench mode:   {args.bench_mode or 'off'}")

    t0 = time.time()
    group = all_gather_files(
        src_dir=args.src_dir,
        dst_dir=args.dst,
        chunk_size=chunk_bytes,
        num_buffers=args.num_buffers,
        bench_mode=args.bench_mode,
    )
    elapsed = time.time() - t0
    print(f"All-gather complete in {elapsed:.1f}s")

    ray.get(group.teardown.remote())
    print("Worker group torn down.")


if __name__ == "__main__":
    main()
