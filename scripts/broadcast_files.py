#!/usr/bin/env python3
"""
Broadcast files/directories from the driver node to all GPU nodes via NCCL.

Uses shared.fileops.FileTransferGroup which does chunked, double-buffered
GPU-to-GPU transfers over torch.distributed.

Usage:
    # Broadcast a directory
    ray job submit --address=auto --working-dir . -- \
        python -m scripts.broadcast_files /tmp/instance_storage/model_weights/

    # Broadcast to a different destination path
    ray job submit --address=auto --working-dir . -- \
        python -m scripts.broadcast_files /tmp/instance_storage/model_weights/ \
            --dst /data/weights/

    # Tune chunk size (default 256 MB)
    ray job submit --address=auto --working-dir . -- \
        python -m scripts.broadcast_files /tmp/instance_storage/big_dataset/ \
            --chunk-size 512
"""

import argparse
import logging
import time

import ray

from shared.fileops import broadcast_files

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Broadcast files from driver node to all GPU nodes via NCCL")
    parser.add_argument(
        "src",
        type=str,
        help="Source file or directory to broadcast (on the driver node)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Destination directory on all nodes (defaults to same path as src)",
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
        help="Buffer pool size — controls overlap depth (default: 10)",
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

    print(f"Broadcasting: {args.src}")
    print(f"Destination:  {args.dst or args.src}")
    print(f"GPU nodes:    {num_nodes}")
    print(f"Chunk size:   {args.chunk_size} MB")
    print(f"Buffers:      {args.num_buffers}")
    print(f"Bench mode:   {args.bench_mode or 'off'}")

    t0 = time.time()
    group = broadcast_files(
        src_path=args.src,
        dst_dir=args.dst,
        chunk_size=chunk_bytes,
        num_buffers=args.num_buffers,
        bench_mode=args.bench_mode,
    )
    elapsed = time.time() - t0
    print(f"Broadcast complete in {elapsed:.1f}s")

    ray.get(group.teardown.remote())
    print("Worker group torn down.")


if __name__ == "__main__":
    main()
