"""
nccl_file_ops.py

Broadcast files/directories across Ray actors using torch.distributed + NCCL.
Supports async pipelined chunked transfers to overlap disk I/O with network.

Usage:
    import ray
    from nccl_file_ops import broadcast_files

    ray.init()

    # Broadcast a directory from this node to all others
    group = broadcast_files("/data/model_weights/")

    # Reuse for another broadcast
    ray.get(group.broadcast.remote("/data/dataset/"))

    # Cleanup
    ray.get(group.teardown.remote())
"""

import ctypes
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from ray.experimental.tqdm_ray import tqdm as ray_tqdm

from shared.utils import get_random_free_port

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# libc pread / pwrite — bypass Python buffer allocation
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL("libc.so.6", use_errno=True)

_libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_libc.pread.restype = ctypes.c_ssize_t

_libc.pwrite.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_libc.pwrite.restype = ctypes.c_ssize_t


def _libc_pread_all(fd: int, ptr: int, length: int, offset: int) -> int:
    """pread() loop until all bytes read. ptr is a raw C pointer (int)."""
    total = 0
    while total < length:
        n = _libc.pread(fd, ptr + total, length - total, offset + total)
        if n <= 0:
            if n < 0:
                errno = ctypes.get_errno()
                raise OSError(errno, f"pread failed: {os.strerror(errno)}")
            break  # EOF
        total += n
    return total


def _libc_pwrite_all(fd: int, ptr: int, length: int, offset: int) -> int:
    """pwrite() loop until all bytes written. ptr is a raw C pointer (int)."""
    total = 0
    while total < length:
        n = _libc.pwrite(fd, ptr + total, length - total, offset + total)
        if n <= 0:
            if n < 0:
                errno = ctypes.get_errno()
                raise OSError(errno, f"pwrite failed: {os.strerror(errno)}")
            break
        total += n
    return total


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BroadcastConfig:
    """Configuration for file broadcast."""

    chunk_size: int = 1024 * 1024 * 1024  # 1 GB per chunk
    num_buffers: int = 10  # buffer pool size (controls overlap depth)
    device: torch.device | None = None  # defaults to cuda:current

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")


@dataclass
class FileEntry:
    """Metadata for a single file to broadcast."""

    rel_path: str
    size: int
    num_chunks: int

    def chunk_ranges(self, chunk_size: int) -> list[tuple[int, int]]:
        ranges = []
        for i in range(self.num_chunks):
            offset = i * chunk_size
            length = min(chunk_size, self.size - offset)
            ranges.append((offset, length))
        return ranges


@dataclass
class BroadcastManifest:
    """Manifest describing all files to broadcast."""

    files: list[FileEntry] = field(default_factory=list)
    chunk_size: int = 1024 * 1024 * 1024
    src_is_file: bool = False  # True when source was a single file (not a directory)

    def to_json(self) -> str:
        return json.dumps(
            {
                "chunk_size": self.chunk_size,
                "src_is_file": self.src_is_file,
                "files": [{"rel_path": f.rel_path, "size": f.size, "num_chunks": f.num_chunks} for f in self.files],
            }
        )

    @staticmethod
    def from_json(s: str) -> "BroadcastManifest":
        d = json.loads(s)
        return BroadcastManifest(
            chunk_size=d["chunk_size"],
            src_is_file=d.get("src_is_file", False),
            files=[
                FileEntry(
                    rel_path=f["rel_path"],
                    size=f["size"],
                    num_chunks=f["num_chunks"],
                )
                for f in d["files"]
            ],
        )


@dataclass
class ChunkDescriptor:
    """One chunk in the flattened broadcast pipeline."""

    file_idx: int  # index into manifest.files
    chunk_idx: int  # chunk index within the file
    num_chunks: int  # total chunks in this file
    rel_path: str  # for logging
    offset: int  # byte offset in the file
    length: int  # bytes in this chunk


@dataclass
class AllGatherTransfer:
    """One broadcast in the all_gather plan: a src_rank broadcasts files."""

    src_rank: int
    files: list[FileEntry] = field(default_factory=list)


@dataclass
class AllGatherPlan:
    """Plan for all_gather: sequence of broadcasts, each from a different rank."""

    transfers: list[AllGatherTransfer] = field(default_factory=list)
    chunk_size: int = 1024 * 1024 * 1024

    def to_json(self) -> str:
        return json.dumps(
            {
                "chunk_size": self.chunk_size,
                "transfers": [
                    {
                        "src_rank": t.src_rank,
                        "files": [
                            {
                                "rel_path": f.rel_path,
                                "size": f.size,
                                "num_chunks": f.num_chunks,
                            }
                            for f in t.files
                        ],
                    }
                    for t in self.transfers
                ],
            }
        )

    @staticmethod
    def from_json(s: str) -> "AllGatherPlan":
        d = json.loads(s)
        return AllGatherPlan(
            chunk_size=d["chunk_size"],
            transfers=[
                AllGatherTransfer(
                    src_rank=t["src_rank"],
                    files=[
                        FileEntry(
                            rel_path=f["rel_path"],
                            size=f["size"],
                            num_chunks=f["num_chunks"],
                        )
                        for f in t["files"]
                    ],
                )
                for t in d["transfers"]
            ],
        )


# ---------------------------------------------------------------------------
# Worker actor — one per node
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class FileTransferWorker:
    """
    A single worker in the file transfer group.
    Each worker corresponds to one rank in the torch.distributed group.
    """

    def __init__(self):
        self.rank: int | None = None
        self.world_size: int | None = None
        self._initialized = False
        # Configure logging in this actor process so logger.info() is visible
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            force=True,
        )

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        """Initialize torch.distributed for this worker."""
        self.rank = rank
        self.world_size = world_size

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        logger.info(f"[Rank {rank}] init_process_group: master={master_addr}:{master_port} world_size={world_size}")
        t0 = time.monotonic()
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._initialized = True
        logger.info(f"[Rank {rank}] init_process_group: done in {elapsed_ms:.0f}ms")

        # Warm up the PyTorch CUDA context now so it doesn't add latency
        # to the first broadcast.  init_process_group uses NCCL's own CUDA
        # context, but PyTorch's context is lazy-initialized on first use.
        t0 = time.monotonic()
        _dummy = torch.zeros(1, device="cuda")
        del _dummy
        cuda_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Rank {rank}] init_process_group: cuda warmup={cuda_ms:.0f}ms")

    def teardown(self):
        if self._initialized:
            logger.info(f"[Rank {self.rank}] teardown: destroying process group")
            t0 = time.monotonic()
            dist.destroy_process_group()
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._initialized = False
            logger.info(f"[Rank {self.rank}] teardown: done in {elapsed_ms:.0f}ms")
        else:
            logger.info(f"[Rank {self.rank}] teardown: not initialized, skipping")

    def get_node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_device(self, config: BroadcastConfig) -> torch.device:
        if config.device is not None:
            return config.device
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    def _build_manifest(self, src_path: str, chunk_size: int) -> BroadcastManifest:
        t0 = time.monotonic()
        src = Path(src_path)
        manifest = BroadcastManifest(chunk_size=chunk_size)

        if src.is_file():
            manifest.src_is_file = True
            size = src.stat().st_size
            manifest.files.append(
                FileEntry(
                    rel_path=src.name,
                    size=size,
                    num_chunks=max(1, math.ceil(size / chunk_size)),
                )
            )
        elif src.is_dir():
            for fp in sorted(src.rglob("*")):
                if fp.is_file():
                    size = fp.stat().st_size
                    manifest.files.append(
                        FileEntry(
                            rel_path=str(fp.relative_to(src)),
                            size=size,
                            num_chunks=max(1, math.ceil(size / chunk_size)),
                        )
                    )
        else:
            raise FileNotFoundError(f"Source path not found: {src_path}")

        total_bytes = sum(f.size for f in manifest.files)
        total_chunks = sum(f.num_chunks for f in manifest.files)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            f"[Rank {self.rank}] _build_manifest: {len(manifest.files)} files, "
            f"{total_bytes / 1e9:.2f} GB, {total_chunks} total chunks, "
            f"src_is_file={manifest.src_is_file}, took {elapsed_ms:.0f}ms"
        )
        return manifest

    def _broadcast_manifest(
        self,
        manifest: BroadcastManifest | None,
        src_rank: int,
        device: torch.device,
    ) -> BroadcastManifest:
        logger.info(f"[Rank {self.rank}] _broadcast_manifest: begin (is_src={self.rank == src_rank})")
        t_total = time.monotonic()

        if self.rank == src_rank:
            data = manifest.to_json().encode("utf-8")
            size_t = torch.tensor([len(data)], dtype=torch.long, device=device)
            logger.info(f"[Rank {self.rank}] _broadcast_manifest: manifest json={len(data)} bytes")
        else:
            size_t = torch.tensor([0], dtype=torch.long, device=device)
        logger.info(f"[Rank {self.rank}] _broadcast_manifest: size_t={size_t}; {time.monotonic() - t_total:.1f}s")
        dist.barrier()
        logger.info(f"[Rank {self.rank}] _broadcast_manifest: barrier done {time.monotonic() - t_total:.1f}s")
        t0 = time.monotonic()
        dist.broadcast(size_t, src=src_rank)
        ms = (time.monotonic() - t0) * 1000
        manifest_size = size_t.item()
        logger.info(
            f"[Rank {self.rank}] _broadcast_manifest: size_broadcast={ms:.1f}ms (manifest_size={manifest_size} bytes)"
        )

        if self.rank == src_rank:
            payload = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
        else:
            payload = torch.empty(size_t.item(), dtype=torch.uint8, device=device)

        t0 = time.monotonic()
        dist.broadcast(payload, src=src_rank)
        ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Rank {self.rank}] _broadcast_manifest: payload_broadcast={ms:.1f}ms")

        if self.rank != src_rank:
            json_str = payload.cpu().numpy().tobytes().decode("utf-8")
            manifest = BroadcastManifest.from_json(json_str)
            logger.info(
                f"[Rank {self.rank}] _broadcast_manifest: "
                f"received {len(manifest.files)} files, "
                f"src_is_file={manifest.src_is_file}"
            )

        total_ms = (time.monotonic() - t_total) * 1000
        logger.info(f"[Rank {self.rank}] _broadcast_manifest: done in {total_ms:.0f}ms")
        return manifest

    def _broadcast_file_chunked(
        self,
        entry: FileEntry,
        src_path: Path | None,
        dst_path: Path,
        src_rank: int,
        config: BroadcastConfig,
    ):
        """Legacy single-file wrapper. Use _broadcast_all_chunks for pipelined transfers."""
        raise NotImplementedError("Use _broadcast_all_chunks instead")

    # ------------------------------------------------------------------
    # Chunk-based pipeline: flattened across all files
    # ------------------------------------------------------------------

    def _flatten_chunks(self, manifest: BroadcastManifest) -> list[ChunkDescriptor]:
        """Flatten all (file, chunk) pairs into a single ordered list."""
        chunks: list[ChunkDescriptor] = []
        for file_idx, entry in enumerate(manifest.files):
            if entry.size == 0:
                continue  # empty files handled separately
            for chunk_idx, (offset, length) in enumerate(entry.chunk_ranges(manifest.chunk_size)):
                chunks.append(
                    ChunkDescriptor(
                        file_idx=file_idx,
                        chunk_idx=chunk_idx,
                        num_chunks=entry.num_chunks,
                        rel_path=entry.rel_path,
                        offset=offset,
                        length=length,
                    )
                )
        return chunks

    # Header: 8 bytes prepended to every broadcast payload
    #   [0:4] chunk_id  (int32 LE)
    #   [4:8] chunk_len (int32 LE)
    CHUNK_HEADER_SIZE = 8

    def _broadcast_all_chunks(
        self,
        manifest: BroadcastManifest,
        src_root: Path | None,
        dst_root: Path,
        src_rank: int,
        config: BroadcastConfig,
        pbar=None,
        bench_mode: str | None = None,
    ):
        """
        Broadcast all files using a pool of worker threads, each owning one
        CPU pinned buffer + one GPU buffer.  Chunks are sent in arbitrary
        order; a small header (chunk_id, chunk_len) is prepended so receivers
        can identify each chunk without prior knowledge of the send order.

        Sender:
            Workers: pick chunk from work queue, disk_read into cpu_buf
                     (after header), H2D copy whole buffer, signal ready.
            Main:    pop ready_q (unordered), NCCL broadcast, give slot back.

        Receiver:
            Main:    grab free slot, NCCL broadcast into gpu_buf, parse header
                     to learn chunk_id, hand (slot, chunk_id) to worker.
            Workers: D2H copy → pwrite to disk, return slot to free pool.

        All ranks broadcast the same fixed size (HEADER + max_chunk) so that
        NCCL stays in sync regardless of actual chunk lengths.
        """
        import queue
        import struct
        import threading

        HEADER = self.CHUNK_HEADER_SIZE

        is_src = self.rank == src_rank
        device = self._get_device(config)
        num_workers = config.num_buffers
        tag = "SND" if is_src else "RCV"

        flat_chunks = self._flatten_chunks(manifest)
        total_chunks = len(flat_chunks)
        total_bytes = sum(f.size for f in manifest.files)

        if total_chunks == 0:
            logger.info(f"[Rank {self.rank} {tag}] no chunks to transfer")
            if pbar is not None:
                pbar.update(0)
            return

        max_chunk = max(c.length for c in flat_chunks)
        bcast_size = HEADER + max_chunk  # fixed broadcast size for every call
        logger.info(
            f"[Rank {self.rank} {tag}] pipeline: {total_chunks} chunks across "
            f"{len(manifest.files)} files, {total_bytes / 1e9:.2f} GB, "
            f"max_chunk={max_chunk / 1e6:.0f} MB, bcast={bcast_size / 1e6:.0f} MB, "
            f"workers={num_workers}"
        )

        # ------------------------------------------------------------------
        # Allocate buffer pool: one (pinned CPU + GPU) pair per worker
        # Each buffer is HEADER + max_chunk bytes.
        # Each worker also gets its own CUDA stream for H2D / D2H copies,
        # with CUDA events to synchronize against the default stream's
        # NCCL broadcasts.
        # ------------------------------------------------------------------
        t0 = time.monotonic()
        cpu_bufs = [torch.empty(bcast_size, dtype=torch.uint8, pin_memory=True) for _ in range(num_workers)]
        gpu_bufs = [torch.empty(bcast_size, dtype=torch.uint8, device=device) for _ in range(num_workers)]
        copy_streams = [torch.cuda.Stream(device=device) for _ in range(num_workers)]

        alloc_ms = (time.monotonic() - t0) * 1000
        logger.info(
            f"[Rank {self.rank} {tag}] allocated {num_workers}x{bcast_size / 1e6:.0f} MB "
            f"pinned + GPU in {alloc_ms:.0f}ms"
        )

        # NCCL warmup
        t0 = time.monotonic()
        dummy = torch.zeros(1, dtype=torch.uint8, device=device)
        dist.broadcast(dummy, src=src_rank)
        warmup_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Rank {self.rank} {tag}] nccl warmup={warmup_ms:.1f}ms")

        pipeline_t0 = time.monotonic()

        # ==================================================================
        # SENDER
        # ==================================================================
        if is_src:
            if bench_mode == "sender":
                # Sender bench: reuse one GPU buffer, only update header.
                # No disk read, no H2D, no worker threads.
                gpu_buf = gpu_bufs[0]
                header_np = cpu_bufs[0].numpy()
                for i in range(total_chunks):
                    iter_t0 = time.monotonic()
                    desc = flat_chunks[i]
                    chunk_len = desc.length

                    # Stamp header into GPU via tiny CPU→GPU
                    struct.pack_into("<II", header_np, 0, i, chunk_len)
                    gpu_buf[:HEADER].copy_(cpu_bufs[0][:HEADER])

                    t0 = time.monotonic()
                    dist.broadcast(gpu_buf, src=src_rank)
                    nccl_ms = (time.monotonic() - t0) * 1000

                    elapsed = (time.monotonic() - pipeline_t0) * 1000
                    logger.info(
                        f"[Rank {self.rank} {tag}] [{i}/{total_chunks}] "
                        f"{desc.rel_path} c{desc.chunk_idx}/{desc.num_chunks} "
                        f"slot=0 chunk_id={i} BENCH "
                        f"nccl={nccl_ms:.1f}ms "
                        f"iter={(time.monotonic() - iter_t0) * 1000:.1f}ms "
                        f"T+{elapsed:.0f}ms ({chunk_len / 1e6:.1f} MB)"
                    )
                    if pbar is not None:
                        pbar.update(chunk_len)
            else:
                import os as _os_snd

                # ready_q: workers put (slot, chunk_index, copy_done_event) after read+h2d
                ready_q: queue.Queue = queue.Queue()
                # Per-worker queues: main sends (chunk_index, bcast_done_event|None)
                # so a worker only starts new work after its slot is broadcast
                work_qs: list[queue.Queue] = [queue.Queue() for _ in range(num_workers)]
                worker_error = None

                # Open file descriptors for pread (no seek, no lock needed)
                file_fds_snd: dict[int, int] = {}
                for file_idx, entry in enumerate(manifest.files):
                    if entry.size == 0:
                        continue
                    file_fds_snd[file_idx] = _os_snd.open(str(src_root / entry.rel_path), _os_snd.O_RDONLY)

                def _sender_worker(worker_id):
                    """Each worker owns slot == worker_id."""
                    nonlocal worker_error
                    slot = worker_id
                    try:
                        while True:
                            item = work_qs[slot].get()
                            if item is None:
                                return
                            chunk_i, bcast_done_evt = item
                            desc = flat_chunks[chunk_i]

                            # Write header into cpu_buf[slot][0:HEADER]
                            struct.pack_into("<II", cpu_bufs[slot].numpy(), 0, chunk_i, desc.length)

                            # Disk read via libc pread directly into pinned buf
                            fd = file_fds_snd[desc.file_idx]
                            buf_ptr = cpu_bufs[slot].data_ptr() + HEADER
                            t_read = time.monotonic()
                            _libc_pread_all(fd, buf_ptr, desc.length, desc.offset)
                            read_ms = (time.monotonic() - t_read) * 1000

                            # H2D copy on worker's own CUDA stream.
                            # Wait for the previous broadcast on the default stream
                            # to finish reading gpu_bufs[slot] before overwriting it.
                            t_h2d = time.monotonic()
                            with torch.cuda.stream(copy_streams[slot]):
                                if bcast_done_evt is not None:
                                    copy_streams[slot].wait_event(bcast_done_evt)
                                gpu_bufs[slot].copy_(cpu_bufs[slot])
                                copy_done_evt = torch.cuda.Event()
                                copy_done_evt.record(copy_streams[slot])
                            copy_streams[slot].synchronize()
                            h2d_ms = (time.monotonic() - t_h2d) * 1000

                            logger.info(
                                f"[Rank {self.rank} {tag}] PREP [{chunk_i}/{total_chunks}] "
                                f"{desc.rel_path} c{desc.chunk_idx}/{desc.num_chunks} "
                                f"slot={slot} read={read_ms:.0f}ms h2d={h2d_ms:.0f}ms "
                                f"({desc.length / 1e6:.1f} MB)"
                            )
                            ready_q.put((slot, chunk_i, copy_done_evt))
                    except Exception as e:
                        worker_error = e
                        ready_q.put(None)

                # Start worker threads
                workers = []
                for wid in range(num_workers):
                    t = threading.Thread(target=_sender_worker, args=(wid,), daemon=True)
                    t.start()
                    workers.append(t)

                # Seed initial work: one chunk per worker (up to total_chunks)
                # No bcast_done event for initial chunks (no prior broadcast)
                next_chunk = 0
                for wid in range(min(num_workers, total_chunks)):
                    work_qs[wid].put((next_chunk, None))
                    next_chunk += 1

                # Main loop: pop ready (unordered), NCCL broadcast, give slot back
                for i in range(total_chunks):
                    iter_t0 = time.monotonic()
                    item = ready_q.get()
                    if item is None:
                        if worker_error is not None:
                            raise worker_error
                        break
                    slot, chunk_i, copy_done_evt = item
                    desc = flat_chunks[chunk_i]
                    chunk_len = desc.length
                    prep_wait_ms = (time.monotonic() - iter_t0) * 1000

                    # Make the default stream wait for the H2D copy to finish
                    # on copy_streams[slot] before NCCL reads gpu_bufs[slot].
                    torch.cuda.current_stream(device).wait_event(copy_done_evt)

                    # NCCL broadcast (fixed size — all ranks use bcast_size)
                    t0 = time.monotonic()
                    dist.broadcast(gpu_bufs[slot], src=src_rank)

                    # Record event so the worker knows when broadcast is done
                    # reading gpu_bufs[slot] and it is safe to overwrite.
                    bcast_done_evt = torch.cuda.Event()
                    bcast_done_evt.record(torch.cuda.current_stream(device))
                    nccl_ms = (time.monotonic() - t0) * 1000

                    # Dispatch next chunk to the worker that owns this slot
                    if next_chunk < total_chunks:
                        work_qs[slot].put((next_chunk, bcast_done_evt))
                        next_chunk += 1

                    elapsed = (time.monotonic() - pipeline_t0) * 1000
                    logger.info(
                        f"[Rank {self.rank} {tag}] [{i}/{total_chunks}] "
                        f"{desc.rel_path} c{desc.chunk_idx}/{desc.num_chunks} "
                        f"slot={slot} chunk_id={chunk_i} prep_wait={prep_wait_ms:.1f}ms "
                        f"nccl={nccl_ms:.1f}ms "
                        f"iter={(time.monotonic() - iter_t0) * 1000:.1f}ms "
                        f"T+{elapsed:.0f}ms ({chunk_len / 1e6:.1f} MB)"
                    )
                    if pbar is not None:
                        pbar.update(chunk_len)

                # Shut down workers
                for wid in range(num_workers):
                    work_qs[wid].put(None)
                for t in workers:
                    t.join()
                # Close file descriptors
                for fd in file_fds_snd.values():
                    _os_snd.close(fd)

        # ==================================================================
        # RECEIVER
        # ==================================================================
        else:
            if bench_mode == "receiver":
                # Bench mode: NCCL recv only, no D2H / disk write.
                # Reuse a single GPU buffer (slot 0) for all receives.
                for i in range(total_chunks):
                    iter_t0 = time.monotonic()

                    t0 = time.monotonic()
                    dist.broadcast(gpu_bufs[0], src=src_rank)
                    nccl_ms = (time.monotonic() - t0) * 1000

                    # Parse header to report progress
                    header_cpu = gpu_bufs[0][:HEADER].cpu().numpy().tobytes()
                    chunk_id, chunk_len = struct.unpack_from("<II", header_cpu, 0)
                    desc = flat_chunks[chunk_id]

                    elapsed = (time.monotonic() - pipeline_t0) * 1000
                    logger.info(
                        f"[Rank {self.rank} {tag}] [{i}/{total_chunks}] "
                        f"{desc.rel_path} c{desc.chunk_idx}/{desc.num_chunks} "
                        f"slot=0 chunk_id={chunk_id} BENCH "
                        f"nccl={nccl_ms:.1f}ms "
                        f"iter={(time.monotonic() - iter_t0) * 1000:.1f}ms "
                        f"T+{elapsed:.0f}ms ({chunk_len / 1e6:.1f} MB)"
                    )
                    if pbar is not None:
                        pbar.update(chunk_len)
            else:
                import os as _os

                # Pre-allocate output files with .incomplete suffix
                file_fds: dict[int, int] = {}
                incomplete_paths: dict[int, tuple[Path, Path]] = {}  # idx -> (incomplete, final)
                t0 = time.monotonic()
                for entry_idx, entry in enumerate(manifest.files):
                    if entry.size == 0:
                        continue
                    final_path = dst_root / entry.rel_path
                    incomplete_path = final_path.with_suffix(final_path.suffix + ".incomplete")
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    fd = _os.open(str(incomplete_path), _os.O_CREAT | _os.O_WRONLY | _os.O_TRUNC)
                    _os.ftruncate(fd, entry.size)
                    file_fds[entry_idx] = fd
                    incomplete_paths[entry_idx] = (incomplete_path, final_path)
                prealloc_ms = (time.monotonic() - t0) * 1000
                logger.info(f"[Rank {self.rank} {tag}] pre-allocated {len(file_fds)} files in {prealloc_ms:.0f}ms")

                # free_slots: slots available for main thread to recv into
                free_slots: queue.Queue[int] = queue.Queue()
                for s in range(num_workers):
                    free_slots.put(s)

                # write_q: main thread puts (slot, chunk_id, chunk_len, bcast_done_evt)
                write_q: queue.Queue = queue.Queue()
                worker_error = None

                def _recv_worker(worker_id):
                    """Worker does D2H + pwrite, then returns slot."""
                    nonlocal worker_error
                    try:
                        while True:
                            item = write_q.get()
                            if item is None:
                                return
                            slot, chunk_id, chunk_len, bcast_done_evt = item
                            desc = flat_chunks[chunk_id]

                            # D2H copy on worker's own CUDA stream.
                            # Wait for broadcast on the default stream to finish
                            # writing into gpu_bufs[slot] before reading it.
                            payload_end = HEADER + chunk_len
                            t_d2h = time.monotonic()
                            with torch.cuda.stream(copy_streams[slot]):
                                copy_streams[slot].wait_event(bcast_done_evt)
                                cpu_bufs[slot][:payload_end].copy_(gpu_bufs[slot][:payload_end])
                            copy_streams[slot].synchronize()
                            d2h_ms = (time.monotonic() - t_d2h) * 1000

                            # pwrite to disk via libc (skip header)
                            fd = file_fds[desc.file_idx]
                            buf_ptr = cpu_bufs[slot].data_ptr() + HEADER
                            t_w = time.monotonic()
                            _libc_pwrite_all(fd, buf_ptr, chunk_len, desc.offset)
                            write_ms = (time.monotonic() - t_w) * 1000

                            logger.info(
                                f"[Rank {self.rank} {tag}] WRITE [{chunk_id}/{total_chunks}] "
                                f"{desc.rel_path} c{desc.chunk_idx}/{desc.num_chunks} "
                                f"slot={slot} d2h={d2h_ms:.0f}ms write={write_ms:.0f}ms "
                                f"({chunk_len / 1e6:.1f} MB)"
                            )

                            # Return slot to main thread
                            free_slots.put(slot)
                    except Exception as e:
                        worker_error = e
                        free_slots.put(-1)  # unblock main

                # Start worker threads
                workers = []
                for wid in range(num_workers):
                    t = threading.Thread(target=_recv_worker, args=(wid,), daemon=True)
                    t.start()
                    workers.append(t)

                # Main loop: grab free slot, NCCL recv, parse header, hand to worker
                for i in range(total_chunks):
                    iter_t0 = time.monotonic()

                    t0 = time.monotonic()
                    slot = free_slots.get()
                    if slot == -1 and worker_error is not None:
                        raise worker_error
                    slot_wait_ms = (time.monotonic() - t0) * 1000

                    # NCCL receive (fixed size)
                    t0 = time.monotonic()
                    dist.broadcast(gpu_bufs[slot], src=src_rank)

                    # Record event so worker knows broadcast finished writing gpu_bufs[slot]
                    bcast_done_evt = torch.cuda.Event()
                    bcast_done_evt.record(torch.cuda.current_stream(device))
                    nccl_ms = (time.monotonic() - t0) * 1000

                    # Parse header on GPU — copy just 8 bytes to CPU
                    header_cpu = gpu_bufs[slot][:HEADER].cpu().numpy().tobytes()
                    chunk_id, chunk_len = struct.unpack_from("<II", header_cpu, 0)
                    desc = flat_chunks[chunk_id]

                    # Hand off to worker for D2H + write
                    write_q.put((slot, chunk_id, chunk_len, bcast_done_evt))

                    elapsed = (time.monotonic() - pipeline_t0) * 1000
                    logger.info(
                        f"[Rank {self.rank} {tag}] [{i}/{total_chunks}] "
                        f"{desc.rel_path} c{desc.chunk_idx}/{desc.num_chunks} "
                        f"slot={slot} chunk_id={chunk_id} slot_wait={slot_wait_ms:.1f}ms "
                        f"nccl={nccl_ms:.1f}ms "
                        f"iter={(time.monotonic() - iter_t0) * 1000:.1f}ms "
                        f"T+{elapsed:.0f}ms ({chunk_len / 1e6:.1f} MB)"
                    )
                    if pbar is not None:
                        pbar.update(chunk_len)

                # Shut down workers and wait
                for _ in workers:
                    write_q.put(None)
                t0 = time.monotonic()
                for t in workers:
                    t.join()
                if worker_error is not None:
                    raise worker_error
                # Close file descriptors
                for fd in file_fds.values():
                    _os.close(fd)
                # Atomically rename .incomplete -> final
                for incomplete_path, final_path in incomplete_paths.values():
                    _os.rename(str(incomplete_path), str(final_path))
                flush_ms = (time.monotonic() - t0) * 1000
                tp = total_bytes / (flush_ms / 1000) / 1e9 if flush_ms > 0 else 0
                logger.info(f"[Rank {self.rank} {tag}] flush wait: {flush_ms:.0f}ms ({tp:.1f} GB/s)")

        torch.cuda.synchronize(device)
        pipeline_ms = (time.monotonic() - pipeline_t0) * 1000
        tp = total_bytes / (pipeline_ms / 1000) / 1e9 if pipeline_ms > 0 else 0
        logger.info(
            f"[Rank {self.rank} {tag}] pipeline done: {total_chunks} chunks in {pipeline_ms:.0f}ms ({tp:.1f} GB/s)"
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def broadcast(
        self,
        src_path: str,
        dst_dir: str | None = None,
        src_rank: int = 0,
        chunk_size: int = 1024 * 1024 * 1024,
        num_buffers: int = 10,
        bench_mode: str | None = None,
    ):
        """
        Broadcast file(s) from src_rank to all other ranks.

        Args:
            src_path: Path to file or directory on source rank.
            dst_dir:  Destination directory on all ranks.
                      Defaults to same path as src_path.
            src_rank: Rank that owns the source files.
            chunk_size: Bytes per chunk.
            num_buffers: Buffer pool size (controls overlap depth).
            bench_mode: "sender" to skip disk reads (send junk data),
                        "receiver" to skip D2H + disk writes.
                        Both measure pure NCCL throughput from respective side.
        """
        assert self._initialized, "Call init_process_group first"

        broadcast_t0 = time.monotonic()
        config = BroadcastConfig(chunk_size=chunk_size, num_buffers=num_buffers)
        device = self._get_device(config)
        is_src = self.rank == src_rank

        logger.info(
            f"[Rank {self.rank}] broadcast: begin src_path={src_path} "
            f"dst_dir={dst_dir} is_src={is_src} "
            f"chunk_size={chunk_size / 1e6:.0f}MB num_buffers={num_buffers}"
        )

        # Phase 1: manifest
        if is_src:
            manifest = self._build_manifest(src_path, config.chunk_size)
            total_bytes = sum(f.size for f in manifest.files)
            logger.info(
                f"[Rank {self.rank}] broadcast: built manifest, "
                f"{len(manifest.files)} file(s), {total_bytes / 1e9:.2f} GB"
            )
        else:
            manifest = None

        manifest = self._broadcast_manifest(manifest, src_rank, device)

        # Phase 2: compute dst_root
        # For single files, dst_root is the parent directory so that
        # dst_root / rel_path recreates the file at the right location.
        # For directories, dst_root is the directory itself.
        if dst_dir is not None:
            dst_root = Path(dst_dir)
        elif manifest.src_is_file:
            dst_root = Path(src_path).parent
        else:
            dst_root = Path(src_path)

        # src_root: for files, use parent so src_root / rel_path = original path
        src_root = Path(src_path)
        if is_src and manifest.src_is_file:
            src_root = src_root.parent

        logger.info(
            f"[Rank {self.rank}] broadcast: src_root={src_root} dst_root={dst_root} "
            f"transferring {len(manifest.files)} file(s)"
        )

        # Handle empty files (0 bytes) that need to be created on receivers
        for entry in manifest.files:
            if entry.size == 0:
                if not is_src:
                    dst_file = dst_root / entry.rel_path
                    incomplete_file = dst_file.with_suffix(dst_file.suffix + ".incomplete")
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    incomplete_file.touch()
                    os.rename(str(incomplete_file), str(dst_file))
                logger.info(
                    f"[Rank {self.rank}] broadcast: {'skipped' if is_src else 'created'} empty file {entry.rel_path}"
                )

        # Chunk-based pipeline for all non-empty files
        total_bytes = sum(f.size for f in manifest.files)
        pbar = ray_tqdm(
            total=total_bytes,
            desc=f"[Rank {self.rank}] broadcast",
            unit="B",
        )
        self._broadcast_all_chunks(
            manifest=manifest,
            src_root=src_root if is_src else None,
            dst_root=dst_root,
            src_rank=src_rank,
            config=config,
            pbar=pbar,
            bench_mode=bench_mode,
        )
        pbar.close()

        t0 = time.monotonic()
        dist.barrier()
        barrier_ms = (time.monotonic() - t0) * 1000
        total_ms = (time.monotonic() - broadcast_t0) * 1000
        throughput = total_bytes / (total_ms / 1000) / 1e9 if total_ms > 0 else 0
        logger.info(
            f"[Rank {self.rank}] broadcast: complete "
            f"total={total_ms:.0f}ms barrier={barrier_ms:.0f}ms "
            f"throughput={throughput:.2f} GB/s"
        )

    def all_gather(
        self,
        src_dir: str,
        dst_dir: str | None = None,
        chunk_size: int = 1024 * 1024 * 1024,
        num_buffers: int = 10,
        bench_mode: str | None = None,
    ):
        """
        All-gather files: each rank contributes files from src_dir,
        and at the end every rank has all files in dst_dir.

        If src_dir == dst_dir (or dst_dir is None), each rank already has
        its own files and only receives files from other ranks.

        Args:
            src_dir:     Directory containing this rank's files.
            dst_dir:     Destination directory. Defaults to src_dir.
            chunk_size:  Bytes per NCCL chunk.
            num_buffers: Buffer pool size.
            bench_mode:  If True, receivers only do NCCL recv without
                         D2H copy or disk writes (for throughput benchmarking).
        """
        assert self._initialized, "Call init_process_group first"

        all_gather_t0 = time.monotonic()
        config = BroadcastConfig(chunk_size=chunk_size, num_buffers=num_buffers)
        device = self._get_device(config)

        if dst_dir is None:
            dst_dir = src_dir
        src_root = Path(src_dir)
        dst_root = Path(dst_dir)

        logger.info(
            f"[Rank {self.rank}] all_gather: begin src_dir={src_dir} "
            f"dst_dir={dst_dir} chunk_size={chunk_size / 1e6:.0f}MB "
            f"num_buffers={num_buffers}"
        )

        # Phase 1: each rank builds local manifest
        t0 = time.monotonic()
        if src_root.is_dir():
            local_manifest = self._build_manifest(src_dir, config.chunk_size)
        else:
            # Directory doesn't exist or is empty — empty manifest
            local_manifest = BroadcastManifest(chunk_size=config.chunk_size)
        local_manifest.src_is_file = False
        logger.info(
            f"[Rank {self.rank}] all_gather: local manifest "
            f"{len(local_manifest.files)} files, "
            f"{sum(f.size for f in local_manifest.files) / 1e9:.2f} GB "
            f"in {(time.monotonic() - t0) * 1000:.0f}ms"
        )

        # Phase 2: gather all manifests to rank 0
        all_manifests = self._gather_manifests(local_manifest, device)

        # Phase 3: rank 0 builds plan, broadcasts to all
        plan = self._build_and_broadcast_plan(all_manifests, config.chunk_size, device)

        logger.info(
            f"[Rank {self.rank}] all_gather: plan has "
            f"{len(plan.transfers)} transfers, "
            f"{sum(len(t.files) for t in plan.transfers)} total files"
        )

        # Phase 4: execute each transfer (broadcast from src_rank)
        total_bytes_all = sum(f.size for t in plan.transfers for f in t.files)
        pbar = ray_tqdm(
            total=total_bytes_all,
            desc=f"[Rank {self.rank}] all_gather {self.rank}",
            unit="B",
        )

        total_transferred = 0
        for ti, transfer in enumerate(plan.transfers):
            if not transfer.files:
                continue

            is_src = self.rank == transfer.src_rank
            manifest = BroadcastManifest(
                chunk_size=plan.chunk_size,
                src_is_file=False,
                files=transfer.files,
            )
            transfer_bytes = sum(f.size for f in transfer.files)

            logger.info(
                f"[Rank {self.rank}] all_gather: transfer {ti}/{len(plan.transfers)} "
                f"from rank {transfer.src_rank}, "
                f"{len(transfer.files)} files, {transfer_bytes / 1e9:.2f} GB"
            )

            # Handle empty files
            for entry in manifest.files:
                if entry.size == 0 and not is_src:
                    dst_file = dst_root / entry.rel_path
                    incomplete_file = dst_file.with_suffix(dst_file.suffix + ".incomplete")
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    incomplete_file.touch()
                    os.rename(str(incomplete_file), str(dst_file))

            # Transfer non-empty files
            self._broadcast_all_chunks(
                manifest=manifest,
                src_root=src_root if is_src else None,
                dst_root=dst_root,
                src_rank=transfer.src_rank,
                config=config,
                pbar=pbar,
                bench_mode=bench_mode,
            )

            total_transferred += transfer_bytes

        pbar.close()

        t0 = time.monotonic()
        dist.barrier()
        barrier_ms = (time.monotonic() - t0) * 1000
        total_ms = (time.monotonic() - all_gather_t0) * 1000
        throughput = total_transferred / (total_ms / 1000) / 1e9 if total_ms > 0 else 0
        logger.info(
            f"[Rank {self.rank}] all_gather: complete "
            f"total={total_ms:.0f}ms barrier={barrier_ms:.0f}ms "
            f"transferred={total_transferred / 1e9:.2f} GB "
            f"throughput={throughput:.2f} GB/s"
        )

    def _gather_manifests(
        self,
        local_manifest: BroadcastManifest,
        device: torch.device,
    ) -> list[BroadcastManifest] | None:
        """
        Gather all ranks' manifests to rank 0.

        Returns list of manifests on rank 0, None on other ranks.
        """
        t0 = time.monotonic()

        # Serialize local manifest
        local_json = local_manifest.to_json().encode("utf-8")
        local_size = len(local_json)

        # All-gather sizes so each rank knows how much data to expect
        size_tensor = torch.tensor([local_size], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(self.world_size)]
        dist.all_gather(size_list, size_tensor)
        sizes = [int(s.item()) for s in size_list]
        max_size = max(sizes)

        logger.info(f"[Rank {self.rank}] _gather_manifests: sizes={sizes} max_size={max_size}")

        # Pad local payload to max_size and all-gather
        local_payload = torch.zeros(max_size, dtype=torch.uint8, device=device)
        local_bytes = torch.frombuffer(bytearray(local_json), dtype=torch.uint8)
        local_payload[:local_size] = local_bytes.to(device)

        payload_list = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(self.world_size)]
        dist.all_gather(payload_list, local_payload)

        gather_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Rank {self.rank}] _gather_manifests: done in {gather_ms:.0f}ms")

        # Only rank 0 needs to decode all manifests
        if self.rank == 0:
            manifests = []
            for r in range(self.world_size):
                json_bytes = payload_list[r][: sizes[r]].cpu().numpy().tobytes()
                manifests.append(BroadcastManifest.from_json(json_bytes.decode("utf-8")))
            return manifests
        return None

    def _build_and_broadcast_plan(
        self,
        all_manifests: list[BroadcastManifest] | None,
        chunk_size: int,
        device: torch.device,
    ) -> AllGatherPlan:
        """
        Rank 0 builds the all_gather plan and broadcasts it to all ranks.
        """
        if self.rank == 0:
            assert all_manifests is not None
            plan = AllGatherPlan(chunk_size=chunk_size)

            # Deduplicate: first rank with a file wins
            seen_files: dict[str, int] = {}  # rel_path -> src_rank
            per_rank_files: dict[int, list[FileEntry]] = {r: [] for r in range(self.world_size)}

            for rank, manifest in enumerate(all_manifests):
                for entry in manifest.files:
                    if entry.rel_path not in seen_files:
                        seen_files[entry.rel_path] = rank
                        per_rank_files[rank].append(entry)

            # Build transfers — one per rank that has files to contribute
            for rank in range(self.world_size):
                files = per_rank_files[rank]
                if files:
                    plan.transfers.append(AllGatherTransfer(src_rank=rank, files=files))

            total_files = sum(len(t.files) for t in plan.transfers)
            total_bytes = sum(f.size for t in plan.transfers for f in t.files)
            logger.info(
                f"[Rank 0] _build_all_gather_plan: "
                f"{total_files} unique files from "
                f"{len(plan.transfers)} ranks, "
                f"{total_bytes / 1e9:.2f} GB total"
            )

            plan_json = plan.to_json().encode("utf-8")
            plan_size = torch.tensor([len(plan_json)], dtype=torch.long, device=device)
        else:
            plan_json = b""
            plan_size = torch.tensor([0], dtype=torch.long, device=device)

        # Broadcast plan size then payload
        dist.broadcast(plan_size, src=0)
        size = int(plan_size.item())

        if self.rank == 0:
            payload = torch.frombuffer(bytearray(plan_json), dtype=torch.uint8).to(device)
        else:
            payload = torch.empty(size, dtype=torch.uint8, device=device)

        dist.broadcast(payload, src=0)

        if self.rank != 0:
            json_str = payload.cpu().numpy().tobytes().decode("utf-8")
            plan = AllGatherPlan.from_json(json_str)

        logger.info(f"[Rank {self.rank}] _build_and_broadcast_plan: done, {len(plan.transfers)} transfers")
        return plan


# ---------------------------------------------------------------------------
# Coordinator actor — manages the worker group
# ---------------------------------------------------------------------------


@ray.remote
class FileTransferGroup:
    """
    Manages a group of FileTransferWorker actors across the Ray cluster.
    Automatically spawns one worker per GPU node using node affinity.

    Usage:
        group = FileTransferGroup.remote()
        ray.get(group.setup.remote())
        ray.get(group.broadcast.remote(src_path="/data/weights/"))
        ray.get(group.teardown.remote())
    """

    def __init__(self):
        self.workers: list = []
        self.world_size: int = 0
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            force=True,
        )

    def setup(
        self,
        master_port: int | None = None,
        worker_options: dict | None = None,
        src_node_ip: str | None = None,
    ):
        """
        Create one worker per node and initialize torch.distributed.

        Automatically discovers all GPU nodes in the Ray cluster and pins
        one worker to each node via NodeAffinitySchedulingStrategy.

        Args:
            master_port: Port for torch.distributed rendezvous.
            worker_options: Extra kwargs for FileTransferWorker.options()
                           e.g. {"num_cpus": 4}
            src_node_ip: If provided, the worker on this node gets rank 0.
                         Defaults to driver node IP in broadcast_files().
        """
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        setup_t0 = time.monotonic()
        logger.info(f"[Group] setup: begin master_port={master_port} src_node_ip={src_node_ip}")

        if worker_options is None:
            worker_options = {}

        if master_port is None:
            master_port = get_random_free_port()
            logger.info(f"[Group] setup: allocated random port {master_port}")

        # Discover all live GPU nodes
        nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
        if not nodes:
            raise RuntimeError("No alive GPU nodes found in the Ray cluster")

        self.world_size = len(nodes)
        node_ips = [n["NodeManagerAddress"] for n in nodes]
        logger.info(f"[Group] setup: found {self.world_size} GPU nodes: {node_ips}")

        # Sort so src_node gets index 0
        if src_node_ip is not None:
            nodes.sort(key=lambda n: n["NodeManagerAddress"] != src_node_ip)
            if nodes[0]["NodeManagerAddress"] != src_node_ip:
                available = {n["NodeManagerAddress"] for n in nodes}
                raise ValueError(f"No GPU node found at {src_node_ip}. Available: {available}")

        sorted_ips = [n["NodeManagerAddress"] for n in nodes]
        logger.info(f"[Group] setup: rank assignment: {sorted_ips} (rank 0 = {sorted_ips[0]})")

        # Spawn one worker per node, pinned via node affinity
        t0 = time.monotonic()
        self.workers = []
        for node in nodes:
            strategy = NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"],
                soft=False,
            )
            worker = FileTransferWorker.options(
                scheduling_strategy=strategy,
                **worker_options,
            ).remote()
            self.workers.append(worker)
        spawn_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Group] setup: spawned {len(self.workers)} workers in {spawn_ms:.0f}ms")

        # Use rank 0's IP as master
        master_addr = nodes[0]["NodeManagerAddress"]

        # Initialize torch.distributed on all workers in parallel
        logger.info(f"[Group] setup: init_process_group on all workers (master={master_addr}:{master_port})")
        t0 = time.monotonic()
        init_futures = [
            w.init_process_group.remote(
                rank=i,
                world_size=self.world_size,
                master_addr=master_addr,
                master_port=master_port,
            )
            for i, w in enumerate(self.workers)
        ]
        ray.get(init_futures)
        init_ms = (time.monotonic() - t0) * 1000

        total_ms = (time.monotonic() - setup_t0) * 1000
        logger.info(
            f"[Group] setup: done in {total_ms:.0f}ms "
            f"(spawn={spawn_ms:.0f}ms init={init_ms:.0f}ms) "
            f"{self.world_size} workers on {sorted_ips}"
        )

    def broadcast(
        self,
        src_path: str,
        dst_dir: str | None = None,
        chunk_size: int = 1024 * 1024 * 1024,
        num_buffers: int = 10,
        bench_mode: str | None = None,
    ):
        """
        Broadcast file(s) from rank 0 to all ranks.
        All workers execute in parallel; returns when all finish.

        dst_dir defaults to same path as src_path.
        """
        logger.info(
            f"[Group] broadcast: begin src_path={src_path} dst_dir={dst_dir} "
            f"chunk_size={chunk_size / 1e6:.0f}MB num_buffers={num_buffers} "
            f"bench_mode={bench_mode} workers={len(self.workers)}"
        )
        t0 = time.monotonic()
        futures = [
            w.broadcast.remote(
                src_path=src_path,
                dst_dir=dst_dir,
                src_rank=0,
                chunk_size=chunk_size,
                num_buffers=num_buffers,
                bench_mode=bench_mode,
            )
            for w in self.workers
        ]
        ray.get(futures)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Group] broadcast: done in {elapsed_ms:.0f}ms")

    def all_gather(
        self,
        src_dir: str,
        dst_dir: str | None = None,
        chunk_size: int = 1024 * 1024 * 1024,
        num_buffers: int = 10,
        bench_mode: str | None = None,
    ):
        """
        All-gather files from all ranks.
        Each rank contributes files from src_dir. At the end every rank
        has all files in dst_dir.

        dst_dir defaults to src_dir.
        """
        logger.info(
            f"[Group] all_gather: begin src_dir={src_dir} dst_dir={dst_dir} "
            f"chunk_size={chunk_size / 1e6:.0f}MB num_buffers={num_buffers} "
            f"bench_mode={bench_mode} workers={len(self.workers)}"
        )
        t0 = time.monotonic()
        futures = [
            w.all_gather.remote(
                src_dir=src_dir,
                dst_dir=dst_dir,
                chunk_size=chunk_size,
                num_buffers=num_buffers,
                bench_mode=bench_mode,
            )
            for w in self.workers
        ]
        ray.get(futures)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Group] all_gather: done in {elapsed_ms:.0f}ms")

    def teardown(self):
        """Destroy process groups and kill workers."""
        logger.info(f"[Group] teardown: begin ({len(self.workers)} workers)")
        t0 = time.monotonic()
        if self.workers:
            ray.get([w.teardown.remote() for w in self.workers])
            for w in self.workers:
                ray.kill(w)
            self.workers = []
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[Group] teardown: done in {elapsed_ms:.0f}ms")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def broadcast_files(
    src_path: str,
    dst_dir: str | None = None,
    chunk_size: int = 1024 * 1024 * 1024,
    num_buffers: int = 10,
    master_port: int | None = None,
    worker_options: dict | None = None,
    bench_mode: str | None = None,
) -> FileTransferGroup:
    """
    One-shot convenience: create group, broadcast, return group for reuse.

    Call this on the node that has the source files. That node becomes
    rank 0 automatically. One worker is created per GPU node in the cluster.

    Args:
        src_path:  File or directory to broadcast.
        dst_dir:   Destination directory on all ranks. Defaults to src_path.
        chunk_size: Bytes per NCCL chunk.
        num_buffers: GPU double/triple buffering.
        master_port: torch.distributed rendezvous port.
        worker_options: Extra ray.remote options for workers.

    Returns:
        The FileTransferGroup actor (reusable for more operations).

    Example:
        group = broadcast_files("/data/model/")
        ray.get(group.broadcast.remote("/data/dataset/"))
        ray.get(group.teardown.remote())
    """
    driver_ip = ray.util.get_node_ip_address()
    logger.info(
        f"broadcast_files: begin src_path={src_path} dst_dir={dst_dir} "
        f"chunk_size={chunk_size / 1e6:.0f}MB num_buffers={num_buffers} "
        f"driver_ip={driver_ip}"
    )
    overall_t0 = time.monotonic()

    group = FileTransferGroup.remote()

    t0 = time.monotonic()
    ray.get(
        group.setup.remote(
            master_port=master_port,
            worker_options=worker_options,
            src_node_ip=driver_ip,
        )
    )
    setup_ms = (time.monotonic() - t0) * 1000
    logger.info(f"broadcast_files: setup done in {setup_ms:.0f}ms")

    t0 = time.monotonic()
    ray.get(
        group.broadcast.remote(
            src_path=src_path,
            dst_dir=dst_dir,
            chunk_size=chunk_size,
            num_buffers=num_buffers,
            bench_mode=bench_mode,
        )
    )
    broadcast_ms = (time.monotonic() - t0) * 1000
    total_ms = (time.monotonic() - overall_t0) * 1000
    logger.info(f"broadcast_files: done total={total_ms:.0f}ms (setup={setup_ms:.0f}ms broadcast={broadcast_ms:.0f}ms)")
    return group


def all_gather_files(
    src_dir: str,
    dst_dir: str | None = None,
    chunk_size: int = 1024 * 1024 * 1024,
    num_buffers: int = 10,
    master_port: int | None = None,
    worker_options: dict | None = None,
    bench_mode: str | None = None,
) -> FileTransferGroup:
    """
    All-gather files across all GPU nodes.

    Each node contributes files from src_dir. At the end every node has
    all files in dst_dir (defaults to src_dir).

    Args:
        src_dir:     Directory containing this node's files.
        dst_dir:     Destination directory. Defaults to src_dir.
        chunk_size:  Bytes per NCCL chunk.
        num_buffers: Buffer pool size.
        master_port: torch.distributed rendezvous port.
        worker_options: Extra ray.remote options for workers.

    Returns:
        The FileTransferGroup actor (reusable for more operations).
    """
    driver_ip = ray.util.get_node_ip_address()
    logger.info(
        f"all_gather_files: begin src_dir={src_dir} dst_dir={dst_dir} "
        f"chunk_size={chunk_size / 1e6:.0f}MB num_buffers={num_buffers} "
        f"driver_ip={driver_ip}"
    )
    overall_t0 = time.monotonic()

    group = FileTransferGroup.remote()

    t0 = time.monotonic()
    ray.get(
        group.setup.remote(
            master_port=master_port,
            worker_options=worker_options,
            src_node_ip=driver_ip,
        )
    )
    setup_ms = (time.monotonic() - t0) * 1000
    logger.info(f"all_gather_files: setup done in {setup_ms:.0f}ms")

    t0 = time.monotonic()
    ray.get(
        group.all_gather.remote(
            src_dir=src_dir,
            dst_dir=dst_dir,
            chunk_size=chunk_size,
            num_buffers=num_buffers,
            bench_mode=bench_mode,
        )
    )
    gather_ms = (time.monotonic() - t0) * 1000
    total_ms = (time.monotonic() - overall_t0) * 1000
    logger.info(f"all_gather_files: done total={total_ms:.0f}ms (setup={setup_ms:.0f}ms gather={gather_ms:.0f}ms)")
    return group
