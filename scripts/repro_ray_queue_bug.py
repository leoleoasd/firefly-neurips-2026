#!/usr/bin/env python3
"""Reproduce the ray.util.queue desync: qsize() counts items that get()/get_nowait()
can never retrieve.

Hypothesis: _QueueActor wraps a (non-thread-safe) asyncio.Queue but exposes BOTH
async methods (put/get -> event-loop thread) and sync methods (put_nowait/get_nowait/
qsize -> thread-pool threads on an async actor). With maxsize=0, put_async falls to
put_nowait and the collector uses get_nowait — all sync -> concurrent cross-thread
access to the asyncio.Queue under high producer concurrency corrupts it.

tau2 hits this (16 workers x high concurrency, maxsize=0 output queue); the input
queue (maxsize>0 -> async put/get) and tool_call_agent's lighter load do not.

Run:  uv run python scripts/repro_ray_queue_bug.py   (connects to the running cluster)
"""

import asyncio
import time

import ray
from ray.util.queue import Empty, Queue


def _make_group(toklen=4000, n_samples=8):
    """A realistic-ish "group": list of n_samples dicts each carrying long token /
    logprob lists + nested metadata, like a tau2 group of 8 trajectories (~0.5MB)."""
    return [
        {
            "tokens": list(range(toklen)),
            "log_probs": [0.123456] * toklen,
            "loss_mask": [1] * toklen,
            "metadata": {"messages": [{"role": "assistant", "content": "x" * 200}] * 20, "idx": s},
            "reward": float(s % 2),
        }
        for s in range(n_samples)
    ]


@ray.remote(num_cpus=0.1)
class Producer:
    """Mimics an AsyncRolloutWorkerActor pushing completed groups."""

    async def produce(self, q, n, big_item, slow):
        item = _make_group() if big_item else (b"x" * 64)
        for i in range(n):
            if slow:
                # simulate episode latency between puts (groups trickle in)
                await asyncio.sleep(0.05)
            # block=True is the default; for maxsize<=0 this routes to put_nowait
            # (sync method), for maxsize>0 to the async put.
            await q.put_async((i, item))
        return n


async def run_case(maxsize, num_producers, items_each, big_item, drain_mode, label, slow=False):
    q = Queue(maxsize=maxsize)
    producers = [Producer.remote() for _ in range(num_producers)]
    put_futs = [p.produce.remote(q, items_each, big_item, slow) for p in producers]

    total_put = num_producers * items_each
    total_got = 0
    stuck_obs = 0  # times we saw qsize>0 but drained 0
    nonempty_errs = 0
    start = time.time()

    while time.time() - start < 60:
        drained = 0
        # drain everything currently retrievable
        while True:
            try:
                if drain_mode == "get_nowait":
                    q.get_nowait()
                else:  # async get with tiny timeout (event-loop path)
                    await q.get_async(block=True, timeout=0.02)
                total_got += 1
                drained += 1
            except Empty:
                break
            except Exception as e:
                if type(e).__name__ == "Empty" or "Empty" in type(e).__name__:
                    break
                nonempty_errs += 1
                if nonempty_errs <= 2:
                    print(f"  [{label}] drain raised non-Empty {type(e).__name__}: {e}")
                break

        qs = q.qsize()
        if drained == 0 and qs > 0:
            stuck_obs += 1

        producers_done = len(ray.wait(put_futs, timeout=0, num_returns=len(put_futs))[0]) == len(put_futs)
        if producers_done and qs == 0 and total_got >= total_put:
            break
        await asyncio.sleep(0.005)

    final_qsize = q.qsize()
    leaked = total_put - total_got
    verdict = "BUG (items stuck: qsize>0 but undrainable)" if (leaked > 0 and stuck_obs > 5) else "ok"
    print(
        f"[{label}] maxsize={maxsize} producers={num_producers} big_item={big_item} drain={drain_mode}\n"
        f"    put={total_put} got={total_got} leaked={leaked} final_qsize={final_qsize} "
        f"stuck_obs={stuck_obs} nonempty_errs={nonempty_errs}  -> {verdict}"
    )
    with __import__("contextlib").suppress(Exception):
        q.shutdown()
    for p in producers:
        ray.kill(p)
    return leaked, stuck_obs


async def main():
    print("=== ray.util.queue desync reproduction ===\n")
    # F: tau2-faithful — maxsize=0, 16 producers, LARGE complex group items (~0.5MB),
    #    slow trickle (episode latency), get_nowait drain (the real collector pattern).
    await run_case(
        0, 16, 30, True, "get_nowait", "F:maxsize0/16prod/BIG-group/slow/get_nowait (TAU2-FAITHFUL)", slow=True
    )
    # G: same but async get drain.
    await run_case(0, 16, 30, True, "get_async", "G:maxsize0/16prod/BIG-group/slow/get_async", slow=True)
    # H: big items, fast (no trickle), high volume.
    await run_case(0, 16, 60, True, "get_nowait", "H:maxsize0/16prod/BIG-group/fast/get_nowait")
    # A: original simple case (control).
    await run_case(0, 16, 50, False, "get_nowait", "A:maxsize0/16prod/small-item (control)")
    # D: input-queue style (the working pattern).
    await run_case(
        1_000_000, 16, 30, True, "get_async", "D:maxsizeBIG/16prod/BIG-group/get_async (input style)", slow=True
    )


if __name__ == "__main__":
    ray.init(address="auto")
    asyncio.run(main())
