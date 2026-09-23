#!/usr/bin/env python3
"""
Monitor RM (Reward Model) servers via sglang router or registry.

Queries the rm_router's /workers and /get_loads endpoints for real-time stats.
Falls back to direct worker queries if no router is found.

Usage:
    python monitor_rm.py [--interval SECONDS] [--debug]
"""

import argparse
import time
from datetime import datetime

import ray
import requests


def get_router_url() -> str | None:
    """Get RM router URL from sglang registry."""
    ray.init(address="auto", namespace="sglang", ignore_reinit_error=True)
    registry = ray.get_actor("sglang_registry")
    urls = ray.get(registry.get_all.remote("rm_router"))
    return urls[0] if urls else None


def get_worker_urls() -> list[str]:
    """Get RM worker URLs from sglang registry."""
    ray.init(address="auto", namespace="sglang", ignore_reinit_error=True)
    registry = ray.get_actor("sglang_registry")
    urls = ray.get(registry.get_all.remote("rm_worker"))
    return urls


def get_router_metrics(router_url: str, debug: bool = False) -> list[dict]:
    """Fetch worker metrics from the router, then query each worker for details."""
    # Get workers list from router
    try:
        resp = requests.get(f"{router_url}/workers", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if debug:
            print(f"DEBUG /workers: {data}")
        workers = data.get("workers", [])
    except Exception as e:
        return [{"error": f"Failed to fetch /workers: {e}"}]

    results = []
    for worker in workers:
        url = worker["url"]
        m = {
            "url": url,
            "healthy": worker.get("is_healthy", False),
        }

        # Query each worker directly for running/queued/throughput
        try:
            resp = requests.get(f"{url}/get_load", timeout=5)
            resp.raise_for_status()
            load_data = resp.json()
            if debug:
                print(f"DEBUG get_load {url}: {load_data}")

            if isinstance(load_data, list) and load_data:
                m["running"] = sum(d.get("num_reqs", 0) for d in load_data)
                m["queued"] = sum(d.get("num_waiting_reqs", 0) for d in load_data)
                m["tokens"] = sum(d.get("num_tokens", 0) for d in load_data)
            else:
                m["running"] = load_data.get("num_reqs", 0) if isinstance(load_data, dict) else 0
                m["queued"] = load_data.get("num_waiting_reqs", 0) if isinstance(load_data, dict) else 0
                m["tokens"] = load_data.get("num_tokens", 0) if isinstance(load_data, dict) else 0
        except Exception:
            m["running"] = 0
            m["queued"] = 0
            m["tokens"] = 0

        # Query throughput
        try:
            resp = requests.get(f"{url}/get_server_info", timeout=5)
            resp.raise_for_status()
            info = resp.json()
            internal = info.get("internal_states", [])
            m["throughput"] = internal[0].get("last_gen_throughput", 0) if internal else 0
        except Exception as e:
            print(f"ERROR get_server_info {url}: {e}")
            m["throughput"] = 0

        results.append(m)

    return results


def get_direct_metrics(worker_urls: list[str], debug: bool = False) -> list[dict]:
    """Fetch metrics directly from each worker via /get_load."""
    results = []
    for url in worker_urls:
        try:
            resp = requests.get(f"{url}/get_load", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if debug:
                print(f"DEBUG get_load {url}: {data}")

            if isinstance(data, list) and data:
                num_reqs = sum(d.get("num_reqs", 0) for d in data)
                num_waiting = sum(d.get("num_waiting_reqs", 0) for d in data)
                num_tokens = sum(d.get("num_tokens", 0) for d in data)
            else:
                num_reqs = data.get("num_reqs", 0) if isinstance(data, dict) else 0
                num_waiting = data.get("num_waiting_reqs", 0) if isinstance(data, dict) else 0
                num_tokens = data.get("num_tokens", 0) if isinstance(data, dict) else 0

            results.append(
                {
                    "url": url,
                    "running": num_reqs,
                    "queued": num_waiting,
                    "tokens": num_tokens,
                }
            )
        except Exception as e:
            results.append({"url": url, "error": str(e)})
    return results


def monitor(interval: float = 2.0, debug: bool = False):
    """Monitor RM servers continuously."""
    router_url = get_router_url()
    use_router = router_url is not None

    if use_router:
        print(f"Using RM router: {router_url}\n")
    else:
        worker_urls = get_worker_urls()
        if not worker_urls:
            print("No RM router or workers found in registry")
            return
        print(f"No router found, querying {len(worker_urls)} worker(s) directly\n")

    first_run = True
    while True:
        print("\033[2J\033[H", end="")  # Clear screen
        print(f"RM Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Refresh interval: {interval}s")

        if use_router:
            print(f"Router: {router_url}\n")
            print(f"{'Worker':<45} {'Running':>8} {'Queued':>8} {'Tokens':>10} {'Tput':>12} {'Tput/Req':>10}")
            print("-" * 100)

            metrics = get_router_metrics(router_url, debug=debug and first_run)
            total_running = 0
            total_queued = 0
            total_tokens = 0
            total_throughput = 0.0
            for m in metrics:
                if "error" in m:
                    print(f"ERROR: {m['error']}")
                    continue
                total_running += m["running"]
                total_queued += m["queued"]
                total_tokens += m["tokens"]
                total_throughput += m["throughput"]
                tput_per_req = m["throughput"] / m["running"] if m["running"] > 0 else 0
                print(
                    f"{m['url']:<45} "
                    f"{m['running']:>8} "
                    f"{m['queued']:>8} "
                    f"{m['tokens']:>10} "
                    f"{m['throughput']:>12.1f} "
                    f"{tput_per_req:>10.1f}"
                )
            print("-" * 100)
            total_tput_per_req = total_throughput / total_running if total_running > 0 else 0
            print(
                f"{'TOTAL':<45} "
                f"{total_running:>8} "
                f"{total_queued:>8} "
                f"{total_tokens:>10} "
                f"{total_throughput:>12.1f} "
                f"{total_tput_per_req:>10.1f}"
            )
        else:
            print("Direct worker queries\n")
            print(f"{'Worker':<45} {'Running':>8} {'Queued':>8} {'Tokens':>10}")
            print("-" * 75)

            metrics = get_direct_metrics(worker_urls, debug=debug and first_run)
            for m in metrics:
                if "error" in m:
                    print(f"{m['url']:<45} ERROR: {m['error']}")
                    continue
                print(f"{m['url']:<45} {m['running']:>8} {m['queued']:>8} {m['tokens']:>10}")

        first_run = False
        print("\nPress Ctrl+C to exit")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor sglang RM servers")
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Print raw API response on first run",
    )
    args = parser.parse_args()

    try:
        monitor(interval=args.interval, debug=args.debug)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
