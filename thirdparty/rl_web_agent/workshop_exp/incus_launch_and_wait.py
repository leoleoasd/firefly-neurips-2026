#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
import uuid

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Use Incus management server to launch a container from a base, then poll " "the container's port 80 until it responds with status_code < 400. Repeat N times."))
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8001",
        help="Incus management server URL (default: http://127.0.0.1:8001)",
    )
    parser.add_argument(
        "--base-name",
        required=True,
        help="Base container name to copy from (required)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for a successful response per run (default: 120)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between polls (default: 0.2)",
    )
    return parser.parse_args()


def launch_container(server_url: str, base_name: str, container_name: str) -> str:
    url = f"{server_url.rstrip('/')}/containers/launch"
    payload = {"base_name": base_name, "container_name": container_name}
    response = requests.post(url, json=payload, timeout=60.0)
    if response.status_code != 200:
        raise RuntimeError(f"Launch failed: {response.status_code} {response.text}")
    data = response.json()
    ip_address = data["ip_address"]
    return ip_address


def delete_container(server_url: str, container_name: str) -> None:
    url = f"{server_url.rstrip('/')}/containers/{container_name}"
    # Best-effort; ignore errors
    try:
        requests.delete(url, timeout=60.0)
    except requests.RequestException:
        pass


def run_once(server_url: str, base_name: str, timeout_s: float, sleep_s: float) -> float:
    container_name = f"incus-http-{uuid.uuid4().hex[:12]}"
    ip_address = launch_container(server_url, base_name, container_name)

    start_time = time.monotonic()
    deadline = start_time + timeout_s
    url = f"http://{ip_address}:80/"

    last_status_code: int | None = None

    try:
        while True:
            try:
                response = requests.get(url, timeout=1.0, allow_redirects=False)
                last_status_code = response.status_code
                if response.status_code < 400:
                    elapsed = time.monotonic() - start_time
                    return elapsed
            except requests.RequestException:
                pass

            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for {url}. " f"Last observed status: {last_status_code}")

            time.sleep(sleep_s)
    finally:
        delete_container(server_url, container_name)


def main() -> None:
    args = parse_args()

    times: list[float] = []
    for idx in range(1, args.runs + 1):
        elapsed = run_once(
            server_url=args.server_url,
            base_name=args.base_name,
            timeout_s=args.timeout,
            sleep_s=args.sleep,
        )
        times.append(elapsed)
        print(f"Run {idx}: {elapsed:.3f}s")

    last_n = min(8, len(times))
    avg_last_n = sum(times[-last_n:]) / float(last_n)
    print(f"Average of last {last_n} runs: {avg_last_n:.3f}s")


if __name__ == "__main__":
    main()
