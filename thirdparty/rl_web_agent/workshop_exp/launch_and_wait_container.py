#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import time
import uuid

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Launch a Docker container (HTTP server), then poll its HTTP port until " "it responds with status_code < 400. Reports elapsed time."))
    parser.add_argument(
        "--image",
        default="nginx:alpine",
        help="Docker image to run (default: nginx:alpine)",
    )
    parser.add_argument(
        "--container-port",
        type=int,
        default=80,
        help="Container HTTP port to expose (default: 80)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs to execute (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for a successful response (default: 120)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between polls (default: 0.2)",
    )
    return parser.parse_args()


def run_once(image: str, container_port: int, timeout_s: float, sleep_s: float) -> float:
    # Pick a random host port and retry a few times if docker run fails (e.g., port in use)
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        host_port = random.randint(20000, 60000)
        container_name = f"temp_http_{uuid.uuid4().hex[:12]}"
        run_cmd = f"docker run -d -p {host_port}:{container_port} " f"--name {container_name} {image}"

        exit_code = os.system(run_cmd)
        if exit_code != 0:
            if attempt == max_attempts:
                raise SystemExit(f"docker run failed with exit code {exit_code}")
            continue

        start_time = time.monotonic()
        deadline = start_time + timeout_s
        url = f"http://127.0.0.1:{host_port}/"
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
            os.system(f"docker stop {container_name} > /dev/null 2>&1")
            os.system(f"docker rm {container_name} > /dev/null 2>&1")


def main() -> None:
    args = parse_args()

    times: list[float] = []
    for idx in range(1, args.runs + 1):
        elapsed = run_once(
            image=args.image,
            container_port=args.container_port,
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
