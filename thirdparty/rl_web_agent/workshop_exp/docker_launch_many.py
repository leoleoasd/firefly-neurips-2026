#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class LaunchResult:
    container_name: str
    host_port: int | None
    http_url: str | None
    elapsed_s: float | None
    status: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Launch many Docker containers with bounded concurrency, measure time to first HTTP " "response (<400) on port mapping, write per-container timings to JSON, then delete all."))
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
        "--count",
        type=int,
        default=200,
        help="Total number of containers to launch (default: 200)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent launches (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-container timeout in seconds to wait for HTTP (<400) (default: 180)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between HTTP polls (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docker_launch_report.json"),
        help="Path to output JSON report (default: docker_launch_report.json)",
    )
    return parser.parse_args()


def run_container(image: str, container_port: int, container_name: str, host_port: int) -> tuple[bool, str | None]:
    run_cmd = f"docker run -d -p {host_port}:{container_port} " f"--name {container_name} {image}"
    exit_code = os.system(run_cmd)
    if exit_code != 0:
        return False, f"docker_run_exit_{exit_code}"
    return True, None


def wait_http_ready(host_port: int, timeout_s: float, sleep_s: float) -> tuple[float | None, str | None]:
    start_time = time.monotonic()
    deadline = start_time + timeout_s
    url = f"http://127.0.0.1:{host_port}/"
    last_status_code: int | None = None

    while True:
        try:
            response = requests.get(url, timeout=1.0, allow_redirects=False)
            last_status_code = response.status_code
            if response.status_code < 400:
                elapsed = time.monotonic() - start_time
                return elapsed, None
        except requests.RequestException:
            pass

        if time.monotonic() > deadline:
            err = f"timeout_waiting_http: waited {timeout_s:.1f}s for {url}; " f"last_status={last_status_code}"
            return None, err

        time.sleep(sleep_s)


def stop_and_remove(container_name: str) -> None:
    os.system(f"docker stop {container_name} > /dev/null 2>&1")
    os.system(f"docker rm {container_name} > /dev/null 2>&1")


def launch_and_measure(
    image: str,
    container_port: int,
    timeout_s: float,
    sleep_s: float,
) -> LaunchResult:
    container_name = f"docker-batch-{uuid.uuid4().hex[:12]}"

    # Start timer BEFORE attempting docker run
    overall_start = time.monotonic()

    # Try a few host ports for robustness under concurrency
    max_attempts = 20
    host_port: int | None = None
    run_error: str | None = None
    for _ in range(max_attempts):
        candidate_port = random.randint(20000, 60000)
        ok, err = run_container(image=image, container_port=container_port, container_name=container_name, host_port=candidate_port)
        if ok:
            host_port = candidate_port
            break
        run_error = err

    if host_port is None:
        return LaunchResult(
            container_name=container_name,
            host_port=None,
            http_url=None,
            elapsed_s=None,
            status="run_error",
            error=run_error,
        )

    # Wait for HTTP readiness
    _, wait_error = wait_http_ready(host_port, timeout_s=timeout_s, sleep_s=sleep_s)
    if wait_error is not None:
        return LaunchResult(
            container_name=container_name,
            host_port=host_port,
            http_url=f"http://127.0.0.1:{host_port}/",
            elapsed_s=None,
            status="timeout",
            error=wait_error,
        )

    total_elapsed_s = time.monotonic() - overall_start
    return LaunchResult(
        container_name=container_name,
        host_port=host_port,
        http_url=f"http://127.0.0.1:{host_port}/",
        elapsed_s=total_elapsed_s,
        status="ok",
        error=None,
    )


def main() -> None:
    args = parse_args()

    image: str = args.image
    container_port: int = args.container_port
    total_count: int = args.count
    max_concurrency: int = args.concurrency
    timeout_s: float = args.timeout
    sleep_s: float = args.sleep
    output_path: Path = args.output

    print(f"Launching {total_count} Docker containers from image '{image}' with concurrency={max_concurrency}")

    results: list[LaunchResult] = []

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(launch_and_measure, image, container_port, timeout_s, sleep_s) for _ in range(total_count)]

        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result.status == "ok" and result.elapsed_s is not None:
                print(f"[{idx:03d}/{total_count}] {result.container_name} ready in {result.elapsed_s:.3f}s")
            else:
                print(f"[{idx:03d}/{total_count}] {result.container_name} {result.status}: {result.error}")

    end_all = time.time()

    # Prepare JSON report
    json_results: list[dict[str, Any]] = []
    for r in results:
        json_results.append(
            {
                "container_name": r.container_name,
                "host_port": r.host_port,
                "http_url": r.http_url,
                "elapsed_s": r.elapsed_s,
                "status": r.status,
                "error": r.error,
            }
        )

    num_ok = sum(1 for r in results if r.status == "ok")
    num_timeout = sum(1 for r in results if r.status == "timeout")
    num_run_error = sum(1 for r in results if r.status == "run_error")

    report: dict[str, Any] = {
        "image": image,
        "container_port": container_port,
        "count": total_count,
        "concurrency": max_concurrency,
        "timeout_s": timeout_s,
        "sleep_s": sleep_s,
        "started_at": start_all,
        "finished_at": end_all,
        "duration_s": end_all - start_all,
        "summary": {
            "ok": num_ok,
            "timeout": num_timeout,
            "run_error": num_run_error,
        },
        "results": json_results,
    }

    # Write report before deletion
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote report to {output_path}")

    # Delete all containers that were created (best-effort)
    containers_to_delete = [r.container_name for r in results]
    print(f"Deleting {len(containers_to_delete)} containers...")
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        delete_futs = [executor.submit(stop_and_remove, name) for name in containers_to_delete]
        for _ in as_completed(delete_futs):
            pass
    print("Deletion requests completed.")


if __name__ == "__main__":
    main()
