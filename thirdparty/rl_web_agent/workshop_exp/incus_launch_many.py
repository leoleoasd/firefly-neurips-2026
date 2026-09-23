#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
    ip_address: str | None
    http_url: str | None
    elapsed_s: float | None
    status: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Launch many Incus containers through the management server with bounded concurrency, " "measure time to first HTTP response (<400) on port 80, write per-container timings to JSON, " "then delete all containers at the end."))
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
        default=Path("incus_launch_report.json"),
        help="Path to output JSON report (default: incus_launch_report.json)",
    )
    return parser.parse_args()


def launch_container(server_url: str, base_name: str, container_name: str) -> tuple[str | None, str | None]:
    url = f"{server_url.rstrip('/')}/containers/launch"
    payload: dict[str, Any] = {"base_name": base_name, "container_name": container_name}
    try:
        response = requests.post(url, json=payload, timeout=120.0)
    except requests.RequestException as exc:
        return None, f"request_error: {exc}"

    if response.status_code != 200:
        return None, f"launch_failed: {response.status_code} {response.text}"

    try:
        data = response.json()
    except ValueError as exc:
        return None, f"invalid_json: {exc}"

    try:
        ip_address = data["ip_address"]
    except KeyError:
        return None, "missing_ip_address"

    if not isinstance(ip_address, str):
        return None, "invalid_ip_type"

    return ip_address, None


def wait_http_ready(ip_address: str, timeout_s: float, sleep_s: float) -> tuple[float | None, str | None]:
    start_time = time.monotonic()
    deadline = start_time + timeout_s
    url = f"http://{ip_address}:80/"
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


def delete_container(server_url: str, container_name: str) -> None:
    url = f"{server_url.rstrip('/')}/containers/{container_name}"
    try:
        requests.delete(url, timeout=60.0)
    except requests.RequestException:
        pass


def launch_and_measure(
    server_url: str,
    base_name: str,
    timeout_s: float,
    sleep_s: float,
) -> LaunchResult:
    container_name = f"incus-batch-{uuid.uuid4().hex[:12]}"

    overall_start = time.monotonic()
    ip_address, launch_error = launch_container(server_url, base_name, container_name)
    if launch_error is not None:
        return LaunchResult(
            container_name=container_name,
            ip_address=None,
            http_url=None,
            elapsed_s=None,
            status="launch_error",
            error=launch_error,
        )

    if ip_address == "unknown" or ip_address is None or len(ip_address.strip()) == 0:
        return LaunchResult(
            container_name=container_name,
            ip_address=ip_address,
            http_url=None,
            elapsed_s=None,
            status="no_ip",
            error="ip_address_unavailable",
        )

    elapsed_s, wait_error = wait_http_ready(ip_address, timeout_s=timeout_s, sleep_s=sleep_s)
    if wait_error is not None:
        return LaunchResult(
            container_name=container_name,
            ip_address=ip_address,
            http_url=f"http://{ip_address}:80/",
            elapsed_s=None,
            status="timeout",
            error=wait_error,
        )

    total_elapsed_s = time.monotonic() - overall_start
    return LaunchResult(
        container_name=container_name,
        ip_address=ip_address,
        http_url=f"http://{ip_address}:80/",
        elapsed_s=total_elapsed_s,
        status="ok",
        error=None,
    )


def main() -> None:
    args = parse_args()

    server_url: str = args.server_url
    base_name: str = args.base_name
    total_count: int = args.count
    max_concurrency: int = args.concurrency
    timeout_s: float = args.timeout
    sleep_s: float = args.sleep
    output_path: Path = args.output

    print(f"Launching {total_count} containers from base '{base_name}' with concurrency={max_concurrency} " f"against {server_url}")

    results: list[LaunchResult] = []

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(launch_and_measure, server_url, base_name, timeout_s, sleep_s) for _ in range(total_count)]

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
                "ip_address": r.ip_address,
                "http_url": r.http_url,
                "elapsed_s": r.elapsed_s,
                "status": r.status,
                "error": r.error,
            }
        )

    num_ok = sum(1 for r in results if r.status == "ok")
    num_timeout = sum(1 for r in results if r.status == "timeout")
    num_no_ip = sum(1 for r in results if r.status == "no_ip")
    num_launch_error = sum(1 for r in results if r.status == "launch_error")

    report: dict[str, Any] = {
        "server_url": server_url,
        "base_name": base_name,
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
            "no_ip": num_no_ip,
            "launch_error": num_launch_error,
        },
        "results": json_results,
    }

    # Write report before deletion as requested
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote report to {output_path}")

    # Delete all containers that were created (best-effort)
    containers_to_delete = [r.container_name for r in results]
    print(f"Deleting {len(containers_to_delete)} containers...")
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        delete_futs = [executor.submit(delete_container, server_url, name) for name in containers_to_delete]
        for _ in as_completed(delete_futs):
            pass
    print("Deletion requests completed.")


if __name__ == "__main__":
    main()
