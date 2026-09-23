#!/usr/bin/env python3
"""
RM Router - Launches a sglang router for reward model servers.

Runs as a regular Ray job (dies when the job is stopped).
Registers itself to sglang_registry with key 'rm_router'.

Usage:
python scripts/rm_router.py
"""

import logging
import os
import subprocess
import sys
import time

import requests

logger = logging.getLogger(__name__)


def _get_free_port() -> int:
    """Get a free port, setting SO_REUSEADDR to avoid race conditions."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


ROUTER_REGISTRY_KEY = "rm_router"


def wait_for_router(url: str, timeout: int = 30) -> bool:
    """Wait for the router to be healthy."""
    for _ in range(timeout):
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _run_as_ray_job():
    """Submit this script as a Ray job."""
    cmd = [
        "ray",
        "job",
        "submit",
        "--address=auto",
        "--no-wait",
        "--",
        sys.executable,
        "scripts/rm_router.py",
        "--_ray-job",
    ]
    print(f"Submitting Ray job: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


def _run_router():
    """Run the router inside the Ray job."""
    import ray
    from slime.utils.misc import get_current_node_ip

    from shared.sglang_registry import get_or_create_registry

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ray.init(address="auto", namespace="sglang")

    node_ip = get_current_node_ip()
    port = _get_free_port()
    prometheus_port = _get_free_port()
    url = f"http://{node_ip}:{port}"

    # Launch sglang router as a subprocess, inherit stdout/stderr
    cmd = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--prometheus-port",
        str(prometheus_port),
        "--policy",
        "random",
    ]

    logger.info(f"Starting router: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    # Wait for router to be ready
    if not wait_for_router(url):
        if proc.poll() is not None:
            logger.error(f"Router process died with exit code {proc.returncode}")
        else:
            logger.error("Router did not become healthy in time")
            proc.terminate()
        sys.exit(1)

    # Register to sglang_registry
    registry = get_or_create_registry("sglang_registry")
    ray.get(registry.add.remote(ROUTER_REGISTRY_KEY, url))
    logger.info(f"Router registered: {url}")

    # Add existing rm_worker entries to the router
    worker_urls = ray.get(registry.get_all.remote("rm_worker"))
    for worker_url in worker_urls:
        try:
            health_resp = requests.get(f"{worker_url}/health", timeout=5)
            health_resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Worker {worker_url} is not healthy, skipping: {e}")
            continue
        try:
            resp = requests.post(
                f"{url}/workers",
                json={"url": worker_url},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(f"Added existing worker to router: {worker_url}")
        except Exception as e:
            logger.error(f"Failed to add worker {worker_url} to router: {e}")

    # Wait for subprocess to exit (keeps the ray job alive)
    try:
        proc.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted, stopping router...")
        proc.terminate()
        proc.wait(timeout=10)

    logger.info(f"Router exited with code {proc.returncode}")
    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    if "--_ray-job" in sys.argv:
        _run_router()
    else:
        _run_as_ray_job()
