#!/usr/bin/env python3
"""
Incus Container Management HTTP Server (throttled)

- Throttles container creation so only one new container can start every N seconds.
- Feature flag to enable/disable throttling via env vars or runtime API.
- Uses the Incus REST API over Unix socket instead of spawning CLI subprocesses,
  eliminating the FD-leak / zombie-process issues inherent in asyncio subprocess
  management under Quart.

API reference: https://linuxcontainers.org/incus/docs/main/rest-api/
"""

import asyncio
import contextlib
import json
import logging
import os

import httpx
from quart import Quart, jsonify, request


# =========================================================================
# Incus REST API client (talks to the daemon over Unix socket)
# =========================================================================
INCUS_SOCKET = "/var/lib/incus/unix.socket"
INCUS_POOL = "default"

# Per-request HTTP timeout (for the socket read/write itself).
# Sync Incus endpoints respond in milliseconds; this is a safety net.
INCUS_HTTP_TIMEOUT = 30.0

# Operation wait timeouts by operation class (seconds).
INCUS_WAIT_STATE_CHANGE = 120.0  # start / stop
INCUS_WAIT_COPY = 600.0  # copy (ZFS clone can be slow on large containers)
INCUS_WAIT_DELETE = 120.0  # delete


class IncusError(Exception):
    """Base exception for Incus API errors."""

    def __init__(self, status_code: int, message: str, response_body: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"Incus error {status_code}: {message}")


class IncusNotFound(IncusError):
    """Raised when the target resource (instance, volume, operation) is 404."""

    def __init__(self, message: str, response_body: dict | None = None):
        super().__init__(404, message, response_body)


class _IncusAPI:
    """Async client for the Incus REST API over a Unix socket."""

    def __init__(self, socket_path: str = INCUS_SOCKET):
        transport = httpx.AsyncHTTPTransport(uds=socket_path)
        self._client = httpx.AsyncClient(
            transport=transport,
            base_url="http://incus",
            timeout=httpx.Timeout(INCUS_HTTP_TIMEOUT, connect=10.0),
        )
        self._logger = logging.getLogger(f"{__name__}.incus_api")

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        http_timeout: float = INCUS_HTTP_TIMEOUT,
        wait_timeout_s: float | None = None,
    ) -> dict | None:
        """
        Send a request to the Incus API and handle sync/async/error responses.

        For async operations (HTTP 202), automatically waits for the operation
        to complete using the /wait endpoint.

        Returns the ``metadata`` dict from the response (may be None for
        operations that produce no metadata on completion).
        """
        resp = await self._client.request(
            method,
            path,
            params=params,
            json=json_body,
            timeout=http_timeout,
        )

        # Non-JSON responses (shouldn't happen, but guard)
        try:
            body = resp.json()
        except Exception:
            if resp.status_code >= 400:
                raise IncusError(resp.status_code, resp.text)
            return None

        resp_type = body["type"]

        # --- error response ---
        if resp_type == "error" or resp.status_code >= 400:
            error_code = body["error_code"]
            error_msg = body["error"]
            if error_code == 404 or resp.status_code == 404:
                raise IncusNotFound(error_msg, body)
            raise IncusError(error_code, error_msg, body)

        # --- sync response ---
        if resp_type == "sync":
            return body["metadata"]

        # --- async response (HTTP 202) ---
        if resp_type == "async":
            op_path = body["operation"]
            if not op_path:
                # Shouldn't happen, but treat as sync success
                return body["metadata"]
            timeout = wait_timeout_s if wait_timeout_s is not None else INCUS_WAIT_STATE_CHANGE
            return await self._await_operation(op_path, timeout)

        # Unknown type — treat as error
        raise IncusError(0, f"Unknown response type: {resp_type}", body)

    async def _await_operation(self, op_path: str, wait_timeout_s: float) -> dict | None:
        """
        Block until an Incus async operation completes.

        Uses the server-side blocking ``/wait`` endpoint so we don't need to
        poll.  The ``timeout`` query parameter tells the Incus daemon how long
        to block before returning a timeout error.

        If the operation has already completed and been garbage-collected by
        Incus (404), we optimistically treat it as success — the race window
        between issuing the request and calling /wait is sub-millisecond in
        practice.
        """
        # The wait endpoint itself can take a long time (up to wait_timeout_s),
        # so give httpx a generous read timeout.
        http_timeout = wait_timeout_s + 30.0

        resp = await self._client.get(
            f"{op_path}/wait",
            params={"timeout": str(int(wait_timeout_s))},
            timeout=http_timeout,
        )

        try:
            body = resp.json()
        except Exception:
            if resp.status_code >= 400:
                raise IncusError(resp.status_code, resp.text)
            return None

        # Operation already gone — optimistic success
        if body["type"] == "error" and body["error_code"] == 404:
            self._logger.warning(f"Operation {op_path} already gone (404), treating as success")
            return None

        if body["type"] == "error":
            raise IncusError(body["error_code"], body["error"], body)

        # Successful wait response: body is sync, metadata is the operation object
        metadata = body["metadata"]
        if metadata is None:
            return None

        op_status_code = metadata["status_code"]
        op_err = metadata["err"]

        if op_status_code == 200:
            # Success
            return metadata

        # Operation failed
        raise IncusError(op_status_code, op_err or f"Operation failed with status_code {op_status_code}", body)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def server_info(self) -> dict:
        """GET /1.0 — server info including environment.server_version."""
        return await self._request("GET", "/1.0")

    async def get_instance(self, name: str) -> dict:
        """
        GET /1.0/instances/<name>?recursion=1

        Returns the full instance metadata including state, network, etc.
        Raises IncusNotFound if the instance does not exist.
        """
        return await self._request("GET", f"/1.0/instances/{name}", params={"recursion": "1"})

    async def list_instance_names(self) -> list[str]:
        """
        GET /1.0/instances

        Returns a list of instance names (extracted from URL paths).
        """
        metadata = await self._request("GET", "/1.0/instances")
        # metadata is a list of URL strings like "/1.0/instances/shopping"
        if not metadata:
            return []
        return [url.rsplit("/", 1)[-1] for url in metadata]

    async def change_instance_state(
        self,
        name: str,
        action: str,
        *,
        force: bool = False,
        timeout_s: int = 30,
        stateful: bool = False,
        wait_timeout_s: float = INCUS_WAIT_STATE_CHANGE,
    ) -> None:
        """
        PUT /1.0/instances/<name>/state

        action: "start" | "stop" | "restart" | "freeze" | "unfreeze"
        """
        await self._request(
            "PUT",
            f"/1.0/instances/{name}/state",
            json_body={
                "action": action,
                "timeout": timeout_s,
                "force": force,
                "stateful": stateful,
            },
            wait_timeout_s=wait_timeout_s,
        )

    async def copy_instance(
        self,
        source: str,
        dest: str,
        *,
        wait_timeout_s: float = INCUS_WAIT_COPY,
    ) -> None:
        """
        POST /1.0/instances — copy (clone) an existing instance.

        The source instance should be stopped for data consistency.
        """
        await self._request(
            "POST",
            "/1.0/instances",
            json_body={
                "name": dest,
                "source": {
                    "type": "copy",
                    "source": source,
                },
            },
            wait_timeout_s=wait_timeout_s,
        )

    async def delete_instance(
        self,
        name: str,
        *,
        wait_timeout_s: float = INCUS_WAIT_DELETE,
    ) -> None:
        """
        DELETE /1.0/instances/<name>

        The instance must be stopped first; otherwise Incus returns 400.
        Raises IncusNotFound if the instance does not exist (caller can
        treat this as idempotent success).
        """
        await self._request(
            "DELETE",
            f"/1.0/instances/{name}",
            wait_timeout_s=wait_timeout_s,
        )

    async def storage_volume_exists(
        self,
        name: str,
        *,
        pool: str = INCUS_POOL,
        volume_type: str = "container",
    ) -> bool:
        """
        Check whether a storage volume exists in the given pool.

        GET /1.0/storage-pools/<pool>/volumes/<type>/<name>
        Returns True (200) or False (404).
        """
        try:
            await self._request("GET", f"/1.0/storage-pools/{pool}/volumes/{volume_type}/{name}")
            return True
        except IncusNotFound:
            return False


# ----------------------------------------
# Config: Feature flag + interval
# ----------------------------------------
def _env_truthy(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


THROTTLE_ENABLED = _env_truthy(os.getenv("THROTTLE_CREATE_CONTAINERS"), True)
try:
    THROTTLE_SECONDS = float(os.getenv("THROTTLE_CREATE_INTERVAL_SECONDS", "3"))
except ValueError:
    THROTTLE_SECONDS = 3.0

# Global throttle state
_throttle_lock = asyncio.Lock()
_last_launch_time = 0.0  # loop.time() of the last launch slot

# ----------------------------------------
# Container lifetime (auto-expiry)
# ----------------------------------------
# Default container lifetime if the client does not specify one. After this
# many seconds elapse past a successful launch, the server force-deletes the
# container (equivalent to calling DELETE /containers/<name>).
DEFAULT_LIFETIME_SECONDS = 2 * 60 * 60  # 2 hours

# Registry of outstanding reaper tasks keyed by container name, so we can
# cancel them on re-launch with the same name or on an explicit DELETE.
_reapers: dict[str, asyncio.Task] = {}

# ----------------------------------------
# Logging / App
# ----------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Quart(__name__)

# ----------------------------------------
# Incus client (global, initialized on startup)
# ----------------------------------------
_client: _IncusAPI | None = None


@app.before_serving
async def _startup():
    global _client
    _client = _IncusAPI()
    logger.info("Incus API client initialized")


@app.after_serving
async def _shutdown():
    global _client
    if _client:
        await _client.aclose()
        _client = None
        logger.info("Incus API client closed")


# ----------------------------------------
# Safe subprocess wrapper (ZFS only)
# ----------------------------------------
async def run_subprocess_safe(cmd: list[str], *, timeout_s: float = 10.0) -> tuple[int, str, str]:
    """
    Run a subprocess with proper timeout and cleanup to prevent FD leaks.

    Only used for `zfs list` in health checks — all Incus operations go
    through IncusClient instead.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        return process.returncode, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        logger.warning(f"Subprocess timed out after {timeout_s}s: {cmd}")
        return 1, "", f"Timed out after {timeout_s}s"
    finally:
        # If still running (timeout or exception), kill and reap so pipe fds are released.
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()


# ----------------------------------------
# Helpers
# ----------------------------------------
async def _maybe_throttle():
    """
    Enforce a minimum interval between container launches.
    This serializes launches and spaces them at least THROTTLE_SECONDS apart.
    """
    global _last_launch_time
    if not THROTTLE_ENABLED:
        return

    async with _throttle_lock:
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait = THROTTLE_SECONDS - (now - _last_launch_time)
        if wait > 0:
            logger.info(f"Throttling: waiting {wait:.2f}s before next container launch")
            await asyncio.sleep(wait)
        # Mark the slot as used at the time we *start* this launch
        _last_launch_time = loop.time()


def _read_free_memory_ratio() -> float:
    """
    Read /proc/meminfo and return the fraction of available memory (0.0-1.0).

    Uses MemAvailable (kernel's estimate of memory available for starting new
    applications, without swapping) over MemTotal. Linux-only.

    Raises:
        RuntimeError: if /proc/meminfo cannot be parsed.
    """
    values: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            # Values are like "12345 kB"; take the integer part
            tokens = parts[1].strip().split()
            if not tokens:
                continue
            try:
                values[key] = int(tokens[0])
            except ValueError:
                continue

    if "MemTotal" not in values or "MemAvailable" not in values or values["MemTotal"] <= 0:
        raise RuntimeError(f"Failed to parse MemTotal/MemAvailable from /proc/meminfo: {values}")

    return values["MemAvailable"] / values["MemTotal"]


# HTTP status code returned when the server refuses to clone a container due to
# low host memory. 507 = Insufficient Storage (closest standard code for
# "server resource exhausted, try later").
INSUFFICIENT_MEMORY_STATUS = 507
INSUFFICIENT_MEMORY_REASON = "insufficient_memory"
# Minimum fraction of MemAvailable/MemTotal required to accept a launch.
MIN_FREE_MEMORY_RATIO = 0.10


async def get_container_ip(container_name: str) -> str | None:
    """Get IP address of a running container via Incus API."""
    try:
        data = await _client.get_instance(container_name)
    except IncusNotFound:
        return None
    except IncusError as e:
        logger.error(f"Failed to get container info for {container_name}: {e.message}")
        return None

    network = data["state"]["network"]
    if network is None:
        # Container is stopped — no network info
        return None

    for interface_name, interface_info in network.items():
        if interface_name == "lo":
            continue
        for addr in interface_info["addresses"]:
            if addr["family"] == "inet" and addr["scope"] == "global":
                return addr["address"]
    return None


async def get_container_status(container_name: str) -> str | None:
    """Get status of a container via Incus API. Returns lowercase status or None."""
    try:
        data = await _client.get_instance(container_name)
    except IncusNotFound:
        return None
    except IncusError as e:
        logger.error(f"Failed to get container status for {container_name}: {e.message}")
        return None

    return data["status"].lower()


# ----------------------------------------
# Routes
# ----------------------------------------
@app.route("/containers/launch", methods=["POST"])
async def launch_container():
    """
    Launch a new container by copying from base and starting it.
    Throttled: ensures at least THROTTLE_SECONDS between launches when enabled.

    Request body:
        base_name (str): required, the base container to clone from.
        container_name (str): required, the new container name.
        lifetime_seconds (number, optional): how long the container is allowed
            to live after a successful launch, in seconds. After this elapses
            the server force-deletes (kills + removes) the container. Defaults
            to DEFAULT_LIFETIME_SECONDS (2 hours). Must be > 0.
    """
    data = await request.get_json()
    base_name = data["base_name"]
    container_name = data["container_name"]

    # Parse / validate lifetime_seconds (optional).
    if "lifetime_seconds" in data and data["lifetime_seconds"] is not None:
        try:
            lifetime_seconds = float(data["lifetime_seconds"])
        except (TypeError, ValueError):
            return jsonify({"error": "lifetime_seconds must be a number"}), 400
        if lifetime_seconds <= 0:
            return jsonify({"error": "lifetime_seconds must be > 0"}), 400
    else:
        lifetime_seconds = float(DEFAULT_LIFETIME_SECONDS)

    # Apply throttle first thing
    await _maybe_throttle()

    # Refuse to clone if host memory is too low. This is checked after the
    # throttle (so slots aren't burned on rejected requests only in bursty
    # moments) but before any incus work is done.
    free_ratio = _read_free_memory_ratio()
    if free_ratio < MIN_FREE_MEMORY_RATIO:
        logger.warning(f"Refusing to launch {container_name}: free memory {free_ratio:.1%} < {MIN_FREE_MEMORY_RATIO:.0%}")
        return (
            jsonify(
                {
                    "error": f"Insufficient host memory: {free_ratio:.1%} available, need >= {MIN_FREE_MEMORY_RATIO:.0%}",
                    "reason": INSUFFICIENT_MEMORY_REASON,
                    "free_memory_ratio": free_ratio,
                    "min_free_memory_ratio": MIN_FREE_MEMORY_RATIO,
                }
            ),
            INSUFFICIENT_MEMORY_STATUS,
        )

    logger.info(f"Launching container {container_name} from base {base_name}")

    # Step 1: Stop base container if it's running (required for data consistency —
    # ensures in-memory state is flushed to disk before the ZFS snapshot/clone).
    base_status = await get_container_status(base_name)
    if base_status == "running":
        logger.info(f"Stopping base container {base_name} before copying")
        try:
            await _client.change_instance_state(base_name, "stop", timeout_s=30, wait_timeout_s=120)
        except IncusError as e:
            logger.error(f"Failed to stop base container: {e.message}")
            return jsonify({"error": f"Failed to stop base container {base_name}: {e.message}"}), 500

    # Step 2: Copy base container
    try:
        await _client.copy_instance(base_name, container_name)
    except IncusError as e:
        logger.error(f"Failed to copy container: {e.message}")
        return jsonify({"error": f"Failed to copy container from {base_name}: {e.message}"}), 500

    logger.info(f"Successfully copied {base_name} to {container_name}")

    # Step 3: Start the container
    try:
        await _client.change_instance_state(container_name, "start", timeout_s=30, wait_timeout_s=120)
    except IncusError as e:
        logger.error(f"Failed to start container: {e.message}")
        # Cleanup best-effort
        try:
            await _client.delete_instance(container_name)
        except (IncusError, IncusNotFound):
            pass
        return jsonify({"error": f"Failed to start container {container_name}: {e.message}"}), 500

    logger.info(f"Successfully started container {container_name}")

    # Step 4: Wait for container to get IP address
    max_retries = 30  # 30 seconds timeout
    ip_address = None
    for _ in range(max_retries):
        ip_address = await get_container_ip(container_name)
        if ip_address:
            break
        await asyncio.sleep(1)

    if not ip_address:
        logger.error(f"Container {container_name} started but no IP address found")
        ip_address = "unknown"  # non-fatal

    # Arm the lifetime reaper: force-delete the container after lifetime_seconds.
    _schedule_reaper(container_name, lifetime_seconds)
    expires_at = asyncio.get_running_loop().time() + lifetime_seconds

    logger.info(f"Container {container_name} launched successfully with IP {ip_address} (lifetime {lifetime_seconds:.0f}s)")
    return jsonify(
        {
            "container_name": container_name,
            "ip_address": ip_address,
            "status": "running",
            "lifetime_seconds": lifetime_seconds,
            "expires_at_monotonic": expires_at,
        }
    )


async def _force_delete_instance(name: str) -> None:
    """
    Force-stop then delete an instance. Equivalent to `incus rm -f <name>`.

    Idempotent: if the instance is already gone, silently succeeds.
    """
    # Force-stop first (Incus DELETE requires the instance to be stopped).
    try:
        await _client.change_instance_state(name, "stop", force=True, timeout_s=30, wait_timeout_s=120)
    except IncusNotFound:
        return  # already gone
    except IncusError as e:
        # "Instance is not running" or similar — benign, proceed to delete
        logger.debug(f"[{name}] force-stop before delete: {e.message}")

    try:
        await _client.delete_instance(name)
    except IncusNotFound:
        pass  # already gone


async def _retry_delete_container(container_name: str, max_attempts: int = 100, delay_seconds: int = 10) -> None:
    """
    Try to force-delete a container up to max_attempts times.
    First attempt happens immediately; subsequent attempts wait delay_seconds.
    """
    for attempt in range(1, max_attempts + 1):
        logger.info(f"[{container_name}] Delete attempt {attempt}/{max_attempts}")
        try:
            await _force_delete_instance(container_name)
            logger.info(f"[{container_name}] Successfully removed container on attempt {attempt}")
            return
        except IncusError as e:
            logger.warning(f"[{container_name}] Failed to remove container on attempt {attempt}: {e.message}")

        if attempt < max_attempts:
            await asyncio.sleep(delay_seconds)

    logger.error(f"[{container_name}] Exhausted {max_attempts} delete attempts. Container may remain.")


async def _reap_after_lifetime(container_name: str, lifetime_seconds: float) -> None:
    """
    Wait `lifetime_seconds`, then force-delete the container via the standard
    retry-delete path. Designed to be scheduled as a background task by
    launch_container.

    Cancellation-safe: if another launch reuses the same container name, or if
    an explicit DELETE arrives first, the owner is expected to cancel this
    task. asyncio.CancelledError propagates out without attempting deletion.
    """
    logger.info(f"[{container_name}] Reaper armed: will force-delete in {lifetime_seconds:.0f}s")
    try:
        await asyncio.sleep(lifetime_seconds)
    except asyncio.CancelledError:
        logger.info(f"[{container_name}] Reaper cancelled before expiry")
        raise

    logger.info(f"[{container_name}] Lifetime expired after {lifetime_seconds:.0f}s; force-deleting")
    try:
        await _retry_delete_container(container_name, max_attempts=100, delay_seconds=10)
    finally:
        # Drop self from the registry so we don't hold a reference to a
        # completed task indefinitely. Guard against a re-launch having
        # already replaced the entry under this name.
        current = _reapers.get(container_name)
        if current is asyncio.current_task():
            _reapers.pop(container_name, None)


def _schedule_reaper(container_name: str, lifetime_seconds: float) -> None:
    """Schedule (or re-schedule) the auto-delete reaper for a container."""
    existing = _reapers.pop(container_name, None)
    if existing is not None and not existing.done():
        logger.info(f"[{container_name}] Cancelling previous reaper before rescheduling")
        existing.cancel()

    task = asyncio.create_task(_reap_after_lifetime(container_name, lifetime_seconds))
    _reapers[container_name] = task


def _cancel_reaper(container_name: str) -> bool:
    """Cancel the reaper for a container if one is pending. Returns True if cancelled."""
    existing = _reapers.pop(container_name, None)
    if existing is not None and not existing.done():
        existing.cancel()
        logger.info(f"[{container_name}] Reaper cancelled (explicit delete or relaunch)")
        return True
    return False


@app.route("/containers/<container_name>", methods=["DELETE"])
async def delete_container(container_name: str):
    """
    Request deletion of a container.
    Immediately return success (200 OK) and perform deletion in the background
    with retry policy: up to 100 attempts, 10 seconds between attempts.
    """
    logger.info(f"Scheduled deletion for container {container_name}")

    # Cancel any pending lifetime reaper so it doesn't race/duplicate work.
    _cancel_reaper(container_name)

    # Fire-and-forget background task
    asyncio.create_task(_retry_delete_container(container_name, max_attempts=100, delay_seconds=10))

    # Return immediately
    return jsonify({"message": f"Deletion for container {container_name} scheduled.", "retry_policy": {"max_attempts": 100, "delay_seconds": 10}}), 200


@app.route("/containers/<container_name>/status", methods=["GET"])
async def get_status(container_name: str):
    """Get status of a container"""
    status = await get_container_status(container_name)
    if status is None:
        return jsonify({"error": f"Container {container_name} not found"}), 404

    ip_address = None
    if status == "running":
        ip_address = await get_container_ip(container_name)
    return jsonify({"container_name": container_name, "status": status, "ip_address": ip_address})


REQUIRED_BASE_CONTAINERS = ["shopping", "shopping-admin", "gitlab"]

REQUIRED_ZFS_DATASETS = [
    "default/containers/gitlab",
    "default/containers/shopping",
    "default/containers/shopping-admin",
]


@app.route("/containers/cleanup", methods=["POST"])
async def cleanup_containers():
    """
    Delete every container except the required base containers.

    Intended for forcibly reclaiming host resources when too many task
    containers have accumulated. Runs deletions sequentially and reports
    per-container results.
    """
    preserved = set(REQUIRED_BASE_CONTAINERS)

    try:
        all_names = await _client.list_instance_names()
    except IncusError as e:
        logger.error(f"cleanup: failed to list containers: {e.message}")
        return jsonify({"error": f"Failed to list containers: {e.message}"}), 500

    targets = [name for name in all_names if name not in preserved]
    logger.info(f"cleanup: preserving {sorted(preserved)}, deleting {targets}")

    deleted: list[str] = []
    failed: list[dict] = []
    for name in targets:
        # Cancel any pending lifetime reaper for this container.
        _cancel_reaper(name)
        try:
            await _force_delete_instance(name)
            logger.info(f"cleanup: removed {name}")
            deleted.append(name)
        except IncusError as e:
            logger.warning(f"cleanup: failed to remove {name}: {e.message}")
            failed.append({"container_name": name, "error": e.message})

    return jsonify(
        {
            "preserved": sorted(preserved),
            "deleted": deleted,
            "failed": failed,
        }
    )


@app.route("/health", methods=["GET"])
async def health_check():
    """Health check endpoint"""
    # 1) Incus daemon alive (via API)
    try:
        info = await _client.server_info()
    except IncusError as e:
        return jsonify({"error": f"Incus not available: {e.message}"}), 503

    # 2) Required base containers exist
    missing = []
    for name in REQUIRED_BASE_CONTAINERS:
        status = await get_container_status(name)
        if status is None:
            missing.append(name)

    if missing:
        return jsonify({"error": f"Required base containers not found: {missing}"}), 503

    # 3) ZFS datasets actually exist on the zfs layer (ground truth, not Incus metadata).
    #    This catches the race where Incus has registered an instance but the
    #    underlying ZFS clone hasn't finished yet.
    missing_zfs = []
    for dataset in REQUIRED_ZFS_DATASETS:
        exit_code, _, _ = await run_subprocess_safe(["zfs", "list", "-H", "-o", "name", dataset], timeout_s=10)
        if exit_code != 0:
            missing_zfs.append(dataset)

    if missing_zfs:
        return jsonify({"error": f"Required ZFS datasets not found: {missing_zfs}"}), 503

    incus_version = info["environment"]["server_version"]
    return jsonify({"status": "healthy", "incus_version": incus_version, "throttle_enabled": THROTTLE_ENABLED, "throttle_interval_seconds": THROTTLE_SECONDS})


# ----------------------------------------
# Runtime toggle (feature flag management)
# ----------------------------------------
@app.route("/config/throttle", methods=["GET"])
async def get_throttle_config():
    return jsonify({"enabled": THROTTLE_ENABLED, "interval_seconds": THROTTLE_SECONDS})


@app.route("/config/throttle", methods=["POST"])
async def set_throttle_config():
    global THROTTLE_ENABLED, THROTTLE_SECONDS
    body = await request.get_json(force=True, silent=True) or {}
    if "enabled" in body:
        THROTTLE_ENABLED = bool(body["enabled"])
    if "interval_seconds" in body:
        try:
            val = float(body["interval_seconds"])
            if val <= 0:
                return jsonify({"error": "interval_seconds must be > 0"}), 400
            THROTTLE_SECONDS = val
        except (TypeError, ValueError):
            return jsonify({"error": "interval_seconds must be a number"}), 400
    logger.info(f"Throttle config updated: enabled={THROTTLE_ENABLED}, interval={THROTTLE_SECONDS}s")
    return jsonify({"enabled": THROTTLE_ENABLED, "interval_seconds": THROTTLE_SECONDS})


if __name__ == "__main__":
    # Quart's built-in server; consider Hypercorn/Gunicorn for production.
    app.run(host="0.0.0.0", port=8001, debug=False)
