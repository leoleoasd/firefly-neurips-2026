"""
Incus Container Management Functions

Stateless async functions for communicating with Incus HTTP server to manage containers
for WebArena environments from the agent environment.
"""

import logging

import httpx


# HTTP status the incus server returns when it refuses to clone due to low
# host memory. Must stay in sync with incus_server.INSUFFICIENT_MEMORY_STATUS.
INSUFFICIENT_MEMORY_STATUS = 507
INSUFFICIENT_MEMORY_REASON = "insufficient_memory"


class InsufficientMemoryError(RuntimeError):
    """
    Raised when the Incus server rejects a container launch because the host
    does not have enough free memory. Callers may want to back off / retry or
    trigger a cleanup rather than treat this as a generic launch failure.
    """

    def __init__(self, message: str, *, free_memory_ratio: float | None = None, min_free_memory_ratio: float | None = None):
        super().__init__(message)
        self.free_memory_ratio = free_memory_ratio
        self.min_free_memory_ratio = min_free_memory_ratio


def _get_httpx_client_kwargs(proxy_server: str | None = None, timeout: int = 300, extra_headers: dict | None = None) -> dict:
    kwargs = {"timeout": timeout}
    if proxy_server:
        kwargs["proxy"] = proxy_server
    if extra_headers:
        kwargs["headers"] = extra_headers
    return kwargs


DEFAULT_LIFETIME_SECONDS = 2 * 60 * 60  # keep in sync with incus_server.DEFAULT_LIFETIME_SECONDS


async def launch_container(
    server_url: str,
    base_name: str,
    container_name: str,
    timeout: int = 300,
    proxy_server: str | None = None,
    extra_headers: dict | None = None,
    lifetime_seconds: float | None = DEFAULT_LIFETIME_SECONDS,
) -> str:
    """
    Launch a new container by copying from base and starting it.

    Args:
        server_url: Incus server URL (e.g., "http://localhost:8001")
        base_name: Name of the base container to copy from
        container_name: Name for the new container instance
        timeout: Request timeout in seconds
        proxy_server: Optional proxy server URL for HTTP requests
        extra_headers: Extra HTTP headers (e.g. proxy session id)
        lifetime_seconds: Max lifetime of the container in seconds. After this
            elapses, the server will force-delete it. Defaults to 2 hours.
            Pass None to omit the field and let the server use its own default.

    Returns:
        str: IP address of the launched container

    Raises:
        InsufficientMemoryError: If the server rejects the launch because the
            host does not have enough free memory.
        RuntimeError: If container launch fails for any other reason.
    """
    logger = logging.getLogger(__name__)
    server_url = server_url.rstrip("/")
    url = f"{server_url}/containers/launch"
    payload: dict = {"base_name": base_name, "container_name": container_name}
    if lifetime_seconds is not None:
        payload["lifetime_seconds"] = lifetime_seconds

    logger.info(f"Launching container {container_name} from base {base_name}")
    if proxy_server:
        logger.debug(f"Using proxy server: {proxy_server}")

    client_kwargs = _get_httpx_client_kwargs(proxy_server, timeout, extra_headers)
    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                ip_address = data["ip_address"]
                logger.info(f"Container {container_name} launched with IP {ip_address}")
                return ip_address
            elif response.status_code == INSUFFICIENT_MEMORY_STATUS:
                # Server refused to clone due to low host memory. Surface a
                # distinct exception so callers can back off or clean up.
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                message = body["error"] if "error" in body else response.text
                free_ratio = body["free_memory_ratio"] if "free_memory_ratio" in body else None
                min_ratio = body["min_free_memory_ratio"] if "min_free_memory_ratio" in body else None
                logger.error(f"Refused to launch {container_name}: insufficient host memory ({message})")
                raise InsufficientMemoryError(
                    f"Incus server refused to launch {container_name}: {message}",
                    free_memory_ratio=free_ratio,
                    min_free_memory_ratio=min_ratio,
                )
            else:
                error_text = response.text
                logger.error(f"Failed to launch container: {error_text}")
                raise RuntimeError(f"Failed to launch container {container_name}: {error_text}")

        except httpx.TimeoutException as e:
            logger.error(f"Timeout launching container {container_name}")
            raise RuntimeError(f"Timeout launching container {container_name}") from e
        except httpx.RequestError as e:
            logger.error(f"Network error launching container: {e}")
            raise RuntimeError(f"Network error launching container {container_name}: {e}") from e


async def delete_container(server_url: str, container_name: str, timeout: int = 300, proxy_server: str | None = None, extra_headers: dict | None = None) -> None:
    """
    Stop and remove a container.

    Args:
        server_url: Incus server URL (e.g., "http://localhost:8001")
        container_name: Name of the container to delete
        timeout: Request timeout in seconds
        proxy_server: Optional proxy server URL for HTTP requests

    Raises:
        RuntimeError: If container deletion fails
    """
    logger = logging.getLogger(__name__)
    server_url = server_url.rstrip("/")
    url = f"{server_url}/containers/{container_name}"

    logger.info(f"Deleting container {container_name}")
    if proxy_server:
        logger.debug(f"Using proxy server: {proxy_server}")

    client_kwargs = _get_httpx_client_kwargs(proxy_server, timeout, extra_headers)
    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            response = await client.delete(url)
            if response.status_code == 200:
                logger.info(f"Container {container_name} deleted successfully")
            else:
                error_text = response.text
                logger.error(f"Failed to delete container: {error_text}")
                raise RuntimeError(f"Failed to delete container {container_name}: {error_text}")

        except httpx.TimeoutException as e:
            logger.error(f"Timeout deleting container {container_name}")
            raise RuntimeError(f"Timeout deleting container {container_name}") from e
        except httpx.RequestError as e:
            logger.error(f"Network error deleting container: {e}")
            raise RuntimeError(f"Network error deleting container {container_name}: {e}") from e


async def get_container_status(server_url: str, container_name: str, timeout: int = 300, proxy_server: str | None = None, extra_headers: dict | None = None) -> dict | None:
    """
    Get status of a container.

    Args:
        server_url: Incus server URL (e.g., "http://localhost:8001")
        container_name: Name of the container
        timeout: Request timeout in seconds
        proxy_server: Optional proxy server URL for HTTP requests

    Returns:
        dict: Container status information or None if not found
    """
    logger = logging.getLogger(__name__)
    server_url = server_url.rstrip("/")
    url = f"{server_url}/containers/{container_name}/status"

    if proxy_server:
        logger.debug(f"Using proxy server: {proxy_server}")

    client_kwargs = _get_httpx_client_kwargs(proxy_server, timeout, extra_headers)
    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                error_text = response.text
                logger.error(f"Failed to get container status: {error_text}")
                return None

        except (httpx.TimeoutException, httpx.RequestError) as e:
            logger.error(f"Error getting container status: {e}")
            return None


async def health_check(server_url: str, timeout: int = 10, proxy_server: str | None = None, extra_headers: dict | None = None) -> bool:
    """
    Check if the Incus server is healthy and available.

    Args:
        server_url: Incus server URL (e.g., "http://localhost:8001")
        timeout: Request timeout in seconds
        proxy_server: Optional proxy server URL for HTTP requests

    Returns:
        bool: True if server is healthy, False otherwise
    """
    server_url = server_url.rstrip("/")
    url = f"{server_url}/health"

    client_kwargs = _get_httpx_client_kwargs(proxy_server, timeout, extra_headers)
    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            response = await client.get(url)
            return response.status_code == 200
        except (httpx.TimeoutException, httpx.RequestError) as e:
            # print traceback
            import traceback

            traceback.print_exc()
            return False


async def cleanup_containers(server_url: str, timeout: int = 600, proxy_server: str | None = None, extra_headers: dict | None = None) -> dict:
    """
    Ask the Incus server to delete every container except the required base
    containers. Intended as a recovery action after an InsufficientMemoryError
    or between batch runs.

    Args:
        server_url: Incus server URL (e.g., "http://localhost:8001")
        timeout: Request timeout in seconds (deletions can be slow, default 600)
        proxy_server: Optional proxy server URL for HTTP requests
        extra_headers: Extra HTTP headers (e.g. proxy session id)

    Returns:
        dict: {"preserved": [...], "deleted": [...], "failed": [...]}

    Raises:
        RuntimeError: If the server returns a non-200 response or is unreachable.
    """
    logger = logging.getLogger(__name__)
    server_url = server_url.rstrip("/")
    url = f"{server_url}/containers/cleanup"

    logger.info(f"Requesting container cleanup on {server_url}")
    client_kwargs = _get_httpx_client_kwargs(proxy_server, timeout, extra_headers)
    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            response = await client.post(url)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Cleanup complete: deleted={len(data['deleted'])}, failed={len(data['failed'])}, preserved={data['preserved']}")
                return data
            error_text = response.text
            logger.error(f"Cleanup request failed: {error_text}")
            raise RuntimeError(f"Cleanup request failed with status {response.status_code}: {error_text}")
        except httpx.TimeoutException as e:
            logger.error("Timeout during cleanup request")
            raise RuntimeError("Timeout during cleanup request") from e
        except httpx.RequestError as e:
            logger.error(f"Network error during cleanup request: {e}")
            raise RuntimeError(f"Network error during cleanup request: {e}") from e
