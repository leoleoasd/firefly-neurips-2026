import asyncio
import logging
import random
from collections import defaultdict

import aiohttp
import ray

logger = logging.getLogger(__name__)

# Number of consecutive health check failures before removing a URL
MAX_FAILURES = 3
# Health check interval in seconds
HEALTH_CHECK_INTERVAL = 5
# Timeout for each /health request in seconds
HEALTH_CHECK_TIMEOUT = 5


@ray.remote
class SGLangRegistry:
    def __init__(self):
        self._store: dict[str, list[str]] = {}
        # Track consecutive failures per URL: (key, url) -> failure count
        self._failure_counts: dict[tuple, int] = defaultdict(int)
        self._health_check_task: asyncio.Task | None = None

    def add(self, key: str, url: str):
        urls = self._store.setdefault(key, [])
        if url not in urls:
            urls.append(url)
        # Reset failure count when (re-)adding
        self._failure_counts.pop((key, url), None)

    def get_one(self, key: str) -> str | None:
        urls = self._store.get(key)
        return random.choice(urls) if urls else None

    def get_all(self, key: str) -> list[str]:
        return list(self._store.get(key, []))

    def dump(self):
        return {k: list(v) for k, v in self._store.items()}

    def remove(self, key: str, url: str) -> bool:
        """Remove a single URL from a key. Returns True if it was found and removed."""
        urls = self._store.get(key)
        if urls and url in urls:
            urls.remove(url)
            if not urls:
                del self._store[key]
            self._failure_counts.pop((key, url), None)
            return True
        return False

    def remove_key(self, key: str) -> list[str]:
        """Remove all URLs for a key. Returns the removed URLs."""
        removed = self._store.pop(key, [])
        for url in removed:
            self._failure_counts.pop((key, url), None)
        return removed

    def clear(self):
        """Clear all entries from the registry."""
        self._store.clear()
        self._failure_counts.clear()

    # ── Health checking ──────────────────────────────────────────────────

    def start_health_check(self):
        """Start the background health check loop. Idempotent."""
        if self._health_check_task is not None and not self._health_check_task.done():
            return
        self._health_check_task = asyncio.ensure_future(self._health_check_loop())
        logger.info("Registry health check started")

    def stop_health_check(self):
        """Stop the background health check loop."""
        if self._health_check_task is not None and not self._health_check_task.done():
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("Registry health check stopped")

    async def _health_check_loop(self):
        """Periodically check /health on all registered URLs."""
        while True:
            try:
                await self._check_all()
            except Exception:
                logger.exception("Error in health check loop")
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _check_all(self):
        """Check health of every registered URL."""
        # Snapshot current entries to avoid mutation during iteration
        snapshot: list[tuple[str, str]] = []
        for key, urls in self._store.items():
            for url in urls:
                snapshot.append((key, url))

        if not snapshot:
            return

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)) as session:
            tasks = [self._check_one(session, key, url) for key, url in snapshot]
            await asyncio.gather(*tasks)

    async def _check_one(self, session: aiohttp.ClientSession, key: str, url: str):
        """Check a single URL's /health endpoint."""
        try:
            async with session.get(f"{url}/health") as resp:
                if resp.status == 200:
                    # Healthy — reset failure count
                    self._failure_counts.pop((key, url), None)
                    return
        except Exception:
            pass

        # Failed
        self._failure_counts[(key, url)] += 1
        count = self._failure_counts[(key, url)]
        logger.warning(f"Health check failed for {key}={url} ({count}/{MAX_FAILURES})")

        if count >= MAX_FAILURES:
            logger.error(f"Removing {key}={url} after {count} consecutive failures")
            self.remove(key, url)


def get_or_create_registry(name: str = "sglang_registry"):
    """
    Ray-official, atomic get-or-create for a named detached actor.
    Automatically starts the background health check loop.
    """
    registry = SGLangRegistry.options(
        name=name,
        lifetime="detached",
        namespace="sglang",
        get_if_exists=True,
    ).remote()
    registry.start_health_check.remote()
    return registry
