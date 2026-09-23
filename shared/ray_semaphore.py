"""
Global semaphores for Ray cluster to limit concurrent operations.

Supports multiple named semaphores for different purposes:
- container_launch: Limits concurrent container launches (default: 8)
- (add more as needed)

Usage:
    # At training job start (in main process):
    initialize_semaphore("container_launch", max_concurrent=8)

    # In async code that needs rate limiting:
    async with acquire_semaphore("container_launch"):
        await expensive_operation()
"""

import asyncio
import logging
import os
import uuid

import ray

logger = logging.getLogger(__name__)


@ray.remote
class GlobalSemaphore:
    """
    Ray actor that implements a distributed semaphore.

    Uses asyncio internally to handle concurrent acquire/release requests.

    Cancellation safety
    -------------------
    ``acquire`` returns a unique ``ticket`` string that the caller must pass
    to ``release``.  The actor tracks tickets in ``_granted_tickets``; a
    ``release`` with an unknown (or already-released) ticket is a no-op.
    This prevents over-release when a client's ``acquire.remote()`` call is
    cancelled after the actor granted the slot — a subsequent unconditional
    release would otherwise leak the slot to a phantom holder.

    Client-side cancellation is handled by the ``SemaphoreContext`` wrapper,
    which shields the acquire RPC and — if the caller is cancelled while the
    acquire is in flight — schedules a detached release once the ticket
    arrives.  The actor itself therefore needs only ticket-aware release.
    """

    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self.max_concurrent = max_concurrent
        self.current_count = 0
        self._waiters: list[tuple[str, asyncio.Future]] = []
        # Tickets that have been granted and not yet released.
        self._granted_tickets: set[str] = set()
        self._lock = asyncio.Lock()
        logger.info(f"GlobalSemaphore '{name}' initialized with max_concurrent={max_concurrent}")

    def _generate_ticket(self) -> str:
        return uuid.uuid4().hex

    async def acquire(self) -> str:
        """
        Acquire the semaphore, blocking until a slot is available.

        Returns:
            A ticket string that must be passed back to ``release``.
        """
        ticket = self._generate_ticket()

        async with self._lock:
            if self.current_count < self.max_concurrent:
                self.current_count += 1
                self._granted_tickets.add(ticket)
                logger.debug(
                    f"Semaphore '{self.name}' acquired immediately: "
                    f"{self.current_count}/{self.max_concurrent} ticket={ticket[:8]}"
                )
                return ticket

            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._waiters.append((ticket, future))

        logger.debug(
            f"Semaphore '{self.name}' waiting: "
            f"{self.current_count}/{self.max_concurrent}, waiters={len(self._waiters)} "
            f"ticket={ticket[:8]}"
        )
        await future

        async with self._lock:
            self._granted_tickets.add(ticket)

        return ticket

    async def release(self, ticket: str | None = None) -> None:
        """Release the semaphore using the ticket returned by ``acquire``.

        A ``None`` or unknown ticket is silently ignored; this makes the
        call idempotent and safe under client-side cancellation races.
        """
        async with self._lock:
            if ticket is None or ticket not in self._granted_tickets:
                logger.debug(
                    f"Semaphore '{self.name}' release ignored (ticket unknown): "
                    f"count={self.current_count}/{self.max_concurrent} "
                    f"ticket={str(ticket)[:8] if ticket else 'None'}"
                )
                return

            self._granted_tickets.discard(ticket)
            self.current_count -= 1
            logger.debug(
                f"Semaphore '{self.name}' released: {self.current_count}/{self.max_concurrent} ticket={ticket[:8]}"
            )

            # Wake the next waiter, if any.
            while self._waiters and self.current_count < self.max_concurrent:
                _next_ticket, waiter = self._waiters.pop(0)
                if waiter.done():
                    # Waiter was already cancelled/completed elsewhere; skip.
                    continue
                self.current_count += 1
                waiter.set_result(True)
                break

    def get_status(self) -> dict:
        """Get current semaphore status."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "current_count": self.current_count,
            "waiters": len(self._waiters),
            "granted": len(self._granted_tickets),
        }


# Semaphore configuration: name -> (env_var, default_value)
SEMAPHORE_CONFIGS = {
    "container_launch": ("MAX_CONCURRENT_CONTAINER_LAUNCHES", 8),
    "container_running": ("MAX_CONCURRENT_CONTAINERS_RUNNING", 32),
}


def _get_actor_name(semaphore_name: str) -> str:
    """Get Ray actor name for a semaphore."""
    return f"global_semaphore_{semaphore_name}"


def initialize_semaphore(name: str, max_concurrent: int | None = None) -> ray.actor.ActorHandle:
    """
    Initialize a named semaphore actor.

    Should be called once at the start of the training job for each semaphore type.

    Args:
        name: Semaphore name (e.g., "container_launch")
        max_concurrent: Maximum concurrent operations. If None, uses
                       environment variable or default from SEMAPHORE_CONFIGS.

    Returns:
        Ray actor handle for the semaphore
    """
    if max_concurrent is None:
        if name in SEMAPHORE_CONFIGS:
            env_var, default = SEMAPHORE_CONFIGS[name]
            max_concurrent = int(os.environ.get(env_var, default))
        else:
            max_concurrent = 8  # Fallback default

    actor_name = _get_actor_name(name)

    actor = GlobalSemaphore.options(
        name=actor_name,
        lifetime="detached",
        get_if_exists=True,
    ).remote(name, max_concurrent)

    logger.info(f"Semaphore '{name}' initialized: {actor_name} with max_concurrent={max_concurrent}")
    return actor


def get_semaphore_actor(name: str) -> ray.actor.ActorHandle:
    """Get a named semaphore actor, creating it if it does not exist."""
    actor_name = _get_actor_name(name)
    try:
        return ray.get_actor(actor_name)
    except ValueError:
        return initialize_semaphore(name)


class SemaphoreContext:
    """
    Async context manager for semaphore acquisition.

    Cancellation-safe:

    - ``__aenter__`` shields the remote acquire RPC.  If the caller is
      cancelled while we are waiting for the acquire to return, we launch a
      detached task that awaits the in-flight acquire and releases the
      ticket the actor eventually grants us.  This guarantees a cancelled
      caller never leaks a semaphore slot.

    - ``__aexit__`` releases with the stored ticket.  Release is also
      shielded so it completes even if the enclosing task is being
      unwound under cancellation.

    Usage:
        async with SemaphoreContext("container_launch"):
            await expensive_operation()
    """

    def __init__(self, name: str):
        self.name = name
        self._actor: ray.actor.ActorHandle | None = None
        self._ticket: str | None = None

    async def __aenter__(self):
        self._actor = get_semaphore_actor(self.name)

        # Wrap the remote call in a coroutine so we can ensure_future it.
        # Ray's ObjectRef is awaitable but asyncio.shield expects a Future;
        # going through a helper coroutine produces a proper Task.
        actor_ref = self._actor

        async def _do_acquire() -> str:
            return await actor_ref.acquire.remote()

        acquire_fut = asyncio.ensure_future(_do_acquire())
        try:
            self._ticket = await asyncio.shield(acquire_fut)
        except asyncio.CancelledError:
            # The outer task was cancelled while we were waiting for the
            # acquire to return.  The actor may still grant us the slot
            # asynchronously; launch a detached release so the slot is
            # returned to the pool instead of leaking.
            self._schedule_orphan_release(acquire_fut)
            self._actor = None
            self._ticket = None
            raise
        return self

    def _schedule_orphan_release(self, acquire_fut: "asyncio.Future[str]") -> None:
        """Release any ticket the actor grants us after our ``__aenter__`` was cancelled.

        Runs as a detached task tied to the event loop, not to the current
        (cancelled) task, so cancellation does not propagate into it.
        """
        actor = self._actor
        if actor is None:
            return
        name = self.name

        async def _release_orphan() -> None:
            try:
                ticket = await acquire_fut
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"Semaphore '{name}' orphan acquire failed: {e}")
                return

            try:
                await actor.release.remote(ticket)
                logger.debug(f"Semaphore '{name}' orphan slot released after cancellation: ticket={ticket[:8]}")
            except Exception as e:
                logger.warning(f"Semaphore '{name}' orphan release failed: {e}")

        asyncio.get_event_loop().create_task(_release_orphan())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._actor is not None and self._ticket is not None:
            actor_ref = self._actor
            ticket = self._ticket

            async def _do_release() -> None:
                await actor_ref.release.remote(ticket)

            try:
                await asyncio.shield(asyncio.ensure_future(_do_release()))
            except asyncio.CancelledError:
                # release.remote() was already dispatched; the actor will
                # process it asynchronously.  Re-raise the cancellation.
                raise
            except Exception as e:
                logger.warning(f"Semaphore '{self.name}' release failed: {e}")
        self._actor = None
        self._ticket = None
        return False  # Don't suppress exceptions


def acquire_semaphore(name: str = "container_launch") -> SemaphoreContext:
    """
    Return an async context manager for semaphore acquisition.

    Args:
        name: Semaphore name (default: "container_launch")

    Usage:
        async with acquire_semaphore("container_launch"):
            await expensive_operation()
    """
    return SemaphoreContext(name)


async def get_semaphore_status(name: str) -> dict:
    """Get current semaphore status."""
    actor = get_semaphore_actor(name)
    return await actor.get_status.remote()


# Convenience function for backward compatibility
def initialize_global_semaphore(max_concurrent: int | None = None):
    """Initialize the container_launch semaphore (backward compatibility)."""
    return initialize_semaphore("container_launch", max_concurrent)
