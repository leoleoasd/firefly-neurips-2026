"""Shared utility functions."""

import socket


def get_random_free_port() -> int:
    """Get a random free port by binding to port 0 and letting the OS assign one.

    This is thread/process-safe: the OS guarantees the returned port is free
    at the moment of allocation. The socket is closed before returning, so
    there is a small TOCTOU window, but it's far safer than scanning ports
    sequentially.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
