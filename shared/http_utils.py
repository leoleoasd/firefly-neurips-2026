"""Instrumented HTTP POST used by our rollout code.

Replaces ``slime.utils.http_utils.post`` at the call site. Splits the work
into three phase counters so we can see where time goes inside an
``assistant_generate`` scope:

  assistant_generate_encode   : JSON-encoding the request payload
  assistant_generate_inflight : from first byte of request to last byte of response
  assistant_generate_decode   : JSON-decoding the response body

The sum of these three should equal the outer ``assistant_generate`` scope.
A large gap between ``*_inflight`` and router-side ``running + queued`` means
time is spent in the router (dispatch/response streaming) rather than at
workers; a large ``*_decode`` points at client-side JSON parsing.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import orjson

from shared.global_counter import counter_scope

logger = logging.getLogger(__name__)


async def post(url: str, payload, max_retries: int = 60):
    """Drop-in replacement for ``slime.utils.http_utils.post``.

    Assumes ``slime.utils.http_utils.init_http_client(args)`` has already been
    called in this process so that ``_http_client`` is a live ``httpx.AsyncClient``.
    """
    # Re-read on each call so we see the client initialized after import.
    from slime.utils.http_utils import _http_client as client

    if client is None:
        raise RuntimeError("shared.http_utils.post: httpx client not initialized; call init_http_client(args) first")

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Phase 1: JSON-encode the request body.
            async with counter_scope("assistant_generate_encode"):
                body = orjson.dumps(payload or {})

            # Phase 2: HTTP roundtrip — this is what the router "sees".
            async with counter_scope("assistant_generate_inflight"):
                response = await client.post(
                    url,
                    content=body,
                    headers={"content-type": "application/json"},
                )
                response.raise_for_status()

            # Phase 3: JSON-decode the response body. For /generate with
            # return_routed_experts=True this is a large base64 blob so
            # we offload to a worker thread.
            async with counter_scope("assistant_generate_decode"):
                try:
                    output = await asyncio.to_thread(orjson.loads, response.content)
                except (orjson.JSONDecodeError, ValueError):
                    output = response.text
        except Exception as e:
            retry_count += 1
            response_text = e.response.text if isinstance(e, httpx.HTTPStatusError) else None
            logger.info(
                "Error: %s, retrying... (attempt %d/%d, url=%s, response=%s)",
                e,
                retry_count,
                max_retries,
                url,
                response_text,
            )
            if retry_count >= max_retries:
                logger.info("Max retries (%d) reached, failing... (url=%s)", max_retries, url)
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output
