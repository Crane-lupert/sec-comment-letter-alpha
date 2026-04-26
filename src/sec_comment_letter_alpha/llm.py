"""LLM call wrapper around `shared_utils.openrouter_client.OpenRouterClient`.

Day 1 found a deterministic bug: when `OpenRouterClient.complete` is invoked
through its decorator `@retry(stop_after_attempt(5), wait_exponential_jitter(initial=2, max=60))`,
httpx raises `LocalProtocolError` on every retry attempt for non-trivial
payloads on Windows. The same call with `retry_with(stop=stop_after_attempt(1))`
succeeds first try.

Reproducer:
    from shared_utils.openrouter_client import OpenRouterClient
    import tenacity
    c = OpenRouterClient(project="X")
    # 5-attempt retry → RetryError[LocalProtocolError] every call
    c.complete(model="google/gemma-3-27b-it", prompt="<real SEC excerpt ~1KB>", max_tokens=400)
    # 1-attempt retry → 200 OK every call
    c.complete.retry_with(stop=tenacity.stop_after_attempt(1), reraise=True)(
        c, model="...", prompt="...", max_tokens=400, temperature=0.0)

Until shared-utils is fixed (cross-repo coordination), this module is the
canonical entry point for LLM calls in Project X. It preserves everything the
shared client provides (file-lock semaphore, budget enforcement, usage logging,
project tagging) and only replaces the broken retry layer with a plain backoff
loop that does not depend on tenacity wrapping the same FileLock.acquire path.
"""

from __future__ import annotations

import time

import tenacity
from shared_utils.openrouter_client import OpenRouterClient


def call_one(
    client: OpenRouterClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.0,
    attempts: int = 4,
    backoff_base_sec: float = 2.0,
) -> dict:
    """Synchronous LLM call with own linear backoff. Single-attempt per call.

    Raises the last seen exception if all attempts fail.
    """
    single = client.complete.retry_with(stop=tenacity.stop_after_attempt(1), reraise=True)
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return single(
                client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001
            last = e
            if i + 1 < attempts:
                time.sleep(backoff_base_sec * (i + 1))
    assert last is not None
    raise last
