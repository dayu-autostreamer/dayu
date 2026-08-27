"""Timeout normalization for the pinned Kubernetes Python client."""

import math


def kubernetes_request_timeout(seconds: float) -> int:
    """Return a total timeout understood by ``kubernetes==20.11.0a1``.

    That client silently ignores scalar floats.  Use a positive integer total
    timeout, rounding down so a request cannot exceed its caller's budget.  A
    sub-second budget maps to the smallest timeout the client can represent.
    """

    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Kubernetes request timeout must be numeric") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Kubernetes request timeout must be finite and positive")
    return max(1, int(math.floor(seconds)))
