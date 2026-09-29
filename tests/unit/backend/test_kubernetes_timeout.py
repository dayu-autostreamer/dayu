import math

import pytest

from kubernetes_timeout import kubernetes_request_timeout


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(30, 30), (4.9, 4), (0.1, 1)],
)
def test_kubernetes_request_timeout_is_an_integer_total_timeout(seconds, expected):
    timeout = kubernetes_request_timeout(seconds)

    assert timeout == expected
    assert type(timeout) is int


@pytest.mark.parametrize("seconds", [0, -1, math.inf, math.nan, "invalid"])
def test_kubernetes_request_timeout_rejects_invalid_budgets(seconds):
    with pytest.raises(ValueError, match="Kubernetes request timeout"):
        kubernetes_request_timeout(seconds)
