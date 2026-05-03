import numpy as np
import pytest
from scaling_laws import (
    ScalingLaw, estimate_compute, optimal_allocation_chinchilla,
    optimal_allocation_kaplan, compare_allocations,
)


def test_scaling_law_fit():
    law = ScalingLaw()
    x = np.array([1e6, 1e7, 1e8, 1e9])
    loss = np.array([3.5, 3.0, 2.7, 2.5])
    law.fit(x, loss)
    assert law.alpha is not None
    assert law.a is not None


def test_scaling_law_predict():
    law = ScalingLaw()
    x = np.array([1e6, 1e7, 1e8, 1e9])
    loss = np.array([3.5, 3.0, 2.7, 2.5])
    law.fit(x, loss)

    # Prediction should be reasonable
    pred = law.predict(1e8)
    assert 2.0 < pred < 4.0


def test_scaling_law_monotonic():
    """Larger scale should predict lower loss."""
    law = ScalingLaw()
    x = np.array([1e6, 1e7, 1e8, 1e9])
    loss = np.array([3.5, 3.0, 2.7, 2.5])
    law.fit(x, loss)

    loss_small = law.predict(1e6)
    loss_large = law.predict(1e10)
    assert loss_large < loss_small


def test_estimate_compute():
    # 1B params, 20B tokens → 6 * 1e9 * 20e9 = 1.2e20
    flops = estimate_compute(1e9, 20e9)
    assert abs(flops - 1.2e20) < 1e18


def test_chinchilla_allocation():
    # For a reasonable compute budget
    compute = 1e18  # ~1 exaFLOP
    n, d = optimal_allocation_chinchilla(compute)
    assert n > 0
    assert d > 0
    # D should be roughly 20 * N
    ratio = d / n
    assert 15 < ratio < 25


def test_kaplan_allocation():
    compute = 1e18
    n, d = optimal_allocation_kaplan(compute)
    assert n > 0
    assert d > 0


def test_compare_allocations():
    compute = 1e18
    result = compare_allocations(compute)
    assert "chinchilla" in result
    assert "kaplan" in result
    # Both should have positive values
    for name, (n, d) in result.items():
        assert n > 0
        assert d > 0


def test_chinchilla_scales_with_compute():
    """Larger compute budget should produce larger model and more tokens."""
    n1, d1 = optimal_allocation_chinchilla(1e18)
    n2, d2 = optimal_allocation_chinchilla(1e20)
    assert n2 > n1
    assert d2 > d1
