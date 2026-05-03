"""Scaling Laws — Complete reference solution.

Based on Kaplan et al. (2020) and Hoffmann et al. (2022) "Chinchilla".
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ScalingLawResult:
    """Result of a scaling law prediction."""
    predicted_loss: float
    optimal_params: Optional[int] = None
    optimal_tokens: Optional[int] = None
    compute_budget: Optional[float] = None


class ScalingLaw:
    """Power-law scaling law: L(x) = a * x^(-alpha) + L_inf

    Fits to empirical data points to predict loss at different scales.
    """

    def __init__(self):
        self.alpha: Optional[float] = None
        self.a: Optional[float] = None
        self.l_inf: Optional[float] = None

    def fit(self, x_values: np.ndarray, loss_values: np.ndarray, l_inf: float = 1.5):
        """Fit power-law: L(x) = a * x^(-alpha) + L_inf

        Args:
            x_values: Scaling variable (params or tokens).
            loss_values: Observed loss values.
            l_inf: Irreducible loss floor (default 1.5, typical for language).
        """
        self.l_inf = l_inf

        # Linearize: log(L - L_inf) = log(a) - alpha * log(x)
        adjusted = loss_values - l_inf
        mask = adjusted > 0
        log_x = np.log(x_values[mask])
        log_y = np.log(adjusted[mask])

        # Linear regression: slope = -alpha, intercept = log(a)
        coeffs = np.polyfit(log_x, log_y, 1)
        self.alpha = -coeffs[0]
        self.a = np.exp(coeffs[1])

    def predict(self, x: float) -> float:
        """Predict loss at given scale."""
        if self.alpha is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.a * (x ** (-self.alpha)) + self.l_inf

    def predict_from_params(self, n_params: float) -> float:
        """Predict loss from model parameter count (Kaplan et al.)."""
        return self.predict(n_params)

    def predict_from_tokens(self, n_tokens: float) -> float:
        """Predict loss from training token count (Kaplan et al.)."""
        return self.predict(n_tokens)


def estimate_compute(n_params: float, n_tokens: float) -> float:
    """Estimate total FLOPs for training.

    Approximation: C ≈ 6 * N * D
    Where N = params, D = tokens, factor of 6 accounts for forward+backward.

    Args:
        n_params: Number of model parameters.
        n_tokens: Number of training tokens.

    Returns:
        Estimated FLOPs.
    """
    return 6 * n_params * n_tokens


def optimal_allocation_chinchilla(
    compute_budget: float,
    params_per_flop: float = 1.0 / 6e8,
    tokens_per_param: float = 20.0,
) -> Tuple[int, int]:
    """Chinchilla-optimal allocation of compute between params and tokens.

    Chinchilla (Hoffmann et al. 2022) found that for a given compute budget,
    the optimal split is roughly: D ≈ 20 * N (tokens ≈ 20 × params).

    This function finds the N and D that:
    1. Satisfy C ≈ 6 * N * D
    2. Maintain D ≈ 20 * N ratio

    Args:
        compute_budget: Total FLOPs available.
        tokens_per_param: Chinchilla ratio (default 20).

    Returns:
        (optimal_params, optimal_tokens)
    """
    # From C = 6 * N * D and D = k * N:
    # C = 6 * N * k * N = 6k * N^2
    # N = sqrt(C / (6k))
    k = tokens_per_param
    n_params = math.sqrt(compute_budget / (6 * k))
    n_tokens = k * n_params

    return int(n_params), int(n_tokens)


def optimal_allocation_kaplan(
    compute_budget: float,
    alpha_n: float = 0.076,
    alpha_d: float = 0.095,
    alpha_c: float = 0.050,
) -> Tuple[int, int]:
    """Kaplan et al. optimal allocation.

    Kaplan found that N should scale faster than D:
    N_opt ∝ C^alpha_n, D_opt ∝ C^alpha_d

    Args:
        compute_budget: Total FLOPs.
        alpha_n: Scaling exponent for params.
        alpha_d: Scaling exponent for tokens.
        alpha_c: Scaling exponent for compute.

    Returns:
        (optimal_params, optimal_tokens)
    """
    # Normalize: C = 6 * N * D
    # N_opt = a * C^alpha_n, D_opt = b * C^alpha_d
    # Since alpha_n + alpha_d ≈ 1 (roughly)
    n_params = (compute_budget / 6) ** (alpha_n / (alpha_n + alpha_d))
    n_tokens = compute_budget / (6 * n_params)

    return int(n_params), int(n_tokens)


def compare_allocations(compute_budget: float) -> Dict[str, Tuple[int, int]]:
    """Compare Chinchilla vs Kaplan allocation strategies.

    Args:
        compute_budget: Total FLOPs.

    Returns:
        Dict with strategy name → (params, tokens).
    """
    return {
        "chinchilla": optimal_allocation_chinchilla(compute_budget),
        "kaplan": optimal_allocation_kaplan(compute_budget),
    }
