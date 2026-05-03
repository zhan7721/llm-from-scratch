"""Scaling Laws — Exercise (stubs for you to fill in).

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

        # TODO: Linearize the power law.
        # The model is: L(x) = a * x^(-alpha) + L_inf
        # Rearrange:     L(x) - L_inf = a * x^(-alpha)
        # Take log:      log(L - L_inf) = log(a) - alpha * log(x)
        #
        # Steps:
        # 1. Compute adjusted = loss_values - l_inf
        # 2. Filter to only positive adjusted values (mask)
        # 3. Compute log_x = np.log(x_values[mask])
        # 4. Compute log_y = np.log(adjusted[mask])
        # 5. Fit a line with np.polyfit(log_x, log_y, 1)
        # 6. Extract: self.alpha = -slope, self.a = exp(intercept)

        raise NotImplementedError("Implement the power-law fitting in fit()")

    def predict(self, x: float) -> float:
        """Predict loss at given scale."""
        if self.alpha is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.a * (x ** (-self.alpha)) + self.l_inf


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
    # TODO: Implement the FLOPs estimation formula.
    # C ≈ 6 * N * D (6 multiply-adds per token per parameter)

    raise NotImplementedError("Implement estimate_compute()")


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
    # TODO: Solve for optimal N and D.
    # Given:
    #   C = 6 * N * D  (compute constraint)
    #   D = k * N      (Chinchilla ratio, k = tokens_per_param)
    #
    # Substitute D = k * N into the compute equation:
    #   C = 6 * N * k * N = 6k * N^2
    #
    # Solve for N:
    #   N = sqrt(C / (6k))
    #
    # Then: D = k * N

    raise NotImplementedError("Implement optimal_allocation_chinchilla()")
