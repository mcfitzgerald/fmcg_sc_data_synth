"""Statistical distribution helpers for generating realistic FMCG networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


def zipf_weights(n: int, alpha: float = 1.05) -> np.ndarray:
    """
    Generate Zipf distribution weights (popularity).

    Useful for:
    - SKU popularity (few items drive most volume)
    - Supplier spend (Pareto principle)
    - City sizes

    Args:
        n: Number of items
        alpha: Skew parameter (higher = more skew). Default 1.05 (near Zipf's law).

    Returns:
        Array of normalized weights summing to 1.0
    """
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / np.sum(weights)


def preferential_attachment(
    existing_degrees: list[int] | np.ndarray,
    m: int = 1,
    rng: Generator | None = None,
) -> list[int]:
    """
    Barabasi-Albert preferential attachment ("Rich get richer").

    Used for network topology where hubs (RDCs) connect to many stores,
    but new nodes prefer connecting to existing hubs.

    Args:
        existing_degrees: Current degree count for each existing node.
        m: Number of connections to make.
        rng: NumPy random generator.

    Returns:
        Indices of nodes to connect to.
    """
    if rng is None:
        rng = np.random.default_rng()

    degrees = np.asarray(existing_degrees)

    # Add 1 to avoid division by zero for isolated nodes (if any)
    # or to give them a non-zero chance to be picked
    weights = (degrees + 1).astype(float)
    total_weight = weights.sum()

    if total_weight == 0:
        # Fallback to uniform if no degrees
        probs = np.ones(len(degrees)) / len(degrees)
    else:
        probs = weights / total_weight

    # Ensure m is not greater than population size
    m = min(m, len(degrees))

    # Select m indices based on probability
    selected = rng.choice(len(degrees), size=m, replace=False, p=probs)
    return [int(x) for x in selected]
