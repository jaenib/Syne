"""Small numeric helpers shared across the pipeline."""

from __future__ import annotations

import numpy as np


def normalize01(x: np.ndarray, *, robust: bool = True) -> np.ndarray:
    """Scale a signal into 0..1.

    With ``robust`` the 5th/95th percentiles define the range, which keeps a
    few transient spikes from squashing everything else toward zero — handy
    for envelopes a renderer will bind to visual parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if robust:
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    else:
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def to_list(x: np.ndarray, *, decimals: int = 5) -> list[float]:
    """Round and convert an array to a JSON-friendly list of floats."""
    return [round(float(v), decimals) for v in np.asarray(x).ravel()]
