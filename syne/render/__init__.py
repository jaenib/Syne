"""Rendering: map a semantic profile onto a morphing Lorenz attractor.

``lorenz`` holds the semantic -> Lorenz-parameter mapping (the connectors) and
the integrator; ``raster`` renders the resulting trajectory to a GIF offline.
A browser-based, audio-synced viewer lives in ``web/`` and consumes the same
exported control JSON.
"""

from syne.render.lorenz import LorenzControl, Trajectory, build_control, integrate

__all__ = ["LorenzControl", "Trajectory", "build_control", "integrate"]
