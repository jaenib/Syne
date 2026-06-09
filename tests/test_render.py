"""Tests for the semantic -> Lorenz mapping and integrator."""

from __future__ import annotations

import numpy as np

from syne.pipeline import analyze_clip
from syne.render.lorenz import build_control, integrate


def test_control_channels_aligned(synth_clip):
    profile = analyze_clip(synth_clip)
    c = build_control(profile)
    n = len(c.times)
    for ch in (c.rho, c.sigma, c.beta, c.speed, c.kick, c.jitter, c.glow, c.hue):
        assert len(ch) == n
    # every connector should be documented
    for key in ("rho", "sigma", "beta", "speed", "kick", "jitter", "glow", "hue"):
        assert key in c.bindings


def test_parameters_in_meaningful_ranges(synth_clip):
    c = build_control(analyze_clip(synth_clip))
    assert all(10.0 <= r <= 80.0 for r in c.rho)        # plausibly chaotic regime
    assert all(4.0 <= s <= 20.0 for s in c.sigma)
    assert all(1.5 <= b <= 4.0 for b in c.beta)
    assert all(0.0 <= h <= 1.0 for h in c.hue)
    assert 0.0 <= c.saturation <= 1.0


def test_rho_tracks_energy():
    """Higher arousal/energy must push rho higher (more chaos)."""
    from dataclasses import replace
    from syne.semantics.schema import MoodTags

    base = analyze_clip  # placeholder to keep import grouping tidy

    # build two profiles differing only in arousal/energy by editing tags
    from tests.conftest import _synth_track  # type: ignore
    from syne.audio import from_array

    clip = from_array(_synth_track(), 22050)
    profile = analyze_clip(clip)

    calm = profile
    calm.tags.mood = replace(calm.tags.mood, arousal=0.05)
    calm.tags.energy.level = 0.05
    rho_calm = np.mean(build_control(calm).rho)

    profile2 = analyze_clip(clip)
    profile2.tags.mood = replace(profile2.tags.mood, arousal=0.95)
    profile2.tags.energy.level = 0.95
    rho_hot = np.mean(build_control(profile2).rho)

    assert rho_hot > rho_calm + 5.0


def test_integrate_is_finite_and_bounded(synth_clip):
    c = build_control(analyze_clip(synth_clip))
    traj = integrate(c, fps=20, substeps=4, max_seconds=3.0)
    assert traj.points.ndim == 2 and traj.points.shape[1] == 3
    assert np.all(np.isfinite(traj.points))
    # Lorenz stays bounded; nothing should blow up to absurd magnitudes
    assert np.max(np.abs(traj.points)) < 200.0
    # frame_end is monotonic non-decreasing and indexes into points
    assert np.all(np.diff(traj.frame_end) >= 0)
    assert traj.frame_end[-1] <= len(traj.points)
