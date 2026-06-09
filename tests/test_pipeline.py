"""End-to-end pipeline tests on synthetic audio."""

from __future__ import annotations

from syne.morph.drivers import derive_drivers
from syne.pipeline import analyze_clip
from syne.semantics.schema import SCHEMA_VERSION


def test_profile_structure(synth_clip):
    profile = analyze_clip(synth_clip)
    assert profile.schema_version == SCHEMA_VERSION
    assert profile.meta.duration > 0
    assert profile.meta.sample_rate == synth_clip.sample_rate


def test_track_tags_in_range(synth_clip):
    t = analyze_clip(synth_clip).tags
    assert t.rhythm.tempo_bpm > 0
    assert t.rhythm.tempo_category in {"slow", "moderate", "fast", "very fast"}
    assert 0.0 <= t.mood.valence <= 1.0
    assert 0.0 <= t.mood.arousal <= 1.0
    assert 0.0 <= t.energy.level <= 1.0
    assert 0.0 <= t.timbre.brightness <= 1.0
    assert t.tonal.mode in {"major", "minor"}


def test_detects_tempo_near_120(synth_clip):
    tempo = analyze_clip(synth_clip).tags.rhythm.tempo_bpm
    # allow octave errors (60/120/240) common in beat tracking
    assert any(abs(tempo - cand) < 8 for cand in (60, 120, 240))


def test_detects_c_major(synth_clip):
    tonal = analyze_clip(synth_clip).tags.tonal
    assert tonal.key == "C"
    assert tonal.mode == "major"


def test_timelines_aligned(synth_clip):
    tl = analyze_clip(synth_clip).timelines
    n = len(tl.times)
    assert n > 0
    for sig in (tl.energy, tl.brightness, tl.flux, tl.onset_strength):
        assert len(sig) == n
        assert all(0.0 <= v <= 1.0 for v in sig)
    assert len(tl.chroma) == n
    assert all(len(frame) == 12 for frame in tl.chroma)
    assert len(tl.segments) >= 1


def test_morph_drivers(synth_clip):
    profile = analyze_clip(synth_clip)
    drivers = derive_drivers(profile)
    assert set(drivers.dynamic) >= {"scale", "pulse", "turbulence", "brightness"}
    for channel in drivers.dynamic.values():
        assert len(channel) == len(drivers.times)
    assert 0.0 <= drivers.static["base_hue"] <= 360.0
    # every dynamic channel should have a documented binding
    for name in drivers.dynamic:
        assert name in drivers.bindings
