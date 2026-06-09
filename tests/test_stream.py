"""Real-time guarantees of the streaming tagger."""

from __future__ import annotations

import numpy as np

from syne.stream import AdaptiveNormalizer, HeuristicStreamingTagger


def _run(samples, sr, hop, blocks):
    t = HeuristicStreamingTagger(sr, hop_length=hop)
    t.reset()
    frames = []
    if blocks is None:
        frames += t.push(samples)
    else:
        for i in range(0, len(samples), blocks):
            frames += t.push(samples[i:i + blocks])
    return frames


def test_causal_prefix_stable(synth_clip):
    """A prefix of the audio yields exactly the prefix of the full-run frames."""
    y, sr, hop = synth_clip.samples, synth_clip.sample_rate, synth_clip.hop_length
    full = _run(y, sr, hop, None)
    half = _run(y[: len(y) // 2], sr, hop, None)
    assert len(half) > 0
    for i in range(len(half)):
        assert half[i].to_dict() == full[i].to_dict()


def test_block_size_invariant(synth_clip):
    """Frames are identical regardless of how the audio is chunked."""
    y, sr, hop = synth_clip.samples, synth_clip.sample_rate, synth_clip.hop_length
    whole = _run(y, sr, hop, None)
    chunked = _run(y, sr, hop, 997)         # awkward block size
    assert len(whole) == len(chunked)
    for a, b in zip(whole, chunked):
        assert a.to_dict() == b.to_dict()


def test_frame_fields_in_range(synth_clip):
    frames = _run(synth_clip.samples, synth_clip.sample_rate, synth_clip.hop_length, None)
    f = frames[len(frames) // 2]
    for v in (f.energy, f.brightness, f.flux, f.onset, f.flatness, f.valence, f.arousal):
        assert 0.0 <= v <= 1.0
    assert len(f.bands) == 3 and all(0.0 <= b <= 1.0 for b in f.bands)
    assert len(f.chroma) == 12
    assert f.tempo_bpm > 0
    assert f.mode in {"major", "minor"}


def test_adaptive_normalizer_causal_range():
    norm = AdaptiveNormalizer()
    out = [norm(x) for x in [0.0, 1.0, 0.5, 2.0, 0.1]]
    assert all(0.0 <= v <= 1.0 for v in out)
    assert out[0] == 0.0                    # first sample has no range yet
    assert out[3] == 1.0                    # a new peak maps to the top
