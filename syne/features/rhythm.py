"""Rhythm features: tempo, beats, pulse clarity, regularity, swing."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from syne.audio import AudioClip


@dataclass
class RhythmFeatures:
    tempo: float                 # BPM
    beat_times: np.ndarray       # seconds
    onset_envelope: np.ndarray   # per-frame onset strength (raw)
    pulse_clarity: float         # 0..1, autocorrelation peak prominence
    regularity: float            # 0..1, steadiness of inter-beat intervals
    swing: float                 # 0..1, off-beat displacement estimate


def extract(clip: AudioClip) -> RhythmFeatures:
    y, sr, hop = clip.samples, clip.sample_rate, clip.hop_length

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop
    )
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    pulse_clarity = _pulse_clarity(onset_env)
    regularity = _regularity(beat_times)
    swing = _swing(onset_env, beat_frames)

    return RhythmFeatures(
        tempo=tempo,
        beat_times=beat_times,
        onset_envelope=onset_env,
        pulse_clarity=pulse_clarity,
        regularity=regularity,
        swing=swing,
    )


def _pulse_clarity(onset_env: np.ndarray) -> float:
    """Prominence of the strongest periodicity in the onset envelope."""
    if onset_env.size < 4 or not np.any(onset_env):
        return 0.0
    env = onset_env - onset_env.mean()
    ac = librosa.autocorrelate(env)
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    # ignore lag 0; strongest secondary peak ~ how clear the pulse is
    return float(np.clip(np.max(ac[1:]), 0.0, 1.0))


def _regularity(beat_times: np.ndarray) -> float:
    """1 - normalized variability of inter-beat intervals."""
    if beat_times.size < 3:
        return 0.0
    ibi = np.diff(beat_times)
    if ibi.mean() <= 0:
        return 0.0
    cv = ibi.std() / ibi.mean()          # coefficient of variation
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def _swing(onset_env: np.ndarray, beat_frames: np.ndarray) -> float:
    """Crude swing estimate: energy of the off-beat (mid-point) subdivision.

    Compares onset energy at the halfway point between consecutive beats
    against the on-beat energy. Higher -> more pronounced shuffle/swing feel.
    """
    if beat_frames.size < 3 or not np.any(onset_env):
        return 0.0
    n = onset_env.size
    on_vals, off_vals = [], []
    for a, b in zip(beat_frames[:-1], beat_frames[1:]):
        mid = (a + b) // 2
        if 0 <= a < n:
            on_vals.append(onset_env[a])
        if 0 <= mid < n:
            off_vals.append(onset_env[mid])
    if not on_vals or not off_vals:
        return 0.0
    on = float(np.mean(on_vals)) + 1e-9
    off = float(np.mean(off_vals))
    return float(np.clip(off / (on + off), 0.0, 1.0))
