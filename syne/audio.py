"""Audio loading and preprocessing.

Thin wrapper around librosa so the rest of the pipeline works with a small,
predictable object instead of juggling sample-rate / mono-mixing concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

DEFAULT_SR = 22050
DEFAULT_HOP = 512
DEFAULT_N_FFT = 2048


@dataclass
class AudioClip:
    """A loaded, mono, float32 audio signal plus analysis parameters."""

    samples: np.ndarray          # mono float32, shape (n,)
    sample_rate: int
    source: str
    channels: int                # original channel count before downmix
    hop_length: int = DEFAULT_HOP
    n_fft: int = DEFAULT_N_FFT

    @property
    def duration(self) -> float:
        return float(len(self.samples) / self.sample_rate)

    @property
    def hop_seconds(self) -> float:
        return self.hop_length / self.sample_rate


def load_audio(
    path: str,
    *,
    sample_rate: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP,
    n_fft: int = DEFAULT_N_FFT,
) -> AudioClip:
    """Load ``path`` as mono at ``sample_rate``.

    librosa transparently handles most common formats (wav, flac, ogg, mp3,
    m4a, ...) via soundfile / audioread.
    """
    # mono=False first so we can report the original channel count, then mix.
    y, sr = librosa.load(path, sr=sample_rate, mono=False)
    if y.ndim == 1:
        channels = 1
        mono = y
    else:
        channels = y.shape[0]
        mono = librosa.to_mono(y)

    mono = np.ascontiguousarray(mono, dtype=np.float32)
    return AudioClip(
        samples=mono,
        sample_rate=sr,
        source=path,
        channels=channels,
        hop_length=hop_length,
        n_fft=n_fft,
    )


def from_array(
    samples: np.ndarray,
    sample_rate: int,
    *,
    source: str = "<array>",
    hop_length: int = DEFAULT_HOP,
    n_fft: int = DEFAULT_N_FFT,
) -> AudioClip:
    """Build an :class:`AudioClip` directly from a numpy array (for tests)."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        samples = librosa.to_mono(samples)
    return AudioClip(
        samples=samples,
        sample_rate=sample_rate,
        source=source,
        channels=1,
        hop_length=hop_length,
        n_fft=n_fft,
    )
