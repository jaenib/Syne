"""Real-time streaming analysis: causal feature extraction + streaming tagger.

This is the heart of the real-time-first design. Audio arrives in blocks (from
a microphone, system capture, or file playback); the extractor maintains a
short ring buffer and emits one :class:`~syne.semantics.schema.SemanticFrame`
per hop using **only past samples** — no whole-file statistics, no look-ahead.

Two pieces:

* :class:`AdaptiveNormalizer` — replaces the offline 5/95-percentile scaling
  (which needed the whole track) with a causal AGC-style range tracker.
* :class:`StreamingFeatureExtractor` — per-hop rFFT features (energy, bands,
  chroma, brightness, flux, onset) plus running tempo / key / novelty.

The :class:`StreamingTagger` protocol is the swap point requested for the
architecture: the heuristic mock (:class:`HeuristicStreamingTagger`) implements
``push(samples) -> list[SemanticFrame]``; a data-driven model later implements
the same method and nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from syne.semantics.schema import SemanticFrame

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (major / minor).
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# --------------------------------------------------------------------------- #
class AdaptiveNormalizer:
    """Causal 0..1 normalizer with fast attack, slow release on the range.

    ``hi`` jumps up instantly to a new peak and decays back slowly; ``lo``
    mirrors it. This adapts to changing loudness/spectral conditions the way an
    automatic gain control does, without ever looking at future samples.
    """

    def __init__(self, release: float = 6e-3, floor: float = 1e-5):
        self.release = release
        self.floor = floor
        self.lo: float | None = None
        self.hi: float | None = None

    def reset(self) -> None:
        self.lo = self.hi = None

    def __call__(self, x: float) -> float:
        if self.hi is None:
            self.lo, self.hi = x, x + self.floor
            return 0.0
        self.hi = x if x > self.hi else self.hi + (x - self.hi) * self.release
        self.lo = x if x < self.lo else self.lo + (x - self.lo) * self.release
        rng = self.hi - self.lo
        if rng < self.floor:
            return 0.0
        return float(min(1.0, max(0.0, (x - self.lo) / rng)))


# --------------------------------------------------------------------------- #
@dataclass
class FrameFeatures:
    """Raw per-hop features the heuristic tagger interprets (internal)."""

    t: float
    energy: float                # 0..1 (adaptive)
    energy_raw: float            # linear RMS (for running dynamics)
    brightness: float            # 0..1 (adaptive)
    flux: float                  # 0..1 (adaptive)
    onset: float                 # 0..1 (adaptive)
    flatness: float              # 0..1 (noisiness)
    bands: list[float]           # 0..1 per band (adaptive)
    chroma: list[float]          # 12, L1-normalized
    tempo_bpm: float             # running estimate
    key: str
    mode: str
    novelty: float               # 0..1 running spectral change


class StreamingFeatureExtractor:
    """Causal, hop-synchronous feature extraction over a ring buffer."""

    def __init__(
        self,
        sample_rate: int,
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_bands: int = 3,
        tempo_window_s: float = 6.0,
    ):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.fps = sample_rate / hop_length
        self.window = np.hanning(n_fft).astype(np.float32)

        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self._freqs = freqs
        # log-spaced band masks (bass / mid / treble ...)
        fmin = max(freqs[1], 20.0)
        edges = np.logspace(np.log10(fmin), np.log10(sample_rate / 2), n_bands + 1)
        self._band_masks = [(freqs >= edges[b]) & (freqs < edges[b + 1])
                            for b in range(n_bands)]
        self._chroma_fb = _chroma_filterbank(sample_rate, n_fft)

        self.norm_energy = AdaptiveNormalizer()
        self.norm_bright = AdaptiveNormalizer()
        self.norm_flux = AdaptiveNormalizer()
        self.norm_onset = AdaptiveNormalizer()
        self.norm_bands = [AdaptiveNormalizer() for _ in range(n_bands)]

        self._tempo_len = int(tempo_window_s * self.fps)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._prev_mag: np.ndarray | None = None
        self._onset_hist = np.zeros(self._tempo_len, dtype=np.float32)
        self._run_chroma = np.zeros(12, dtype=np.float32)
        self._slow_spec = None          # long-timescale band+chroma profile
        self._fast_spec = None          # short-timescale profile
        self._tempo = 120.0
        self._n = 0                     # hop counter
        for nrm in [self.norm_energy, self.norm_bright, self.norm_flux,
                    self.norm_onset, *self.norm_bands]:
            nrm.reset()

    # ------------------------------------------------------------------ #
    def push(self, samples: np.ndarray) -> list[FrameFeatures]:
        """Feed an audio block; return the FrameFeatures completed by it."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        self._buf = np.concatenate([self._buf, samples])
        out = []
        while self._buf.shape[0] >= self.n_fft:
            out.append(self._frame(self._buf[: self.n_fft]))
            self._buf = self._buf[self.hop:]
        return out

    def _frame(self, frame: np.ndarray) -> FrameFeatures:
        t = self._n * self.hop / self.sr
        mag = np.abs(np.fft.rfft(frame * self.window))
        power = mag * mag

        energy_raw = float(np.sqrt(np.mean(frame * frame)) + 1e-12)
        total = float(mag.sum()) + 1e-9
        centroid = float((self._freqs * mag).sum() / total)
        bands_raw = [float(np.sqrt(power[m].mean())) if m.any() else 0.0
                     for m in self._band_masks]

        # flux / onset: half-wave-rectified spectral difference (causal)
        if self._prev_mag is None:
            flux_raw = 0.0
        else:
            flux_raw = float(np.sqrt(np.sum(np.maximum(mag - self._prev_mag, 0.0) ** 2)))
        self._prev_mag = mag

        # spectral flatness (geometric/arithmetic mean) -> noisiness 0..1
        flatness = float(np.exp(np.mean(np.log(mag + 1e-9))) / (mag.mean() + 1e-9))

        # chroma (causal, from this frame's spectrum)
        chroma = self._chroma_fb @ power
        cs = chroma.sum()
        chroma = chroma / cs if cs > 0 else chroma

        # running key (decaying chroma accumulation)
        self._run_chroma = 0.96 * self._run_chroma + 0.04 * chroma
        key, mode = _estimate_key(self._run_chroma)

        # running tempo from the onset-envelope history (autocorrelation)
        self._onset_hist = np.roll(self._onset_hist, -1)
        self._onset_hist[-1] = flux_raw
        if self._n % 8 == 0 and self._n > self.fps:      # refresh ~5x/sec
            self._tempo = _estimate_tempo(self._onset_hist, self.fps, self._tempo)

        # online novelty: divergence of fast vs slow spectral profile
        prof = np.concatenate([np.asarray(bands_raw), chroma])
        if self._slow_spec is None:
            self._slow_spec = prof.copy()
            self._fast_spec = prof.copy()
        self._fast_spec = 0.6 * self._fast_spec + 0.4 * prof
        self._slow_spec = 0.985 * self._slow_spec + 0.015 * prof
        novelty = _cosine_distance(self._fast_spec, self._slow_spec)

        ff = FrameFeatures(
            t=t,
            energy=self.norm_energy(energy_raw),
            energy_raw=energy_raw,
            brightness=self.norm_bright(centroid),
            flux=self.norm_flux(flux_raw),
            onset=self.norm_onset(flux_raw),
            flatness=float(np.clip(flatness, 0.0, 1.0)),
            bands=[n(b) for n, b in zip(self.norm_bands, bands_raw)],
            chroma=[round(float(c), 4) for c in chroma],
            tempo_bpm=round(self._tempo, 2),
            key=key,
            mode=mode,
            novelty=float(np.clip(novelty, 0.0, 1.0)),
        )
        self._n += 1
        return ff


# --------------------------------------------------------------------------- #
@runtime_checkable
class StreamingTagger(Protocol):
    """Push audio blocks, get back SemanticFrames. The mock <-> model seam."""

    name: str
    sample_rate: int
    hop_length: int

    def reset(self) -> None: ...
    def push(self, samples: np.ndarray) -> list[SemanticFrame]: ...


class HeuristicStreamingTagger:
    """Causal heuristic tagger — a mock to be replaced by a data-driven one.

    Wraps :class:`StreamingFeatureExtractor` and maps each frame's features to
    instantaneous semantics (valence/arousal, beat/section events). All state is
    causal and running, so it behaves identically live and over a file.
    """

    name = "heuristic-stream"

    def __init__(self, sample_rate: int, *, hop_length: int = 512, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fx = StreamingFeatureExtractor(sample_rate, n_fft=n_fft, hop_length=hop_length)
        self.reset()

    def reset(self) -> None:
        self.fx.reset()
        self._val = 0.5
        self._aro = 0.5
        self._onset_floor = 0.0
        self._refractory = 0           # frames since last beat
        self._sect_refractory = 0

    def push(self, samples: np.ndarray) -> list[SemanticFrame]:
        return [self._tag(ff) for ff in self.fx.push(samples)]

    def _tag(self, ff: FrameFeatures) -> SemanticFrame:
        tempo_n = float(np.clip((ff.tempo_bpm - 50.0) / 130.0, 0.0, 1.0))
        # instantaneous arousal/valence, smoothed (running) for stability
        aro = 0.45 * ff.energy + 0.30 * tempo_n + 0.15 * ff.onset + 0.10 * ff.flux
        mode_term = 1.0 if ff.mode == "major" else 0.0
        val = 0.45 * mode_term + 0.35 * ff.brightness + 0.20 * (1.0 - ff.flatness)
        self._aro += 0.2 * (float(np.clip(aro, 0, 1)) - self._aro)
        self._val += 0.2 * (float(np.clip(val, 0, 1)) - self._val)

        # beat: onset rises above an adaptive floor, with a refractory gap
        self._onset_floor = max(ff.onset, self._onset_floor * 0.92)
        self._refractory += 1
        beat = ff.onset > 0.55 and ff.onset >= self._onset_floor * 0.9 and self._refractory > 4
        if beat:
            self._refractory = 0

        # section change: sustained novelty, rate-limited
        self._sect_refractory += 1
        section = ff.novelty > 0.25 and self._sect_refractory > int(2 * self.fx.fps)
        if section:
            self._sect_refractory = 0

        return SemanticFrame(
            t=round(ff.t, 4),
            energy=round(ff.energy, 4),
            brightness=round(ff.brightness, 4),
            flux=round(ff.flux, 4),
            onset=round(ff.onset, 4),
            flatness=round(ff.flatness, 4),
            bands=[round(b, 4) for b in ff.bands],
            chroma=ff.chroma,
            tempo_bpm=ff.tempo_bpm,
            key=ff.key,
            mode=ff.mode,
            valence=round(self._val, 4),
            arousal=round(self._aro, 4),
            beat=bool(beat),
            section_change=bool(section),
        )


# --------------------------------------------------------------------------- #
def _chroma_filterbank(sr: int, n_fft: int) -> np.ndarray:
    """(12, n_bins) matrix mapping power-spectrum bins to pitch classes."""
    import librosa
    return librosa.filters.chroma(sr=sr, n_fft=n_fft).astype(np.float32)


def _estimate_key(chroma: np.ndarray) -> tuple[str, str]:
    if not np.any(chroma):
        return "C", "major"
    best = (-2.0, 0, "major")
    for name, base in (("major", _MAJOR), ("minor", _MINOR)):
        for tonic in range(12):
            r = _pearson(chroma, np.roll(base, tonic))
            if r > best[0]:
                best = (r, tonic, name)
    return PITCH_CLASSES[best[1]], best[2]


def _estimate_tempo(onset_hist: np.ndarray, fps: float, prev: float) -> float:
    env = onset_hist - onset_hist.mean()
    if not np.any(env):
        return prev
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lag_min = int(fps * 60.0 / 200.0)        # 200 BPM
    lag_max = min(int(fps * 60.0 / 40.0), len(ac) - 1)   # 40 BPM
    if lag_max <= lag_min:
        return prev
    lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
    bpm = 60.0 * fps / max(lag, 1)
    return float(0.8 * prev + 0.2 * bpm)     # smooth


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float((1.0 - np.dot(a, b) / denom) / 2.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


__all__ = [
    "AdaptiveNormalizer",
    "FrameFeatures",
    "StreamingFeatureExtractor",
    "StreamingTagger",
    "HeuristicStreamingTagger",
]
