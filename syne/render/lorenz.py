"""Map a semantic profile onto the Lorenz attractor — the "connectors".

The Lorenz system

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

is a good target for *meaningful* sonification because its parameters have
well-understood effects on the shape of the curve, so the mapping is not
arbitrary:

* **rho**   — the Rayleigh number, the system's bifurcation knob. Below ~24.74
  trajectories spiral into fixed points (ordered, near-periodic); above it the
  classic chaotic "butterfly" emerges and the wings grow with rho. We drive it
  from **arousal + energy**: calm music -> ordered spiral, intense music ->
  wide, chaotic wings.
* **sigma** — the Prandtl number, sets how fast trajectories are pulled toward
  the swirl (crispness/tightness of the rotation). Driven by **danceability**
  (regular, pulse-y music -> tighter, more defined swirl).
* **beta**  — geometric factor controlling vertical pinch / wing separation.
  Driven by **timbre brightness** (brighter -> more vertical stretch).
* **speed** — integration step per frame = how fast the comet head travels.
  Driven by **tempo** (base) and **energy** (per-frame), so motion accelerates
  in loud sections.
* **kick**  — per-frame impulse that jolts the state, driven by **onset
  strength** (visible flutter on note attacks / beats).
* **jitter**— per-step positional noise, driven by **spectral flux** (surface
  agitation when the spectrum is changing fast).
* **glow / hue** — render intensity from **energy**, color from **key/chroma**.
* **re-seeds** — at **section boundaries** the state is nudged, giving a visual
  "scene change" when the music's structure changes.

``build_control`` produces a :class:`LorenzControl` (all the per-frame
connector signals, resolution-independent on the profile's time grid).
``integrate`` turns that into a concrete 3-D :class:`Trajectory`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from syne.semantics.schema import SemanticProfile

# circle of fifths -> hue, so harmonically near keys get near colors
_CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]


# --------------------------------------------------------------------------- #
@dataclass
class LorenzControl:
    """Per-frame Lorenz connector signals derived from a semantic profile."""

    times: list[float]               # shared time grid (seconds)
    base_dt: float                   # lorenz-time advanced per video frame (pre-speed)
    rotation_speed: float            # camera yaw rate, 0..1
    fade: float                      # trail persistence, 0..1
    saturation: float                # color saturation from valence, 0..1
    # per-frame channels (same length as ``times``)
    rho: list[float]
    sigma: list[float]
    beta: list[float]
    speed: list[float]               # dt multiplier, ~0.5..1.5
    kick: list[float]                # 0..1 onset impulse
    jitter: list[float]              # 0..1 flux noise
    glow: list[float]                # 0..1 intensity
    hue: list[float]                 # 0..1 color
    seeds: list[float] = field(default_factory=list)   # section boundary times
    bindings: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class Trajectory:
    points: np.ndarray               # (N, 3)
    hue: np.ndarray                  # (N,) 0..1
    glow: np.ndarray                 # (N,) 0..1
    frame_end: np.ndarray            # (F,) index into points for each video frame
    fps: int
    rotation_speed: float
    fade: float
    saturation: float


# --------------------------------------------------------------------------- #
def build_control(profile: SemanticProfile) -> LorenzControl:
    tags = profile.tags
    tl = profile.timelines

    energy = np.asarray(tl.energy, dtype=float)
    brightness = np.asarray(tl.brightness, dtype=float)
    flux = np.asarray(tl.flux, dtype=float)
    onset = np.asarray(tl.onset_strength, dtype=float)
    n = len(tl.times)
    if n == 0:                       # degenerate guard
        energy = brightness = flux = onset = np.zeros(1)
        n = 1

    # --- track-level base character ----------------------------------- #
    chaos = float(np.clip(0.55 * tags.mood.arousal + 0.45 * tags.energy.level, 0, 1))
    rho_base = 14.0 + 32.0 * chaos                 # 14 (ordered) .. 46 (wild)
    sigma_base = 6.0 + 10.0 * tags.energy.danceability   # 6 .. 16
    beta_base = 1.9 + 1.6 * tags.timbre.brightness       # 1.9 .. 3.5 (8/3 ~ 2.67)
    base_dt = 0.05 * (tags.rhythm.tempo_bpm / 120.0)     # tempo sets traversal speed
    base_dt = float(np.clip(base_dt, 0.02, 0.12))

    # --- per-frame modulation ----------------------------------------- #
    rho = rho_base * (0.85 + 0.30 * energy) + 6.0 * flux         # energy widens, flux spikes
    sigma = sigma_base * (0.90 + 0.20 * energy)
    beta = beta_base * (0.95 + 0.10 * brightness)
    speed = 0.6 + 0.9 * energy                                  # 0.6 .. 1.5
    kick = onset
    jitter = flux
    glow = 0.2 + 0.8 * energy

    # color: base hue from key, drifting with the per-frame dominant pitch class
    base_hue = _key_to_hue01(tags.tonal.key)
    chroma_shift = _chroma_shift(tl.chroma, n)
    hue = (base_hue + 0.15 * chroma_shift) % 1.0

    saturation = float(np.clip(0.45 + 0.5 * tags.mood.valence, 0, 1))
    fade = float(np.clip(0.78 + 0.18 * tags.energy.dynamic_range, 0, 0.97))
    rotation_speed = float(np.clip((tags.rhythm.tempo_bpm - 40.0) / 160.0, 0.05, 1.0))
    seeds = [s.start for s in tl.segments if s.start > 0.0]

    bindings = {
        "rho": "wing size & chaos  <- arousal + energy (+ flux spikes)",
        "sigma": "swirl tightness   <- danceability",
        "beta": "vertical pinch     <- timbre brightness",
        "speed": "head velocity     <- tempo (base) + energy (per frame)",
        "kick": "impulse flutter    <- onset strength",
        "jitter": "surface noise    <- spectral flux",
        "glow": "line intensity     <- energy",
        "hue": "color              <- key + chroma drift",
        "saturation": "color saturation <- valence",
        "fade": "trail persistence  <- dynamic range",
        "rotation_speed": "camera yaw <- tempo",
        "seeds": "state re-seed      <- section boundaries",
    }

    return LorenzControl(
        times=list(tl.times) if n == len(tl.times) else [0.0],
        base_dt=base_dt,
        rotation_speed=rotation_speed,
        fade=fade,
        saturation=saturation,
        rho=_r(rho),
        sigma=_r(sigma),
        beta=_r(beta),
        speed=_r(speed),
        kick=_r(kick),
        jitter=_r(jitter),
        glow=_r(glow),
        hue=_r(hue),
        seeds=[round(float(s), 4) for s in seeds],
        bindings=bindings,
    )


# --------------------------------------------------------------------------- #
def integrate(
    control: LorenzControl,
    *,
    fps: int = 30,
    substeps: int = 6,
    max_seconds: float | None = None,
    seed: int = 7,
) -> Trajectory:
    """Integrate the Lorenz ODE (RK4) under time-varying control -> Trajectory."""
    times = np.asarray(control.times, dtype=float)
    if times.size < 2:
        times = np.array([0.0, 1.0])
    total = times[-1] if max_seconds is None else min(times[-1], max_seconds)
    n_frames = max(int(total * fps), 1)
    vt = np.arange(n_frames) / fps

    def rs(arr: list[float]) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.size != times.size:                 # robust to length mismatch
            a = np.resize(a, times.size)
        return np.interp(vt, times, a)

    rho, sigma, beta = rs(control.rho), rs(control.sigma), rs(control.beta)
    speed, kick, jitter = rs(control.speed), rs(control.kick), rs(control.jitter)
    glow, hue = rs(control.glow), rs(control.hue)

    rng = np.random.default_rng(seed)
    state = np.array([0.1, 0.0, 0.0])
    pts, hcol, gcol, frame_end = [], [], [], []
    seed_times = sorted(control.seeds)
    si = 0

    for f in range(n_frames):
        # section re-seed: a gentle nudge the attractor will reabsorb
        while si < len(seed_times) and seed_times[si] <= vt[f]:
            state = state + rng.standard_normal(3) * 0.6
            si += 1

        dt = control.base_dt * max(speed[f], 0.05) / substeps
        for _ in range(substeps):
            state = _rk4(state, sigma[f], rho[f], beta[f], dt)
            if jitter[f] > 0:
                state = state + rng.standard_normal(3) * jitter[f] * 0.12
            pts.append(state.copy())
            hcol.append(hue[f])
            gcol.append(glow[f])

        if kick[f] > 0.2:                          # visible jolt on onsets
            state = state + rng.standard_normal(3) * kick[f] * 0.7

        frame_end.append(len(pts))

    return Trajectory(
        points=np.asarray(pts),
        hue=np.asarray(hcol),
        glow=np.asarray(gcol),
        frame_end=np.asarray(frame_end),
        fps=fps,
        rotation_speed=control.rotation_speed,
        fade=control.fade,
        saturation=control.saturation,
    )


# --------------------------------------------------------------------------- #
def _deriv(s: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def _rk4(s: np.ndarray, sigma: float, rho: float, beta: float, dt: float) -> np.ndarray:
    k1 = _deriv(s, sigma, rho, beta)
    k2 = _deriv(s + 0.5 * dt * k1, sigma, rho, beta)
    k3 = _deriv(s + 0.5 * dt * k2, sigma, rho, beta)
    k4 = _deriv(s + dt * k3, sigma, rho, beta)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _key_to_hue01(key: str) -> float:
    try:
        return _CIRCLE_OF_FIFTHS.index(key) / 12.0
    except ValueError:
        return 0.0


def _chroma_shift(chroma: list[list[float]], n: int) -> np.ndarray:
    out = np.zeros(n)
    for i, frame in enumerate(chroma[:n]):
        if frame and max(frame) > 0:
            out[i] = int(np.argmax(frame)) / 12.0
    return out


def _r(x: np.ndarray, d: int = 5) -> list[float]:
    return [round(float(v), d) for v in np.asarray(x).ravel()]


__all__ = ["LorenzControl", "Trajectory", "build_control", "integrate"]
