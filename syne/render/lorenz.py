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
    # per-frame channels (same length as ``times``)
    rho: list[float]
    sigma: list[float]
    beta: list[float]
    speed: list[float]               # dt multiplier, ~0.5..1.5
    kick: list[float]                # 0..1 onset impulse
    jitter: list[float]              # 0..1 flux noise
    glow: list[float]                # 0..1 intensity
    hue: list[float]                 # 0..1 color (from full chroma)
    sat: list[float]                 # 0..1 saturation (from tonal clarity)
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
    sat: np.ndarray                  # (N,) 0..1
    glow: np.ndarray                 # (N,) 0..1
    frame_end: np.ndarray            # (F,) index into points for each video frame
    fps: int
    rotation_speed: float
    fade: float
    scale: float = 1.0               # relative size when compositing multiple curves
    frame_pulse: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (F,) beat punch
    frame_angle: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (F,) camera yaw (rad)


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

    # --- per-frame modulation (wide ranges so the shape actually morphs) -- #
    e2 = energy ** 1.4                                          # punchier (gamma) energy
    rho = rho_base * (0.55 + 1.05 * e2) + 9.0 * onset           # energy widens, hits flare
    sigma = sigma_base * (0.80 + 0.45 * energy)
    beta = beta_base * (0.78 + 0.55 * brightness)               # real vertical movement
    speed = 0.5 + 1.4 * e2                                      # 0.5 .. 1.9
    kick = onset
    jitter = flux
    glow = 0.12 + 0.95 * e2

    # color: continuous hue from the FULL 12-D chroma (weighted circular mean
    # over the circle of fifths) so harmony — not just the loudest note —
    # paints the curve, with a gentle valence warm/cool tint.
    hue, clarity = _chroma_to_hue(tl.chroma, n)
    tint = (0.5 - tags.mood.valence) * 0.10                     # warm major / cool minor
    hue = (hue + tint) % 1.0

    # saturation per frame from tonal clarity (peaky chroma = more vivid), with
    # a high floor so even percussive / noisy passages stay colorful.
    sat = np.clip(0.62 + 0.38 * clarity - 0.15 * tags.timbre.roughness, 0.5, 1.0)

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
        "hue": "color              <- full chroma (circle of fifths) + valence tint",
        "sat": "color saturation   <- tonal clarity - timbre roughness",
        "fade": "trail persistence  <- dynamic range",
        "rotation_speed": "camera yaw <- tempo",
        "seeds": "state re-seed      <- section boundaries",
    }

    return LorenzControl(
        times=list(tl.times) if n == len(tl.times) else [0.0],
        base_dt=base_dt,
        rotation_speed=rotation_speed,
        fade=fade,
        rho=_r(rho),
        sigma=_r(sigma),
        beta=_r(beta),
        speed=_r(speed),
        kick=_r(kick),
        jitter=_r(jitter),
        glow=_r(glow),
        hue=_r(hue),
        sat=_r(sat),
        seeds=[round(float(s), 4) for s in seeds],
        bindings=bindings,
    )


def build_band_controls(
    profile: SemanticProfile, *, hue_spread: float = 0.5
) -> list[LorenzControl]:
    """One :class:`LorenzControl` per frequency band (bass/mid/treble...).

    All curves share the rhythm/harmony connectors, but each band drives its
    own wing-size (``rho``) and intensity (``glow``) from that band's energy,
    gets a hue offset for color separation, and a different ``beta`` so the
    butterflies don't perfectly coincide. Together they show the spectral
    balance morphing over time.
    """
    base = build_control(profile)
    bands = profile.timelines.bands
    if not bands:
        return [base]

    arr = np.asarray(bands, dtype=float)             # (n_frames, n_bands)
    n = len(base.times)
    if arr.ndim != 2 or arr.shape[0] != n:
        return [base]
    n_bands = arr.shape[1]
    if n_bands < 2:
        return [base]

    tags = profile.tags
    chaos = float(np.clip(0.55 * tags.mood.arousal + 0.45 * tags.energy.level, 0, 1))
    rho_base = 14.0 + 32.0 * chaos
    flux = np.asarray(profile.timelines.flux, dtype=float)
    flux = np.resize(flux, n)
    hue0 = np.asarray(base.hue, dtype=float)
    beta0 = np.asarray(base.beta, dtype=float)

    offsets = np.linspace(-hue_spread / 2, hue_spread / 2, n_bands)
    beta_mul = np.linspace(0.8, 1.25, n_bands)        # bass taller, treble pinched

    controls = []
    for b in range(n_bands):
        be = arr[:, b]
        # per-band onset: each curve punches on its OWN band's hits (bass kick,
        # treble hats) instead of all sharing one global onset.
        onset_b = np.clip(np.diff(be, prepend=be[:1]), 0.0, None)
        mx = onset_b.max()
        onset_b = onset_b / mx if mx > 0 else onset_b
        e2 = be ** 1.4
        rho_b = rho_base * (0.55 + 1.0 * e2) + 8.0 * onset_b
        glow_b = np.clip(0.10 + 0.95 * e2, 0.0, 1.0)
        speed_b = 0.5 + 1.3 * e2
        hue_b = (hue0 + offsets[b]) % 1.0
        beta_b = beta0 * beta_mul[b]
        controls.append(
            LorenzControl(
                times=base.times, base_dt=base.base_dt,
                rotation_speed=base.rotation_speed, fade=base.fade,
                rho=_r(rho_b), sigma=base.sigma, beta=_r(beta_b),
                speed=_r(speed_b), kick=_r(onset_b), jitter=base.jitter,
                glow=_r(glow_b), hue=_r(hue_b), sat=base.sat,
                seeds=base.seeds, bindings=base.bindings,
            )
        )
    return controls


def build_band_trajectories(
    profile: SemanticProfile,
    *,
    fps: int = 30,
    substeps: int = 12,
    speed_scale: float = 3.0,
    max_seconds: float | None = None,
) -> list[Trajectory]:
    """Integrate one trajectory per band, sized so they read as distinct."""
    controls = build_band_controls(profile)
    scales = (
        np.linspace(1.15, 0.72, len(controls)) if len(controls) > 1 else np.array([1.0])
    )
    trajs = []
    for i, c in enumerate(controls):
        tr = integrate(c, fps=fps, substeps=substeps, speed_scale=speed_scale,
                       max_seconds=max_seconds, seed=7 + i * 5)
        tr.scale = float(scales[i])
        trajs.append(tr)
    return trajs


# --------------------------------------------------------------------------- #
def integrate(
    control: LorenzControl,
    *,
    fps: int = 30,
    substeps: int = 6,
    max_seconds: float | None = None,
    speed_scale: float = 1.0,
    seed: int = 7,
) -> Trajectory:
    """Integrate the Lorenz ODE (RK4) under time-varying control -> Trajectory.

    ``speed_scale`` multiplies how much Lorenz-time is advanced per video frame,
    making the head travel faster (so the full attractor is traced quickly).
    """
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
    glow, hue, sat = rs(control.glow), rs(control.hue), rs(control.sat)

    # reactive envelopes: a sharp-attack / fast-decay pulse on each onset (the
    # core "punch"), and a steady camera yaw whose rate rises with tempo.
    pulse = np.zeros(n_frames)
    p = 0.0
    for f in range(n_frames):
        p = max(float(kick[f]), p * 0.80)
        pulse[f] = p
    ang_vel = 0.18 + 1.25 * control.rotation_speed             # rad/sec
    angle = np.arange(n_frames) / fps * ang_vel

    rng = np.random.default_rng(seed)
    state = np.array([0.1, 0.0, 0.0])
    pts, hcol, scol, gcol, frame_end = [], [], [], [], []
    seed_times = sorted(control.seeds)
    si = 0

    for f in range(n_frames):
        # section re-seed: a gentle nudge the attractor will reabsorb
        while si < len(seed_times) and seed_times[si] <= vt[f]:
            state = state + rng.standard_normal(3) * 0.35
            si += 1

        # head darts forward on a hit (pulse), so onsets are visible as motion
        dt = control.base_dt * max(speed[f], 0.05) * (1.0 + 1.1 * pulse[f]) \
            * speed_scale / substeps
        for _ in range(substeps):
            state = _rk4(state, sigma[f], rho[f], beta[f], dt)
            if jitter[f] > 0:
                state = state + rng.standard_normal(3) * jitter[f] * 0.12
            pts.append(state.copy())
            hcol.append(hue[f])
            scol.append(sat[f])
            gcol.append(glow[f])
        frame_end.append(len(pts))

    return Trajectory(
        points=np.asarray(pts),
        hue=np.asarray(hcol),
        sat=np.asarray(scol),
        glow=np.asarray(gcol),
        frame_end=np.asarray(frame_end),
        fps=fps,
        rotation_speed=control.rotation_speed,
        fade=control.fade,
        frame_pulse=pulse,
        frame_angle=angle,
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


# hue position of each pitch class on the circle of fifths (so fifth-related
# notes are adjacent in color): pitch class pc -> ((pc*7) % 12) / 12
_PC_HUE = ((np.arange(12) * 7) % 12) / 12.0
_PC_ANGLE = 2 * np.pi * _PC_HUE


def _chroma_to_hue(chroma: list[list[float]], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Continuous hue + tonal clarity from the full 12-D chroma per frame.

    Hue is the chroma-weighted circular mean over the circle-of-fifths color
    wheel; clarity is the peak share of the dominant pitch class (1.0 = a single
    pure tone, low = energy spread across many notes / noise).
    """
    hue = np.zeros(n)
    clarity = np.zeros(n)
    for i, frame in enumerate(chroma[:n]):
        w = np.asarray(frame, dtype=float)
        total = w.sum()
        if total <= 0:
            continue
        x = float(np.sum(w * np.cos(_PC_ANGLE)))
        y = float(np.sum(w * np.sin(_PC_ANGLE)))
        hue[i] = (np.arctan2(y, x) % (2 * np.pi)) / (2 * np.pi)
        clarity[i] = float(w.max() / total)
    return hue, clarity


def _r(x: np.ndarray, d: int = 5) -> list[float]:
    return [round(float(v), d) for v in np.asarray(x).ravel()]


__all__ = [
    "LorenzControl", "Trajectory", "build_control", "integrate",
    "build_band_controls", "build_band_trajectories",
]
