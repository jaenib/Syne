"""Morph drivers: the bridge from semantics to a future geometry renderer.

This module does **not** render anything. It translates a
:class:`~syne.semantics.schema.SemanticProfile` into a compact set of
named, renderer-agnostic *control channels* — the suggested contract a
morphing-geometry renderer can bind to without re-deriving anything.

Two kinds of channels:

* **static**  — one value for the whole track (e.g. base rotation speed from
  tempo, palette from key/mode). Good for initializing a scene.
* **dynamic** — a 0..1 signal per frame on the timeline grid (e.g. ``scale``
  from energy, ``pulse`` from onsets). Good for animating geometry over time.

Keeping this here means the renderer stays a thin consumer: read channels,
map them to geometry parameters, draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from syne.semantics.schema import SemanticProfile

# A renderer-friendly hue per pitch class (degrees on the color wheel),
# spiralling the circle of fifths so harmonically near keys sit near in hue.
_CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]


@dataclass
class MorphDrivers:
    """Normalized control channels derived from a semantic profile."""

    hop_seconds: float
    times: list[float]
    static: dict[str, Any] = field(default_factory=dict)
    dynamic: dict[str, list[float]] = field(default_factory=dict)
    # human-readable note on what each channel is intended to drive
    bindings: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_seconds": self.hop_seconds,
            "times": self.times,
            "static": self.static,
            "dynamic": self.dynamic,
            "bindings": self.bindings,
        }


def derive_drivers(profile: SemanticProfile) -> MorphDrivers:
    tl = profile.timelines
    tags = profile.tags

    # --- static channels (scene initialization) ------------------------ #
    hue = _key_to_hue(tags.tonal.key)
    static = {
        # rotation: slow for calm tracks, faster as tempo climbs
        "rotation_speed": round(_tempo_to_speed(tags.rhythm.tempo_bpm), 4),
        "base_hue": hue,                                  # 0..360
        "saturation": round(0.4 + 0.5 * tags.mood.valence, 4),   # brighter = more saturated
        "palette_warmth": round(tags.timbre.warmth, 4),
        "symmetry": _mode_to_symmetry(tags.tonal.mode),   # major=even, minor=odd
        "base_complexity": round(0.3 + 0.7 * tags.timbre.roughness, 4),
        "mood": tags.mood.label,
    }

    # --- dynamic channels (per-frame animation) ------------------------ #
    dynamic = {
        "scale": tl.energy,                # overall size pulses with energy
        "pulse": tl.onset_strength,        # sharp impulses on attacks
        "turbulence": tl.flux,             # surface agitation from spectral motion
        "brightness": tl.brightness,       # emissive / glow from spectral centroid
        "hue_shift": _chroma_hue_shift(tl.chroma),  # color drift from harmony
    }

    bindings = {
        "rotation_speed": "angular velocity of the whole form",
        "base_hue": "starting color (degrees, 0..360)",
        "saturation": "color saturation",
        "palette_warmth": "blend toward warm tones",
        "symmetry": "rotational symmetry order of the geometry",
        "base_complexity": "subdivision / detail level of the mesh",
        "scale": "global size / radius over time",
        "pulse": "impulse for vertex displacement on note onsets",
        "turbulence": "noise / deformation amount",
        "brightness": "emissive intensity / glow",
        "hue_shift": "color rotation offset over time (0..1 -> 0..360)",
    }

    return MorphDrivers(
        hop_seconds=tl.hop_seconds,
        times=tl.times,
        static=static,
        dynamic=dynamic,
        bindings=bindings,
    )


# --------------------------------------------------------------------------- #
def _tempo_to_speed(bpm: float) -> float:
    # map 40..200 BPM onto 0..1
    return max(0.0, min(1.0, (bpm - 40.0) / 160.0))


def _key_to_hue(key: str) -> float:
    try:
        idx = _CIRCLE_OF_FIFTHS.index(key)
    except ValueError:
        idx = 0
    return round(idx * (360.0 / 12.0), 2)


def _mode_to_symmetry(mode: str) -> int:
    return 6 if mode == "major" else 5


def _chroma_hue_shift(chroma: list[list[float]]) -> list[float]:
    """Per-frame dominant pitch class mapped to a 0..1 hue offset."""
    out = []
    for frame in chroma:
        if not frame or max(frame) <= 0:
            out.append(0.0)
            continue
        dom = max(range(len(frame)), key=lambda i: frame[i])
        out.append(round(dom / 12.0, 5))
    return out


__all__ = ["MorphDrivers", "derive_drivers"]
