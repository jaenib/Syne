"""Interpretable vocabularies and the rules that map features to words.

Kept separate from the tagger so the vocabulary can be tuned, translated, or
extended without touching the feature math.
"""

from __future__ import annotations


def tempo_category(bpm: float) -> str:
    if bpm < 70:
        return "slow"
    if bpm < 110:
        return "moderate"
    if bpm < 150:
        return "fast"
    return "very fast"


def time_feel(swing: float, regularity: float) -> str:
    if regularity < 0.4:
        return "loose"
    if swing > 0.4:
        return "swung"
    return "straight"


def energy_label(level: float) -> str:
    if level < 0.33:
        return "calm"
    if level < 0.66:
        return "driving"
    return "intense"


# Valence/arousal quadrant labels (Russell's circumplex), plus a few nuances.
def mood_label(valence: float, arousal: float) -> str:
    if arousal >= 0.5 and valence >= 0.5:
        return "energetic"          # happy + active
    if arousal >= 0.5 and valence < 0.5:
        return "tense"              # angry / agitated
    if arousal < 0.5 and valence >= 0.5:
        return "serene"             # peaceful / content
    return "melancholic"            # sad / subdued


def mood_descriptors(valence: float, arousal: float, mode: str) -> list[str]:
    out: list[str] = []
    if valence >= 0.65:
        out.append("bright")
    elif valence <= 0.35:
        out.append("dark")
    if arousal >= 0.65:
        out.append("excited")
    elif arousal <= 0.35:
        out.append("relaxed")
    out.append("uplifting" if mode == "major" else "somber")
    return out


def timbre_descriptors(brightness: float, warmth: float, roughness: float) -> list[str]:
    out: list[str] = []
    out.append("bright" if brightness >= 0.5 else "dark")
    if warmth >= 0.5:
        out.append("warm")
    if roughness >= 0.5:
        out.append("rough")
    elif roughness <= 0.25:
        out.append("smooth")
    return out
