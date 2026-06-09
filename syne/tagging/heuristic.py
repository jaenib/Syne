"""Heuristic DSP tagger.

Maps the extracted feature groups onto interpretable track-level tags using
transparent rules (no training data required). Each tag is therefore fully
explainable in terms of its inputs — the point of a "semantic-aware
interpretable" tagger. A learned model can later replace this class behind the
:class:`~syne.tagging.base.Tagger` interface.
"""

from __future__ import annotations

import numpy as np

from syne.features import FeatureBundle
from syne.semantics.schema import (
    EnergyTags,
    GenreHint,
    MoodTags,
    RhythmTags,
    TimbreTags,
    TonalTags,
    TrackTags,
)
from syne.tagging import vocab


class HeuristicTagger:
    name = "heuristic-dsp"

    def tag(self, features: FeatureBundle) -> TrackTags:
        rhythm = self._rhythm(features)
        tonal = self._tonal(features)
        timbre = self._timbre(features)
        energy = self._energy(features)
        mood = self._mood(features, tonal)
        genre_hints = self._genre_hints(features, rhythm, timbre, energy)
        return TrackTags(
            rhythm=rhythm,
            tonal=tonal,
            mood=mood,
            energy=energy,
            timbre=timbre,
            genre_hints=genre_hints,
        )

    # ------------------------------------------------------------------ #
    def _rhythm(self, f: FeatureBundle) -> RhythmTags:
        r = f.rhythm
        return RhythmTags(
            tempo_bpm=round(r.tempo, 2),
            tempo_category=vocab.tempo_category(r.tempo),
            beat_strength=round(r.pulse_clarity, 4),
            regularity=round(r.regularity, 4),
            time_feel=vocab.time_feel(r.swing, r.regularity),
        )

    def _tonal(self, f: FeatureBundle) -> TonalTags:
        t = f.tonal
        return TonalTags(
            key=t.key,
            mode=t.mode,
            key_confidence=round(t.key_confidence, 4),
            chroma_centroid=round(t.chroma_centroid, 4),
        )

    def _timbre(self, f: FeatureBundle) -> TimbreTags:
        t = f.timbre
        return TimbreTags(
            brightness=round(t.brightness, 4),
            warmth=round(t.warmth, 4),
            roughness=round(t.roughness, 4),
            descriptors=vocab.timbre_descriptors(t.brightness, t.warmth, t.roughness),
        )

    def _energy(self, f: FeatureBundle) -> EnergyTags:
        level = f.dynamics.energy_level
        danceability = float(
            np.clip(f.rhythm.regularity * f.rhythm.pulse_clarity * (0.5 + 0.5 * level),
                    0.0, 1.0)
        )
        return EnergyTags(
            level=round(level, 4),
            dynamic_range=round(f.dynamics.dynamic_range, 4),
            danceability=round(danceability, 4),
            label=vocab.energy_label(level),
        )

    def _mood(self, f: FeatureBundle, tonal: TonalTags) -> MoodTags:
        # Arousal: physical activeness — energy, tempo, attack density, motion.
        tempo_n = float(np.clip((f.rhythm.tempo - 50.0) / 130.0, 0.0, 1.0))
        flux_n = float(np.clip(np.mean(f.structure.flux) / (np.max(f.structure.flux) + 1e-9), 0.0, 1.0))
        arousal = float(np.clip(
            0.45 * f.dynamics.energy_level
            + 0.30 * tempo_n
            + 0.15 * f.rhythm.pulse_clarity
            + 0.10 * flux_n,
            0.0, 1.0,
        ))

        # Valence: positive affect — major mode, brightness, wide dynamics.
        mode_term = 1.0 if tonal.mode == "major" else 0.0
        valence = float(np.clip(
            0.45 * mode_term
            + 0.35 * f.timbre.brightness
            + 0.20 * f.dynamics.dynamic_range,
            0.0, 1.0,
        ))

        return MoodTags(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            label=vocab.mood_label(valence, arousal),
            descriptors=vocab.mood_descriptors(valence, arousal, tonal.mode),
        )

    # ------------------------------------------------------------------ #
    def _genre_hints(
        self, f: FeatureBundle, rhythm: RhythmTags, timbre: TimbreTags, energy: EnergyTags
    ) -> list[GenreHint]:
        """Coarse, low-confidence genre leanings from feature combinations.

        These are explicitly *hints* — flagged with modest confidence so a
        renderer (or a user) treats them as suggestive, not authoritative.
        """
        bpm = rhythm.tempo_bpm
        bright = timbre.brightness
        rough = timbre.roughness
        warm = timbre.warmth
        lvl = energy.level
        dance = energy.danceability

        scored = {
            "electronic": 0.6 * dance + 0.3 * (bpm >= 115) + 0.2 * bright,
            "ambient": 0.6 * (lvl < 0.35) + 0.3 * warm + 0.2 * (bpm < 90),
            "rock": 0.5 * rough + 0.3 * (lvl > 0.6) + 0.2 * (90 <= bpm <= 160),
            "acoustic": 0.5 * f.dynamics.dynamic_range + 0.3 * warm + 0.2 * (rough < 0.3),
            "hiphop": 0.5 * (70 <= bpm <= 100) + 0.3 * dance + 0.2 * warm,
        }
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        hints = []
        for label, score in ranked[:3]:
            conf = float(np.clip(score, 0.0, 1.0)) * 0.7  # cap: these are hints
            if conf >= 0.15:
                hints.append(GenreHint(label=label, confidence=round(conf, 4)))
        return hints
