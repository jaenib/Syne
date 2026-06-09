"""Schema serialization round-trip tests."""

from __future__ import annotations

from syne.pipeline import analyze_clip
from syne.semantics.schema import SemanticProfile


def test_json_roundtrip(synth_clip, tmp_path):
    profile = analyze_clip(synth_clip)

    path = tmp_path / "track.semantic.json"
    profile.save(str(path))
    reloaded = SemanticProfile.load(str(path))

    assert reloaded.schema_version == profile.schema_version
    assert reloaded.meta.source == profile.meta.source
    assert reloaded.tags.tonal.key == profile.tags.tonal.key
    assert reloaded.tags.rhythm.tempo_bpm == profile.tags.rhythm.tempo_bpm
    assert len(reloaded.timelines.times) == len(profile.timelines.times)
    assert reloaded.to_dict() == profile.to_dict()


def test_from_dict_handles_minimal_optionals(synth_clip):
    d = analyze_clip(synth_clip).to_dict()
    d["tags"]["genre_hints"] = []
    d["timelines"]["beats"] = []
    d["timelines"]["segments"] = []
    profile = SemanticProfile.from_dict(d)
    assert profile.tags.genre_hints == []
    assert profile.timelines.segments == []
