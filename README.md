# Syne

**Semantic music tagging pipeline** — turns an audio file into structured,
interpretable semantic information, designed as a stable contract for a future
renderer that morphs abstract geometry in sync with the music.

Inspired by [SAMAT](https://github.com/andreaspatakis/samat) (*Semantic-Aware
Interpretable Multimodal Music Auto-Tagging*): every feature belongs to a named
perceptual group, and every tag is explainable in terms of its inputs. Where
SAMAT trains models on large datasets, Syne ships a transparent DSP-heuristic
tagger that runs offline out of the box — behind a pluggable interface so a
learned model can drop in later.

The included renderer maps those tags onto a **morphing Lorenz attractor** — the
chaos "butterfly" — so the geometry breathes with the music:

![Lorenz attractor morphing to music](docs/lorenz_demo.gif)

```
audio file ──▶ load ──▶ feature groups ──▶ tagger ──▶ SemanticProfile ──▶ morph drivers ──▶ (renderer)
                         rhythm/tonal/         │            (JSON contract)        │
                         timbre/dynamics/       │                                   └─ static + per-frame channels
                         structure              └─ track tags + timelines
```

## Why two layers

The semantic output is split so a renderer can both *set a scene* and *animate
it*:

- **Track-level tags** — global, grouped descriptors (tempo, key/mode, mood as
  valence/arousal, energy, timbre, genre hints).
- **Timelines** — per-frame signals on a shared time grid (energy, brightness,
  spectral flux, onset strength, 12-D chroma) plus beat times and section
  boundaries — so geometry can morph *with* the music, not just react to one
  global mood.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Core deps: `librosa`, `numpy`, `soundfile`.

## Usage

### CLI

```bash
# print a human-readable summary
syne analyze track.wav --summary

# write the semantic profile, and the renderer-ready morph drivers
syne analyze track.wav -o track.semantic.json --drivers track.drivers.json
```

### Library

```python
from syne import analyze_file, derive_drivers

profile = analyze_file("track.wav")           # -> SemanticProfile
print(profile.tags.mood.label, profile.tags.tonal.key, profile.tags.tonal.mode)

profile.save("track.semantic.json")           # the JSON contract

drivers = derive_drivers(profile)             # renderer-ready control channels
```

## The semantic contract

A `SemanticProfile` serializes to JSON. See
[`examples/example_output.semantic.json`](examples/example_output.semantic.json)
for a full sample. Shape:

```jsonc
{
  "schema_version": "1.0",
  "meta": { "source": "...", "duration": 6.0, "sample_rate": 22050, "channels": 1 },
  "tags": {
    "rhythm": { "tempo_bpm": 117.5, "tempo_category": "fast", "beat_strength": 0.7,
                "regularity": 0.9, "time_feel": "straight" },
    "tonal":  { "key": "C", "mode": "major", "key_confidence": 0.77, "chroma_centroid": 2.1 },
    "mood":   { "valence": 0.51, "arousal": 0.58, "label": "energetic",
                "descriptors": ["uplifting"] },
    "energy": { "level": 0.65, "dynamic_range": 0.4, "danceability": 0.63, "label": "driving" },
    "timbre": { "brightness": 0.08, "warmth": 0.73, "roughness": 0.1,
                "descriptors": ["dark", "warm", "smooth"] },
    "genre_hints": [{ "label": "electronic", "confidence": 0.49 }]
  },
  "timelines": {
    "hop_seconds": 0.023,
    "times":          [ ... ],   // shared time grid (seconds)
    "energy":         [ ... ],   // 0..1 per frame
    "brightness":     [ ... ],
    "flux":           [ ... ],
    "onset_strength": [ ... ],
    "chroma":         [ [12 floats], ... ],
    "beats":          [ ... ],   // beat times (s)
    "segments":       [ { "start": 0.0, "end": 3.1, "label": "section_0", "novelty": 0.0 } ]
  }
}
```

## Morph drivers — the renderer hand-off

`derive_drivers(profile)` translates the profile into renderer-agnostic control
channels so a renderer never has to re-derive anything — it just binds channels
to geometry parameters and draws.

- **static** (scene init): `rotation_speed`, `base_hue`, `saturation`,
  `palette_warmth`, `symmetry`, `base_complexity`, `mood`.
- **dynamic** (per-frame, 0..1): `scale` ← energy, `pulse` ← onsets,
  `turbulence` ← spectral flux, `brightness` ← spectral centroid,
  `hue_shift` ← harmony.

Each channel ships with a `bindings` note describing its intended geometric
meaning. See
[`examples/example_output.drivers.json`](examples/example_output.drivers.json).

## Renderer — the Lorenz butterfly

The Lorenz system

```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

is a meaningful target because its parameters have well-understood effects on
the curve, so the mapping isn't arbitrary. `syne/render/lorenz.py` owns the
**connectors** (tags → Lorenz parameters); two renderers consume them — an
offline GIF rasterizer (no GPU / ffmpeg) and an audio-synced
[web viewer](web/).

| connector | controls the curve | driven by | why |
|---|---|---|---|
| `rho` | wing size & chaos | arousal + energy (+ flux) | below ~24.74 the system spirals into fixed points (ordered); above it the chaotic wings grow with `rho` |
| `sigma` | swirl tightness | danceability | sets how fast trajectories are pulled into the rotation |
| `beta` | vertical pinch / wing separation | timbre brightness | geometric factor on the `z` axis |
| `speed` | comet-head velocity | tempo (base) + energy | integration step per frame |
| `kick` | impulse flutter | onset strength | jolts the state on attacks/beats |
| `jitter` | surface agitation | spectral flux | per-step positional noise |
| `glow` | line intensity | energy | brightness of the trail |
| `hue` / `saturation` | color | key + chroma / valence | circle-of-fifths hue, valence saturation |
| `seeds` | state re-seed | section boundaries | a visual "scene change" when structure changes |

So a calm, sparse track yields an ordered, slow spiral; a loud, fast, dynamic
track yields wide, fluttering, fast chaotic wings — and the shape morphs
continuously as those qualities change through the piece.

The GIF renderer uses a **long-exposure persistence buffer**: the whole
attractor stays visible at all times and fades slowly, while a fast head keeps
re-tracing it (so energy surges that widen the wings bloom the shape), finished
with filmic tone-mapping and bloom over a supersampled buffer.

### Render

```bash
# audio -> animated GIF (analyze + map + integrate + rasterize)
syne render track.wav -o track.lorenz.gif

# tune the look: faster head, slower fade, bigger canvas
syne render track.wav --speed 4 --decay 0.994 --size 600

# export per-frame control JSON for the web viewer (no GIF)
syne render track.wav --controls track.lorenz.json --no-gif
```

Key knobs: `--speed` (head velocity), `--decay` (trail persistence, higher =
slower fade; default derived from the track's dynamic range), `--substeps`
(curve density), `--size`, `--fps`, `--seconds`.

```python
from syne.pipeline import analyze_file
from syne.render.lorenz import build_control, integrate
from syne.render.raster import render_gif

profile = analyze_file("track.wav")
control = build_control(profile)            # tags -> Lorenz connectors
traj = integrate(control, fps=30)           # RK4 under time-varying control
render_gif(traj, "track.lorenz.gif")
```

The [web viewer](web/) (`web/`) loads the control JSON + the audio and
integrates the attractor live in the browser, synced to playback. Rendering
needs the optional Pillow dependency: `pip install -e ".[render]"`.

## Architecture

```
syne/
  audio.py            # load & downmix to mono float32
  features/           # interpretable feature groups (STFT computed once, shared)
    rhythm.py         #   tempo, beats, pulse clarity, regularity, swing
    tonal.py          #   chroma, key/mode (Krumhansl-Schmuckler)
    timbre.py         #   spectral shape, MFCC, brightness/warmth/roughness
    dynamics.py       #   RMS energy, loudness, dynamic range
    structure.py      #   spectral flux, section boundaries
  tagging/
    base.py           # Tagger interface (pluggable)
    heuristic.py      # default DSP-heuristic tagger
    vocab.py          # interpretable label vocabularies + rules
  semantics/schema.py # SemanticProfile dataclasses = the JSON contract
  morph/drivers.py    # SemanticProfile -> generic renderer control channels
  render/
    lorenz.py         # the connectors: tags -> Lorenz params + RK4 integrator
    raster.py         # Trajectory -> animated GIF (Pillow, additive glow)
  pipeline.py         # orchestration: file -> SemanticProfile
  cli.py              # `syne analyze` / `syne render`
web/                  # interactive, audio-synced WebGL viewer (Three.js)
```

## Extending with a learned tagger

Implement the `Tagger` protocol and pass it to the pipeline — everything
downstream (timelines, contract, drivers) is unchanged:

```python
from syne.pipeline import analyze_clip
from syne.audio import load_audio

class MyModelTagger:
    name = "my-model"
    def tag(self, features):     # -> TrackTags
        ...

profile = analyze_clip(load_audio("track.wav"), tagger=MyModelTagger())
```

## Roadmap

- [x] Renderer turning the semantic info into morphing geometry (Lorenz butterfly).
- [x] Audio-synced interactive WebGL viewer.
- [ ] Optional deep-learning tagger adapter (e.g. MTG-Jamendo tags).
- [ ] Beat-synchronous timelines for tighter visual sync.
- [ ] More attractors / geometry modes selectable per track.

## Tests

```bash
pytest          # runs fully offline on synthetic audio
```

## License

MIT
