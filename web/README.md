# Syne web viewer

An interactive, audio-synced renderer of the morphing Lorenz attractor. It
consumes the per-frame control JSON exported by the Python pipeline — the same
semantic → Lorenz mapping the offline GIF renderer uses — so the *meaning*
lives in Python and this page is just an integrator + WebGL renderer.

## Use

1. Export controls (and keep the audio file handy):

   ```bash
   syne render track.wav --controls track.lorenz.json --no-gif
   ```

2. Serve this folder (ES modules need http, not `file://`):

   ```bash
   cd web && python -m http.server 8000
   # open http://localhost:8000
   ```

3. In the page: load `track.lorenz.json`, load `track.wav`, press **play**.
   The attractor integrates in real time and reads its parameters from the
   control timeline at the audio's current playback position, so the geometry
   morphs in sync with the music. With no audio it animates on a synthetic
   clock.

## How the music drives the shape

The control JSON carries per-frame connector signals (see `bindings` inside the
file):

| channel | drives | from |
|---|---|---|
| `rho` | wing size & chaos | arousal + energy (+ flux spikes) |
| `sigma` | swirl tightness | danceability |
| `beta` | vertical pinch | timbre brightness |
| `speed` | head velocity | tempo + energy |
| `kick` | impulse flutter | onset strength |
| `jitter` | surface noise | spectral flux |
| `glow` | line intensity | energy |
| `hue` / `saturation` | color | key/chroma / valence |
| `seeds` | state re-seed | section boundaries |

Dependencies: Three.js, loaded from a CDN via an import map (needs network).
