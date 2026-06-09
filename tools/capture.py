"""Render an audio file to GIF and quantify visual reactivity:
- motion: mean abs frame-to-frame pixel change (overall dynamism)
- beat lock: is motion higher right after beats than between them?
- saves a montage strip of frames around a beat.
Usage: python tools/capture.py IN.wav OUT.gif [seconds]
"""
import sys, numpy as np
from PIL import Image
from syne.audio import load_audio
from syne.pipeline import analyze_clip
from syne.render.lorenz import build_band_trajectories
from syne.render.raster import render_gif

inp, out = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv)>3 else None
clip = load_audio(inp); prof = analyze_clip(clip)
fps = 30
trajs = build_band_trajectories(prof, fps=fps, max_seconds=secs)
render_gif(trajs, out, size=300)

im = Image.open(out); n = im.n_frames
fr = []
for i in range(n):
    im.seek(i); fr.append(np.asarray(im.convert("RGB"), float))
fr = np.array(fr)
diff = np.abs(np.diff(fr, axis=0)).mean(axis=(1,2,3))      # per-frame motion
motion = diff.mean()
# beat lock: motion in 0..120ms after a beat vs elsewhere
beats = [b for b in prof.timelines.beats if (secs is None or b < secs)]
onb = []; 
for b in beats:
    f = int(b*fps)
    if 0 < f < len(diff): onb.append(diff[f:f+4].max())
onb = np.array(onb) if onb else np.array([0])
print(f"frames={n}  mean_motion={motion:.2f}  motion_std={diff.std():.2f}")
print(f"peak_motion={diff.max():.2f}  on-beat_motion={onb.mean():.2f}  ratio_onbeat/mean={onb.mean()/ (motion+1e-9):.2f}")
print(f"contrast(p95-p5 of frame brightness, last frame)={np.percentile(fr[-1].mean(2),95)-np.percentile(fr[-1].mean(2),5):.1f}")
