"""Offline rasterizer: a Lorenz :class:`Trajectory` -> animated GIF.

Uses a **persistence (long-exposure) buffer**: every frame the accumulation is
multiplied by a decay factor and the newly-integrated segment is splatted in
additively. With a slow decay the whole butterfly stays visible at all times
and fades gently, while a fast-travelling head keeps redrawing it — so energy
surges (which widen the Lorenz wings via ``rho``) visibly bloom the shape.

Pipeline per frame: decay -> splat new segment -> filmic tone-map -> bloom ->
downscale (supersampled for clean anti-aliasing). No GPU or ffmpeg needed.
"""

from __future__ import annotations

import colorsys

import numpy as np

from syne.render.lorenz import Trajectory

# 5x5 soft splat kernel for the glow
_KERNEL = np.array(
    [
        [0.05, 0.12, 0.18, 0.12, 0.05],
        [0.12, 0.40, 0.65, 0.40, 0.12],
        [0.18, 0.65, 1.00, 0.65, 0.18],
        [0.12, 0.40, 0.65, 0.40, 0.12],
        [0.05, 0.12, 0.18, 0.12, 0.05],
    ],
    dtype=np.float32,
)
_KR = _KERNEL.shape[0] // 2

_Z_CENTER = 25.0
_SPAN = 30.0


def render_gif(
    traj: Trajectory,
    path: str,
    *,
    size: int = 460,
    supersample: int = 2,
    background: tuple[int, int, int] = (4, 5, 13),
    decay: float | None = None,
    exposure: float = 2.6,
    bloom: float = 0.7,
) -> str:
    """Render ``traj`` to a long-exposure animated GIF at ``path``."""
    from PIL import Image, ImageChops, ImageFilter

    n_video = len(traj.frame_end)
    ss = size * supersample
    bg = np.asarray(background, dtype=np.float32)
    bg_head = 255.0 - bg

    # slow persistence: map the track's fade tag into a high decay factor
    if decay is None:
        decay = float(np.clip(0.985 + 0.011 * traj.fade, 0.985, 0.996))

    rgb_lut = _hue_to_rgb(traj.hue, traj.saturation)

    accum = np.zeros((ss, ss, 3), dtype=np.float32)
    half = ss / 2.0
    scale = (ss * 0.40) / _SPAN
    blur_radius = ss * 0.0065

    frames = []
    prev_end = 0
    for f in range(n_video):
        accum *= decay

        end = int(traj.frame_end[f])
        lo = max(0, prev_end - 1)            # overlap one point for continuity
        prev_end = end
        if end - lo >= 2:
            pts = traj.points[lo:end]
            glow = traj.glow[lo:end]
            cols = rgb_lut[lo:end]

            px, py = _project(pts, half, scale)
            depth = _depth_shade(pts)        # subtle 3-D form cue
            inten = (0.35 + 0.75 * glow) * depth
            # brighten the freshest points so the head reads as a hot core
            inten[-6:] *= 2.2
            _draw_curve(accum, px, py, cols * inten[:, None], ss)

        # filmic tone-map: 1 - exp(-x) -> graceful highlights, no harsh clip
        toned = 1.0 - np.exp(-accum * exposure)
        rgb = bg + bg_head * toned
        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

        if bloom > 0:
            halo = img.filter(ImageFilter.GaussianBlur(blur_radius))
            halo = halo.point(lambda v: int(v * bloom))
            img = ImageChops.add(img, halo)

        if supersample != 1:
            img = img.resize((size, size), Image.LANCZOS)
        frames.append(img)

    if not frames:
        frames = [Image.new("RGB", (size, size), tuple(background))]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / max(traj.fps, 1)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path


# --------------------------------------------------------------------------- #
def _project(pts: np.ndarray, half: float, scale: float):
    """Static orthographic projection with a gentle tilt (z up)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    screen_x = x
    screen_y = -(z - _Z_CENTER) * 0.95 + y * 0.30
    return half + screen_x * scale, half + screen_y * scale


def _depth_shade(pts: np.ndarray) -> np.ndarray:
    """Brighten points nearer the camera (along +y) for a 3-D read."""
    y = pts[:, 1]
    d = (y + 25.0) / 50.0
    return np.clip(0.6 + 0.5 * d, 0.4, 1.2)


def _draw_curve(buf, px, py, colors, size):
    """Splat each point, interpolating between samples for a continuous curve."""
    k = _KERNEL[:, :, None]
    m = len(px)
    for i in range(m):
        _splat_one(buf, px[i], py[i], colors[i], size, k)
        if i + 1 < m:
            dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
            dist = (dx * dx + dy * dy) ** 0.5
            # don't connect across teleports (re-seed / onset kicks); just gap
            if dist > size * 0.12:
                continue
            steps = int(dist / 1.3)
            if steps > 0:
                cc = 0.5 * (colors[i] + colors[i + 1])
                for t in np.linspace(0.0, 1.0, steps + 2)[1:-1]:
                    _splat_one(buf, px[i] + dx * t, py[i] + dy * t, cc, size, k)


def _splat_one(buf, x, y, color, size, k):
    x0, y0 = int(round(x)), int(round(y))
    if _KR <= x0 < size - _KR and _KR <= y0 < size - _KR:
        buf[y0 - _KR:y0 + _KR + 1, x0 - _KR:x0 + _KR + 1, :] += k * color[None, None, :]


def _hue_to_rgb(hue: np.ndarray, sat: float) -> np.ndarray:
    out = np.empty((hue.size, 3), dtype=np.float32)
    for i, h in enumerate(hue):
        out[i] = colorsys.hsv_to_rgb(float(h) % 1.0, sat, 1.0)
    return out


__all__ = ["render_gif"]
