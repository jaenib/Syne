"""Offline rasterizer: a Lorenz :class:`Trajectory` -> animated GIF.

Renders a glowing "comet" — a fading tail of recent trajectory points — with a
slowly rotating camera. Points are splatted additively onto a float buffer so
overlapping curve segments bloom, giving the attractor its luminous look
without any GPU or ffmpeg dependency.
"""

from __future__ import annotations

import colorsys

import numpy as np

from syne.render.lorenz import Trajectory

# 5x5 soft splat kernel (normalized-ish gaussian) for the glow
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

# Lorenz attractor roughly spans x,y in [-25,25], z in [0,50]; center z.
_Z_CENTER = 25.0
_SPAN = 30.0


def render_gif(
    traj: Trajectory,
    path: str,
    *,
    size: int = 480,
    tail: int = 260,
    background: tuple[int, int, int] = (6, 7, 16),
    gain: float = 200.0,
) -> str:
    """Render ``traj`` to an animated GIF at ``path``; returns ``path``."""
    from PIL import Image

    n_video = len(traj.frame_end)
    bg = np.asarray(background, dtype=np.float32)
    rot_k = traj.rotation_speed * 0.045          # radians per frame
    sat = traj.saturation

    # precompute HSV->RGB per point (value handled later via intensity)
    rgb_lut = _hue_to_rgb(traj.hue, sat)

    frames = []
    half = size / 2.0
    scale = (size * 0.42) / _SPAN
    for f in range(n_video):
        head = int(traj.frame_end[f])
        lo = max(0, head - tail)
        m = head - lo
        if m <= 1:
            frames.append(Image.new("RGB", (size, size), tuple(background)))
            continue

        pts = traj.points[lo:head]
        glow = traj.glow[lo:head]
        cols = rgb_lut[lo:head]

        px, py = _project(pts, rot_k * f, half, scale)
        buf = np.zeros((size, size, 3), dtype=np.float32)

        # along-tail fade: newest points brightest
        ramp = np.linspace(0.12, 1.0, m) ** 1.6
        inten = ramp * (0.3 + 0.8 * glow)
        # whiten the comet head so the leading point reads clearly
        head_cols = cols.copy()
        head_cols[-12:] = 0.55 * head_cols[-12:] + 0.45
        _draw_curve(buf, px, py, head_cols * inten[:, None], size)

        img = np.clip(bg + buf * gain, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img, "RGB"))

    if not frames:
        frames = [Image.new("RGB", (size, size), tuple(background))]
    duration_ms = int(1000 / max(traj.fps, 1))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path


# --------------------------------------------------------------------------- #
def _project(pts: np.ndarray, angle: float, half: float, scale: float):
    """Yaw about the vertical axis, then orthographic project with a slight tilt."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ca, sa = np.cos(angle), np.sin(angle)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    screen_x = xr
    screen_y = -(z - _Z_CENTER) * 0.95 + yr * 0.32      # z up, gentle tilt
    px = half + screen_x * scale
    py = half + screen_y * scale
    return px, py


def _draw_curve(buf: np.ndarray, px: np.ndarray, py: np.ndarray, colors: np.ndarray, size: int):
    """Splat each point, interpolating between consecutive points so the
    integration samples join into a continuous glowing curve."""
    k = _KERNEL[:, :, None]
    m = len(px)
    for i in range(m):
        _splat_one(buf, px[i], py[i], colors[i], size, k)
        if i + 1 < m:                       # fill the gap to the next sample
            dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
            dist = (dx * dx + dy * dy) ** 0.5
            steps = int(dist / 1.3)
            if 0 < steps < 60:
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
