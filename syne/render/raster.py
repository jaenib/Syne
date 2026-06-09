"""Offline rasterizer: a Lorenz :class:`Trajectory` -> animated GIF.

Long-exposure persistence buffer with **color-preserving accumulation**: rather
than summing RGB (which clips toward white wherever the curve overlaps itself),
each frame accumulates two things separately —

* ``col`` — intensity-weighted hue,  and
* ``den`` — scalar density (intensity),

both decaying slowly. The output hue is ``col / den`` (so overlapping same-hue
windings stay that hue instead of washing to white) and the brightness is a
tone-mapped function of ``den``. Finished with bloom over a supersampled buffer.
No GPU or ffmpeg needed.
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
    traj: "Trajectory | list[Trajectory]",
    path: str,
    *,
    size: int = 460,
    supersample: int = 2,
    background: tuple[int, int, int] = (4, 5, 13),
    decay: float | None = None,
    exposure: float = 3.2,
    bloom: float = 0.6,
) -> str:
    """Render one or several Lorenz trajectories to a long-exposure GIF.

    Multiple trajectories (e.g. one per frequency band) are composited into the
    same color-preserving buffers, so overlapping wings blend their hues.
    """
    from PIL import Image, ImageChops, ImageFilter

    trajs = [traj] if isinstance(traj, Trajectory) else list(traj)
    n_video = min(len(t.frame_end) for t in trajs)
    ss = size * supersample
    bg = np.asarray(background, dtype=np.float32)
    bg_head = 255.0 - bg

    if decay is None:
        decay = float(np.clip(0.985 + 0.011 * trajs[0].fade, 0.985, 0.996))

    luts = [_hue_to_rgb(t.hue, t.sat) for t in trajs]   # per-point vivid colors

    # one (color, density) buffer pair PER curve, so curves keep their own hue;
    # they are combined by lighten (max) at output time — averaging different
    # hues into a shared buffer would grey out the overlap region.
    cols = [np.zeros((ss, ss, 3), dtype=np.float32) for _ in trajs]
    dens = [np.zeros((ss, ss), dtype=np.float32) for _ in trajs]
    half = ss / 2.0
    base_scale = (ss * 0.40) / _SPAN
    blur_radius = ss * 0.0065

    frames = []
    prev_end = [0] * len(trajs)
    nb = len(trajs)
    for f in range(n_video):
        for ti, t in enumerate(trajs):
            cols[ti] *= decay
            dens[ti] *= decay
            end = int(t.frame_end[f])
            lo = max(0, prev_end[ti] - 1)
            prev_end[ti] = end
            if end - lo >= 2:
                pts = t.points[lo:end]
                glow = t.glow[lo:end]
                cc = luts[ti][lo:end]
                px, py = _project(pts, half, base_scale * t.scale)
                depth = _depth_shade(pts)
                inten = (0.35 + 0.75 * glow) * depth
                inten[-6:] *= 2.0                      # hot (bright) head, same hue
                _draw_curve(cols[ti], dens[ti], px, py, cc, inten, ss)

        # occlusion compositing: each pixel takes the PURE color of the densest
        # curve there (like depth ordering), so overlaps stay saturated instead
        # of blending toward grey/white.
        if nb == 1:
            mean_col = cols[0] / np.maximum(dens[0][:, :, None], 1e-6)
            lum = 1.0 - np.exp(-dens[0] * exposure)
            rgb = bg + bg_head * mean_col * lum[:, :, None]
        else:
            dstack = np.stack(dens)                    # (nb, ss, ss)
            win = dstack.argmax(0)                      # densest curve per pixel
            wsel = win[None, :, :, None]
            mean_cols = np.stack([cols[i] / np.maximum(dens[i][:, :, None], 1e-6)
                                  for i in range(nb)])
            win_col = np.take_along_axis(mean_cols, wsel, 0)[0]
            win_den = np.take_along_axis(dstack, win[None], 0)[0]
            lum = 1.0 - np.exp(-win_den * exposure)
            rgb = bg + bg_head * win_col * lum[:, :, None]
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
        duration=int(1000 / max(trajs[0].fps, 1)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path


# --------------------------------------------------------------------------- #
def _project(pts: np.ndarray, half: float, scale: float):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    screen_x = x
    screen_y = -(z - _Z_CENTER) * 0.95 + y * 0.30
    return half + screen_x * scale, half + screen_y * scale


def _depth_shade(pts: np.ndarray) -> np.ndarray:
    y = pts[:, 1]
    d = (y + 25.0) / 50.0
    return np.clip(0.6 + 0.5 * d, 0.4, 1.2)


def _draw_curve(col, den, px, py, cols, inten, size):
    """Splat intensity-weighted color into ``col`` and intensity into ``den``,
    interpolating between samples for a continuous curve."""
    k = _KERNEL
    m = len(px)
    for i in range(m):
        _splat(col, den, px[i], py[i], cols[i], inten[i], size, k)
        if i + 1 < m:
            dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > size * 0.12:                     # don't connect teleports
                continue
            steps = int(dist / 1.3)
            if steps > 0:
                cc = 0.5 * (cols[i] + cols[i + 1])
                ic = 0.5 * (inten[i] + inten[i + 1])
                for t in np.linspace(0.0, 1.0, steps + 2)[1:-1]:
                    _splat(col, den, px[i] + dx * t, py[i] + dy * t, cc, ic, size, k)


def _splat(col, den, x, y, color, inten, size, k):
    x0, y0 = int(round(x)), int(round(y))
    if _KR <= x0 < size - _KR and _KR <= y0 < size - _KR:
        sl = (slice(y0 - _KR, y0 + _KR + 1), slice(x0 - _KR, x0 + _KR + 1))
        ki = k * inten
        col[sl[0], sl[1], :] += ki[:, :, None] * color[None, None, :]
        den[sl] += ki


def _hue_to_rgb(hue: np.ndarray, sat: np.ndarray) -> np.ndarray:
    out = np.empty((hue.size, 3), dtype=np.float32)
    for i in range(hue.size):
        out[i] = colorsys.hsv_to_rgb(float(hue[i]) % 1.0, float(sat[i]), 1.0)
    return out


__all__ = ["render_gif"]
