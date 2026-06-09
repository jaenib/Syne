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
    tail: int = 1000,
    exposure: float = 1.1,
    bloom: float = 0.45,
    flash_gain: float = 0.9,
    zoom_gain: float = 0.12,
    **_legacy,
) -> str:
    """Render one or several Lorenz trajectories to a reactive GIF.

    Each frame **redraws the recent tail** at the current camera angle (so the
    butterfly stays clean while it tumbles in 3-D, instead of smearing). The
    tail fades from head to tip; onsets drive a sharp **pulse** that brightens
    the head, flashes the frame, and punches the zoom — so beats are
    unmistakable while the attractor shape stays legible.
    """
    from PIL import Image, ImageChops, ImageFilter

    trajs = [traj] if isinstance(traj, Trajectory) else list(traj)
    n_video = min(len(t.frame_end) for t in trajs)
    ss = size * supersample
    bg = np.asarray(background, dtype=np.float32)
    bg_head = 255.0 - bg

    luts = [_hue_to_rgb(t.hue, t.sat) for t in trajs]
    have_fx = trajs[0].frame_angle.size == n_video
    half = ss / 2.0
    base_scale = (ss * 0.40) / _SPAN
    blur_radius = ss * 0.0065
    nb = len(trajs)

    frames = []
    for f in range(n_video):
        angle = float(trajs[0].frame_angle[f]) if have_fx else 0.0
        pulse_g = max((float(t.frame_pulse[f]) for t in trajs
                       if t.frame_pulse.size == n_video), default=0.0)

        cols = [np.zeros((ss, ss, 3), dtype=np.float32) for _ in trajs]
        dens = [np.zeros((ss, ss), dtype=np.float32) for _ in trajs]
        for ti, t in enumerate(trajs):
            end = int(t.frame_end[f])
            lo = max(0, end - tail)
            m = end - lo
            if m < 2:
                continue
            pts = t.points[lo:end]
            glow = t.glow[lo:end]
            cc = luts[ti][lo:end]
            px, py = _project(pts, half, base_scale * t.scale, angle)
            depth = _depth_shade(pts, angle)
            ramp = np.linspace(0.06, 1.0, m) ** 1.5          # head bright, tip dim
            inten = ramp * (0.3 + 0.8 * glow) * depth
            cp = float(t.frame_pulse[f]) if t.frame_pulse.size == n_video else 0.0
            inten[-14:] *= (2.0 + 4.0 * cp)                  # head pops on its band's hit
            _draw_curve(cols[ti], dens[ti], px, py, cc, inten, ss)

        if nb == 1:
            mean_col = cols[0] / np.maximum(dens[0][:, :, None], 1e-6)
            lum = 1.0 - np.exp(-dens[0] * exposure)
            rgb = bg + bg_head * mean_col * lum[:, :, None]
        else:
            dstack = np.stack(dens)
            win = dstack.argmax(0)
            wsel = win[None, :, :, None]
            mean_cols = np.stack([cols[i] / np.maximum(dens[i][:, :, None], 1e-6)
                                  for i in range(nb)])
            win_col = np.take_along_axis(mean_cols, wsel, 0)[0]
            win_den = np.take_along_axis(dstack, win[None], 0)[0]
            lum = 1.0 - np.exp(-win_den * exposure)
            rgb = bg + bg_head * win_col * lum[:, :, None]

        rgb *= (1.0 + flash_gain * pulse_g)                  # beat brightness flash
        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

        if bloom > 0:
            halo = img.filter(ImageFilter.GaussianBlur(blur_radius))
            halo = halo.point(lambda v: int(v * bloom))
            img = ImageChops.add(img, halo)
        frames.append(_resize_zoom(img, size, 1.0 + zoom_gain * pulse_g))

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
def _project(pts: np.ndarray, half: float, scale: float, angle: float = 0.0):
    """Yaw about the vertical (z) axis by ``angle``, then orthographic project."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if angle:
        ca, sa = np.cos(angle), np.sin(angle)
        x, y = x * ca - y * sa, x * sa + y * ca
    screen_x = x
    screen_y = -(z - _Z_CENTER) * 0.95 + y * 0.30
    return half + screen_x * scale, half + screen_y * scale


def _depth_shade(pts: np.ndarray, angle: float = 0.0) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    yr = x * np.sin(angle) + y * np.cos(angle) if angle else y
    d = (yr + 25.0) / 50.0
    return np.clip(0.55 + 0.6 * d, 0.35, 1.25)


def _resize_zoom(img, size: int, zoom: float):
    from PIL import Image
    if zoom > 1.001:
        w = img.width
        cw = max(2, int(w / zoom))
        off = (w - cw) // 2
        img = img.crop((off, off, off + cw, off + cw))
    return img.resize((size, size), Image.LANCZOS)


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
