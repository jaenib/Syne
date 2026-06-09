"""Command-line entry point.

    syne analyze track.wav -o track.semantic.json
    syne analyze track.wav --drivers           # also emit morph drivers
    syne analyze track.wav --summary           # print a human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys

from syne import __version__


def _cmd_analyze(args: argparse.Namespace) -> int:
    # imported lazily so `syne --version` doesn't pay the librosa import cost
    from syne.morph.drivers import derive_drivers
    from syne.pipeline import analyze_file

    profile = analyze_file(args.input)

    if args.output:
        profile.save(args.output)
        print(f"wrote semantic profile -> {args.output}", file=sys.stderr)
    if args.drivers:
        drivers = derive_drivers(profile)
        path = args.drivers if isinstance(args.drivers, str) else _suffix(args, ".drivers.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(drivers.to_dict(), fh, indent=2)
        print(f"wrote morph drivers   -> {path}", file=sys.stderr)

    if args.summary or not args.output:
        _print_summary(profile)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from syne.pipeline import analyze_file
    from syne.render.lorenz import build_control, integrate
    from syne.render.raster import render_gif
    from syne.semantics.schema import SemanticProfile

    if args.from_profile:
        profile = SemanticProfile.load(args.input)
    else:
        profile = analyze_file(args.input)

    control = build_control(profile)

    if args.controls:
        control.save(args.controls)
        print(f"wrote lorenz controls -> {args.controls}", file=sys.stderr)

    out = args.output or _strip_ext(args.input) + ".lorenz.gif"
    if not args.no_gif:
        traj = integrate(control, fps=args.fps, max_seconds=args.seconds)
        render_gif(traj, out, size=args.size, tail=args.tail)
        print(f"wrote animation       -> {out}", file=sys.stderr)

    if args.summary:
        _print_summary(profile)
    return 0


def _strip_ext(path: str) -> str:
    for ext in (".semantic.json", ".lorenz.json", ".json", ".wav", ".flac",
                ".mp3", ".ogg", ".m4a"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def _suffix(args: argparse.Namespace, suffix: str) -> str:
    base = args.output or args.input
    for ext in (".semantic.json", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base + suffix


def _print_summary(profile) -> None:
    t = profile.tags
    print(f"\n  source     : {profile.meta.source}")
    print(f"  duration   : {profile.meta.duration:.1f}s @ {profile.meta.sample_rate} Hz")
    print(f"  tempo      : {t.rhythm.tempo_bpm:.1f} BPM ({t.rhythm.tempo_category}, "
          f"{t.rhythm.time_feel})")
    print(f"  key        : {t.tonal.key} {t.tonal.mode} "
          f"(conf {t.tonal.key_confidence:.2f})")
    print(f"  mood       : {t.mood.label}  "
          f"[valence {t.mood.valence:.2f} / arousal {t.mood.arousal:.2f}]  "
          f"{', '.join(t.mood.descriptors)}")
    print(f"  energy     : {t.energy.label}  "
          f"[level {t.energy.level:.2f}, dance {t.energy.danceability:.2f}]")
    print(f"  timbre     : {', '.join(t.timbre.descriptors)}  "
          f"[bright {t.timbre.brightness:.2f}, warm {t.timbre.warmth:.2f}]")
    if t.genre_hints:
        hints = ", ".join(f"{g.label} ({g.confidence:.2f})" for g in t.genre_hints)
        print(f"  genre hints: {hints}")
    print(f"  timeline   : {len(profile.timelines.times)} frames, "
          f"{len(profile.timelines.beats)} beats, "
          f"{len(profile.timelines.segments)} sections\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syne", description=__doc__)
    parser.add_argument("--version", action="version", version=f"syne {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="tag an audio file")
    analyze.add_argument("input", help="path to an audio file (wav/flac/ogg/mp3/...)")
    analyze.add_argument("-o", "--output", help="write the semantic profile JSON here")
    analyze.add_argument(
        "--drivers", nargs="?", const=True, default=False,
        help="also emit morph drivers JSON (optional path)",
    )
    analyze.add_argument("--summary", action="store_true", help="print a summary")
    analyze.set_defaults(func=_cmd_analyze)

    render = sub.add_parser(
        "render", help="render a morphing Lorenz attractor from the music"
    )
    render.add_argument("input", help="audio file, or a .semantic.json with --from-profile")
    render.add_argument("-o", "--output", help="output GIF path")
    render.add_argument("--from-profile", action="store_true",
                        help="treat INPUT as an existing semantic profile JSON")
    render.add_argument("--controls", help="also export per-frame Lorenz control JSON "
                        "(for the web viewer)")
    render.add_argument("--no-gif", action="store_true",
                        help="skip the GIF (useful with --controls)")
    render.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    render.add_argument("--seconds", type=float, default=None,
                        help="cap the rendered duration")
    render.add_argument("--size", type=int, default=480, help="GIF size in px (default 480)")
    render.add_argument("--tail", type=int, default=260,
                        help="comet tail length in points (default 260)")
    render.add_argument("--summary", action="store_true", help="print a summary")
    render.set_defaults(func=_cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
