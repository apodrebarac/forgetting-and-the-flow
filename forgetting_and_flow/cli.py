"""Command line. ``python -m forgetting_and_flow <command> ...``"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from .eventlog import read_log
from .flow import FlowParams, contribution, frames_to_replace
from .forgetting import ForgettingParams
from .runner import Run, RunConfig


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--archive", required=True, type=Path, help="directory of images (the finite archive)")
    p.add_argument("--out", type=Path, default=None, help="run directory (default runs/<cmd>_<time>)")
    p.add_argument("--size", type=int, default=512, help="canvas height in pixels")
    p.add_argument("--aspect", type=float, default=1.0, help="canvas width/height")
    p.add_argument("--fps", type=float, default=60.0, help="nominal frame rate (the timebase)")
    p.add_argument("--hold", type=float, default=4.0, help="seconds an image is held fully visible")
    p.add_argument("--fade", type=float, default=1.5, help="crossfade seconds")
    p.add_argument("--mode", choices=["performance", "study"], default="performance",
                   help="performance: fresh entropy, seed not recorded; study: seed recorded")
    p.add_argument("--seed", type=int, default=None, help="study mode seed")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--seconds", type=float, default=None, help="stop after wall-clock seconds")
    p.add_argument("--log-every", type=int, default=30, help="frames between metric records")
    p.add_argument("--save-every", type=int, default=0, help="frames between saved PNGs (0 = never)")
    p.add_argument("--display", action="store_true", help="live window (f = forgive, q = quit)")
    p.add_argument("--pace", action="store_true", help="hold nominal fps when running ahead")
    p.add_argument("--label", default="", help="free text recorded in run_start")
    # a few exposed parameters; everything else via the dataclasses in code
    p.add_argument("--k", type=float, default=None, help="Flow injection rate (default 0.06)")
    p.add_argument("--bites-per-minute", type=float, default=None,
                   help="Forgetting: new bites over the whole canvas per minute (default 2)")
    p.add_argument("--creep-px-per-minute", type=float, default=None,
                   help="Forgetting: speed of the void's edge in px/minute (default 1)")
    p.add_argument("--rebirth-every", type=float, default=None,
                   help="optional rebirth cycle, seconds (long installations)")


def _config(args, systems: tuple[str, ...], cmd: str, **extra) -> RunConfig:
    out = args.out or Path("runs") / f"{cmd}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    fp = FlowParams()
    if args.k is not None:
        fp.k = args.k
    gp = ForgettingParams()
    if args.bites_per_minute is not None:
        gp.bites_per_minute = args.bites_per_minute
    if args.creep_px_per_minute is not None:
        gp.creep_px_per_minute = args.creep_px_per_minute
    return RunConfig(archive=args.archive, out_dir=out, systems=systems, size=args.size,
                     aspect=args.aspect, fps=args.fps, hold_seconds=args.hold,
                     fade_seconds=args.fade, mode=args.mode, seed=args.seed,
                     frames=args.frames, seconds=args.seconds, log_every=args.log_every,
                     save_every=args.save_every, display=args.display, pace=args.pace,
                     rebirth_seconds=args.rebirth_every, flow=fp, forgetting=gp,
                     label=args.label, **extra)


def _report(run: Run, final: dict) -> None:
    print(f"run {run.run_id}: {run.frame} frames -> {run.out_dir / 'events.jsonl'}")
    if "forgetting" in final:
        f = final["forgetting"]
        print(f"  memory remaining {f['remaining']:.4f}   void {f['void_fraction']:.4f}   "
              f"edge {f['edge_length']} px   forgiven {f['forgiven']}x")
    if "flow" in final:
        f = final["flow"]
        print(f"  corr to source {f['corr_to_source']:+.3f}   buffer std {f['buffer_std']:.3f}   "
              f"entropy {f['buffer_entropy_bits']:.2f} bits")


def _announce(run: Run) -> None:
    print(f"run {run.run_id} ({run.cfg.mode}) on {len(run.paths)} images, canvas {run.h}x{run.w} "
          f"@ {run.cfg.fps:g} fps -> {run.out_dir}")
    if run.forgetting is not None:
        r = run.forgetting.memory.describe()
        print(f"  forgetting: {r['bites_per_minute']:.3g} bites/min, creep {r['creep_px_per_minute']:.3g} px/min, "
              f"half-life estimate {r['half_life_minutes_estimate']:.1f} min")
    if run.flow is not None:
        k = run.flow.p.k
        print(f"  flow: k = {k:g}; 50% replaced after {frames_to_replace(0.5, k)} frames, "
              f"99% after {frames_to_replace(0.99, k)}")
    if run.forgetting is not None and not run.cfg.withhold_forgiveness:
        print(f"  forgive from outside: touch {run.out_dir / 'FORGIVE'}" +
              ("  (or press f)" if run.cfg.display else ""))


def cmd_run(args, systems: tuple[str, ...], name: str) -> int:
    run = Run(_config(args, systems, name))
    _announce(run)
    _report(run, run.run())
    return 0


def cmd_terminal(args) -> int:
    from .terminal import Consumer, ConsumerParams
    if not args.consume:
        print("terminal: refusing to run.\n"
              "This command OVERWRITES the image files in --archive, in place, generation by\n"
              "generation, until they hold nothing. It never deletes anything, and it cannot be\n"
              "undone from inside the system. If that directory is the one meant to be consumed,\n"
              "run again with --consume.", file=sys.stderr)
        return 2
    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    seconds = args.days * 86400.0 if args.seconds is None else args.seconds
    cfg = _config(args, systems, "terminal", withhold_forgiveness=True)
    cfg.seconds = seconds
    run = Run(cfg)
    Consumer(ConsumerParams(generation_mix=args.generation_mix)).attach(run)
    run.log.write("terminal", days=args.days, note="forgiveness withheld; archive consumed in place")
    _announce(run)
    length = f"{args.days:g} days" if args.seconds is None else f"{args.seconds:g} s"
    print(f"  terminal: {length}; forgiveness withheld; files in {args.archive} are consumed in place")
    _report(run, run.run())
    return 0


def cmd_decay_table(args) -> int:
    k = args.k
    print(f"contribution(N) = 1 - (1 - k)^N    k = {k}")
    for n in (1, 5, 11, 20, 30, 45, 60, 75, 120):
        print(f"  N = {n:4d} ({n / args.fps:6.2f} s)  ->  {100 * contribution(n, k):6.2f}% replaced")
    for frac in (0.5, 0.9, 0.99, 0.999):
        n = frames_to_replace(frac, k)
        print(f"  {100 * frac:5.1f}% replaced after N = {n} frames ({n / args.fps:.2f} s)")
    return 0


def cmd_demo_archive(args) -> int:
    from .archive import make_demo_archive
    paths = make_demo_archive(args.out, n=args.n, size=args.size, seed=args.seed)
    print(f"wrote {len(paths)} synthetic images to {args.out} (stand-ins, not the work's material)")
    return 0


def cmd_summary(args) -> int:
    recs = read_log(args.log)
    runs: dict[str, list[dict]] = {}
    for r in recs:
        runs.setdefault(r["run"], []).append(r)
    for run_id, rs in runs.items():
        start = next((r for r in rs if r["event"] == "run_start"), {})
        end = next((r for r in rs if r["event"] == "run_end"), None)
        mets = [r for r in rs if r["event"] == "metrics"]
        gens = [r for r in rs if r["event"] == "generation"]
        forg = [r for r in rs if r["event"] == "forgive"]
        print(f"run {run_id}  mode={start.get('mode')}  systems={[s['system'] for s in start.get('systems', [])]}"
              f"  images={start.get('archive_images')}  seed={start.get('seed')}")
        if end:
            print(f"  ended: {end['reason']} after {end['frames']} frames, {end['elapsed_s']} s")
        if mets and "forgetting" in mets[-1]:
            first, last = mets[0]["forgetting"], mets[-1]["forgetting"]
            print(f"  memory remaining: {first['remaining']:.4f} -> {last['remaining']:.4f}")
        if mets and "flow" in mets[-1]:
            cs = [m["flow"]["corr_to_source"] for m in mets]
            print(f"  corr to source: mean {sum(cs) / len(cs):+.3f}  last {cs[-1]:+.3f}")
        if forg:
            print("  forgiveness: " + ", ".join(f"{f['origin']}={f['result']}" for f in forg))
        if gens:
            by_file: dict[str, int] = {}
            for g in gens:
                by_file[g["file"]] = max(by_file.get(g["file"], 0), g["generation"])
            print(f"  generations written: {len(gens)}  (max per file {max(by_file.values())}); "
                  f"last std {gens[-1]['std']:.4f}, entropy {gens[-1]['entropy_bits']:.2f} bits")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forgetting_and_flow",
                                description="The Forgetting and the Flow — degradation as method.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("flow", help="run The Flow (lossy recursion)")
    _common(s)
    s.set_defaults(fn=lambda a: cmd_run(a, ("flow",), "flow"))

    s = sub.add_parser("forgetting", help="run The Forgetting (irreversible memory mask)")
    _common(s)
    s.set_defaults(fn=lambda a: cmd_run(a, ("forgetting",), "forgetting"))

    s = sub.add_parser("both", help="run both systems side by side on the same deck")
    _common(s)
    s.set_defaults(fn=lambda a: cmd_run(a, ("flow", "forgetting"), "both"))

    s = sub.add_parser("terminal", help="the final form: consume the archive in place (requires --consume)")
    _common(s)
    s.add_argument("--systems", default="forgetting,flow", help="comma list: forgetting, flow")
    s.add_argument("--days", type=float, default=10.0, help="run length in days (default 10)")
    s.add_argument("--generation-mix", type=float, default=0.5,
                   help="share of the system's output written into each file per generation")
    s.add_argument("--consume", action="store_true",
                   help="confirm that the files in --archive are to be overwritten, generation by generation")
    s.set_defaults(fn=cmd_terminal)

    s = sub.add_parser("decay-table", help="print contribution(N) = 1 - (1-k)^N")
    s.add_argument("--k", type=float, default=0.06)
    s.add_argument("--fps", type=float, default=60.0)
    s.set_defaults(fn=cmd_decay_table)

    s = sub.add_parser("demo-archive", help="write synthetic stand-in images for development")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n", type=int, default=12)
    s.add_argument("--size", type=int, default=384)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo_archive)

    s = sub.add_parser("summary", help="summarise an events.jsonl log")
    s.add_argument("--log", type=Path, required=True)
    s.set_defaults(fn=cmd_summary)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
