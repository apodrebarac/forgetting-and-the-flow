"""Tests of the claims in the technical appendix. Run with ``python -m unittest`` or pytest."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgetting_and_flow.archive import Deck, Player, make_demo_archive  # noqa: E402
from forgetting_and_flow.eventlog import read_log  # noqa: E402
from forgetting_and_flow.flow import Flow, FlowParams, contribution, frames_to_replace  # noqa: E402
from forgetting_and_flow.forgetting import (Forgetting, ForgettingParams,  # noqa: E402
                                            ForgivenessWithheld, MemoryMask)
from forgetting_and_flow.noise import CurlField, divergence  # noqa: E402
from forgetting_and_flow.runner import Run, RunConfig  # noqa: E402
from forgetting_and_flow.terminal import Consumer, ConsumerParams  # noqa: E402


class TestMemoryMask(unittest.TestCase):
    def test_update_rule_is_monotone(self):
        """mask(t+1) <= mask(t) everywhere, for every step, under aggressive parameters."""
        m = MemoryMask(64, 64, ForgettingParams(bite_rate=0.01, creep_rate=0.2), seed=1)
        for _ in range(200):
            before = m.mask.copy()
            after = m.step()
            self.assertTrue(np.all(after <= before + 1e-7))
        self.assertLess(m.remaining(), 1.0)

    def test_void_is_absorbing(self):
        """Once a pixel is void it stays void (the state machine is one-way)."""
        m = MemoryMask(32, 32, ForgettingParams(bite_rate=0.05, creep_rate=0.1), seed=2)
        for _ in range(100):
            was_void = m.mask <= 0.0
            m.step()
            self.assertTrue(np.all(m.mask[was_void] <= 0.0))

    def test_edge_creep_only_at_margins(self):
        """With bites off, erosion happens only next to existing void."""
        m = MemoryMask(32, 32, ForgettingParams(bite_rate=0.0, creep_rate=0.5, creep_radius=1), seed=3)
        m.mask[10:14, 10:14] = 0.0
        m.step()
        eroded = m.mask < 1.0
        # nothing far from the void touched
        self.assertTrue(np.all(m.mask[:8, :] == 1.0))
        self.assertTrue(np.all(m.mask[:, 16:] == 1.0))
        # the ring around the void did erode
        self.assertTrue(eroded[9, 10])
        self.assertTrue(eroded[10, 14])

    def test_forgiveness_is_external_and_only_way_back(self):
        m = MemoryMask(32, 32, ForgettingParams(bite_rate=0.05, creep_rate=0.1), seed=4)
        for _ in range(50):
            m.step()
        self.assertLess(m.remaining(), 1.0)
        m.forgive()
        self.assertEqual(m.remaining(), 1.0)
        self.assertEqual(m.forgiven, 1)

    def test_creep_front_speed_calibration(self):
        """The edge of a void advances CREEP_FRONT_SPEED px/frame per unit creep_rate."""
        from forgetting_and_flow.forgetting import CREEP_FRONT_SPEED
        for c in (0.003, 0.01):
            m = MemoryMask(96, 96, ForgettingParams(bite_rate=0.0, creep_rate=c), seed=0)
            m.mask[44:52, 44:52] = 0.0
            r0 = (m.mask[48] < 0.5).sum() / 2
            n = int(20 / c)
            for _ in range(n):
                m.step()
            v = ((m.mask[48] < 0.5).sum() / 2 - r0) / n
            self.assertAlmostEqual(v / c, CREEP_FRONT_SPEED, delta=0.1)

    def test_human_units_and_half_life(self):
        from forgetting_and_flow.forgetting import half_life_minutes
        p = ForgettingParams(bites_per_minute=2.0, creep_px_per_minute=1.0)
        bite, creep = p.rates(512, 512, 60.0)
        self.assertAlmostEqual(bite * 512 * 512 * 3600, 2.0)       # 2 bites per minute
        self.assertAlmostEqual(creep * 1.1 * 3600, 1.0)             # 1 px per minute
        t = half_life_minutes(512, 512, p, 60.0)
        self.assertGreater(t, 20)
        self.assertLess(t, 120)
        # faster settings -> shorter half-life; no bites -> never
        self.assertLess(half_life_minutes(512, 512, ForgettingParams(bites_per_minute=20, creep_px_per_minute=10), 60.0), t)
        self.assertEqual(half_life_minutes(512, 512, ForgettingParams(bites_per_minute=0), 60.0), float("inf"))

    def test_half_life_estimate_matches_simulation(self):
        """The closed form is within a factor of two of a real run (overlap slows the end)."""
        from forgetting_and_flow.forgetting import half_life_minutes
        p = ForgettingParams(bite_rate=1.77e-5, creep_rate=0.003, bite_radius=2)
        est = half_life_minutes(128, 128, p, fps=60.0) * 3600
        m = MemoryMask(128, 128, p, seed=1)
        f = 0
        while m.remaining() >= 0.5:
            m.step()
            f += 1
        self.assertLess(abs(np.log(f / est)), np.log(2.0))

    def test_forgiveness_can_be_withheld(self):
        m = MemoryMask(8, 8, ForgettingParams(), seed=5)
        m.withhold_forgiveness = True
        m.step()
        with self.assertRaises(ForgivenessWithheld):
            m.forgive()

    def test_full_system_runs_and_renders_binary(self):
        f = Forgetting(48, 64, ForgettingParams(), seed=6)
        src = np.random.default_rng(0).random((48, 64, 3)).astype(np.float32)
        out = f.step(src)
        self.assertEqual(out.shape, (48, 64))
        self.assertTrue(set(np.unique(out)).issubset({0.0, 1.0}))  # pure B&W dither


class TestFlow(unittest.TestCase):
    def test_contribution_formula_matches_pure_injection(self):
        """With transport off, the source share after N frames is exactly 1-(1-k)^N."""
        k = 0.06
        h, w = 16, 16
        fl = Flow(h, w, FlowParams.transport_off(), seed=0)
        fl.p.k = k
        src = np.ones((h, w, 3), np.float32)
        for n in range(1, 80):
            fl.step(src)
            self.assertAlmostEqual(float(fl.buf.mean()), contribution(n, k), places=5)

    def test_appendix_numbers(self):
        self.assertEqual(frames_to_replace(0.5, 0.06), 12)   # 11 frames -> 49.4 %, 12 -> 52.4 %
        self.assertGreater(contribution(11, 0.06), 0.49)
        self.assertGreater(contribution(75, 0.06), 0.99)

    def test_loop_forgets_the_image(self):
        """Correlation with a source decays after the source is replaced by a different one."""
        rng = np.random.default_rng(1)
        h, w = 64, 64
        fl = Flow(h, w, FlowParams(), seed=1)
        a = rng.random((h, w, 3)).astype(np.float32)
        b = rng.random((h, w, 3)).astype(np.float32)
        for _ in range(120):
            fl.step(a)
        c0 = fl.metrics(a, 120)["corr_to_source"]
        for _ in range(120):
            fl.step(b)
        c1 = fl.metrics(a, 120)["corr_to_source"]
        self.assertGreater(c0, 0.2)
        self.assertLess(c1, c0 * 0.5)

    def test_values_stay_in_range(self):
        fl = Flow(32, 32, FlowParams(), seed=2)
        src = np.random.default_rng(2).random((32, 32, 3)).astype(np.float32)
        for _ in range(50):
            out = fl.step(src)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
        d = fl.display()
        self.assertGreaterEqual(d.min(), 0.0)
        self.assertLessEqual(d.max(), 1.0)


class TestNoise(unittest.TestCase):
    def test_curl_field_is_divergence_free(self):
        f = CurlField(96, 96, seed=3, scale=40.0, strength=1.0)
        vy, vx = f.velocity(frame=5)
        div = divergence(vy, vx)[2:-2, 2:-2]
        speed = np.sqrt(vy * vy + vx * vx).mean()
        self.assertLess(np.abs(div).mean(), 0.05 * speed)

    def test_field_moves_in_time(self):
        f = CurlField(32, 32, seed=4)
        a = f.potential(0)
        b = f.potential(500)
        self.assertGreater(np.abs(a - b).mean(), 1e-3)


class TestDeck(unittest.TestCase):
    def test_no_immediate_repeats(self):
        for n in (2, 3, 7):
            d = Deck(n, np.random.default_rng(n))
            seq = [d.next() for _ in range(2000)]
            self.assertTrue(all(x != y for x, y in zip(seq, seq[1:])))
            self.assertEqual(set(seq), set(range(n)))

    def test_single_card_deck(self):
        d = Deck(1, np.random.default_rng(0))
        self.assertEqual([d.next() for _ in range(3)], [0, 0, 0])


class TestRunAndTerminal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.archive = self.root / "archive"
        make_demo_archive(self.archive, n=4, size=48, seed=0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_study_run_logs_seed_and_metrics(self):
        cfg = RunConfig(archive=self.archive, out_dir=self.root / "run", systems=("flow", "forgetting"),
                        size=32, fps=10, hold_seconds=0.5, fade_seconds=0.2, mode="study", seed=7,
                        frames=40, log_every=10)
        run = Run(cfg)
        final = run.run()
        recs = read_log(self.root / "run" / "events.jsonl")
        start = recs[0]
        self.assertEqual(start["event"], "run_start")
        self.assertEqual(start["seed"], 7)
        self.assertEqual(recs[-1]["event"], "run_end")
        self.assertIn("flow", final)
        self.assertIn("forgetting", final)
        self.assertTrue(any(r["event"] == "image" and r["kind"] == "enter" for r in recs))

    def test_performance_run_withholds_seed(self):
        cfg = RunConfig(archive=self.archive, out_dir=self.root / "run", size=16, fps=10,
                        mode="performance", frames=5, log_every=0)
        Run(cfg).run()
        start = read_log(self.root / "run" / "events.jsonl")[0]
        self.assertIn("withheld", str(start["seed"]))

    def test_signal_file_forgives_and_is_logged(self):
        cfg = RunConfig(archive=self.archive, out_dir=self.root / "run", size=16, fps=10,
                        mode="study", seed=1, frames=30, log_every=0,
                        forgetting=ForgettingParams(bite_rate=0.05, creep_rate=0.2))
        run = Run(cfg)
        self.assertIn("half_life_minutes_estimate",
                      read_log(self.root / "run" / "events.jsonl")[0]["systems"][0]["rates"])
        for _ in range(20):
            run.step()
        self.assertLess(run.forgetting.memory.remaining(), 1.0)
        (run.out_dir / "FORGIVE").touch()
        run.step()
        self.assertFalse((run.out_dir / "FORGIVE").exists())
        # the gesture restored the mask; one more step erodes it again
        recs = read_log(run.out_dir / "events.jsonl")
        f = [r for r in recs if r["event"] == "forgive"]
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0]["result"], "restored")
        self.assertEqual(f[0]["origin"], "signal-file")
        run.log.close()

    def test_terminal_consumes_archive_in_place_and_deletes_nothing(self):
        before = {p.name: p.read_bytes() for p in sorted(self.archive.iterdir())}
        cfg = RunConfig(archive=self.archive, out_dir=self.root / "run", systems=("forgetting",),
                        size=24, fps=10, hold_seconds=0.3, fade_seconds=0.1, mode="study", seed=3,
                        frames=120, log_every=0, withhold_forgiveness=True,
                        forgetting=ForgettingParams(bite_rate=0.02, creep_rate=0.3))
        run = Run(cfg)
        Consumer(ConsumerParams(generation_mix=0.5)).attach(run)
        (run.out_dir / "FORGIVE").touch()   # the gesture is withheld in this run
        run.run()
        after = {p.name: p.read_bytes() for p in sorted(self.archive.iterdir())}
        self.assertEqual(set(before), set(after), "no file was deleted or added")
        changed = [n for n in before if before[n] != after[n]]
        self.assertTrue(changed, "files were written back")
        recs = read_log(run.out_dir / "events.jsonl")
        gens = [r for r in recs if r["event"] == "generation"]
        self.assertTrue(gens)
        self.assertTrue(all(g["sha256"] for g in gens))
        forg = [r for r in recs if r["event"] == "forgive"]
        self.assertEqual(forg[0]["result"], "withheld")
        # generations climb per file
        maxgen = max(g["generation"] for g in gens)
        self.assertGreaterEqual(maxgen, 2)

    def test_player_crossfade_and_events(self):
        from forgetting_and_flow.archive import list_images
        paths = list_images(self.archive)
        events = []
        pl = Player(paths, 16, 16, np.random.default_rng(0), hold_frames=5, fade_frames=4)
        pl.listeners.append(events.append)
        for _ in range(30):
            s = pl.step()
            self.assertEqual(s.shape, (16, 16, 3))
        kinds = [e.kind for e in events]
        self.assertIn("enter", kinds)
        self.assertIn("surfaced", kinds)
        self.assertIn("exit", kinds)


if __name__ == "__main__":
    unittest.main()
