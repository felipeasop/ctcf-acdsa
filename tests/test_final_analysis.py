#!/usr/bin/env python3
"""Regression tests for the manuscript's final statistical analysis."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/11_final_analysis.py"
SPEC = importlib.util.spec_from_file_location("final_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class FinalAnalysisTests(unittest.TestCase):
    def test_motif_distance_identity_and_reverse_complement(self):
        motif = np.asarray([
            [.7, .1, .1, .1],
            [.1, .2, .6, .1],
            [.1, .1, .2, .6],
            [.2, .2, .2, .4],
        ])
        self.assertAlmostEqual(ANALYSIS.distance(motif, motif), 0.0)
        reverse = ANALYSIS.reverse_complement_matrix(motif)
        self.assertAlmostEqual(ANALYSIS.distance(motif, reverse), 0.0)
        for metric in ("jsd", "tv", "hellinger"):
            value = ANALYSIS.distance(motif, np.roll(motif, 1, axis=1), metric)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_group_centering_removes_only_group_means(self):
        values = [2.0, 4.0, 10.0, 16.0]
        groups = ["a", "a", "b", "b"]
        centered = ANALYSIS.center_within_groups(values, groups)
        np.testing.assert_allclose(centered, [-1.0, 1.0, -3.0, 3.0])
        self.assertAlmostEqual(centered[:2].mean(), 0.0)
        self.assertAlmostEqual(centered[2:].mean(), 0.0)

    def test_conditional_statistic_matches_one_common_slope_fwl_model(self):
        rows = []
        # Four observations per cell make separate per-cell regressions possible
        # The production estimand deliberately uses common control slopes
        for signal, shift in ((55.0, 10.0), (60.0, -7.0), (65.0, 3.0)):
            for sample, control in enumerate((-1.5, -0.5, 0.5, 1.5), start=1):
                disagreement = 0.4 * control + (sample % 2) * 0.3 + shift
                loss = 0.7 * control + 0.5 * disagreement + shift / 5
                rows.append({
                    "cohort": "fixture", "sample": sample, "signal": signal,
                    "width": 19, "jsd": disagreement,
                    "oops_ap": 1.0 - loss, "oops_conf": control + shift,
                    "oops_resampling": -0.2 * control + shift / 2,
                })

        observed = ANALYSIS.conditional_statistic(rows, "oops")
        groups = [(r["cohort"], r["signal"], r["width"]) for r in rows]
        y = ANALYSIS.center_within_groups([1-r["oops_ap"] for r in rows], groups)
        d = ANALYSIS.center_within_groups([r["jsd"] for r in rows], groups)
        x = np.column_stack([
            ANALYSIS.center_within_groups([r["oops_conf"] for r in rows], groups),
            ANALYSIS.center_within_groups([r["oops_resampling"] for r in rows], groups),
        ])
        base = LinearRegression().fit(x, y)
        base_d = LinearRegression().fit(x, d)
        expected_r = np.corrcoef(y - base.predict(x), d - base_d.predict(x))[0, 1]
        full = LinearRegression().fit(np.column_stack([x, d]), y)
        self.assertAlmostEqual(observed["partial_r"], expected_r, places=12)
        self.assertAlmostEqual(observed["baseline_r2"], base.score(x, y), places=12)
        self.assertAlmostEqual(
            observed["delta_r2"], full.score(np.column_stack([x, d]), y) - base.score(x, y),
            places=12,
        )

    def test_cluster_bootstrap_preserves_cohort_sizes(self):
        rows = [
            {"cohort": cohort, "sample": sample, "signal": signal}
            for cohort in ("a", "b")
            for sample in range(1, 4)
            for signal in (55, 60, 65)
        ]
        sampled = ANALYSIS.resample_trajectories(rows, np.random.default_rng(7))
        self.assertEqual(len(sampled), len(rows))
        for cohort in ("a", "b"):
            self.assertEqual(sum(r["cohort"] == cohort for r in sampled), 9)


if __name__ == "__main__":
    unittest.main()
