#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from synthesize_adaptive_whole_territory_pbr_generalized_mirror_guard import choose_route


def args() -> Namespace:
    return Namespace(
        directional_coherence_min=0.62,
        directional_spectral_second_moment_min=0.55,
        structured_visible_highpass_min=0.012,
        structured_visible_edge_p95_min=0.050,
        structured_periodic_min_corr=0.18,
        structured_lowdetail_total_std_min=0.100,
        structured_axis_ratio_max=0.55,
        mirror_risk_macro_ratio_min=0.970,
        mirror_risk_total_std_max=0.100,
        mirror_risk_lowcontrast_std_max=0.075,
        mirror_risk_edge_p95_min=0.040,
        smooth_highpass_max=0.012,
        smooth_edge_p95_max=0.040,
    )


def metrics(highpass: float, edge: float, periodic: float, axis: float = 1.0) -> dict:
    return {
        "structured_highpass_std": highpass,
        "structured_edge_p95": edge,
        "structured_periodic_max_corr": periodic,
        "structured_axis_energy_min_ratio": axis,
    }


def directional(coherence: float = 0.0, second: float = 0.0) -> dict:
    return {
        "directional_structure_tensor_coherence": coherence,
        "directional_spectral_second_moment": second,
    }


class RouteTests(unittest.TestCase):
    def test_directional_wood_like_evidence_has_priority(self) -> None:
        route = choose_route(metrics(0.02, 0.06, 0.3), directional(0.80, 0.70), 0.5, 0.08, args())
        self.assertEqual(route, "directional_spectral")

    def test_visible_periodic_motif_is_structured(self) -> None:
        route = choose_route(metrics(0.02, 0.03, 0.30), directional(), 0.5, 0.08, args())
        self.assertEqual(route, "structured")

    def test_low_detail_false_periodicity_uses_mirror_guard(self) -> None:
        route = choose_route(metrics(0.004, 0.02, 0.25), directional(), 0.985, 0.08, args())
        self.assertEqual(route, "nonstationary_spectral")

    def test_plain_low_detail_material_is_smooth(self) -> None:
        route = choose_route(metrics(0.004, 0.02, 0.05), directional(), 0.5, 0.02, args())
        self.assertEqual(route, "smooth")

    def test_unstructured_detail_is_stochastic(self) -> None:
        route = choose_route(metrics(0.02, 0.045, 0.05), directional(), 0.3, 0.05, args())
        self.assertEqual(route, "stochastic")


if __name__ == "__main__":
    unittest.main()

