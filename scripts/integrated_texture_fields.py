#!/usr/bin/env python3
"""Shared structure-aware texture analysis and expansion for v3b_v2."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = np.clip(image[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float32) - float(np.mean(first))
    b = second.astype(np.float32) - float(np.mean(second))
    denom = float(np.std(a) * np.std(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.mean(a * b) / denom)


def _best_axis_period(signal: np.ndarray, axis: int) -> tuple[int, float]:
    length = int(signal.shape[axis])
    other = int(signal.shape[1 - axis])
    if length < 24 or other < 8:
        return 0, 0.0
    start = max(4, length // 32)
    stop = max(start + 1, length // 2)
    step = max(1, length // 160)
    best_lag = 0
    best_corr = 0.0
    for lag in range(start, stop + 1, step):
        if axis == 1:
            corr = _normalized_correlation(signal[:, lag:], signal[:, :-lag])
        else:
            corr = _normalized_correlation(signal[lag:, :], signal[:-lag, :])
        # Avoid selecting a tiny local-texture lag when a larger motif has a
        # nearly identical correlation score.
        adjusted = corr + 0.015 * (lag / max(length, 1))
        best_adjusted = best_corr + 0.015 * (best_lag / max(length, 1))
        if adjusted > best_adjusted:
            best_lag = int(lag)
            best_corr = float(corr)
    return best_lag, best_corr


def analyze_structure(image: np.ndarray) -> dict[str, float | int]:
    """Measure long edges and repeated high-pass detail in a complete patch."""
    gray = _gray(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    axis_energy_x = float(np.mean(np.abs(gx)))
    axis_energy_y = float(np.mean(np.abs(gy)))
    axis_energy_ratio = min(axis_energy_x, axis_energy_y) / max(
        max(axis_energy_x, axis_energy_y), 1e-8
    )
    edge_p95 = float(np.quantile(magnitude, 0.95))
    threshold = max(0.02, 0.60 * edge_p95)
    strong = (magnitude >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    if count <= 1:
        max_span = 0.0
        max_area = 0.0
    else:
        components = stats[1:]
        max_span = float(
            np.max(
                np.maximum(
                    components[:, cv2.CC_STAT_WIDTH],
                    components[:, cv2.CC_STAT_HEIGHT],
                )
            )
        )
        max_area = float(np.max(components[:, cv2.CC_STAT_AREA]))
    sigma = max(1.0, min(gray.shape) * (8.0 / 512.0))
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), sigma)
    period_y, corr_y = _best_axis_period(highpass, 0)
    period_x, corr_x = _best_axis_period(highpass, 1)
    periodic_corr = max(corr_x, corr_y)
    return {
        "structured_source_height": int(gray.shape[0]),
        "structured_source_width": int(gray.shape[1]),
        "structured_edge_p95": edge_p95,
        "structured_axis_energy_x": axis_energy_x,
        "structured_axis_energy_y": axis_energy_y,
        "structured_axis_energy_min_ratio": float(axis_energy_ratio),
        "structured_edge_component_threshold": float(threshold),
        "structured_edge_max_span_frac": max_span / float(max(1, min(gray.shape))),
        "structured_edge_max_component_area_frac": max_area / float(max(1, gray.size)),
        "structured_highpass_std": float(np.std(highpass)),
        "structured_periodic_max_corr": float(periodic_corr),
        "structured_period_x": int(period_x),
        "structured_period_y": int(period_y),
        "structured_period_corr_x": float(corr_x),
        "structured_period_corr_y": float(corr_y),
    }


def choose_structure_strategy(
    metrics: dict[str, float | int],
    mode: str,
    edge_p95_threshold: float,
    edge_span_threshold: float,
    highpass_std_threshold: float,
    periodic_corr_threshold: float,
) -> str | None:
    if mode == "off":
        return None
    long_edge = bool(
        float(metrics["structured_edge_p95"]) >= edge_p95_threshold
        and float(metrics["structured_edge_max_span_frac"]) >= edge_span_threshold
    )
    repeated = bool(
        float(metrics["structured_highpass_std"]) >= highpass_std_threshold
        and float(metrics["structured_periodic_max_corr"]) >= periodic_corr_threshold
    )
    if not (long_edge or repeated):
        return None
    # Automatic routing must not turn a finite material exemplar into a
    # deterministic super-tile. Horizontal wainscot edges, lighting gradients,
    # and crop boundaries all satisfy ``long_edge``; a mirror fallback then
    # repeats those accidental structures over an entire room face. Keep these
    # measurements for audit, but let the original realrooms
    # quilting/stochastic/resize branches synthesize the field. Exact
    # full-patch repetition remains an explicit authoring choice only.
    if mode == "auto":
        return None
    if mode == "mirror":
        return "mirror"
    if mode == "periodic":
        return "periodic" if repeated else None
    raise ValueError(f"Unsupported structured texture mode: {mode}")


def _center_period_patch(
    source: np.ndarray,
    metrics: dict[str, float | int],
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = source.shape[:2]
    corr_y = float(metrics.get("structured_period_corr_y", 0.0))
    corr_x = float(metrics.get("structured_period_corr_x", 0.0))
    period_y = int(metrics.get("structured_period_y", 0)) if corr_y > 0.18 else h
    period_x = int(metrics.get("structured_period_x", 0)) if corr_x > 0.18 else w
    period_y = int(np.clip(period_y, max(8, h // 12), h))
    period_x = int(np.clip(period_x, max(8, w // 12), w))
    y0 = max(0, (h - period_y) // 2)
    x0 = max(0, (w - period_x) // 2)
    return source[y0 : y0 + period_y, x0 : x0 + period_x].copy(), (period_y, period_x)


def periodic_repeat_channel(
    source: np.ndarray,
    shape_hw: tuple[int, int],
    metrics: dict[str, float | int],
) -> tuple[np.ndarray, tuple[int, int]]:
    patch, period = _center_period_patch(np.clip(source, 0.0, 1.0), metrics)
    h, w = int(shape_hw[0]), int(shape_hw[1])
    repeats_y = max(1, int(np.ceil(h / patch.shape[0])))
    repeats_x = max(1, int(np.ceil(w / patch.shape[1])))
    tiled = np.tile(patch, (repeats_y, repeats_x, 1))
    return tiled[:h, :w].copy(), period


def mirror_repeat_channel(
    source: np.ndarray,
    shape_hw: tuple[int, int],
    channel: str = "basecolor",
) -> np.ndarray:
    """Mirror a complete patch; fix tangent signs when mirroring normals."""
    source = np.clip(source.astype(np.float32), 0.0, 1.0)
    right = source[:, ::-1].copy()
    bottom = source[::-1].copy()
    bottom_right = source[::-1, ::-1].copy()
    if channel == "normal":
        right[..., 0] = 1.0 - right[..., 0]
        bottom[..., 1] = 1.0 - bottom[..., 1]
        bottom_right[..., 0] = 1.0 - bottom_right[..., 0]
        bottom_right[..., 1] = 1.0 - bottom_right[..., 1]
    mirrored = np.concatenate(
        [np.concatenate([source, right], axis=1), np.concatenate([bottom, bottom_right], axis=1)],
        axis=0,
    )
    h, w = int(shape_hw[0]), int(shape_hw[1])
    repeats_y = max(1, int(np.ceil(h / mirrored.shape[0])))
    repeats_x = max(1, int(np.ceil(w / mirrored.shape[1])))
    tiled = np.tile(mirrored, (repeats_y, repeats_x, 1))
    return tiled[:h, :w].copy()


def expand_structured_channel(
    source: np.ndarray,
    shape_hw: tuple[int, int],
    strategy: str,
    metrics: dict[str, float | int],
    channel: str = "basecolor",
) -> tuple[np.ndarray, dict[str, Any]]:
    if strategy == "periodic":
        field, period = periodic_repeat_channel(source, shape_hw, metrics)
        return field, {
            "strategy": "structured_period_aware_fullpatch_repeat",
            "measured_period_hw": [int(period[0]), int(period[1])],
            "normal_strategy": "chord_normal_periodic_repeat" if channel == "normal" else None,
        }
    field = mirror_repeat_channel(source, shape_hw, channel)
    return field, {
        "strategy": "structured_fullpatch_mirror_repeat",
        "measured_period_hw": None,
        "normal_strategy": (
            "chord_normal_mirror_repeat_with_tangent_sign_fix" if channel == "normal" else None
        ),
    }
