#!/usr/bin/env python3
"""Geometric (depth/density) crop of Object A's trained 3DGS PLY.

Unlike the old color-filter crop (which deleted the doll's pale face and left a
hollow purple shell), this keeps ALL colors inside a tight 3D region centered on
the doll, then removes sparse floaters/background by voxel-density filtering.

Stages:
  1. Locate the doll center from robust purple-hair seed points (median).
  2. Keep visible points within an adaptive ball/box around that center.
  3. Voxel-density filter: drop points whose local voxel is sparsely populated
     (kills the flat door plane behind the doll and scattered floaters).
  4. Optional axis remap for scene insertion.

Use --analyze-only first to inspect the radial distribution before committing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compose_gaussian_scene import dc_to_rgb, read_gaussian_ply, write_gaussian_ply
from compose_scene import downsample, write_binary_ply

AXIS_NAMES = ("x", "y", "z")
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def opacity_values(data: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32)))


def purple_seed(rgb: np.ndarray, opacity: np.ndarray, min_opacity: float) -> np.ndarray:
    r = rgb[:, 0].astype(np.int16)
    g = rgb[:, 1].astype(np.int16)
    b = rgb[:, 2].astype(np.int16)
    sat = rgb.max(axis=1).astype(np.int16) - rgb.min(axis=1).astype(np.int16)
    bright = rgb.mean(axis=1)
    visible = opacity > min_opacity
    purple = (b > r + 15) & (r > g + 8) & (b > g + 25) & (sat > 32)
    dark_purple = (bright < 95) & (b >= g + 12) & (r >= g + 5) & (sat > 18)
    return visible & (purple | dark_purple)


def voxel_density_keep(xyz: np.ndarray, voxel: float, min_count: int) -> np.ndarray:
    """Keep points whose voxel cell contains >= min_count points."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    # hash voxel triplets to a single int
    k = keys - keys.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    return counts[inv] >= min_count


def remap_axes(xyz: np.ndarray, axis_map: str, axis_signs: list[float]) -> np.ndarray:
    if len(axis_map) != 3 or sorted(axis_map) != ["x", "y", "z"]:
        raise ValueError("--axis-map must be a permutation of x,y,z")
    idx = [AXIS_TO_INDEX[c] for c in axis_map]
    signs = np.asarray(axis_signs, dtype=np.float32)
    return xyz[:, idx].astype(np.float32) * signs


def write_projection(path: Path, xyz: np.ndarray, rgb: np.ndarray, axes: tuple[int, int]) -> None:
    import cv2

    size, pad = 1100, 45
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260529)
    pts, colors = downsample(xyz, rgb, 90_000, rng)
    xy = pts[:, axes].astype(np.float32)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (size - 2 * pad) / float(span.max())
    pix = np.round((xy - lo) * scale + pad).astype(np.int32)
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    y = size - 1 - pix[:, 1]
    x = pix[:, 0]
    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    img[y[valid], x[valid]] = colors[valid][:, ::-1]
    cv2.putText(img, f"{AXIS_NAMES[axes[0]]}/{AXIS_NAMES[axes[1]]}", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-ply", type=Path)
    p.add_argument("--preview-ply", type=Path)
    p.add_argument("--preview-dir", type=Path)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--min-opacity", type=float, default=0.1)
    p.add_argument("--radius-multiplier", type=float, default=2.5,
                   help="ball radius = seed 95th-pct spread * this")
    p.add_argument("--min-radius", type=float, default=0.2)
    p.add_argument("--max-radius", type=float, default=0.0, help="hard cap on radius, 0=off")
    p.add_argument("--voxel-divisor", type=float, default=48.0,
                   help="voxel size = crop bbox diag / this")
    p.add_argument("--min-voxel-count", type=int, default=4)
    p.add_argument("--axis-map", default="xyz")
    p.add_argument("--axis-signs", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    data, properties = read_gaussian_ply(args.input)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    opacity = opacity_values(data)
    rgb = dc_to_rgb(data)
    visible = opacity > args.min_opacity

    seed = purple_seed(rgb, opacity, args.min_opacity)
    if seed.sum() < 32:
        raise SystemExit(f"Too few purple seeds: {int(seed.sum())}")
    center = np.median(xyz[seed], axis=0)

    vis_xyz = xyz[visible]
    dist = np.linalg.norm(vis_xyz - center, axis=1)
    seed_spread = np.quantile(np.linalg.norm(xyz[seed] - center, axis=1), 0.95)
    radius = max(seed_spread * args.radius_multiplier, args.min_radius)
    if args.max_radius > 0:
        radius = min(radius, args.max_radius)

    report = {
        "input": str(args.input),
        "total": int(data.shape[0]),
        "visible": int(visible.sum()),
        "full_visible_bbox_min": vis_xyz.min(axis=0).tolist(),
        "full_visible_bbox_max": vis_xyz.max(axis=0).tolist(),
        "seed_count": int(seed.sum()),
        "doll_center": center.tolist(),
        "seed_95pct_spread": float(seed_spread),
        "dist_percentiles": {
            str(q): float(np.quantile(dist, q / 100.0))
            for q in (10, 25, 50, 75, 90, 95, 99)
        },
        "chosen_radius": float(radius),
        "radial_histogram": {
            f"<= {r:.2f}": int((dist <= r).sum())
            for r in (radius * 0.25, radius * 0.5, radius, radius * 1.5,
                      radius * 2.0, radius * 4.0)
        },
    }

    if args.analyze_only:
        print(json.dumps(report, indent=2))
        return 0

    ball = visible & (np.linalg.norm(xyz - center, axis=1) <= radius)
    ball_xyz = xyz[ball]
    diag = float(np.linalg.norm(ball_xyz.max(axis=0) - ball_xyz.min(axis=0)))
    voxel = max(diag / args.voxel_divisor, 1e-4)
    dense_local = voxel_density_keep(ball_xyz, voxel, args.min_voxel_count)
    keep_idx = np.where(ball)[0][dense_local]
    crop = np.zeros(data.shape[0], dtype=bool)
    crop[keep_idx] = True
    if crop.sum() < 500:
        raise SystemExit(f"Crop too small after density filter: {int(crop.sum())}")

    out_xyz = remap_axes(xyz[crop], args.axis_map, args.axis_signs)
    report.update({
        "voxel_size": voxel,
        "min_voxel_count": args.min_voxel_count,
        "ball_count": int(ball.sum()),
        "final_crop_count": int(crop.sum()),
        "crop_bbox_min": xyz[crop].min(axis=0).tolist(),
        "crop_bbox_max": xyz[crop].max(axis=0).tolist(),
        "crop_rgb_mean": rgb[crop].mean(axis=0).tolist(),
        "axis_map": args.axis_map,
        "axis_signs": args.axis_signs,
    })
    print(json.dumps(report, indent=2))

    if args.out_ply:
        out = data[crop].copy()
        out["x"], out["y"], out["z"] = out_xyz[:, 0], out_xyz[:, 1], out_xyz[:, 2]
        write_gaussian_ply(args.out_ply, out, properties)
    if args.preview_ply:
        write_binary_ply(args.preview_ply, out_xyz, rgb[crop])
    if args.preview_dir:
        for name, ax in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            write_projection(args.preview_dir / f"{name}.png", out_xyz, rgb[crop], ax)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
