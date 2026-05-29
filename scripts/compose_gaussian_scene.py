#!/usr/bin/env python3
"""Compose A/B/C and background as a native 3DGS Gaussian PLY model."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from compose_scene import (
    SH_C0,
    normalize_object,
    rotate_z,
    sample_textured_obj,
    stats,
    write_binary_ply,
)

PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def parse_ply_header(path: Path) -> tuple[int, int, list[tuple[str, str]]]:
    with path.open("rb") as f:
        header_len = 0
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        fmt = ""
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            header_len += len(line)
            text = line.decode("ascii", errors="replace").strip()
            parts = text.split()
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                if parts[1] == "list":
                    raise ValueError(f"Unsupported list property in vertex element: {text}")
                properties.append((parts[2], parts[1]))
            if text == "end_header":
                break

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got {fmt}: {path}")
    if vertex_count <= 0:
        raise ValueError(f"No vertex data found in PLY: {path}")
    return header_len, vertex_count, properties


def gaussian_dtype(properties: list[tuple[str, str]]) -> np.dtype:
    return np.dtype([(name, PLY_TYPES[typ]) for name, typ in properties])


def read_gaussian_ply(path: Path) -> tuple[np.ndarray, list[tuple[str, str]]]:
    header_len, vertex_count, properties = parse_ply_header(path)
    dtype = gaussian_dtype(properties)
    with path.open("rb") as f:
        f.seek(header_len)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
    return data, properties


def visible_xyz(data: np.ndarray, opacity_threshold: float) -> np.ndarray:
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    if "opacity" not in data.dtype.names:
        return xyz
    opacity = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32)))
    keep = opacity > opacity_threshold
    return xyz[keep] if keep.any() else xyz


def rgb_to_dc(rgb: np.ndarray) -> np.ndarray:
    rgb01 = rgb.astype(np.float32) / 255.0
    return (rgb01 - 0.5) / SH_C0


def dc_to_rgb(data: np.ndarray) -> np.ndarray:
    sh = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)
    rgb01 = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)
    return np.round(rgb01 * 255.0).astype(np.uint8)


def transform_gaussians(
    data: np.ndarray,
    placed_xyz: np.ndarray,
    scale_factor: float,
    opacity_threshold: float,
    gaussian_scale_multiplier: float = 1.0,
    opacity_multiplier: float = 1.0,
) -> np.ndarray:
    opacity = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32)))
    keep = opacity > opacity_threshold
    if not keep.any():
        keep = np.ones(data.shape[0], dtype=bool)
    out = data[keep].copy()
    out["x"], out["y"], out["z"] = placed_xyz[keep, 0], placed_xyz[keep, 1], placed_xyz[keep, 2]
    log_scale = math.log(max(scale_factor * gaussian_scale_multiplier, 1.0e-8))
    for name in ("scale_0", "scale_1", "scale_2"):
        out[name] = out[name].astype(np.float32) + log_scale
    if opacity_multiplier != 1.0:
        boosted = np.clip(opacity[keep] * opacity_multiplier, 1.0e-5, 0.995)
        out["opacity"] = np.log(boosted / (1.0 - boosted)).astype(np.float32)
    return out


def make_mesh_gaussians(
    xyz: np.ndarray,
    rgb: np.ndarray,
    dtype: np.dtype,
    gaussian_scale: float,
    opacity: float,
) -> np.ndarray:
    out = np.zeros(xyz.shape[0], dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["nx"], out["ny"], out["nz"] = 0.0, 0.0, 0.0
    dc = rgb_to_dc(rgb)
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    for name in dtype.names or ():
        if name.startswith("f_rest_"):
            out[name] = 0.0
    logit = math.log(opacity / (1.0 - opacity))
    out["opacity"] = logit
    log_scale = math.log(gaussian_scale)
    out["scale_0"], out["scale_1"], out["scale_2"] = log_scale, log_scale, log_scale
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = 1.0, 0.0, 0.0, 0.0
    return out


def write_gaussian_ply(path: Path, data: np.ndarray, properties: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {data.shape[0]}",
    ]
    lines.extend(f"property {typ} {name}" for name, typ in properties)
    lines.append("end_header")
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        data.tofile(f)


def update_cfg_args(template: Path, out_path: Path, model_path: Path) -> None:
    text = template.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"model_path='[^']*'", f"model_path='{model_path.as_posix()}'", text)
    out_path.write_text(text, encoding="utf-8")


def compose(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    bg, properties = read_gaussian_ply(args.bg_ply)
    a, a_properties = read_gaussian_ply(args.a_ply)
    if properties != a_properties:
        raise ValueError("A and background Gaussian PLY properties do not match.")

    bg_xyz_visible = visible_xyz(bg, args.opacity_threshold)
    a_xyz_all = np.column_stack([a["x"], a["y"], a["z"]]).astype(np.float32)
    a_xyz_visible = visible_xyz(a, args.opacity_threshold)

    b_xyz, b_rgb = sample_textured_obj(args.b_obj, args.b_texture, args.mesh_points, rng)
    c_xyz, c_rgb = sample_textured_obj(args.c_obj, args.c_texture, args.mesh_points, rng)

    bg_stats = stats("background_visible", bg_xyz_visible)
    bg_span = np.asarray(bg_stats.span, dtype=np.float32)
    bg_center = np.asarray(bg_stats.center, dtype=np.float32)
    placement_meta = None
    placement_by_name = {}
    if args.placement_json.exists():
        placement_meta = json.loads(args.placement_json.read_text(encoding="utf-8"))
        placement_by_name = {item["name"]: item for item in placement_meta["transforms"]}

    if placement_meta is None:
        table_z = float(np.percentile(bg_xyz_visible[:, 2], args.table_z_percentile))
        object_height = float(max(bg_span[2] * args.object_height_ratio, 0.35))
        max_scaled_height = object_height * max(args.a_size_multiplier, args.b_size_multiplier, args.c_size_multiplier)
        spacing = float(max(bg_span[0] * args.spacing_ratio, max_scaled_height * 1.18))
        y = float(bg_center[1] + bg_span[1] * args.y_offset_ratio)
        placements = {
            "A_real_3dgs": ((float(bg_center[0] + args.a_slot * spacing), y), table_z, args.a_rotation),
            "B_text3d": ((float(bg_center[0] + args.b_slot * spacing), y), table_z, args.b_rotation),
            "C_zero123": ((float(bg_center[0] + args.c_slot * spacing), y), table_z, args.c_rotation),
        }
    else:
        table_z = float(placement_meta["table_z"])
        object_height = float(placement_meta["object_height"])
        spacing = float(placement_meta["spacing"])
        placements = {}

    # A is an optimized Gaussian model, so preserve its learned opacity, SH, scale and rotation.
    a_rot_all = rotate_z(a_xyz_all, args.a_rotation)
    a_rot_visible = rotate_z(a_xyz_visible, args.a_rotation)
    if placement_by_name:
        a_record = placement_by_name["A_real_3dgs"]
        a_scale = float(a_record["scale"])
        a_translation_all = np.asarray(a_record["translation"], dtype=np.float32)
        a_center = (a_rot_visible.min(axis=0) + a_rot_visible.max(axis=0)) * 0.5
        placed_a_all = (a_rot_all - a_center) * a_scale + a_translation_all
        placed_a_visible = (a_rot_visible - a_center) * a_scale + a_translation_all
    else:
        a_rot_all = rotate_z(a_xyz_all, placements["A_real_3dgs"][2])
        a_rot_visible = rotate_z(a_xyz_visible, placements["A_real_3dgs"][2])
        placed_a_visible, a_scale, _ = normalize_object(
            a_rot_visible,
            object_height * args.a_size_multiplier,
            placements["A_real_3dgs"][0],
            placements["A_real_3dgs"][1],
        )
        a_bmin = a_rot_visible.min(axis=0)
        a_bmax = a_rot_visible.max(axis=0)
        a_center = (a_bmin + a_bmax) * 0.5
        centered_a_all = (a_rot_all - a_center) * a_scale
        a_after_min = centered_a_all.min(axis=0)
        a_translation_all = np.asarray(
            [
                placements["A_real_3dgs"][0][0],
                placements["A_real_3dgs"][0][1],
                placements["A_real_3dgs"][1] - a_after_min[2],
            ],
            dtype=np.float32,
        )
        placed_a_all = centered_a_all + a_translation_all
    a_out = transform_gaussians(
        a,
        placed_a_all.astype(np.float32),
        a_scale,
        args.opacity_threshold,
        args.a_gaussian_scale_multiplier,
        args.a_opacity_multiplier,
    )

    mesh_outputs = []
    mesh_preview_parts = []
    transform_records = []
    for name, raw_xyz, rgb in [
        ("B_text3d", b_xyz, b_rgb),
        ("C_zero123", c_xyz, c_rgb),
    ]:
        rotation = {"B_text3d": args.b_rotation, "C_zero123": args.c_rotation}[name]
        size_multiplier = {"B_text3d": args.b_size_multiplier, "C_zero123": args.c_size_multiplier}[name]
        rotated = rotate_z(raw_xyz, rotation)
        if placement_by_name:
            record = placement_by_name[name]
            scale = float(record["scale"])
            translation = np.asarray(record["translation"], dtype=np.float32)
            raw_center = (rotated.min(axis=0) + rotated.max(axis=0)) * 0.5
            placed = ((rotated - raw_center) * scale + translation).astype(np.float32)
        else:
            center_xy, bottom_z, _ = placements[name]
            placed, scale, translation = normalize_object(
                rotated,
                object_height * size_multiplier,
                center_xy,
                bottom_z,
            )
        mesh_outputs.append(
            make_mesh_gaussians(
                placed,
                rgb,
                bg.dtype,
                args.mesh_gaussian_scale,
                args.mesh_opacity,
            )
        )
        mesh_preview_parts.append((placed, rgb))
        transform_records.append(
            {
                "name": name,
                "count": int(placed.shape[0]),
                "scale": float(scale),
                "translation": [float(v) for v in translation],
                "bbox_min": [float(v) for v in placed.min(axis=0)],
                "bbox_max": [float(v) for v in placed.max(axis=0)],
            }
        )

    if args.focus_bg_crop:
        object_xyz_for_crop = np.concatenate(
            [
                placed_a_visible.astype(np.float32),
                *(part[0].astype(np.float32) for part in mesh_preview_parts),
            ],
            axis=0,
        )
        crop_min = object_xyz_for_crop.min(axis=0)
        crop_max = object_xyz_for_crop.max(axis=0)
        crop_min[:2] -= args.bg_focus_margin_xy
        crop_max[:2] += args.bg_focus_margin_xy
        crop_min[2] = table_z - args.bg_focus_below_z
        crop_max[2] += args.bg_focus_above_z
        bg_xyz_all = np.column_stack([bg["x"], bg["y"], bg["z"]]).astype(np.float32)
        keep = np.all((bg_xyz_all >= crop_min) & (bg_xyz_all <= crop_max), axis=1)
        if "opacity" in bg.dtype.names:
            bg_opacity = 1.0 / (1.0 + np.exp(-bg["opacity"].astype(np.float32)))
            keep &= bg_opacity > args.bg_focus_opacity_threshold
        if keep.sum() < args.min_focus_bg_gaussians:
            raise ValueError(
                f"Focus background crop kept only {int(keep.sum())} gaussians; "
                "increase margins or lower thresholds."
            )
        bg_out = bg[keep].copy()
    else:
        crop_min = None
        crop_max = None
        bg_out = bg.copy()
    combined = np.concatenate([bg_out, a_out, *mesh_outputs])

    model_path = args.out_dir / "model"
    ply_path = model_path / "point_cloud/iteration_3000/point_cloud.ply"
    write_gaussian_ply(ply_path, combined, properties)

    if args.bg_cfg_args.exists():
        update_cfg_args(args.bg_cfg_args, model_path / "cfg_args", model_path)
    for rel in ("input.ply", "cameras.json", "exposure.json"):
        src = args.bg_model_dir / rel
        if src.exists():
            shutil.copy2(src, model_path / rel)

    # A small colored preview PLY is useful for quick inspection without the rasterizer.
    preview_xyz = np.concatenate(
        [
            np.column_stack([bg["x"], bg["y"], bg["z"]]).astype(np.float32),
            np.column_stack([a_out["x"], a_out["y"], a_out["z"]]).astype(np.float32),
            *(part[0] for part in mesh_preview_parts),
        ],
        axis=0,
    )
    preview_rgb = np.concatenate(
        [
            dc_to_rgb(bg),
            dc_to_rgb(a_out),
            *(part[1] for part in mesh_preview_parts),
        ],
        axis=0,
    )
    write_binary_ply(args.out_dir / "combined_gaussian_preview_rgb.ply", preview_xyz, preview_rgb)

    metadata = {
        "method": "native 3DGS Gaussian PLY composition",
        "model_dir": str(model_path),
        "point_cloud": str(ply_path),
        "counts": {
            "background_gaussians": int(bg_out.shape[0]),
            "A_real_3dgs_gaussians": int(a_out.shape[0]),
            "B_mesh_synthetic_gaussians": int(mesh_outputs[0].shape[0]),
            "C_mesh_synthetic_gaussians": int(mesh_outputs[1].shape[0]),
            "combined_gaussians": int(combined.shape[0]),
        },
        "placement": {
            "table_z_percentile": args.table_z_percentile,
            "table_z": table_z,
            "object_height": object_height,
            "spacing": spacing,
            "size_multipliers": {
                "A_real_3dgs": args.a_size_multiplier,
                "B_text3d": args.b_size_multiplier,
                "C_zero123": args.c_size_multiplier,
            },
            "slots": {
                "A_real_3dgs": args.a_slot,
                "B_text3d": args.b_slot,
                "C_zero123": args.c_slot,
            },
            "a_gaussian_scale_multiplier": args.a_gaussian_scale_multiplier,
            "a_opacity_multiplier": args.a_opacity_multiplier,
            "mesh_gaussian_scale": args.mesh_gaussian_scale,
            "mesh_opacity": args.mesh_opacity,
            "opacity_threshold_for_A": args.opacity_threshold,
            "placement_source": str(args.placement_json) if placement_meta is not None else "computed_from_background",
            "focus_bg_crop": bool(args.focus_bg_crop),
            "focus_bg_crop_min": None if crop_min is None else [float(v) for v in crop_min],
            "focus_bg_crop_max": None if crop_max is None else [float(v) for v in crop_max],
        },
        "A_real_3dgs": {
            "count": int(a_out.shape[0]),
            "scale": float(a_scale),
            "translation": [float(v) for v in a_translation_all],
            "bbox_min": [float(v) for v in placed_a_visible.min(axis=0)],
            "bbox_max": [float(v) for v in placed_a_visible.max(axis=0)],
        },
        "mesh_assets": transform_records,
        "note": "B/C are textured meshes sampled into synthetic Gaussian splats; A/background preserve trained 3DGS parameters.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gaussian_composition_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path("/mnt/d/2026_Spring/deeplearning/hw3/report_materials/local_outputs")
    bg_model = Path("/home/xundaoying/hw3_work/outputs/background_counter_3dgs_3000_clean")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-ply", type=Path, default=Path("/home/xundaoying/hw3_work/outputs/A_real_object_3dgs_3000_clean/point_cloud/iteration_3000/point_cloud.ply"))
    parser.add_argument("--bg-ply", type=Path, default=bg_model / "point_cloud/iteration_3000/point_cloud.ply")
    parser.add_argument("--b-obj", type=Path, default=Path("/home/xundaoying/hw3_work/outputs/threestudio/hw3_object_b_text3d/steps_3000_wandb/save/it3000-export/model.obj"))
    parser.add_argument("--b-texture", type=Path, default=Path("/home/xundaoying/hw3_work/outputs/threestudio/hw3_object_b_text3d/steps_3000_wandb/save/it3000-export/texture_kd.jpg"))
    parser.add_argument("--c-obj", type=Path, default=Path("/home/xundaoying/hw3_work/outputs/threestudio/hw3_object_c_stable_zero123/steps_400_sam3_wandb/save/it400-export/model.obj"))
    parser.add_argument("--c-texture", type=Path, default=Path("/home/xundaoying/hw3_work/outputs/threestudio/hw3_object_c_stable_zero123/steps_400_sam3_wandb/save/it400-export/texture_kd.jpg"))
    parser.add_argument("--out-dir", type=Path, default=root / "scene_gaussian_3dgs_clean")
    parser.add_argument(
        "--placement-json",
        type=Path,
        default=root / "scene_composition_clean/transforms.json",
    )
    parser.add_argument(
        "--bg-model-dir",
        type=Path,
        default=bg_model,
    )
    parser.add_argument(
        "--bg-cfg-args",
        type=Path,
        default=bg_model / "cfg_args",
    )
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--mesh-points", type=int, default=80_000)
    parser.add_argument("--table-z-percentile", type=float, default=58.0)
    parser.add_argument("--object-height-ratio", type=float, default=0.12)
    parser.add_argument("--spacing-ratio", type=float, default=0.22)
    parser.add_argument("--y-offset-ratio", type=float, default=0.02)
    parser.add_argument("--a-size-multiplier", type=float, default=1.0)
    parser.add_argument("--b-size-multiplier", type=float, default=1.0)
    parser.add_argument("--c-size-multiplier", type=float, default=1.0)
    parser.add_argument("--a-slot", type=float, default=-1.0)
    parser.add_argument("--b-slot", type=float, default=0.0)
    parser.add_argument("--c-slot", type=float, default=1.0)
    parser.add_argument("--a-gaussian-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--a-opacity-multiplier", type=float, default=1.0)
    parser.add_argument("--a-rotation", type=float, default=0.0)
    parser.add_argument("--b-rotation", type=float, default=0.0)
    parser.add_argument("--c-rotation", type=float, default=0.0)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--mesh-gaussian-scale", type=float, default=0.018)
    parser.add_argument("--mesh-opacity", type=float, default=0.74)
    parser.add_argument("--focus-bg-crop", action="store_true")
    parser.add_argument("--bg-focus-margin-xy", type=float, default=3.2)
    parser.add_argument("--bg-focus-below-z", type=float, default=0.45)
    parser.add_argument("--bg-focus-above-z", type=float, default=1.2)
    parser.add_argument("--bg-focus-opacity-threshold", type=float, default=0.01)
    parser.add_argument("--min-focus-bg-gaussians", type=int, default=15_000)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    compose(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
