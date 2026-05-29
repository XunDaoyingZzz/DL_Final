#!/usr/bin/env python3
"""Compose A/B/C objects into the reconstructed counter background scene.

The script converts the available assets into a unified colored point cloud:

- 3DGS PLY files are read directly. Their SH DC colors are converted to RGB.
- OBJ meshes are sampled on triangle surfaces and textured through UVs.
- All objects are scaled and placed on a simple row over the counter scene.
- A combined binary PLY and several preview PNGs are exported.

This is meant for report visualization and spatial-placement evidence, not for
physically correct Gaussian rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

SH_C0 = 0.28209479177387814


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


@dataclass
class AssetStats:
    name: str
    count: int
    bbox_min: list[float]
    bbox_max: list[float]
    center: list[float]
    span: list[float]


@dataclass
class TransformInfo:
    name: str
    source: str
    count: int
    scale: float
    translation: list[float]
    bbox_min: list[float]
    bbox_max: list[float]


def parse_ply_header(path: Path) -> tuple[int, int, list[tuple[str, str]], str]:
    with path.open("rb") as f:
        header_lines: list[str] = []
        header_len = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            header_len += len(line)
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break

    fmt = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            if parts[1] == "list":
                raise ValueError(f"Unsupported list property in vertex element: {line}")
            properties.append((parts[2], parts[1]))

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got {fmt}: {path}")
    if vertex_count <= 0:
        raise ValueError(f"No vertices found in PLY: {path}")
    return header_len, vertex_count, properties, fmt


def read_3dgs_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header_len, vertex_count, properties, _ = parse_ply_header(path)
    dtype_fields = []
    for name, typ in properties:
        if typ not in PLY_TYPES:
            raise ValueError(f"Unsupported PLY type {typ!r} for {name!r} in {path}")
        dtype_fields.append((name, PLY_TYPES[typ]))
    dtype = np.dtype(dtype_fields)

    with path.open("rb") as f:
        f.seek(header_len)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    xyz = np.column_stack(
        [
            data["x"].astype(np.float32),
            data["y"].astype(np.float32),
            data["z"].astype(np.float32),
        ]
    )

    names = set(data.dtype.names or [])
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.uint8)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(
            np.float32
        )
        rgb_f = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)
        rgb = np.round(rgb_f * 255.0).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)

    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32)))
        keep = opacity > 0.01
        if keep.any():
            xyz = xyz[keep]
            rgb = rgb[keep]
    return xyz, rgb


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[list[int], list[int]]]]:
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[tuple[list[int], list[int]]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                v_idx: list[int] = []
                vt_idx: list[int] = []
                for token in line.split()[1:]:
                    fields = token.split("/")
                    if not fields[0]:
                        continue
                    vi = int(fields[0])
                    v_idx.append(vi - 1 if vi > 0 else len(vertices) + vi)
                    if len(fields) > 1 and fields[1]:
                        ti = int(fields[1])
                        vt_idx.append(ti - 1 if ti > 0 else len(uvs) + ti)
                    else:
                        vt_idx.append(-1)
                if len(v_idx) >= 3:
                    for i in range(1, len(v_idx) - 1):
                        faces.append(
                            (
                                [v_idx[0], v_idx[i], v_idx[i + 1]],
                                [vt_idx[0], vt_idx[i], vt_idx[i + 1]],
                            )
                        )
    return np.asarray(vertices, dtype=np.float32), np.asarray(uvs, dtype=np.float32), faces


def sample_textured_obj(
    obj_path: Path,
    texture_path: Path | None,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, uvs, faces = parse_obj(obj_path)
    if vertices.size == 0 or not faces:
        raise ValueError(f"OBJ has no sampleable geometry: {obj_path}")

    tri_v = np.asarray([face[0] for face in faces], dtype=np.int64)
    pts = vertices[tri_v]
    areas = 0.5 * np.linalg.norm(
        np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1
    )
    valid = areas > 1.0e-12
    if not valid.any():
        raise ValueError(f"OBJ triangles have zero area: {obj_path}")
    tri_v = tri_v[valid]
    areas = areas[valid]
    probs = areas / areas.sum()

    chosen = rng.choice(len(tri_v), size=count, replace=True, p=probs)
    chosen_v = vertices[tri_v[chosen]]

    r1 = np.sqrt(rng.random(count, dtype=np.float32))
    r2 = rng.random(count, dtype=np.float32)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    xyz = (
        chosen_v[:, 0] * w0[:, None]
        + chosen_v[:, 1] * w1[:, None]
        + chosen_v[:, 2] * w2[:, None]
    ).astype(np.float32)

    rgb = np.full((count, 3), 210, dtype=np.uint8)
    if texture_path is not None and texture_path.exists() and len(uvs) > 0:
        tri_t = np.asarray([face[1] for face in faces], dtype=np.int64)[valid][chosen]
        has_uv = (tri_t >= 0).all(axis=1)
        if has_uv.any():
            uv_tri = uvs[tri_t[has_uv]]
            uv = (
                uv_tri[:, 0] * w0[has_uv, None]
                + uv_tri[:, 1] * w1[has_uv, None]
                + uv_tri[:, 2] * w2[has_uv, None]
            )
            img = Image.open(texture_path).convert("RGB")
            tex = np.asarray(img)
            h, w = tex.shape[:2]
            px = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int32)
            py = np.clip((1.0 - (uv[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(
                np.int32
            )
            rgb[has_uv] = tex[py, px]
    return xyz, rgb


def downsample(
    xyz: np.ndarray, rgb: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] <= count:
        return xyz.astype(np.float32), rgb.astype(np.uint8)
    idx = rng.choice(xyz.shape[0], size=count, replace=False)
    return xyz[idx].astype(np.float32), rgb[idx].astype(np.uint8)


def stats(name: str, xyz: np.ndarray) -> AssetStats:
    bmin = xyz.min(axis=0)
    bmax = xyz.max(axis=0)
    center = (bmin + bmax) * 0.5
    span = bmax - bmin
    return AssetStats(
        name=name,
        count=int(xyz.shape[0]),
        bbox_min=[float(v) for v in bmin],
        bbox_max=[float(v) for v in bmax],
        center=[float(v) for v in center],
        span=[float(v) for v in span],
    )


def normalize_object(
    xyz: np.ndarray,
    target_height: float,
    target_center_xy: tuple[float, float],
    target_bottom_z: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    bmin = xyz.min(axis=0)
    bmax = xyz.max(axis=0)
    span = bmax - bmin
    scale = target_height / float(max(span[2], 1.0e-6))
    centered = (xyz - (bmin + bmax) * 0.5) * scale
    after_min = centered.min(axis=0)
    after_max = centered.max(axis=0)
    translation = np.asarray(
        [
            target_center_xy[0],
            target_center_xy[1],
            target_bottom_z - after_min[2],
        ],
        dtype=np.float32,
    )
    placed = centered + translation
    return placed.astype(np.float32), float(scale), translation


def rotate_z(xyz: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    c, s = math.cos(radians), math.sin(radians)
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return xyz @ rot.T


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    arr = np.empty(xyz.shape[0], dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with path.open("wb") as f:
        f.write(header)
        arr.tofile(f)


def render_view(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    axes: tuple[int, int],
    title: str,
    max_points: int,
    rng: np.random.Generator,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    annotations: list[tuple[str, tuple[float, float]]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_xyz, sample_rgb = downsample(xyz, rgb, max_points, rng)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    ax.scatter(
        sample_xyz[:, axes[0]],
        sample_xyz[:, axes[1]],
        c=sample_rgb.astype(np.float32) / 255.0,
        s=0.08,
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(["x", "y", "z"][axes[0]])
    ax.set_ylabel(["x", "y", "z"][axes[1]])
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    if annotations:
        for label, xy in annotations:
            ax.text(
                xy[0],
                xy[1],
                label,
                fontsize=11,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.7},
            )
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_assets(args: argparse.Namespace, rng: np.random.Generator):
    bg_xyz, bg_rgb = read_3dgs_ply(args.bg_ply)
    a_xyz, a_rgb = read_3dgs_ply(args.a_ply)
    b_xyz, b_rgb = sample_textured_obj(args.b_obj, args.b_texture, args.mesh_points, rng)
    c_xyz, c_rgb = sample_textured_obj(args.c_obj, args.c_texture, args.mesh_points, rng)
    return {
        "background": (bg_xyz, bg_rgb),
        "A_real_3dgs": (a_xyz, a_rgb),
        "B_text3d": (b_xyz, b_rgb),
        "C_zero123": (c_xyz, c_rgb),
    }


def compose(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    assets = load_assets(args, rng)

    raw_stats = [stats(name, xyz) for name, (xyz, _) in assets.items()]
    if args.inspect_only:
        print(json.dumps([asdict(s) for s in raw_stats], indent=2))
        return

    bg_xyz, bg_rgb = downsample(*assets["background"], args.bg_points, rng)
    a_xyz, a_rgb = downsample(*assets["A_real_3dgs"], args.a_points, rng)
    b_xyz, b_rgb = assets["B_text3d"]
    c_xyz, c_rgb = assets["C_zero123"]

    # Use the dense central part of the counter scene as a robust placement area.
    bg_center = np.median(bg_xyz, axis=0)
    bg_span = np.percentile(bg_xyz, 95, axis=0) - np.percentile(bg_xyz, 5, axis=0)
    table_z = float(np.percentile(bg_xyz[:, 2], args.table_z_percentile))
    object_height = float(max(bg_span[2] * args.object_height_ratio, 0.25))
    spacing = float(max(bg_span[0] * args.spacing_ratio, object_height * 1.35))
    y = float(bg_center[1] + bg_span[1] * args.y_offset_ratio)

    placements = {
        "A_real_3dgs": ((float(bg_center[0] - spacing), y), table_z, args.a_rotation),
        "B_text3d": ((float(bg_center[0]), y), table_z, args.b_rotation),
        "C_zero123": ((float(bg_center[0] + spacing), y), table_z, args.c_rotation),
    }

    transformed: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("background", bg_xyz, bg_rgb)
    ]
    transforms: list[TransformInfo] = [
        TransformInfo(
            name="background",
            source=str(args.bg_ply),
            count=int(bg_xyz.shape[0]),
            scale=1.0,
            translation=[0.0, 0.0, 0.0],
            bbox_min=[float(v) for v in bg_xyz.min(axis=0)],
            bbox_max=[float(v) for v in bg_xyz.max(axis=0)],
        )
    ]

    for name, (xyz, rgb) in [
        ("A_real_3dgs", (a_xyz, a_rgb)),
        ("B_text3d", (b_xyz, b_rgb)),
        ("C_zero123", (c_xyz, c_rgb)),
    ]:
        center_xy, bottom_z, rot = placements[name]
        rotated = rotate_z(xyz, rot)
        placed, scale, translation = normalize_object(
            rotated, object_height, center_xy, bottom_z
        )
        transformed.append((name, placed, rgb))
        transforms.append(
            TransformInfo(
                name=name,
                source=str(
                    {
                        "A_real_3dgs": args.a_ply,
                        "B_text3d": args.b_obj,
                        "C_zero123": args.c_obj,
                    }[name]
                ),
                count=int(placed.shape[0]),
                scale=scale,
                translation=[float(v) for v in translation],
                bbox_min=[float(v) for v in placed.min(axis=0)],
                bbox_max=[float(v) for v in placed.max(axis=0)],
            )
        )

    combined_xyz = np.concatenate([item[1] for item in transformed], axis=0)
    combined_rgb = np.concatenate([item[2] for item in transformed], axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_binary_ply(args.out_dir / "combined_scene.ply", combined_xyz, combined_rgb)

    metadata = {
        "raw_stats": [asdict(s) for s in raw_stats],
        "combined_stats": asdict(stats("combined_scene", combined_xyz)),
        "table_z_percentile": args.table_z_percentile,
        "table_z": table_z,
        "object_height": object_height,
        "spacing": spacing,
        "transforms": [asdict(t) for t in transforms],
        "note": "Point-cloud composition for report visualization; B/C meshes are texture-sampled.",
    }
    (args.out_dir / "transforms.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    preview_rng = np.random.default_rng(args.seed + 7)
    render_view(
        args.out_dir / "render_top_xy.png",
        combined_xyz,
        combined_rgb,
        (0, 1),
        "Top view (x-y)",
        args.preview_points,
        preview_rng,
    )
    render_view(
        args.out_dir / "render_front_xz.png",
        combined_xyz,
        combined_rgb,
        (0, 2),
        "Front view (x-z)",
        args.preview_points,
        preview_rng,
    )
    render_view(
        args.out_dir / "render_side_yz.png",
        combined_xyz,
        combined_rgb,
        (1, 2),
        "Side view (y-z)",
        args.preview_points,
        preview_rng,
    )

    object_xyz = np.concatenate([item[1] for item in transformed if item[0] != "background"])
    object_min = object_xyz.min(axis=0)
    object_max = object_xyz.max(axis=0)
    margin = max(float((object_max - object_min).max()) * 0.35, 0.5)
    object_centers = {
        item.name: (np.asarray(item.bbox_min) + np.asarray(item.bbox_max)) * 0.5
        for item in transforms
        if item.name != "background"
    }

    def focus_limits(axis_a: int, axis_b: int):
        return (
            (float(object_min[axis_a] - margin), float(object_max[axis_a] + margin)),
            (float(object_min[axis_b] - margin), float(object_max[axis_b] + margin)),
        )

    def focus_annotations(axis_a: int, axis_b: int):
        labels = {"A_real_3dgs": "A", "B_text3d": "B", "C_zero123": "C"}
        return [
            (labels[name], (float(center[axis_a]), float(center[axis_b])))
            for name, center in object_centers.items()
        ]

    render_view(
        args.out_dir / "render_focus_top_xy.png",
        combined_xyz,
        combined_rgb,
        (0, 1),
        "Composed objects on counter (top x-y)",
        args.preview_points,
        preview_rng,
        limits=focus_limits(0, 1),
        annotations=focus_annotations(0, 1),
    )
    render_view(
        args.out_dir / "render_focus_front_xz.png",
        combined_xyz,
        combined_rgb,
        (0, 2),
        "Composed objects on counter (front x-z)",
        args.preview_points,
        preview_rng,
        limits=focus_limits(0, 2),
        annotations=focus_annotations(0, 2),
    )
    render_view(
        args.out_dir / "render_focus_side_yz.png",
        combined_xyz,
        combined_rgb,
        (1, 2),
        "Composed objects on counter (side y-z)",
        args.preview_points,
        preview_rng,
        limits=focus_limits(1, 2),
        annotations=focus_annotations(1, 2),
    )

    print(json.dumps(metadata, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path("report_materials/local_outputs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-ply", type=Path, default=root / "A_3dgs/point_cloud_iteration_3000.ply")
    parser.add_argument(
        "--bg-ply",
        type=Path,
        default=root / "BG_counter_3dgs/point_cloud_iteration_3000.ply",
    )
    parser.add_argument(
        "--b-obj",
        type=Path,
        default=root
        / "B_text3d/steps_3000/it3000-export/model_textured_auto.obj",
    )
    parser.add_argument(
        "--b-texture",
        type=Path,
        default=root / "B_text3d/steps_3000/it3000-export/texture_kd.jpg",
    )
    parser.add_argument(
        "--c-obj", type=Path, default=root / "C_zero123/it400-export/model.obj"
    )
    parser.add_argument(
        "--c-texture",
        type=Path,
        default=root / "C_zero123/it400-export/texture_kd.jpg",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("report_materials/local_outputs/scene_composition"),
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--bg-points", type=int, default=260_000)
    parser.add_argument("--a-points", type=int, default=80_000)
    parser.add_argument("--mesh-points", type=int, default=70_000)
    parser.add_argument("--preview-points", type=int, default=180_000)
    parser.add_argument("--table-z-percentile", type=float, default=58.0)
    parser.add_argument("--object-height-ratio", type=float, default=0.12)
    parser.add_argument("--spacing-ratio", type=float, default=0.22)
    parser.add_argument("--y-offset-ratio", type=float, default=0.02)
    parser.add_argument("--a-rotation", type=float, default=0.0)
    parser.add_argument("--b-rotation", type=float, default=0.0)
    parser.add_argument("--c-rotation", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    compose(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
