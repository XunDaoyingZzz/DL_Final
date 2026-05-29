#!/usr/bin/env python3
"""Compose A (cropped 3DGS) + B/C (mesh->Gaussian) onto the garden 3DGS background.

Garden is y-up (verified from its training cameras). A is also y-up, so objects stay
upright with no axis remap. Objects are placed on the garden table near the scene
center, side by side. A keeps its trained Gaussian params (with a CORRECT quaternion
rotation about the up axis, unlike old2 which rotated positions only); B/C are textured
meshes sampled into isotropic Gaussian splats.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from compose_gaussian_scene import (
    make_mesh_gaussians, read_gaussian_ply, transform_gaussians, write_gaussian_ply,
    dc_to_rgb,
)
from compose_scene import sample_textured_obj, write_binary_ply


def rot_matrix(axis: str, deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, quats as (w,x,y,z), b is (N,4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def place_points(xyz, rot_x, rot_y, target_size, cx, cz, table_y, size_mult):
    R = rot_matrix("y", rot_y) @ rot_matrix("x", rot_x)
    x = xyz @ R.T
    bmin, bmax = x.min(0), x.max(0)
    center = (bmin + bmax) * 0.5
    span = float((bmax - bmin).max())
    scale = (target_size * size_mult) / max(span, 1e-6)
    x = (x - center) * scale
    out = x.copy()
    out[:, 0] += cx
    out[:, 2] += cz
    out[:, 1] += table_y - out[:, 1].min()  # bottom rests on the table
    return out.astype(np.float32), R, scale


def sample_surfels(obj_path, count, rng):
    """Sample surface points + per-point unit normals from an OBJ (area-weighted)."""
    from compose_scene import parse_obj
    V, _UV, faces = parse_obj(obj_path)
    tri = np.array([f[0] for f in faces], dtype=np.int64)
    p = V[tri]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    ok = area > 1e-12
    tri, p, nrm, area = tri[ok], p[ok], nrm[ok], area[ok]
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    ch = rng.choice(len(tri), size=count, replace=True, p=area / area.sum())
    r1 = np.sqrt(rng.random(count, dtype=np.float32))
    r2 = rng.random(count, dtype=np.float32)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    pts = p[ch, 0] * w0[:, None] + p[ch, 1] * w1[:, None] + p[ch, 2] * w2[:, None]
    return pts.astype(np.float32), nrm[ch].astype(np.float32)


def quats_from_normals(n):
    """Quaternions (w,x,y,z) rotating local +z onto each unit normal."""
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    d = np.clip(n @ ref, -1.0, 1.0)
    axis = np.cross(np.broadcast_to(ref, n.shape), n)
    an = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(an > 1e-8, axis / np.maximum(an, 1e-9),
                    np.array([1.0, 0.0, 0.0], dtype=np.float32))
    half = np.arccos(d) * 0.5
    s = np.sin(half)
    return np.column_stack([np.cos(half), axis * s[:, None]]).astype(np.float32)


def make_surfel_gaussians(xyz, quats, rgb, dtype, tangent_scale, normal_scale, opacity):
    """Flat, surface-aligned Gaussians (crisp opaque surfels) instead of fuzzy balls."""
    from compose_gaussian_scene import rgb_to_dc
    out = np.zeros(xyz.shape[0], dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if "nx" in (dtype.names or ()):
        out["nx"], out["ny"], out["nz"] = 0.0, 0.0, 0.0
    dc = rgb_to_dc(rgb)
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    for name in dtype.names or ():
        if name.startswith("f_rest_"):
            out[name] = 0.0
    out["opacity"] = math.log(opacity / (1.0 - opacity))
    out["scale_0"] = math.log(tangent_scale)
    out["scale_1"] = math.log(tangent_scale)
    out["scale_2"] = math.log(normal_scale)  # thin along the surface normal
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bg-ply", type=Path, required=True)
    p.add_argument("--bg-model-dir", type=Path, required=True)
    p.add_argument("--a-ply", type=Path, required=True)
    p.add_argument("--b-obj", type=Path, required=True)
    p.add_argument("--b-texture", type=Path, required=True)
    p.add_argument("--c-obj", type=Path, required=True)
    p.add_argument("--c-texture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--center-xz", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--table-y", type=float, default=1.0)
    p.add_argument("--object-size", type=float, default=0.6, help="target max dim, world units")
    p.add_argument("--spacing", type=float, default=0.85)
    p.add_argument("--layout", choices=["line", "ring"], default="ring")
    p.add_argument("--layout-axis", choices=["x", "z"], default="x")
    p.add_argument("--ring-radius", type=float, default=1.1)
    p.add_argument("--ring-azimuth0", type=float, default=0.0, help="deg, A's azimuth on the ring")
    p.add_argument("--face-outward", action="store_true",
                   help="rotate each object so its front points radially outward")
    p.add_argument("--a-size-mult", type=float, default=1.0)
    p.add_argument("--b-size-mult", type=float, default=1.0)
    p.add_argument("--c-size-mult", type=float, default=1.0)
    p.add_argument("--a-rot-y", type=float, default=0.0)
    p.add_argument("--b-rot-x", type=float, default=0.0)
    p.add_argument("--b-rot-y", type=float, default=0.0)
    p.add_argument("--c-rot-x", type=float, default=0.0)
    p.add_argument("--c-rot-y", type=float, default=0.0)
    p.add_argument("--mesh-points", type=int, default=200000)
    p.add_argument("--surfel-tangent", type=float, default=0.007, help="in-plane surfel radius")
    p.add_argument("--surfel-normal", type=float, default=0.0015, help="thin axis along normal")
    p.add_argument("--mesh-opacity", type=float, default=0.99)
    # threestudio's exported mesh textures are fragmented noise atlases (bad albedo bake),
    # so by default we color B/C with a clean representative solid color (coarse generated
    # objects, honest for the 3-route comparison). Pass negative RGB to keep the texture.
    p.add_argument("--b-rgb", type=int, nargs=3, default=[128, 80, 165])
    p.add_argument("--c-rgb", type=int, nargs=3, default=[96, 76, 104])
    p.add_argument("--color-jitter", type=float, default=6.0)
    p.add_argument("--a-opacity-threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260529)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    bg, props = read_gaussian_ply(args.bg_ply)
    a, a_props = read_gaussian_ply(args.a_ply)
    if props != a_props:
        raise SystemExit("A and bg PLY properties differ")

    cx, cz = args.center_xz
    order = ["A", "B", "C"]
    positions, facing = {}, {}
    if args.layout == "ring":
        for i, name in enumerate(order):
            adeg = args.ring_azimuth0 + i * 120.0
            az = math.radians(adeg)
            positions[name] = (cx + args.ring_radius * math.cos(az),
                               cz + args.ring_radius * math.sin(az))
            facing[name] = adeg if args.face_outward else 0.0
    else:
        off = {"A": -args.spacing, "B": 0.0, "C": args.spacing}
        for name in order:
            d = off[name]
            positions[name] = (cx + d, cz) if args.layout_axis == "x" else (cx, cz + d)
            facing[name] = 0.0

    def obj_center(name):
        return positions[name]

    records = {}

    # ---- A: cropped 3DGS, keep trained params, rotate about up=y (quat-correct) ----
    a_xyz = np.column_stack([a["x"], a["y"], a["z"]]).astype(np.float32)
    acx, acz = obj_center("A")
    a_roty = facing["A"] + args.a_rot_y
    placed_a, Ra, a_scale = place_points(a_xyz, 0.0, a_roty, args.object_size,
                                         acx, acz, args.table_y, args.a_size_mult)
    a_out = transform_gaussians(a, placed_a, a_scale, args.a_opacity_threshold)
    # rotate the Gaussian orientations (quaternions) about y to match the position rotation
    if a_roty != 0.0:
        th = np.radians(a_roty)
        qr = np.array([np.cos(th / 2), 0.0, np.sin(th / 2), 0.0], dtype=np.float32)
        q = np.stack([a_out["rot_0"], a_out["rot_1"], a_out["rot_2"], a_out["rot_3"]], axis=1).astype(np.float32)
        q = quat_mul(qr, q)
        a_out["rot_0"], a_out["rot_1"], a_out["rot_2"], a_out["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    records["A"] = {"count": int(a_out.shape[0]), "scale": float(a_scale),
                    "bbox_min": placed_a.min(0).tolist(), "bbox_max": placed_a.max(0).tolist()}

    # ---- B/C: textured meshes -> isotropic Gaussian splats ----
    mesh_out = []
    preview_parts = [(np.column_stack([bg["x"], bg["y"], bg["z"]]).astype(np.float32), dc_to_rgb(bg)),
                     (placed_a, dc_to_rgb(a_out))]
    solid = {"B": args.b_rgb, "C": args.c_rgb}
    for name, obj, tex, rx, ry, sm in [
        ("B", args.b_obj, args.b_texture, args.b_rot_x, args.b_rot_y, args.b_size_mult),
        ("C", args.c_obj, args.c_texture, args.c_rot_x, args.c_rot_y, args.c_size_mult),
    ]:
        pts, nrm = sample_surfels(obj, args.mesh_points, rng)
        base = solid[name]
        if min(base) >= 0:  # broken threestudio texture -> clean solid color + slight jitter
            jit = rng.normal(0.0, args.color_jitter, size=(pts.shape[0], 3)).astype(np.float32)
            rgb = np.clip(np.asarray(base, dtype=np.float32)[None, :] + jit, 0, 255).astype(np.uint8)
        else:
            _, rgb = sample_textured_obj(obj, tex, pts.shape[0], rng)
        ocx, ocz = obj_center(name)
        placed, R, sc = place_points(pts, rx, facing[name] + ry, args.object_size, ocx, ocz, args.table_y, sm)
        nrot = nrm @ R.T
        nrot = nrot / np.maximum(np.linalg.norm(nrot, axis=1, keepdims=True), 1e-9)
        quats = quats_from_normals(nrot)
        g = make_surfel_gaussians(placed, quats, rgb, bg.dtype,
                                  args.surfel_tangent, args.surfel_normal, args.mesh_opacity)
        mesh_out.append(g)
        preview_parts.append((placed, rgb))
        records[name] = {"count": int(placed.shape[0]), "scale": float(sc),
                         "bbox_min": placed.min(0).tolist(), "bbox_max": placed.max(0).tolist()}

    combined = np.concatenate([bg, a_out, *mesh_out])
    model_path = args.out_dir / "model"
    ply_path = model_path / "point_cloud/iteration_30000/point_cloud.ply"
    write_gaussian_ply(ply_path, combined, props)

    # copy cfg_args + cameras.json from the garden model so the renderer/viewer work
    cfg = args.bg_model_dir / "cfg_args"
    if cfg.exists():
        txt = cfg.read_text(errors="replace")
        txt = re.sub(r"model_path='[^']*'", f"model_path='{model_path.as_posix()}'", txt)
        (model_path / "cfg_args").write_text(txt)
    for rel in ("cameras.json", "input.ply", "exposure.json"):
        src = args.bg_model_dir / rel
        if src.exists():
            shutil.copy2(src, model_path / rel)

    # quick colored preview point cloud (no rasterizer needed)
    pv_xyz = np.concatenate([q[0] for q in preview_parts])
    pv_rgb = np.concatenate([q[1] for q in preview_parts])
    write_binary_ply(args.out_dir / "combined_preview_rgb.ply", pv_xyz, pv_rgb)

    meta = {
        "method": "garden y-up native 3DGS composition (A=trained Gaussians, B/C=mesh splats)",
        "model_dir": str(model_path), "point_cloud": str(ply_path),
        "center_xz": [cx, cz], "table_y": args.table_y, "object_size": args.object_size,
        "spacing": args.spacing, "layout_axis": args.layout_axis,
        "counts": {"background": int(bg.shape[0]), "A": int(a_out.shape[0]),
                   "B": int(mesh_out[0].shape[0]), "C": int(mesh_out[1].shape[0]),
                   "combined": int(combined.shape[0])},
        "objects": records,
    }
    (args.out_dir / "composition_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["counts"], indent=2))
    print(json.dumps(records, indent=2))
    print("model", model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
