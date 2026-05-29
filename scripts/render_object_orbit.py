#!/usr/bin/env python3
"""Render an object-only orbit + contact sheet from a single 3DGS PLY.

For verifying a standalone Gaussian object (e.g. cropped Object A) BEFORE fusion.
Takes a PLY directly, auto-computes a robust center/radius (percentile bounds to
ignore stray floaters), orbits the native 3DGS rasterizer around a chosen up axis,
and writes a multi-view contact sheet (and optional video).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

AXIS = {"x": 0, "y": 1, "z": 2}


def add_gs_path(gs_root: Path) -> None:
    sys.path.insert(0, str(gs_root))
    os.chdir(gs_root)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-8 else v / n


def look_at(eye, target, width, height, fovy_deg, world_up):
    from utils.graphics_utils import getProjectionMatrix

    forward = normalize(target - eye)
    right = normalize(np.cross(forward, world_up))
    if np.linalg.norm(right) < 1e-5:
        world_up = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = np.stack([right, up, forward], axis=0)
    w2c[:3, 3] = -(w2c[:3, :3] @ eye.astype(np.float32))
    fovy = math.radians(fovy_deg)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * width / height)
    world_view = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam_center = torch.inverse(world_view)[3, :3]
    return SimpleNamespace(
        image_width=width, image_height=height, znear=0.01, zfar=100.0,
        world_view_transform=world_view, full_proj_transform=full_proj,
        FoVx=fovx, FoVy=fovy, camera_center=cam_center,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gs-root", type=Path, default=Path("/home/xundaoying/hw3_repos/gaussian-splatting"))
    p.add_argument("--up", choices=list(AXIS), default="y", help="object vertical axis")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--views", type=int, default=6, help="contact-sheet azimuth views")
    p.add_argument("--elev-deg", type=float, default=12.0)
    p.add_argument("--fov-y", type=float, default=40.0)
    p.add_argument("--radius-mult", type=float, default=2.1)
    p.add_argument("--center", type=float, nargs=3, default=None, help="override orbit target")
    p.add_argument("--radius-override", type=float, default=0.0, help=">0 overrides orbit radius")
    p.add_argument("--arc-deg", type=float, default=360.0)
    p.add_argument("--start-deg", type=float, default=0.0)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--bg", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    add_gs_path(args.gs_root)
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    g = GaussianModel(args.sh_degree)
    g.load_ply(str(args.ply))
    xyz = g.get_xyz.detach().cpu().numpy()

    lo = np.percentile(xyz, 2, axis=0)
    hi = np.percentile(xyz, 98, axis=0)
    center = (lo + hi) * 0.5
    span = hi - lo
    radius = float(np.max(span) * args.radius_mult)
    if args.center is not None:
        center = np.asarray(args.center, dtype=np.float32)
    if args.radius_override > 0:
        radius = float(args.radius_override)
    arc = math.radians(args.arc_deg)
    start = math.radians(args.start_deg)

    def az_of(k: int, n: int) -> float:
        if args.arc_deg >= 359.9:
            return 2.0 * math.pi * k / n
        return start + arc * k / max(n - 1, 1)

    up_i = AXIS[args.up]
    plane = [i for i in range(3) if i != up_i]
    world_up = np.zeros(3, dtype=np.float32)
    world_up[up_i] = 1.0
    elev = math.radians(args.elev_deg)

    pipe = SimpleNamespace(debug=False, antialiasing=False,
                           compute_cov3D_python=False, convert_SHs_python=False)
    bg = torch.tensor(args.bg, dtype=torch.float32, device="cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def eye_at(az: float) -> np.ndarray:
        e = center.copy().astype(np.float32)
        horiz = radius * math.cos(elev)
        e[plane[0]] += horiz * math.sin(az)
        e[plane[1]] += -horiz * math.cos(az)
        e[up_i] += radius * math.sin(elev)
        return e

    def render_az(az: float) -> np.ndarray:
        cam = look_at(eye_at(az), center.astype(np.float32),
                      args.width, args.height, args.fov_y, world_up)
        with torch.no_grad():
            out = render(cam, g, pipe, bg, separate_sh=False)["render"]
        arr = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    # contact sheet
    tiles = []
    for k in range(args.views):
        az = az_of(k, args.views)
        img = render_az(az)
        cv2.putText(img, f"{int(round(math.degrees(az)))} deg", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cols = 3 if args.views % 3 == 0 else args.views
    rows = math.ceil(args.views / cols)
    h, w = tiles[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for idx, t in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    sheet_path = args.out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    print("center", center.tolist(), "span", span.tolist(), "radius", radius)
    print("contact_sheet", sheet_path)

    if args.video:
        vp = args.out_dir / "object_orbit.mp4"
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
        for i in range(args.video_frames):
            az = az_of(i, args.video_frames)
            vw.write(cv2.cvtColor(render_az(az), cv2.COLOR_RGB2BGR))
        vw.release()
        print("video", vp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
