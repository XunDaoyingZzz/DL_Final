#!/usr/bin/env python3
"""Render a 3DGS model from its real training cameras (cameras.json).

This is the anti-smearing fusion renderer: instead of a synthetic orbit that can leave
the trained view distribution (old2's failure -> white smearing), we fly through the
ACTUAL training camera poses, sorted by azimuth around the scene's up axis, so every
frame is an in-distribution viewpoint. Handles any up axis automatically.
"""
from __future__ import annotations
import argparse
import json
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


def find_ply(model_dir: Path) -> Path:
    cands = sorted(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"),
                   key=lambda p: int(p.parent.name.split("_")[-1]))
    if not cands:
        raise SystemExit(f"No point_cloud.ply under {model_dir}")
    return cands[-1]


def build_cam(C, R_c2w, W, H, fx, fy):
    from utils.graphics_utils import getProjectionMatrix, focal2fov
    R_w2c = R_c2w.T.astype(np.float32)
    t = (-R_w2c @ C).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t
    FoVx = focal2fov(fx, W)
    FoVy = focal2fov(fy, H)
    world_view = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
    proj = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).transpose(0, 1).cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return SimpleNamespace(
        image_width=W, image_height=H, znear=0.01, zfar=100.0,
        world_view_transform=world_view, full_proj_transform=full,
        FoVx=FoVx, FoVy=FoVy,
        camera_center=torch.tensor(C, dtype=torch.float32, device="cuda"),
    )


def look_at_R(eye, target, world_up):
    """Camera-to-world rotation (COLMAP convention: cam +x right, +y down, +z forward)."""
    f = target - eye
    f = f / max(np.linalg.norm(f), 1e-9)
    down0 = -world_up
    r = np.cross(down0, f)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(np.array([0.0, 0.0, -1.0], np.float32), f)
    r = r / max(np.linalg.norm(r), 1e-9)
    d = np.cross(f, r)
    d = d / max(np.linalg.norm(d), 1e-9)
    return np.stack([r, d, f], axis=1).astype(np.float32)  # columns = [right, down, forward]


def mat_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q, np.float64)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = math.acos(d)
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(th0 * t) + q2 * math.sin(th0 * t)


def estimate_orbit(cams):
    """From the training cameras estimate the look-at center (least-squares ray
    intersection), the true scene up (mean camera up), and the orbit radius/height."""
    P = np.array([c["position"] for c in cams], np.float64)
    Rs = [np.array(c["rotation"], np.float64) for c in cams]
    fwd = np.array([r[:, 2] for r in Rs])     # camera forward (+z) in world
    up = -np.array([r[:, 1] for r in Rs])     # camera up = -(+y_cam) in world
    true_up = up.sum(0)
    true_up = true_up / np.linalg.norm(true_up)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, f in zip(P, fwd):
        M = np.eye(3) - np.outer(f, f)
        A += M
        b += M @ p
    center = np.linalg.solve(A, b)
    rel = P - center
    h = rel @ true_up
    planar = rel - np.outer(h, true_up)
    radius = float(np.median(np.linalg.norm(planar, axis=1)))
    height = float(np.median(h))
    a0 = np.array([1.0, 0.0, 0.0]) if abs(true_up @ np.array([1.0, 0, 0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a0 - (a0 @ true_up) * true_up
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(true_up, e1)
    return center, true_up, radius, height, e1, e2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="dir with point_cloud/iteration_*/")
    p.add_argument("--cameras", type=Path, required=True, help="3DGS cameras.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gs-root", type=Path, default=Path("/home/xundaoying/hw3_repos/gaussian-splatting"))
    p.add_argument("--up", choices=list(AXIS), default="y")
    p.add_argument("--out-width", type=int, default=960)
    p.add_argument("--stride", type=int, default=1, help="use every Nth sorted camera")
    p.add_argument("--az-center", type=float, default=None, help="deg; keep cameras near this azimuth")
    p.add_argument("--az-half", type=float, default=None, help="deg; half-width of the kept azimuth arc")
    p.add_argument("--pitch-deg", type=float, default=0.0,
                   help="tilt each camera down(+)/up(-) about its right axis to frame table objects")
    p.add_argument("--interp", type=int, default=1,
                   help="substeps per camera segment for a smooth path (needs --look-at)")
    p.add_argument("--look-at", type=float, nargs=3, default=None,
                   help="fixed target; recompute each camera's rotation to aim here (smooth mode)")
    p.add_argument("--h264", action="store_true", help="transcode output to H.264 yuv420p")
    p.add_argument("--ffmpeg", default="/home/xundaoying/miniconda3/envs/llm-26-gpu/bin/ffmpeg")
    # smooth analytic orbit: a perfectly uniform circle fitted to the training cameras'
    # true up / center / radius / height (level horizon, all angles, in-distribution).
    p.add_argument("--smooth-orbit", action="store_true")
    p.add_argument("--orbit-frames", type=int, default=240)
    p.add_argument("--orbit-arc", type=float, default=320.0)
    p.add_argument("--orbit-start", type=float, default=0.0)
    p.add_argument("--orbit-radius-scale", type=float, default=1.0)
    p.add_argument("--orbit-height-scale", type=float, default=1.0)
    p.add_argument("--aim-drop", type=float, default=0.25, help="lower the look-at toward the table")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--bg", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    p.add_argument("--video-name", default="scene_from_cameras.mp4")
    p.add_argument("--contact-n", type=int, default=8)
    args = p.parse_args()

    add_gs_path(args.gs_root)
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    ply = find_ply(args.model)
    g = GaussianModel(args.sh_degree)
    g.load_ply(str(ply))

    cams = json.loads(args.cameras.read_text())
    pos = np.array([c["position"] for c in cams], dtype=np.float64)
    center = np.median(pos, axis=0)
    up_i = AXIS[args.up]
    pa, pb = [i for i in range(3) if i != up_i]
    az = np.arctan2(pos[:, pb] - center[pb], pos[:, pa] - center[pa])
    if args.az_center is not None and args.az_half is not None:
        diff = (np.degrees(az) - args.az_center + 180.0) % 360.0 - 180.0
        sel = np.where(np.abs(diff) <= args.az_half)[0]
        order = sel[np.argsort(az[sel])][::args.stride]
    else:
        order = np.argsort(az)[::args.stride]

    pipe = SimpleNamespace(debug=False, antialiasing=False,
                           compute_cov3D_python=False, convert_SHs_python=False)
    bg = torch.tensor(args.bg, dtype=torch.float32, device="cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = [{"C": np.array(cams[int(ci)]["position"], np.float64),
             "R": np.array(cams[int(ci)]["rotation"], np.float64),
             "fx": float(cams[int(ci)]["fx"]), "fy": float(cams[int(ci)]["fy"]),
             "W0": int(cams[int(ci)]["width"]), "H0": int(cams[int(ci)]["height"]),
             "az": math.degrees(float(az[ci]))} for ci in order]

    render_list = []
    if args.smooth_orbit:
        center, up, radius, height, e1, e2 = estimate_orbit(cams)
        aim = center - args.aim_drop * up
        R0 = radius * args.orbit_radius_scale
        H = height * args.orbit_height_scale
        c0 = cams[0]
        print(f"smooth_orbit: center={center.round(2).tolist()} up={up.round(3).tolist()} "
              f"radius={radius:.2f} height={height:.2f}")
        for i in range(args.orbit_frames):
            azd = args.orbit_start + args.orbit_arc * i / max(args.orbit_frames - 1, 1)
            a = math.radians(azd)
            eye = center + R0 * (math.cos(a) * e1 + math.sin(a) * e2) + H * up
            render_list.append({"C": eye, "R": look_at_R(eye, aim, up),
                                "fx": float(c0["fx"]), "fy": float(c0["fy"]),
                                "W0": int(c0["width"]), "H0": int(c0["height"]), "az": azd})
    # smooth mode: lerp position + SLERP the real camera rotations (keeps the true,
    # slightly-tilted scene orientation -> no canted horizon, stays in-distribution).
    elif args.interp > 1 and len(base) >= 2:
        quats = [mat_to_quat(b["R"]) for b in base]
        for k in range(len(base) - 1):
            a, b = base[k], base[k + 1]
            for s in range(args.interp):
                t = s / args.interp
                render_list.append({"C": (1 - t) * a["C"] + t * b["C"],
                                    "R": quat_to_mat(slerp(quats[k], quats[k + 1], t)),
                                    "fx": (1 - t) * a["fx"] + t * b["fx"],
                                    "fy": (1 - t) * a["fy"] + t * b["fy"],
                                    "W0": a["W0"], "H0": a["H0"], "az": a["az"]})
        render_list.append(base[-1])
    else:
        render_list = base

    frames = []
    frame_az = []
    with torch.no_grad():
        for rc in render_list:
            frame_az.append(rc["az"])
            W0, H0 = rc["W0"], rc["H0"]
            ow = args.out_width
            oh = int(round(ow * H0 / W0))
            C = rc["C"].astype(np.float32)
            R_c2w = np.array(rc["R"], dtype=np.float32)
            if args.pitch_deg != 0.0:
                t = math.radians(args.pitch_deg)
                ct, st = math.cos(t), math.sin(t)
                Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], dtype=np.float32)
                R_c2w = R_c2w @ Rx
            cam = build_cam(C, R_c2w, ow, oh, rc["fx"], rc["fy"])
            out = render(cam, g, pipe, bg, separate_sh=False)["render"]
            img = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))

    if not frames:
        raise SystemExit("no frames rendered")
    h, w = frames[0].shape[:2]
    vp = args.out_dir / args.video_name
    vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    if args.h264:
        import subprocess
        hp = vp.with_name(vp.stem + "_h264.mp4")
        try:
            subprocess.run([args.ffmpeg, "-y", "-i", str(vp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                            str(hp)], check=True)
            print("h264", hp)
        except Exception as e:  # transcode is optional; never lose the render or contact sheet
            print(f"WARN h264 transcode failed ({e}); mp4v kept at {vp}")

    # contact sheet
    n = min(args.contact_n, len(frames))
    idxs = [int(round(i * (len(frames) - 1) / max(n - 1, 1))) for i in range(n)]
    cols = 4 if n % 4 == 0 else n
    rows = math.ceil(n / cols)
    tw = 320
    th = int(round(h * tw / w))
    sheet = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for j, fi in enumerate(idxs):
        t = cv2.resize(frames[fi], (tw, th)).copy()
        cv2.putText(t, f"az{frame_az[fi]:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        r, cc = divmod(j, cols)
        sheet[r * th:(r + 1) * th, cc * tw:(cc + 1) * tw] = t
    sp = args.out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sp), sheet)
    print("ply", ply)
    print("rendered", len(frames), "frames ->", vp)
    print("contact_sheet", sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
