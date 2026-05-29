#!/usr/bin/env python3
"""Segment Object C with a local SAM3 checkpoint.

This script is intentionally offline: pass a local SAM3 repo and checkpoint.
It writes a transparent PNG and quick previews for Zero123 input selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def default_sam3_root() -> Path:
    candidates = [
        Path("/mnt/d/interndir/SAM/sam3"),
        Path(r"D:\interndir\SAM\sam3"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_checkpoint() -> Path:
    root = default_sam3_root()
    return root / "checkpoints" / "sam3.pt"


def clean_mask(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return mask.astype(np.uint8) * 255

    alpha = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha = np.where(labels == largest, 255, 0).astype(np.uint8)

    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2)
    return alpha


def choose_mask(masks, boxes, scores, width: int, height: int) -> tuple[int, dict]:
    masks_np = masks.detach().cpu().numpy().astype(bool)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    boxes_np = boxes.detach().float().cpu().numpy()
    scores_np = scores.detach().float().cpu().numpy()

    image_area = float(width * height)
    ranked: list[tuple[float, int, dict]] = []
    cx0, cy0 = width / 2.0, height / 2.0
    diag = (width * width + height * height) ** 0.5

    for idx, mask in enumerate(masks_np):
        area = float(mask.sum())
        area_ratio = area / image_area
        x0, y0, x1, y1 = boxes_np[idx].tolist()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        center_penalty = (((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5) / diag
        shape_bonus = min(area_ratio, 0.45)
        valid_penalty = 0.0 if 0.08 <= area_ratio <= 0.85 else 0.35
        rank = float(scores_np[idx]) + shape_bonus - 0.7 * center_penalty - valid_penalty
        ranked.append(
            (
                rank,
                idx,
                {
                    "score": float(scores_np[idx]),
                    "area": int(area),
                    "area_ratio": area_ratio,
                    "box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                    "rank": rank,
                },
            )
        )

    if not ranked:
        raise RuntimeError("SAM3 returned no masks for this prompt/threshold.")
    ranked.sort(reverse=True, key=lambda item: item[0])
    return ranked[0][1], ranked[0][2]


def save_outputs(
    image: Image.Image,
    alpha: np.ndarray,
    box_xyxy: list[float],
    out_rgba: Path,
    out_preview: Path,
    out_overlay: Path,
) -> None:
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    out_rgba.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_rgba)

    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    white.convert("RGB").save(out_preview)

    overlay = image.convert("RGBA")
    red = Image.new("RGBA", rgba.size, (255, 40, 40, 0))
    red.putalpha(Image.fromarray((alpha * 0.35).astype(np.uint8), mode="L"))
    overlay.alpha_composite(red)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(box_xyxy, outline=(0, 255, 80, 255), width=3)
    overlay.convert("RGB").save(out_overlay)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-root", type=Path, default=default_sam3_root())
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "C_single_image" / "c_zero123_input_512.png",
    )
    parser.add_argument(
        "--out-rgba",
        type=Path,
        default=ROOT / "data" / "C_single_image" / "c_zero123_sam3_rgba_512.png",
    )
    parser.add_argument(
        "--out-preview",
        type=Path,
        default=ROOT
        / "data"
        / "C_single_image"
        / "c_zero123_sam3_preview_white_512.png",
    )
    parser.add_argument(
        "--out-overlay",
        type=Path,
        default=ROOT / "data" / "C_single_image" / "c_zero123_sam3_overlay_512.jpg",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=ROOT / "data" / "C_single_image" / "c_zero123_sam3_metadata.json",
    )
    parser.add_argument(
        "--prompt",
        default="anime doll head keychain",
        help="Text prompt for SAM3 concept segmentation.",
    )
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable CUDA BF16 autocast. SAM3's fused MLP path usually expects it.",
    )
    parser.add_argument("--dry-import", action="store_true")
    args = parser.parse_args()

    if not args.sam3_root.exists():
        raise SystemExit(f"Missing SAM3 root: {args.sam3_root}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing SAM3 checkpoint: {args.checkpoint}")
    if not args.input.exists():
        raise SystemExit(f"Missing input image: {args.input}")

    sys.path.insert(0, str(args.sam3_root))
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"sam3_root={args.sam3_root}")
    print(f"checkpoint={args.checkpoint}")
    if args.dry_import:
        print("dry import ok")
        return 0

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable; falling back to CPU.")

    image = Image.open(args.input).convert("RGB")
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=device,
        compile=False,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.threshold,
    )
    autocast_enabled = device == "cuda" and not args.no_autocast
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]
    best_idx, best_meta = choose_mask(masks, boxes, scores, *image.size)
    mask = masks.detach().cpu().numpy().astype(bool)
    if mask.ndim == 4:
        mask = mask[:, 0]
    alpha = clean_mask(mask[best_idx])
    save_outputs(
        image,
        alpha,
        best_meta["box_xyxy"],
        args.out_rgba,
        args.out_preview,
        args.out_overlay,
    )

    meta = {
        "prompt": args.prompt,
        "threshold": args.threshold,
        "device": device,
        "autocast_bf16": autocast_enabled,
        "sam3_root": str(args.sam3_root),
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "out_rgba": str(args.out_rgba),
        "out_preview": str(args.out_preview),
        "out_overlay": str(args.out_overlay),
        "best": best_meta,
        "num_masks": int(masks.shape[0]),
    }
    args.out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
