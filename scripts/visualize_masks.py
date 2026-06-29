"""Visualize the SAM2 masks: tint each mask over its RGB frame.

Host-friendly (PIL + numpy only, no cv2). Reads <idx>_color.png + <idx>_mask.png
from a frames dir and writes one overlay per frame as <idx>_mask_viz.png.
Pass --montage to also write a single combined grid.

  python3 scripts/visualize_masks.py [--frames_dir DIR] [--montage]
"""
import os
import glob
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FRAMES = os.path.join(REPO, "results", "infer_res", "0001", "video_stream")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default=DEFAULT_FRAMES)
    ap.add_argument("--montage", action="store_true", help="also write a combined grid")
    ap.add_argument("--out", default=os.path.join(DEFAULT_FRAMES, "_mask_overlay.png"),
                    help="montage path (only with --montage)")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.45, help="mask tint opacity")
    return ap.parse_args()


# distinct tint colors per mask id (id 1, 2, 3, ...)
COLORS = [(255, 40, 40), (40, 160, 255), (60, 220, 60), (255, 210, 0), (220, 60, 255)]


def overlay(color_path, mask_path, alpha):
    rgb = np.asarray(Image.open(color_path).convert("RGB"), np.float32)
    H, W = rgb.shape[:2]
    out = rgb.copy()
    npx = 0
    n_ids = 0
    if os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert("L"))
        ids = [v for v in np.unique(m) if v > 0]  # distinct grey levels = object ids
        n_ids = len(ids)
        for k, v in enumerate(ids):
            sel = m == v
            npx += int(sel.sum())
            col = np.array(COLORS[k % len(COLORS)], np.float32)
            out[sel] = (1 - alpha) * out[sel] + alpha * col
    img = Image.fromarray(out.astype(np.uint8))
    # label: frame name + object pixel count
    d = ImageDraw.Draw(img)
    tag = f"{os.path.basename(color_path).replace('_color.png','')}  {n_ids}obj {npx}px"
    d.rectangle([0, 0, 8 + 6 * len(tag), 14], fill=(0, 0, 0))
    d.text((2, 2), tag, fill=(255, 255, 255))
    return img, npx


def main():
    args = parse_args()
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*_color.png")))
    assert frames, f"no *_color.png in {args.frames_dir}"

    tiles, total_hit = [], 0
    for fp in frames:
        img, npx = overlay(fp, fp.replace("color.png", "mask.png"), args.alpha)
        viz_path = fp.replace("_color.png", "_mask_viz.png")
        img.save(viz_path)
        tiles.append(img)
        total_hit += npx > 0
    print(f"{len(frames)} frames, {total_hit} with an object -> "
          f"<idx>_mask_viz.png in {args.frames_dir}")

    if args.montage:
        tw, th = tiles[0].size
        cols = args.cols
        rows = (len(tiles) + cols - 1) // cols
        grid = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * tw, (i // cols) * th))
        grid.save(args.out)
        print(f"montage -> {args.out} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
