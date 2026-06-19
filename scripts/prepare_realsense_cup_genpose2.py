"""Prepare the custom `realsense_cup` capture for GenPose++ offline inference.

Source: the FoundationPose demo capture (RGB-D + binary masks, frame-indexed).
Target: results/infer_res/0001/video_stream/ in the format `runners/infer_camera.py`
        (USE_CAM=False) expects, i.e. per frame:
            <idx>_color.png   RGB png
            <idx>_depth.exr   float32 depth in METERS   (cutoop.load_depth)
            <idx>_mask.exr    float32 = mask_id / 255    (cutoop.load_mask -> *255)
            <idx>_meta.json   {"camera": {"intrinsics": {...}}}

Units/format facts (verified against cutoop 0.1.0 data_loader.py):
- GenPose2 builds the point cloud from depth and clips depth > 4.0 m, so depth
  must be in metres. FoundationPose depth is uint16 millimetres -> divide by 1000.
- load_mask does `cv2.imread(...).astype(float) * 255`, so to recover integer
  mask id 1 the EXR must store 1/255. Single object -> id 1.
- meta.json hits the dict branch in InferDataset (no cutoop.load_meta), needs
  camera.intrinsics {fx, fy, cx, cy, width, height}.

Run on the host (data lives here; container does not mount it):
    OPENCV_IO_ENABLE_OPENEXR=1 \
    ~/miniconda3/envs/foundationpose/bin/python scripts/prepare_realsense_cup_genpose2.py
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # before cv2 import
import json
import glob
from pathlib import Path

import cv2
import numpy as np

SRC = Path("/home/pose/dipl/FoundationPose/demo_data/realsense_cup")
DST = Path(__file__).resolve().parent.parent / "results" / "infer_res" / "0001" / "video_stream"

# Intrinsics from the capture's cam_K.txt / camera.json (640x360 RealSense D415).
INTRINSICS = {"fx": 456.5, "fy": 456.5, "cx": 320.0, "cy": 180.0, "width": 640, "height": 360}
DEPTH_DIV = 1000.0  # uint16 mm -> float m


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rgb_files = sorted(SRC.glob("rgb/*.png"))
    assert rgb_files, f"no rgb under {SRC}"

    n_with_mask = 0
    for rgb_path in rgb_files:
        stem = rgb_path.stem  # e.g. 000001
        idx = int(stem)
        prefix = DST / f"{idx:04d}_"

        # color
        color = cv2.imread(str(rgb_path))
        assert color is not None, rgb_path
        cv2.imwrite(str(prefix) + "color.png", color)

        # depth: uint16 mm -> float32 m
        depth_raw = cv2.imread(str(SRC / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        assert depth_raw is not None, f"missing depth for {stem}"
        depth_m = depth_raw.astype(np.float32) / DEPTH_DIV
        cv2.imwrite(str(prefix) + "depth.exr", depth_m)

        # mask: binary -> id 1, stored as 1/255 so load_mask*255 recovers 1
        mask_path = SRC / "masks" / f"{stem}.png"
        mask_id = np.zeros((INTRINSICS["height"], INTRINSICS["width"]), np.float32)
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if m.ndim == 3:
                m = m[:, :, 0]
            mask_id[m > 0] = 1.0 / 255.0
            n_with_mask += 1
        cv2.imwrite(str(prefix) + "mask.exr", mask_id)

        # meta
        with open(str(prefix) + "meta.json", "w") as f:
            json.dump({"camera": {"intrinsics": INTRINSICS}}, f, indent=2)

    print(f"wrote {len(rgb_files)} frames ({n_with_mask} with masks) to {DST}")


if __name__ == "__main__":
    main()
