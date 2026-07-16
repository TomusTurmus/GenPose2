"""Prepare a VIDEO as a GenPose++ offline `video_stream` clip.

A video is just a frame sequence, and GenPose++'s offline path estimates pose per
frame, so running "on video" needs no model changes -- only frame extraction plus
wrapping the frames as the format `runners/infer_camera.py` (USE_CAM=0) reads:

    results/infer_res/<data_point>/video_stream/
        <idx>_color.png   RGB png
        <idx>_depth.exr   float32 depth in METRES  (cutoop.load_depth)
        <idx>_mask.exr    float32 = mask_id / 255  (cutoop.load_mask -> *255)
        <idx>_meta.json   {"camera": {"intrinsics": {fx,fy,cx,cy,width,height}}}

Frames are re-indexed 0..N-1 in decode order, so `sorted(glob('*_color.png'))` --
what the runner enumerates -- matches capture order.

GenPose++ IS RGB-D: it builds the point cloud from depth (and drops depth > 4 m).
A plain mono video therefore CANNOT be used -- there is no depth to lift the mask
into 3D, and no amount of plumbing substitutes for it. Supported sources:

  --rgbd_dir DIR   BOP-style capture: rgb/*.png + depth/*.png (uint16 mm)
                   [+ masks/*.png] [+ cam_K.txt | camera.json]
                   (what src/scripts/capture_d555.py writes, and the layout of
                   the FoundationPose realsense_cup demo capture)
  --bag FILE       RealSense .bag recording -- a real video file carrying depth;
                   colour+depth are aligned and intrinsics read from the stream.
                   Needs pyrealsense2 (host env `realsense`).

GenPose++ needs NO per-object assets (no CAD, no templates, no per-object
weights): it is category-level and model-free, so unlike GigaPose there is
nothing to copy or symlink per clip -- prep is frames + intrinsics only.

Masks are the one real dependency (GenPose++ ships no segmenter; the paper uses
GT masks). Either pass --masks_dir, or let this script write empty masks and run
the Grounded-SAM2 command it prints. ORDER MATTERS: prep -> segment -> pose;
prep rewrites mask.exr, so re-running it after segmenting silently clobbers the
SAM2 masks.

Run on the host (source data lives here; the container only mounts the repo):
    OPENCV_IO_ENABLE_OPENEXR=1 \
    ~/miniconda3/envs/foundationpose/bin/python scripts/prepare_video_genpose2.py \
        --rgbd_dir /home/pose/dipl/FoundationPose/demo_data/realsense_cup \
        --data_point 2 --every 1
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # must precede `import cv2` to write/read .exr
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
INFER_RES = REPO / "results" / "infer_res"


def stream_rgbd_dir(rgbd_dir):
    """Yield (color_bgr, depth_raw) from a BOP-style rgb/ + depth/ capture dir."""
    rgbd_dir = Path(rgbd_dir)
    rgb_files = sorted(glob.glob(str(rgbd_dir / "rgb" / "*.png")))
    if not rgb_files:
        raise SystemExit(f"No rgb/*.png under {rgbd_dir}")
    for rgb_path in rgb_files:
        stem = Path(rgb_path).stem
        color = cv2.imread(rgb_path)
        if color is None:
            raise SystemExit(f"Unreadable colour frame: {rgb_path}")
        depth_path = rgbd_dir / "depth" / f"{stem}.png"
        if not depth_path.exists():
            raise SystemExit(f"Missing depth for frame {stem}: {depth_path}\n"
                             "GenPose++ is RGB-D; every frame needs depth.")
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise SystemExit(f"Unreadable depth frame: {depth_path}")
        yield color, depth, stem


def stream_bag(bag_path, width, height):
    """Yield (color_bgr, depth_uint16_mm) from a RealSense .bag, depth aligned to colour."""
    import pyrealsense2 as rs  # lazy: only the --bag path needs it

    pipe = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag_path), repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)  # decode every frame, don't drop
    align = rs.align(rs.stream.color)

    # depth_scale converts raw depth units -> metres; we re-emit as uint16 mm below.
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    idx = 0
    try:
        while True:
            ok, frames = pipe.try_wait_for_frames(5000)
            if not ok:
                break
            frames = align.process(frames)
            cframe, dframe = frames.get_color_frame(), frames.get_depth_frame()
            if not cframe or not dframe:
                continue
            color = np.asanyarray(cframe.get_data())
            depth_mm = (np.asanyarray(dframe.get_data()).astype(np.float32)
                        * depth_scale * 1000.0).astype(np.uint16)
            if idx == 0:
                intr = cframe.profile.as_video_stream_profile().get_intrinsics()
                stream_bag.intrinsics = {"fx": float(intr.fx), "fy": float(intr.fy),
                                         "cx": float(intr.ppx), "cy": float(intr.ppy),
                                         "width": int(intr.width), "height": int(intr.height)}
            yield color, depth_mm, f"{idx:06d}"
            idx += 1
    finally:
        pipe.stop()


def resolve_intrinsics(args, wh, src_dir):
    """Intrinsics, in priority order: explicit --fx/--fy > --cam_from > capture dir > bag stream."""
    W, H = wh
    if args.fx is not None and args.fy is not None:
        intr = {"fx": args.fx, "fy": args.fy,
                "cx": args.cx if args.cx is not None else W / 2.0,
                "cy": args.cy if args.cy is not None else H / 2.0,
                "width": W, "height": H}
        print(f"intrinsics: from --fx/--fy: {intr}")
        return intr

    if args.cam_from:
        intr = _intrinsics_from_cam_from(args.cam_from)
        print(f"intrinsics: reused from '{args.cam_from}': {intr}")
    elif getattr(stream_bag, "intrinsics", None) is not None:
        intr = dict(stream_bag.intrinsics)
        print(f"intrinsics: read from the .bag colour stream: {intr}")
    elif src_dir is not None and (intr := _intrinsics_from_capture_dir(src_dir, W, H)):
        print(f"intrinsics: read from the capture dir: {intr}")
    else:
        raise SystemExit("No intrinsics. Pass --fx/--fy [--cx --cy], or --cam_from "
                         "<data_point|path>, or use a capture dir with cam_K.txt/camera.json.")

    if (intr["width"], intr["height"]) != (W, H):
        raise SystemExit(
            f"Intrinsics are for {intr['width']}x{intr['height']} but frames are {W}x{H}. "
            "Intrinsics are resolution-specific -- re-capture, or pass --fx/--fy for this "
            "resolution (scale focal+principal point by the resize factor).")
    return intr


def _intrinsics_from_cam_from(cam_from):
    """--cam_from is either an existing data_point index, or a path to a meta/cam_K/camera file."""
    if str(cam_from).isdigit():
        metas = sorted(glob.glob(str(INFER_RES / f"{int(cam_from):04d}" / "video_stream" / "*_meta.json")))
        if not metas:
            raise SystemExit(f"No *_meta.json in data_point {int(cam_from):04d}'s video_stream.")
        return json.loads(Path(metas[0]).read_text())["camera"]["intrinsics"]

    p = Path(cam_from)
    if not p.exists():
        raise SystemExit(f"--cam_from not found: {p}")
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        if "camera" in d:                      # a prepared meta.json
            return d["camera"]["intrinsics"]
        return _intr_from_K(np.array(d["cam_K"], float).reshape(3, 3), d.get("width"), d.get("height"))
    return _intr_from_K(np.loadtxt(str(p)).reshape(3, 3), None, None)  # cam_K.txt


def _intr_from_K(K, W, H):
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": int(W) if W else None, "height": int(H) if H else None}


def _intrinsics_from_capture_dir(src_dir, W, H):
    src_dir = Path(src_dir)
    for name in ("camera.json", "cam_K.txt"):
        p = src_dir / name
        if p.exists():
            intr = _intrinsics_from_cam_from(str(p))
            intr["width"] = intr["width"] or W
            intr["height"] = intr["height"] or H
            return intr
    return None


def load_mask_id(masks_dir, stem, H, W):
    """Per-frame mask -> float32 id/255 image. Binary masks collapse to id 1."""
    for cand in (Path(masks_dir) / f"{stem}.png", Path(masks_dir) / f"{stem}.jpg"):
        if cand.exists():
            m = cv2.imread(str(cand), cv2.IMREAD_UNCHANGED)
            if m is None:
                return None
            if m.ndim == 3:
                m = m[:, :, 0]
            if m.shape[:2] != (H, W):
                raise SystemExit(f"Mask {cand} is {m.shape[:2]}, frames are {(H, W)}.")
            out = np.zeros((H, W), np.float32)
            out[m > 0] = 1.0 / 255.0  # binary -> single instance, id 1
            return out
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rgbd_dir", help="BOP-style capture dir: rgb/ + depth/ [+ masks/]")
    src.add_argument("--bag", help="RealSense .bag recording (needs pyrealsense2)")
    src.add_argument("--video", help="plain colour video -- NOT usable: GenPose++ needs depth")
    ap.add_argument("--data_point", type=int, default=2,
                    help="clip index -> results/infer_res/<data_point:04d>/ (default 2; 1 is the cup demo)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame (default 1 = all)")
    ap.add_argument("--max_frames", type=int, default=0, help="cap extracted frames (0 = no cap)")
    ap.add_argument("--fx", type=float, help="focal x (px) at the video resolution")
    ap.add_argument("--fy", type=float, help="focal y (px)")
    ap.add_argument("--cx", type=float, help="principal point x (default W/2)")
    ap.add_argument("--cy", type=float, help="principal point y (default H/2)")
    ap.add_argument("--cam_from", help="reuse intrinsics from an existing data_point index, "
                                       "or a cam_K.txt / camera.json / meta.json path")
    ap.add_argument("--masks_dir", help="per-frame masks named like the source rgb frames. "
                                        "If omitted, empty masks are written and the "
                                        "Grounded-SAM2 command is printed.")
    ap.add_argument("--depth_scale", type=float, default=1000.0,
                    help="divide raw depth by this to get METRES (default 1000 = uint16 mm)")
    ap.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty clip dir")
    args = ap.parse_args()

    if args.video:
        raise SystemExit(
            "--video (mono) cannot drive GenPose++: it is an RGB-D method -- it lifts the\n"
            "masked pixels into a point cloud using depth, so a colour-only clip has nothing\n"
            "to estimate a pose from. Use --bag (RealSense recording, carries depth) or\n"
            "--rgbd_dir (rgb/ + depth/ capture dir). To use a mono video you would need a\n"
            "different, RGB-only pose model.")

    dst = INFER_RES / f"{args.data_point:04d}" / "video_stream"
    if dst.exists() and any(dst.iterdir()) and not args.overwrite:
        raise SystemExit(f"{dst} is not empty -- refusing to clobber it (it may hold another\n"
                         "clip, or SAM2 masks). Pass --overwrite, or pick another --data_point.")
    dst.mkdir(parents=True, exist_ok=True)

    if args.bag:
        stream_bag.intrinsics = None
        frames = stream_bag(args.bag, None, None)
        src_dir = None
    else:
        frames = stream_rgbd_dir(args.rgbd_dir)
        src_dir = args.rgbd_dir

    # --- decode -> re-index 0..N-1 ---
    wh = None
    n_masked = 0
    kept = 0
    src_idx = 0
    stems = []
    for color, depth_raw, stem in frames:
        if src_idx % args.every != 0:
            src_idx += 1
            continue
        src_idx += 1
        H, W = color.shape[:2]
        if wh is None:
            wh = (W, H)
        elif wh != (W, H):
            raise SystemExit(f"Frame size changed mid-clip: {wh} -> {(W, H)}")
        if depth_raw.shape[:2] != (H, W):
            raise SystemExit(f"Depth {depth_raw.shape[:2]} != colour {(H, W)} on frame {stem}. "
                             "Depth must be aligned to colour.")

        prefix = dst / f"{kept:04d}_"
        cv2.imwrite(str(prefix) + "color.png", color)
        cv2.imwrite(str(prefix) + "depth.exr", (depth_raw.astype(np.float32) / args.depth_scale))

        mask_id = None
        if args.masks_dir:
            mask_id = load_mask_id(args.masks_dir, stem, H, W)
        if mask_id is None:
            mask_id = np.zeros((H, W), np.float32)
        else:
            n_masked += 1
        cv2.imwrite(str(prefix) + "mask.exr", mask_id)

        stems.append(stem)
        kept += 1
        if args.max_frames and kept >= args.max_frames:
            break

    if not kept:
        raise SystemExit("No frames extracted.")

    intr = resolve_intrinsics(args, wh, src_dir)
    for i in range(kept):
        (dst / f"{i:04d}_meta.json").write_text(
            json.dumps({"camera": {"intrinsics": intr}}, indent=2))

    # Frame index -> source frame, so results can be traced back after re-indexing.
    (dst / "frame_index.json").write_text(json.dumps(
        {"source": str(args.bag or args.rgbd_dir), "every": args.every,
         "frames": {f"{i:04d}": s for i, s in enumerate(stems)}}, indent=2))

    print(f"frames : kept {kept} (every {args.every}th of {src_idx} read) at {wh[0]}x{wh[1]} -> {dst}")
    print(f"depth  : raw / {args.depth_scale} -> metres (GenPose++ drops depth > 4 m)")
    print(f"masks  : {n_masked}/{kept} frames had a mask"
          f"{'' if args.masks_dir else ' (none requested -> all empty)'}")

    print("\n" + "=" * 78)
    if n_masked < kept:
        print("NEXT: masks. GenPose++ ships no segmenter; frames without one yield no pose.")
        print("Run Grounded-SAM2 over this clip (in the container -- see VIDEO_PIPELINE_GENPOSE2.md):")
        print(f"  python scripts/segment_sam2.py --frames_dir {dst} \\\n"
              f"      --text '<object>' --zoom 2 --max-obj 1")
        print(f"  python3 scripts/visualize_masks.py --frames_dir {dst}   # eyeball them first")
    print("\nTHEN: per-frame pose + jitter report:")
    print(f"  bash run_video_pipeline_genpose2.sh {args.data_point}")
    print("=" * 78)


if __name__ == "__main__":
    main()
