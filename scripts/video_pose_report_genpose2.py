"""Turn per-frame GenPose++ predictions into a POSE VIDEO + a JITTER report.

Run `runners/infer_camera.py` with PER_FRAME=1 and every frame is estimated
independently -- no temporal model, no warm start. The frame-to-frame variation of
the estimate on a (near-)static object IS the method's pose jitter, which is what
this script measures. No ground truth needed.

Reads the poses.csv written by the offline runner (columns im_id, frame, obj_id,
R, t, size, score) and:
  1. overlays the predicted 3D bounding box + XYZ axes on each RGB frame, using the
     clip's intrinsics,
  2. stitches the overlays into pose.mp4 (frames ordered by im_id),
  3. computes consecutive-frame deltas -- geodesic rotation angle (deg) and
     translation change (mm) -- per object, writes jitter.csv + jitter.png, and
     prints summary statistics (a low-jitter method has small, flat deltas).

Two GenPose++ specifics vs the GigaPose original this is ported from:
- UNITS: GenPose++ works in METRES (translation and size), BOP/GigaPose in
  millimetres. t is scaled to mm for the report only, so dt is comparable.
- NO CAD: GenPose++ is category-level and predicts the object's SIZE, so the box is
  built from the predicted extents, not from a mesh. The box therefore reflects
  scale error too -- that is the method's own estimate, not a fixed model.

Runs on the host (no torch/cutoop needed):
    ~/miniconda3/envs/foundationpose/bin/python scripts/video_pose_report_genpose2.py --data_point 2
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import cv2
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INFER_RES = REPO / "results" / "infer_res"

M_TO_MM = 1000.0  # GenPose++ is metres; the report states translation jitter in mm.


def load_poses(csv_path):
    """-> {obj_id: {im_id: (R3x3, t3_mm, size3_mm, score, frame_stem)}}"""
    # frame is a zero-padded file stem ("0000"): keep it a string or pandas reads it as int 0
    # and the colour-frame lookup misses every file.
    df = pd.read_csv(csv_path, dtype={"frame": str})
    poses = {}
    for _, r in df.iterrows():
        R = np.array(list(map(float, str(r["R"]).split()))).reshape(3, 3)
        t = np.array(list(map(float, str(r["t"]).split())))
        size = np.array(list(map(float, str(r["size"]).split())))
        obj_id, im_id = int(r["obj_id"]), int(r["im_id"])
        frame = f'{r["frame"]}' if "frame" in df.columns else f"{im_id:04d}"
        per_obj = poses.setdefault(obj_id, {})
        # keep the best-scoring row if a frame ever has duplicates
        if im_id not in per_obj or float(r["score"]) > per_obj[im_id][3]:
            per_obj[im_id] = (R, t * M_TO_MM, size * M_TO_MM, float(r["score"]), frame)
    return poses


def cam_K_of(stream_dir):
    metas = sorted(glob.glob(str(stream_dir / "*_meta.json")))
    if not metas:
        raise SystemExit(f"No *_meta.json in {stream_dir}")
    i = json.loads(Path(metas[0]).read_text())["camera"]["intrinsics"]
    return np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], dtype=np.float64)


def box_corners(size_mm):
    """Predicted extents -> the 8 corners of the object-centred box."""
    hx, hy, hz = np.asarray(size_mm, dtype=np.float64) / 2.0
    return np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)],
                    dtype=np.float64)


BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def project(K, R, t, X):
    Xc = (R @ X.T).T + t
    uv = (K @ Xc.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv.astype(int)


def draw_pose(img, K, R, t, corners, axis_len, box_color=(0, 255, 0)):
    p = project(K, R, t, corners)
    for a, b in BOX_EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), box_color, 2)
    o = project(K, R, t, np.array([[0, 0, 0], [axis_len, 0, 0],
                                   [0, axis_len, 0], [0, 0, axis_len]], float))
    for end, col in zip(o[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # X r, Y g, Z b
        cv2.line(img, tuple(o[0]), tuple(end), col, 2)


def geodesic_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


OBJ_COLORS = [(0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 255, 255), (128, 255, 128)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_point", type=int, default=2, help="clip index under results/infer_res")
    ap.add_argument("--csv", help="explicit poses.csv (default: the clip's poses.csv)")
    ap.add_argument("--fps", type=float, default=15.0, help="output video fps")
    ap.add_argument("--out", help="output dir (default results/infer_res/<dp>/video_report)")
    args = ap.parse_args()

    res = INFER_RES / f"{args.data_point:04d}"
    stream_dir = res / "video_stream"
    csv_path = Path(args.csv) if args.csv else res / "poses.csv"
    if not csv_path.exists():
        raise SystemExit(f"No poses at {csv_path}. Run the offline pose step first "
                         f"(SAVE_POSES=1 PER_FRAME=1 DATA_POINT={args.data_point}).")
    print(f"CSV: {csv_path}")

    K = cam_K_of(stream_dir)
    out = Path(args.out) if args.out else res / "video_report"
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    poses = load_poses(csv_path)
    if not poses:
        raise SystemExit("No poses in CSV.")
    all_ids = sorted({im for per_obj in poses.values() for im in per_obj})

    # Per-frame mask ids are positional (np.unique over the mask), NOT stable identities:
    # a spurious detection on one frame renumbers the rest and shows up as huge jitter.
    n_obj = {len([1 for per_obj in poses.values() if im in per_obj]) for im in all_ids}
    if len(n_obj) > 1:
        print(f"WARNING: object count varies across frames {sorted(n_obj)} -- mask ids are "
              "positional, so jitter per obj_id may compare different objects. "
              "Re-segment with --max-obj 1 for a single-instance clip.")

    # --- overlays + video ---
    vw = None
    for im_id in all_ids:
        stem = next(per_obj[im_id][4] for per_obj in poses.values() if im_id in per_obj)
        img = cv2.imread(str(stream_dir / f"{stem}_color.png"))
        if img is None:
            print(f"(skipped frame {stem}: no colour image)")
            continue
        if vw is None:
            H, W = img.shape[:2]
            vw = cv2.VideoWriter(str(out / "pose.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        labels = []
        for k, (obj_id, per_obj) in enumerate(sorted(poses.items())):
            if im_id not in per_obj:
                continue
            R, t, size, score, _ = per_obj[im_id]
            draw_pose(img, K, R, t, box_corners(size), float(size.max()) * 0.5,
                      OBJ_COLORS[k % len(OBJ_COLORS)])
            labels.append(f"o{obj_id} s={score:.2f} Z={t[2]:.0f}mm")
        cv2.putText(img, f"f{im_id} " + " | ".join(labels), (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(frames_dir / f"{im_id:04d}.png"), img)
        vw.write(img)
    if vw is not None:
        vw.release()
        print(f"video: {out/'pose.mp4'} ({len(all_ids)} frames @ {args.fps} fps)")
    else:
        print(f"WARNING: no overlay written -- no colour frames matched the CSV in {stream_dir}")

    # --- jitter: consecutive-frame deltas, per object ---
    rows = []
    for obj_id, per_obj in sorted(poses.items()):
        ids = sorted(per_obj)
        for a, b in zip(ids[:-1], ids[1:]):
            Ra, ta, sa, _, _ = per_obj[a]
            Rb, tb, sb_, sc_b, _ = per_obj[b]
            rows.append({
                "obj_id": obj_id, "im_id": b, "gap": b - a,
                "dR_deg": geodesic_deg(Ra, Rb),
                "dt_mm": float(np.linalg.norm(tb - ta)),
                "dsize_mm": float(np.linalg.norm(sb_ - sa)),
                "score": sc_b,
            })
    if not rows:
        raise SystemExit("Only one frame has a pose -- nothing to compare.")
    jdf = pd.DataFrame(rows)
    jdf.to_csv(out / "jitter.csv", index=False)

    def stats(v):
        v = np.asarray(v)
        return f"mean={v.mean():.3f} median={np.median(v):.3f} std={v.std():.3f} max={v.max():.3f}"

    print("\n=== POSE JITTER (frame-to-frame) ===")
    for obj_id, g in jdf.groupby("obj_id"):
        gaps = set(g["gap"]) - {1}
        print(f"\n-- object {obj_id}: {len(g)+1} frames with pose, {len(g)} consecutive pairs"
              + (f"  (WARNING: gaps in frames with a pose: {sorted(gaps)})" if gaps else ""))
        print(f"rotation dR [deg]: {stats(g['dR_deg'])}")
        print(f"translation dt   : {stats(g['dt_mm'])} (mm)")
        print(f"size ds          : {stats(g['dsize_mm'])} (mm)")
        print(f"mean score       : {g['score'].mean():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for obj_id, g in jdf.groupby("obj_id"):
            ax[0].plot(g["im_id"], g["dR_deg"], "-o", ms=3, label=f"obj {obj_id}")
            ax[1].plot(g["im_id"], g["dt_mm"], "-o", ms=3, label=f"obj {obj_id}")
        ax[0].set_ylabel("dR (deg)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax[0].set_title(f"GenPose++ per-frame jitter — data_point {args.data_point:04d}")
        ax[1].set_ylabel("dt (mm)"); ax[1].set_xlabel("frame (im_id)"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "jitter.png", dpi=120)
        print(f"\nplot : {out/'jitter.png'}")
    except Exception as e:  # matplotlib optional
        print(f"(skipped plot: {e})")
    print(f"csv  : {out/'jitter.csv'}")


if __name__ == "__main__":
    main()
