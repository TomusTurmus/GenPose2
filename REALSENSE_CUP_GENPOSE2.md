# Running GenPose++ on custom RealSense data (`realsense_cup`)

Offline 6D pose + size estimation on a custom RGB-D capture, reusing the
FoundationPose `realsense_cup` demo data. GenPose++ is **category-level +
RGB-D**: it needs depth + an instance mask, but **no CAD model / templates** and
**no `obj_meta.json`** for the inference path. Companion to GigaPose's
`PIPELINE_CUES.md`; the meta-lessons there apply, the concrete steps differ.

## Source data
`/home/pose/dipl/FoundationPose/demo_data/realsense_cup/` — 30 frames:
- `rgb/000###.png` (640×360), `depth/000###.png` (uint16 **mm**, max 3000),
  `masks/000###.png` (binary, 25 frames have a mask), `cam_K.txt`/`camera.json`
  (fx=fy=456.5, cx=320, cy=180).
- Quality note: cup is **far/tiny** early (frame 1 ≈2.1 m, ~213 mask px) and
  **close** later (frames ~20–30 ≈0.3–1.3 m). Expect weak poses on far frames
  (GigaPose cues §10 reality check applies).

## Format GenPose++ expects (offline `video_stream`)
`results/infer_res/0001/video_stream/<idx>_{color.png,depth.exr,mask.exr,meta.json}`
- `depth.exr`: float32 **metres** (loader clips >4 m). uint16 mm → ÷1000.
- `mask.exr`: float32 = **mask_id / 255** — `cutoop.load_mask` does `imread()*255`,
  so store 1/255 to recover integer id 1. Single object → id 1.
- `meta.json`: `{"camera":{"intrinsics":{fx,fy,cx,cy,width,height}}}` (the dict
  branch in `InferDataset`; cutoop's `load_meta` is not used here).

## Segmentation: where the masks come from

GenPose++ needs an instance mask but the **paper ships no segmenter** — it
benchmarks with *ground-truth* masks ("we assume ground truth instance
segmentation is known", §5.1) precisely to decouple pose accuracy from any
detector. The **repo's** only real segmentation is the live-camera demo
(`camera/camera.py`): SAM2 prompted by a human **click** on frame 0, then
`predictor.track()` on later frames.

To run on custom data **fully automatically** (no clicks, no GT/borrowed masks)
and still catch small/far objects, `scripts/segment_sam2.py` implements
**Grounded-SAM2**: GroundingDINO (text label → box) → SAM2 (box → pixel mask).

- **Default = per-frame**: detect + segment *every frame independently*. The
  offline `video_stream` path scores each frame's pose independently and uses
  **no temporal info**, so SAM2 video *tracking* buys nothing — and is actively
  harmful here because the camera **pans across the room** (cup a far speck on
  frame 1, two large mugs by frame 25). Prompt-once-then-track seeds on the tiny
  far frame and *loses the object exactly when it gets easy* (verified: masks on
  frames 1–10 only, empty 11–30). Per-frame detection trivially catches the big
  mugs. `--track` keeps the old prompt-once paradigm for parity.
- GroundingDINO via HuggingFace `transformers` (`IDEA-Research/grounding-dino-tiny`,
  **pinned 4.44.2** — newer needs torch≥2.2; image has 2.1) — no custom CUDA
  ops. SAM2 image predictor uses the same `sam2.1_hiera_tiny.pt`.
- Output per frame: `<idx>_mask.exr` (float32 = id/255) + `<idx>_mask.png`.
  Per-frame mode masks **every** detection above threshold → one mask id each.
- Small/far objects: `--zoom N` upscales before detection (boxes mapped back);
  lower `--box-thresh` for recall; `--max-obj K` caps detections per frame.
  The SAM2 `_C` postproc warning is benign (optional, doesn't affect masks).

### Tuning false positives (important before pose)

Per-frame mode masks **every** detection above threshold, and the pose runner
discovers objects with `np.unique(mask)` (`datasets_infer_camera.py:371`) — so it
estimates **one pose per mask id**, *including false positives*. Verified on the
cup capture: `--text 'cup' --zoom 2` finds both foreground mugs on close frames
(good) but also labels a background chair "cup" at low confidence → 4 ids on
frame 25 → 2 spurious poses downstream. Tune before running pose:

- `--max-obj 1` — keep only the highest-scoring detection per frame (use when you
  expect a single instance). `--max-obj 2` for the two-mug capture.
- `--box-thresh 0.35` — raise the confidence bar to drop weak false positives.
- `--text 'white mug'` — a tighter label than the generic category.

Inspect results with the host-only montage (no docker needed):
```bash
python3 scripts/visualize_masks.py            # -> video_stream/_mask_overlay.png
```
It tints each mask id over its RGB frame and labels `<frame> <#obj> <#px>`, so
extra-colour blobs = false positives at a glance.

This **replaces** the borrowed FoundationPose/SAM-6D `mask.exr` written by step 1
below — run the prep first (it also writes depth/meta), then overwrite the masks
with SAM2 via the segmentation command in step 2b.

> **Ordering trap (verified bite):** the strict order is **prep (1) → segment
> (2b) → pose (3)**. The prep script rewrites `mask.exr` with the *borrowed*
> mask; if you run it *after* segmentation it silently clobbers the SAM2
> `mask.exr` and pose then uses the wrong masks. Tell: prep writes `mask.exr`
> but **not** `mask.png`, so a clobber leaves `mask.exr` newer than `mask.png`
> (`ls -l --time-style=+%H:%M *_mask.*`). After step 2b they must share a
> timestamp. The `_mask_overlay.png` is built from `mask.png`, so it can look
> correct while the `.exr` pose actually reads is stale.

### After segmentation: just rerun pose, no path changes

Segmentation overwrites `<idx>_mask.exr` **in place** in the same
`results/infer_res/0001/video_stream/` dir the pose runner reads
(`alternetive_init` → `prefix + 'mask.exr'`). So **step 3 below is unchanged** —
no paths to correct. The only behavioural difference is multi-instance: if SAM2
wrote >1 id, pose runs per id (see false-positive note above).

## Steps
1. **Stage data (host)** — data lives on the host, the container only mounts the repo:
   ```bash
   OPENCV_IO_ENABLE_OPENEXR=1 \
   ~/miniconda3/envs/foundationpose/bin/python scripts/prepare_realsense_cup_genpose2.py
   ```
   Writes 30 frames (25 with masks) to `results/infer_res/0001/video_stream/`.

2. **Build the image (host, sudo — user not in `docker` group)**:
   ```bash
   cd /home/pose/dipl/GenPose2
   sudo docker build -t genpose2:latest -f docker/Dockerfile .
   ```
   (Rebuild needed after this session: `requirements_camera.txt` now adds
   `transformers`+`timm` for GroundingDINO.)

2b. **Segment with Grounded-SAM2** — overwrites the staged masks with real SAM2 masks:
   ```bash
   sudo docker run --gpus all --rm --ipc=host \
     -v /home/pose/dipl/GenPose2:/workspace/GenPose2 \
     -v /home/pose/.cache/torch:/root/.cache/torch \
     -v /home/pose/.cache/huggingface:/root/.cache/huggingface \
     genpose2:latest bash -lc \
     "source /opt/conda/etc/profile.d/conda.sh && conda activate genpose2 && \
      cd /workspace/GenPose2 && \
      python scripts/segment_sam2.py --text 'cup' --zoom 2"
   ```
   First run downloads GroundingDINO-tiny (~700 MB) from HuggingFace → needs
   internet; cached via the mounted `~/.cache/huggingface`. If `NO detection`,
   raise `--zoom 3` or lower `--box-thresh 0.15`.

3. **Run offline inference (headless, one-shot)**:
   ```bash
   sudo docker run --gpus all --rm --ipc=host \
     -v /home/pose/dipl/GenPose2:/workspace/GenPose2 \
     -v /home/pose/.cache/torch:/root/.cache/torch \
     genpose2:latest bash -lc \
     "source /opt/conda/etc/profile.d/conda.sh && conda activate genpose2 && \
      cd /workspace/GenPose2 && \
      USE_CAM=0 SAVE_CAM=0 SAVE_RES=1 HEADLESS=1 DATA_POINT=1 python runners/infer_camera.py"
   ```
   Outputs (bind-mounted back to host):
   `results/infer_res/0001/infered_images/infer_####.png` and `infered_videos/output.mp4`.
   First run downloads `dinov2_vits14` (~85 MB) → needs internet; cached via the
   mounted `~/.cache/torch`.

   For the interactive live-camera path instead: `xhost +local: && cd docker &&
   sudo bash run_container.sh`, then inside run `python runners/infer_camera.py`
   (defaults USE_CAM=1; needs SAM2 installed + the D415).

## Changes made this session
- `scripts/prepare_realsense_cup_genpose2.py` — **new**, builds the `video_stream` dir.
- `runners/infer_camera.py`:
  - `OPENCV_IO_ENABLE_OPENEXR=1` set **before** `import cv2` (else `.exr` depth/mask
    read silently fails — cutoop sets it too late, after cv2 is already imported).
  - `pyrealsense2` / `from camera.camera import RealSenseRobotStream` made **lazy**
    (only in the `USE_CAM` branch) — `camera.camera` imports SAM2 at module load,
    which would crash the offline path (SAM2 not installed, not needed offline).
  - `USE_CAM/SAVE_CAM/SAVE_RES/HEADLESS/DATA_POINT/TRACKING/TRACKING_T0` now read
    from env vars (can't use CLI args — `get_config()` consumes `sys.argv`).
  - GUI calls (`imshow`/`waitKey`) guarded by `HEADLESS`; removed offline `'s'`-key
    `rs_streamer` reference (NameError offline).
- `docker/Dockerfile`: `TORCH_CUDA_ARCH_LIST` `8.9` → `8.6;8.9`. The local GPU is an
  **RTX A2000 (sm_86)**; kernels built only for sm_89 fail with "no kernel image
  available for execution on the device".

## Sanity checks after a run
- `det(R) ≈ 1`, translation in metres and plausible (cup ~0.3–2 m).
- Overlay (axes/box drawn by `visualize_pose` → `DetectMatch._draw_image`) should
  sit on the cup; expect good alignment on close frames, drift on far/tiny ones.
