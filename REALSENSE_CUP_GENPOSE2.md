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
