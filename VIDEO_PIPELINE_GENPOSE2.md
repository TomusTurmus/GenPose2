# Running GenPose++ on a video + measuring pose jitter

Port of GigaPose's `VIDEO_PORTING_GUIDE.md` recipe to GenPose++. Companion to
`REALSENSE_CUP_GENPOSE2.md` (custom-data prep, segmentation, docker); this doc
covers the *video* clip + *jitter* parts.

**Core idea:** a video is just a frame sequence. GenPose++'s offline path has no
temporal model, so running on video needs no model changes — only frame extraction,
wrapping frames as `video_stream/`, and reassembling per-frame poses. Because each
frame is then independent, the **frame-to-frame variation on a static object = the
model's pose jitter** — no ground truth needed.

## Pre-flight: is GenPose++ a candidate? (yes, with two caveats)

- **Per-frame, model-free, instance-agnostic?** Yes — category-level, no CAD, no
  templates, no per-object training. Poses on a novel object of a known category are
  meaningful, so the clip's poses mean something (unlike untrained instance-level
  methods). Nothing to copy or symlink per clip: **prep is frames + intrinsics only.**
- ⚠️ **It is RGB-D.** GenPose++ lifts the masked pixels into a point cloud using
  depth (and drops depth > 4 m). **A mono video cannot be used** — `--video` fails
  with an explanation rather than producing garbage. The clip must carry depth:
  a RealSense `.bag`, or an `rgb/` + `depth/` capture dir.
- ⚠️ **It has a native tracking mode**, and it is **on by default**. Per the porting
  guide, prefer it for quality — but it makes frames correlated, so it *understates*
  jitter. Use `PER_FRAME=1` for the honest intrinsic-jitter number. See below.

### The `PER_FRAME` subtlety (the trap worth knowing)

`inference()` takes `prev_pose` and uses it as the sampler's `init_x`. `TRACKING`
only additionally lowers `T0` (0.55 → 0.3). So:

| Mode | prev pose fed as init_x? | Frames independent? | Use for |
|---|---|---|---|
| `TRACKING=1` (default) | yes, T0=0.3 | no | best-looking tracking |
| `TRACKING=0` alone | **still yes**, T0=0.55 | **no** | — (not what you want) |
| `PER_FRAME=1` | no | **yes** | measuring intrinsic jitter |

**`TRACKING=0` is not enough** to get independent frames — it still warm-starts from
the previous pose. Only `PER_FRAME=1` clears `prev_pose`.

## Steps

Prep + report run on the **host** (`foundationpose` env has cv2/pandas/matplotlib);
pose runs in the **container** (GPU + torch + cutoop). `--bag` needs `pyrealsense2`
(host env `realsense`).

### 1. Stage the clip (host)

```bash
OPENCV_IO_ENABLE_OPENEXR=1 ~/miniconda3/envs/foundationpose/bin/python \
  scripts/prepare_video_genpose2.py \
    --rgbd_dir /home/pose/dipl/FoundationPose/demo_data/realsense_cup \
    --masks_dir /home/pose/dipl/FoundationPose/demo_data/realsense_cup/masks \
    --data_point 2 --every 1
```

Writes `results/infer_res/0002/video_stream/<idx>_{color.png,depth.exr,mask.exr,meta.json}`,
frames re-indexed `0000..N-1` in capture order, plus `frame_index.json` mapping the new
index back to the source frame. Sources: `--rgbd_dir` (rgb/ + depth/ uint16 mm
[+ masks/] [+ cam_K.txt|camera.json]) or `--bag` (RealSense recording; intrinsics read
from the stream). Ergonomics carried over from GigaPose: `--every N`, `--max_frames`,
`--cam_from <data_point|path>` vs `--fx/--fy/--cx/--cy`.

`--data_point` defaults to **2** (1 is the existing cup demo). Writing into a
non-empty clip dir needs `--overwrite`.

### 2. Masks (the one real dependency)

GenPose++ ships no segmenter — the paper assumes GT masks. Either `--masks_dir` above,
or Grounded-SAM2 per frame (in the container):

```bash
python scripts/segment_sam2.py --frames_dir results/infer_res/0002/video_stream \
    --text 'cup' --zoom 2 --max-obj 1
python3 scripts/visualize_masks.py --frames_dir results/infer_res/0002/video_stream
```

> **Ordering trap:** prep → segment → pose. Prep rewrites `mask.exr`, so re-running
> step 1 after segmenting silently clobbers the SAM2 masks (see
> `REALSENSE_CUP_GENPOSE2.md` for the timestamp tell).

> **Jitter caveat (applies to every model):** detector/mask jitter leaks into measured
> pose jitter. These masks are per-frame (not one static box), which is the honest
> setup — but record which masks you used when reporting numbers. Borrowed
> FoundationPose masks vs per-frame SAM2 give different jitter.

### 3 + 4. Pose + report (one shot)

```bash
bash run_video_pipeline_genpose2.sh 2            # per_frame (default): honest jitter
bash run_video_pipeline_genpose2.sh 2 tracking   # native warm start: smoother
```

That runs the container pose step (writes `poses.csv`, `infered_images/`,
`infered_videos/output.mp4`) then the host report. To run them separately:

```bash
# pose (container)
USE_CAM=0 SAVE_CAM=0 SAVE_RES=1 HEADLESS=1 SAVE_POSES=1 PER_FRAME=1 TRACKING=0 \
  DATA_POINT=2 python runners/infer_camera.py
# report (host)
~/miniconda3/envs/foundationpose/bin/python scripts/video_pose_report_genpose2.py --data_point 2
```

Outputs in `results/infer_res/0002/video_report/`: `pose.mp4` (predicted box + axes
overlay), `frames/`, `jitter.csv`, `jitter.png`, and printed stats.

## Reading the jitter report

Per object: `dR_deg` (geodesic rotation delta), `dt_mm` (translation delta),
`dsize_mm` (size delta), each between consecutive frames. **Low jitter = small, flat
deltas.** Expect large deltas on the far/tiny frames of the cup capture (~213 mask px
at 2.1 m) and smaller ones once the mug is close — jitter is not constant across a
clip, so quote the distribution, not just the mean.

- **Units: metres → mm.** GenPose++ works in **metres** (BOP/GigaPose is mm). The
  runner's `poses.csv` stores metres; the report converts to mm so `dt` is comparable
  to GigaPose numbers. Do not mix.
- **The box is predicted, not a CAD model.** GenPose++ estimates size, so the overlay
  box reflects size error too, and `dsize_mm` is a real jitter channel that
  CAD-based methods don't have.
- **Mask ids are positional, not identities.** Objects are discovered with
  `np.unique(mask)`, so a spurious detection on one frame renumbers the rest and shows
  up as huge fake jitter. The report warns when the object count varies. For a
  single-instance clip, segment with `--max-obj 1`.

## Changeset (what this port added)

| File | Action | Note |
|---|---|---|
| `scripts/prepare_video_genpose2.py` | **new** | RGB-D frame extraction + intrinsics → `video_stream/`; refuses mono video. No per-object assets to reuse (category-level). |
| `scripts/video_pose_report_genpose2.py` | **new** | Overlay + `pose.mp4` + jitter. Jitter math/plot/writer reused verbatim from GigaPose; `load_poses`, intrinsics/size source and units are the GenPose++ adaptations. |
| `runners/infer_camera.py` | **edit** | `PER_FRAME` (independent frames), `SAVE_POSES` → `poses.csv` (nothing persisted poses before — only overlay images), `inference(return_scores=True)` exposes retained-sample energy as a per-object confidence. |
| `scripts/visualize_masks.py` | **edit** | `--out` now derives from `--frames_dir` (previously any clip's montage landed in clip 0001). |
| `run_video_pipeline_genpose2.sh` | **new** | pose → report, `per_frame` \| `tracking`. |

Not needed here (unlike GigaPose): no CAD/template copying, and no detections-loader
edit — `segment_sam2.py` already takes `--frames_dir` and the runner already takes
`DATA_POINT`, so a new clip needs no code change.

## Design decisions & invariants (why it is built this way)

Notes for whoever runs or changes this next — these were non-obvious while porting.

### Where each stage runs, and why

| Stage | Runs | Needs |
|---|---|---|
| prep (`prepare_video_genpose2.py`) | host, `foundationpose` env | cv2 + EXR; source data lives on the host, the container only mounts the repo |
| `--bag` decoding | host, `realsense` env | `pyrealsense2` (not in `foundationpose`) |
| pose (`infer_camera.py`) | container `genpose2:latest` | GPU + torch + cutoop |
| report (`video_pose_report_genpose2.py`) | host, `foundationpose` env | cv2/pandas/matplotlib only |

The report is host-side **on purpose**: it deliberately needs no torch/cutoop, so
jitter analysis can be re-run and tweaked without the GPU or the container. Only the
pose step needs either.

`sudo docker` is password-gated here (the user is not in the `docker` group), so the
pose step is interactive and cannot be run unattended. `run_video_pipeline_genpose2.sh`
honours `DOCKER=` and `HOST_PY=` overrides rather than hardcoding that assumption.

`OPENCV_IO_ENABLE_OPENEXR=1` must be set **before** `import cv2` or `.exr` depth/mask
I/O silently fails (cutoop sets it too late). Both new scripts set it at the top.

### Invariants that are easy to break by accident

1. **`obj_id` ↔ pose-row alignment is positional.** `poses.csv` pairs `obj_idxx[j]`
   with `pose[0][j]`. That only holds because `get_objects(only_idx=True)` and the
   full `get_objects()` iterate `np.unique(mask)` applying *identical* filters (skip
   id 0, skip masks < 10 px, skip `get_per_object() -> None`). Change the filtering in
   one path and the CSV silently mislabels objects — wrong per-object jitter, no error.
2. **`get_objects` caches only on the full path.** `self.data` is set at the end of the
   full call; `only_idx=True` returns early and does not cache. So `only_idx` must be
   called **before** the full call — calling it afterwards returns the cached full dict,
   which has no `'idx'` key (KeyError). The offline loop already orders it correctly.
3. **`inference()` must keep returning a 2-tuple by default.** The live-camera branch
   depends on it, and `runners/infer.py` carries a *duplicate* `GenPose2` class that is
   not shared with this file. `return_scores=True` is opt-in for exactly that reason —
   do not "clean it up" into an unconditional 3-tuple.
4. **Frame stems are strings, not numbers.** `poses.csv`'s `frame` is a zero-padded
   stem (`0000`). The report reads it with `dtype={"frame": str}` — without that pandas
   infers int `0`, every colour-frame lookup misses, and you get a jitter report with no
   overlay video. This bit once; the guard is the `dtype` and the "no overlay written"
   warning.

### What `score` actually means

Mean **energy of the retained samples** (both rotation and translation channels), where
`retain_num = eval_repeat_num * retain_ratio` = 50 * 0.4 = 20 of 50 sampled poses. It is
`1.0` if no energy model is loaded (the ones-placeholder). Treat it as a confidence that
is comparable **within** a clip — not a probability, and not comparable across methods.

### What constrains the measurement itself

- **Depth > 4 m is zeroed** by the loader, so far objects lose their points entirely.
  Combined with the cup capture being ~213 mask px at 2.1 m, early frames are weak by
  construction — that is the data, not a bug in the port.
- **Detector/mask jitter leaks into pose jitter** — an upper bound on the model's own
  jitter, never a lower one. Always record which masks produced a number.
- **`data_point` 1 is the existing cup demo**; new clips default to 2, and prep refuses
  to write into a non-empty clip dir without `--overwrite` (protects staged clips and
  SAM2 masks).

## Verified so far

- Prep on the `realsense_cup` capture: 30 frames staged, depth **3000 mm → 3.000 m**,
  `mask.exr` decodes to exactly id `1` under cutoop's `imread()*255`, intrinsics
  auto-read from the capture dir, `--every`/`--max_frames` re-index correctly.
- Report: `poses.csv` → parser round-trip with real tensors (multi-object, metres→mm,
  zero-padded stems), jitter math (2.96° measured on an injected 2.86° rotation),
  `pose.mp4` + `jitter.png` written, overlay projects the box at the labelled depth.
- **Not yet run:** the GPU pose step (needs `sudo docker`), so `poses.csv` from a real
  run and the end-to-end jitter numbers are still pending.
