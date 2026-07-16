#!/usr/bin/env bash
# One-shot GenPose++ pipeline for a VIDEO clip: per-frame pose -> pose video + jitter report.
#
# GenPose++'s offline path has no temporal model, so with PER_FRAME=1 every frame is
# estimated independently -- this is the exact offline image pipeline applied to extracted
# video frames, and the frame-to-frame variation it produces IS the method's intrinsic
# jitter. See VIDEO_PIPELINE_GENPOSE2.md.
#
# Usage:
#   bash run_video_pipeline_genpose2.sh [DATA_POINT] [MODE]
#   # defaults: DATA_POINT=2, MODE=per_frame   (MODE=tracking uses the native warm start)
#
# Prereqs (run once, on the host -- the clip must exist and be segmented):
#   1. OPENCV_IO_ENABLE_OPENEXR=1 ~/miniconda3/envs/foundationpose/bin/python \
#          scripts/prepare_video_genpose2.py --rgbd_dir <capture> --data_point <DP> --every 1
#   2. masks per frame (GenPose++ ships no segmenter):
#          python scripts/segment_sam2.py --frames_dir results/infer_res/<DP>/video_stream \
#              --text '<object>' --zoom 2 --max-obj 1
#      ORDER MATTERS: prep rewrites mask.exr, so never re-run step 1 after step 2.
#   3. no per-object assets needed -- GenPose++ is category-level (no CAD, no templates).
#
# Pose runs in the container (needs the GPU + torch + cutoop); the report runs on the host.
set -euo pipefail

DP="${1:-2}"
MODE="${2:-per_frame}"
DPP=$(printf "%04d" "$DP")
RES="results/infer_res/${DPP}"
STREAM="${RES}/video_stream"
HOST_PY="${HOST_PY:-$HOME/miniconda3/envs/foundationpose/bin/python}"
DOCKER="${DOCKER:-sudo docker}"

if [ ! -d "$STREAM" ]; then
  echo "No clip at ${STREAM} -- run scripts/prepare_video_genpose2.py first (see header)." >&2
  exit 1
fi

case "$MODE" in
  per_frame) PER_FRAME=1; TRACKING=0 ;;   # independent frames: honest intrinsic jitter
  tracking)  PER_FRAME=0; TRACKING=1 ;;   # native warm start: smoother, but frames are correlated
  *) echo "MODE must be per_frame or tracking (got '$MODE')" >&2; exit 1 ;;
esac
echo "== GenPose++ video pipeline: data_point=${DPP} mode=${MODE} =="

# 1. Per-frame pose -> poses.csv + infered_images/ + infered_videos/output.mp4
${DOCKER} run --gpus all --rm --ipc=host \
  -v "$(pwd)":/workspace/GenPose2 \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  genpose2:latest bash -lc \
  "source /opt/conda/etc/profile.d/conda.sh && conda activate genpose2 && \
   cd /workspace/GenPose2 && \
   USE_CAM=0 SAVE_CAM=0 SAVE_RES=1 HEADLESS=1 SAVE_POSES=1 \
   PER_FRAME=${PER_FRAME} TRACKING=${TRACKING} DATA_POINT=${DP} python runners/infer_camera.py"

# 2. Pose video (pose.mp4) + jitter report (jitter.csv / jitter.png + stats)
"$HOST_PY" scripts/video_pose_report_genpose2.py --data_point "$DP"

echo "Done. Pose video + jitter in ${RES}/video_report/"
