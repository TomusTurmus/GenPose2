"""Segmentation stage for GenPose++ custom-data inference: Grounded-SAM2.

GenPose2's *paper* (Omni6DPose, arXiv 2406.04316) does not ship a segmenter --
it evaluates with ground-truth masks ("we assume ground truth instance
segmentation is known"). The *repo's* real-time camera path (`camera/camera.py`)
segments with the SAM2 real-time camera predictor (Gy920/segment-anything-2-real-time,
`sam2.1_hiera_tiny.pt`), but prompts it by an interactive human click on frame 0.

To make that fully automatic (no clicks, no borrowed/GT masks) and able to find
small/far objects, we put a *detector* in front of SAM2 -> Grounded-SAM2:
  GroundingDINO (text label -> box)  ->  SAM2 (box -> pixel mask).
GroundingDINO runs via HuggingFace transformers (IDEA-Research/grounding-dino-tiny),
so there are no custom CUDA ops to compile.

Two modes:
  (default) PER-FRAME: detect + segment every frame independently. This is the
      right choice for the offline `video_stream` path -- GenPose2 scores each
      frame's pose independently and uses NO temporal info, so SAM2's video
      tracking buys nothing and only propagates a bad first-frame seed. Robust
      when the camera pans (object tiny/far early, large/close later).
  --track  PROMPT-ONCE: build the real-time camera predictor, prompt frame 0,
      then track() forward (the live-demo paradigm). Kept for parity; brittle
      when the scene changes a lot.

Prompting:
  --text "cup"     GroundingDINO label (repeat for >1 category). Required for the
                   automatic path. Per-frame mode masks EVERY detection above
                   threshold (-> one mask id each); --track masks the best per label.
  --box X,Y,W,H    manual box (repeat); --point X,Y; --from-mask; --click
                   are frame-0 prompts, only meaningful with --track.
Small objects: --zoom N upscales the frame N x before GroundingDINO (boxes mapped
back); lower --box-thresh raises recall.

Writes, per frame, into --frames_dir (default = the video_stream dir):
  <idx>_mask.exr   float32 = mask_id/255  (cutoop.load_mask does img*255 -> id)
  <idx>_mask.png   uint8 visualization (per-object grey level)

Run in the container (needs torch + SAM2 + GroundingDINO + GPU):
  conda activate genpose2 && cd /workspace/GenPose2
  python scripts/segment_sam2.py --text "cup" --zoom 2
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # before cv2, to write .exr
import sys
import glob
import argparse

import numpy as np
import cv2
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO, "segment-anything-2-real-time"))

DEFAULT_FRAMES = os.path.join(REPO, "results", "infer_res", "0001", "video_stream")
DEFAULT_CKPT = os.path.join(REPO, "segment-anything-2-real-time", "checkpoints", "sam2.1_hiera_tiny.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # resolved by SAM2's hydra search path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default=DEFAULT_FRAMES)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--track", action="store_true", help="prompt-once + SAM2 video track (default: per-frame detect)")
    ap.add_argument("--text", action="append", default=[], help="GroundingDINO label e.g. 'cup' (repeat for >1)")
    ap.add_argument("--box", action="append", default=[], help="X,Y,W,H manual box, --track only (repeat)")
    ap.add_argument("--point", action="append", default=[], help="X,Y positive click, --track only (repeat)")
    ap.add_argument("--from-mask", action="store_true", help="seed frame-0 box from existing mask.exr, --track only")
    ap.add_argument("--click", action="store_true", help="interactive frame-0 click UI, --track only (needs X11)")
    ap.add_argument("--gdino-id", default="IDEA-Research/grounding-dino-tiny", help="HF GroundingDINO model")
    ap.add_argument("--box-thresh", type=float, default=0.25, help="GroundingDINO box confidence threshold")
    ap.add_argument("--text-thresh", type=float, default=0.20, help="GroundingDINO text threshold")
    ap.add_argument("--zoom", type=int, default=1, help="upscale frame NxN before GroundingDINO (small objects)")
    ap.add_argument("--max-obj", type=int, default=0, help="cap detections kept per frame (0 = all)")
    return ap.parse_args()


class GroundingDINO:
    """HF GroundingDINO loaded once; detect labelled boxes on an RGB frame."""

    def __init__(self, model_id, device):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, rgb, labels, box_thresh, text_thresh, zoom):
        """Return all detections as [(label, [x1,y1,x2,y2], score)], sorted by score desc."""
        from PIL import Image
        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb)
        if zoom > 1:
            img = img.resize((W * zoom, H * zoom), Image.BICUBIC)
        caption = " . ".join(l.strip().lower() for l in labels) + " ."  # GDINO wants lowercase '.'-separated
        inputs = self.processor(images=img, text=caption, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], box_threshold=box_thresh,
            text_threshold=text_thresh, target_sizes=[img.size[::-1]])[0]

        dets = []
        for box, score, text in zip(res["boxes"], res["scores"], res["labels"]):
            x1, y1, x2, y2 = (float(v) / zoom for v in box.tolist())
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(W - 1.0, x2), min(H - 1.0, y2)
            if x2 > x1 and y2 > y1:
                dets.append((str(text), [int(x1), int(y1), int(x2), int(y2)], float(score)))
        dets.sort(key=lambda d: d[2], reverse=True)
        return dets


def box_from_mask(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = m[:, :, -1]
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        raise ValueError(f"empty mask: {mask_path}")
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def click_prompts(color_bgr):
    """Replicate camera.py's UI: click positive points, 'n' = next object, 'q' = done."""
    objs, pts = [], []
    disp = color_bgr.copy()
    win = "prompt (click object; n=next obj; q=done)"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow(win, disp)

    cv2.imshow(win, disp)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("n"), ord("q")):
            if pts:
                objs.append(np.array(pts, dtype=np.float32))
                pts = []
            if k == ord("q"):
                break
    cv2.destroyAllWindows()
    return objs


def write_masks(prefix, mask_id):
    """mask_id: float32 HxW with values j/255 (j=object id). Write .exr + .png vis."""
    cv2.imwrite(prefix + "mask.exr", mask_id)
    vis = mask_id * 255.0  # back to integer ids
    if vis.max() > 0:
        vis = vis / vis.max() * 255.0  # spread ids across grey range for visibility
    cv2.imwrite(prefix + "mask.png", vis.astype(np.uint8))


def run_per_frame(args, frames, autocast):
    """Default mode: GroundingDINO + SAM2 image predictor on every frame independently."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    assert args.text, "per-frame mode needs --text (e.g. --text 'cup'); or pass --track for manual prompts"
    print(f"per-frame Grounded-SAM2: labels={args.text} zoom={args.zoom} box_thresh={args.box_thresh}")
    gdino = GroundingDINO(args.gdino_id, args.device)
    sam2_model = build_sam2(MODEL_CFG, args.ckpt, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)

    n_written = n_hit = 0
    with torch.inference_mode(), autocast:
        for i, fpath in enumerate(frames):
            bgr = cv2.imread(fpath)
            H, W = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            dets = gdino.detect(rgb, args.text, args.box_thresh, args.text_thresh, args.zoom)
            if args.max_obj > 0:
                dets = dets[:args.max_obj]

            mask_id = np.zeros((H, W), np.float32)
            if dets:
                predictor.set_image(rgb)
                for j, (lab, box, score) in enumerate(dets):
                    masks, _, _ = predictor.predict(box=np.array(box, np.float32),
                                                    multimask_output=False)
                    mask_id[masks[0].astype(bool)] = (j + 1) / 255.0  # cutoop multiplies by 255
                n_hit += 1

            write_masks(fpath.replace("color.png", ""), mask_id)
            n_written += 1
            top = f"{dets[0][2]:.2f}" if dets else "-"
            print(f"  frame {i:02d} {os.path.basename(fpath)}: {len(dets)} det "
                  f"(top score {top}), {int((mask_id > 0).sum())} px")
    print(f"wrote {n_written} masks ({n_hit} with an object) to {args.frames_dir}")


def run_track(args, frames, autocast):
    """Legacy prompt-once mode: SAM2 real-time camera predictor seeded on frame 0."""
    from sam2.build_sam import build_sam2_camera_predictor
    predictor = build_sam2_camera_predictor(MODEL_CFG, args.ckpt, device=args.device)

    first_bgr = cv2.imread(frames[0])
    H, W = first_bgr.shape[:2]
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), autocast:
        predictor.load_first_frame(first_rgb)
        n_obj = 0
        if args.click:
            for pts in click_prompts(first_bgr):
                n_obj += 1
                predictor.add_new_prompt(frame_idx=0, obj_id=n_obj, points=pts,
                                         labels=np.ones(len(pts), np.int32))
        else:
            boxes = list(args.box)
            if args.text:
                gdino = GroundingDINO(args.gdino_id, args.device)
                for lab, (x1, y1, x2, y2), score in gdino.detect(
                        first_rgb, args.text, args.box_thresh, args.text_thresh, args.zoom):
                    print(f"  GroundingDINO '{lab}': score={score:.3f} box=[{x1},{y1},{x2},{y2}]")
                    boxes.append(f"{x1},{y1},{x2 - x1},{y2 - y1}")
                    break  # track a single best detection
            if args.from_mask:
                x, y, w, h = box_from_mask(frames[0].replace("color.png", "mask.exr"))
                boxes.append(f"{x},{y},{w},{h}")
            for b in boxes:
                x, y, w, h = (int(v) for v in b.split(","))
                n_obj += 1
                predictor.add_new_prompt(frame_idx=0, obj_id=n_obj, bbox=[x, y, x + w, y + h])
            if args.point:
                pts = np.array([[int(v) for v in p.split(",")] for p in args.point], np.float32)
                n_obj += 1
                predictor.add_new_prompt(frame_idx=0, obj_id=n_obj, points=pts,
                                         labels=np.ones(len(pts), np.int32))
        assert n_obj > 0, "no prompt/detection; use --text / --box / --point / --from-mask / --click"
        print(f"tracking {n_obj} object(s) from frame 0")

        n_written = 0
        for i, fpath in enumerate(frames):
            rgb = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
            _obj_ids, mask_logits = predictor.track(rgb)
            masks = (mask_logits[:, 0] > 0.0).cpu().numpy()
            mask_id = np.zeros((H, W), np.float32)
            for j in range(masks.shape[0]):
                mask_id[masks[j]] = (j + 1) / 255.0
            write_masks(fpath.replace("color.png", ""), mask_id)
            n_written += 1
            print(f"  frame {i:02d} {os.path.basename(fpath)}: {int((mask_id > 0).sum())} px")
    print(f"wrote {n_written} masks to {args.frames_dir}")


def main():
    args = parse_args()
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*_color.png")))
    assert frames, f"no *_color.png in {args.frames_dir}"
    print(f"{len(frames)} frames; ckpt={os.path.basename(args.ckpt)} "
          f"mode={'track' if args.track else 'per-frame'}")

    autocast = (torch.autocast(args.device, dtype=torch.bfloat16)
                if args.device == "cuda" else torch.autocast("cpu", enabled=False))
    if args.track:
        run_track(args, frames, autocast)
    else:
        run_per_frame(args, frames, autocast)


if __name__ == "__main__":
    main()
