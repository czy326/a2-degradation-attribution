"""
Shared utilities for the SR4SOD probe family (probe 0).

Conventions follow E:\\论文2\\dataset\\probe*.py:
  - UTF-8 stdout wrapper
  - human-readable tables on stdout, one JSON artifact next to the script
  - pre-registered verdicts only, no post-hoc threshold tuning

This module is dataset-agnostic and CPU-only (numpy + cv2).

Annotation formats supported
  dota_like : "x1 y1 x2 y2 class difficult" per line, one file per image
              (AI-TOD and DOTA-v1 both ship in this layout)
  yolo      : "cls cx cy w h" normalised, one file per image
  coco      : single json with images / annotations (axis-aligned bbox)

Everything downstream works on axis-aligned boxes (x1,y1,x2,y2,cls,diff).
DOTA-style oriented boxes (8-value lines) are NOT handled here on purpose:
probe 0 is about scale and label noise, not about rotation.
"""
import os
import json
import numpy as np

# --------------------------------------------------------------------------
# paths  (WSL view; the script is meant to run inside the emssm env)
# --------------------------------------------------------------------------
WORKDIR_WSL = "/mnt/e/新方向研究/sr4sod"
AITOD_WSL = "/mnt/e/新方向研究/sr4sod/_data"

# short-side bins, in pixels of the INPUT image the detector actually sees.
# The upper edge of a bin is the first edge greater than the value.
SIZE_EDGES = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 1e9]

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # 0.50 .. 0.95


# --------------------------------------------------------------------------
# annotation readers
# --------------------------------------------------------------------------
def read_dota_like(path):
    """Return list of (x1,y1,x2,y2,cls,difficult). Skips malformed lines."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            p = ln.strip().split()
            if len(p) < 6:
                continue
            try:
                v = [float(x) for x in p[:6]]
            except ValueError:
                continue
            out.append((v[0], v[1], v[2], v[3], int(v[4]), int(v[5])))
    return out


def read_yolo(path, img_w, img_h):
    """Return list of (x1,y1,x2,y2,cls,0) from a YOLO label file."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            p = ln.strip().split()
            if len(p) < 5:
                continue
            try:
                c = int(float(p[0]))
                cx, cy, w, h = [float(x) for x in p[1:5]]
            except ValueError:
                continue
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
            out.append((x1, y1, x2, y2, c, 0))
    return out


def read_coco(path, drop_crowd=True):
    """COCO json -> list of dict(name, w, h, boxes(N,4), cats(N,)).

    AI-TOD ships this format (bbox = [x, y, w, h], axis aligned, iscrowd flag).
    Images with no annotations are skipped, so the returned list is exactly the
    set of images that carry at least one usable box.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    by_img = {}
    for a in d.get("annotations", []):
        if drop_crowd and int(a.get("iscrowd", 0)) == 1:
            continue
        try:
            x, y, w, h = [float(v) for v in a["bbox"][:4]]
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        by_img.setdefault(a["image_id"], []).append(
            (x, y, x + w, y + h, int(a.get("category_id", 0))))
    out = []
    for im in d.get("images", []):
        bs = by_img.get(im["id"])
        if not bs:
            continue
        arr = np.array([[b[0], b[1], b[2], b[3]] for b in bs], dtype=np.float32)
        cats = np.array([b[4] for b in bs], dtype=np.int32)
        out.append(dict(name=str(im.get("file_name", im["id"])),
                        w=int(im.get("width", 0)), h=int(im.get("height", 0)),
                        boxes=arr, cats=cats))
    return out


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def box_short_side(b):
    return float(min(b[2] - b[0], b[3] - b[1]))


def box_area(b):
    return float(max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0))


def size_bin(short_side, edges=SIZE_EDGES):
    """Index of the short-side bin. Returns -1 for degenerate boxes."""
    if short_side <= 0:
        return -1
    for i in range(len(edges) - 1):
        if edges[i] <= short_side < edges[i + 1]:
            return i
    return len(edges) - 2


def bin_label(i, edges=SIZE_EDGES):
    if i < 0:
        return "degenerate"
    lo, hi = edges[i], edges[i + 1]
    hi_s = "inf" if hi >= 1e9 else f"{hi:g}"
    return f"{lo:g}-{hi_s}"


def iou_batch(A, B):
    """A:(N,4) B:(M,4) -> (N,M) IoU matrix."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    ax1, ay1, ax2, ay2 = A[:, 0:1], A[:, 1:2], A[:, 2:3], A[:, 3:4]
    bx1, by1, bx2, by2 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0).astype(np.float32)


def jitter_boxes(boxes, delta, rng):
    """Perturb every side of every box by Uniform(-delta, delta) pixels.

    delta <= 0 returns a copy. Boxes are kept non-degenerate (>=1 px).
    """
    B = np.asarray(boxes, dtype=np.float32).copy()
    if delta <= 0 or len(B) == 0:
        return B, np.zeros(len(B), dtype=bool)
    n = len(B)
    d = rng.uniform(-delta, delta, size=(n, 4)).astype(np.float32)
    B[:, 0] += d[:, 0]
    B[:, 1] += d[:, 1]
    B[:, 2] += d[:, 2]
    B[:, 3] += d[:, 3]
    # enforce x1<x2, y1<y2 with at least 1 px
    x1 = np.minimum(B[:, 0], B[:, 2] - 1.0)
    x2 = np.maximum(B[:, 2], B[:, 0] + 1.0)
    y1 = np.minimum(B[:, 1], B[:, 3] - 1.0)
    y2 = np.maximum(B[:, 3], B[:, 1] + 1.0)
    B = np.stack([x1, y1, x2, y2], axis=1)
    degenerate = (B[:, 2] - B[:, 0] <= 0) | (B[:, 3] - B[:, 1] <= 0)
    return B, degenerate


# --------------------------------------------------------------------------
# AP
# --------------------------------------------------------------------------
def ceiling_ap(pred, gt, thresholds=IOU_THRESHOLDS):
    """Oracle-detector AP ceiling under a matching constraint.

    Assumes |pred| == |gt| and no ranking signal, i.e. the best a detector
    could do is match as many ground-truth boxes as the IoU threshold allows.
    Matching is greedy by descending IoU (one-to-one).
    Returns {threshold: matched_fraction}.
    """
    n = len(gt)
    out = {}
    if n == 0:
        return {t: float("nan") for t in thresholds}
    M = iou_batch(pred, gt)
    for t in thresholds:
        cand = np.argwhere(M >= t)
        if len(cand) == 0:
            out[t] = 0.0
            continue
        vals = M[cand[:, 0], cand[:, 1]]
        order = np.argsort(-vals)
        cand = cand[order]
        used_p = set()
        used_g = set()
        cnt = 0
        for pi, gi in cand:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(int(pi))
            used_g.add(int(gi))
            cnt += 1
        out[t] = cnt / float(n)
    return out


def ap_all_points(pred, scores, gt, iou_thr):
    """Standard all-points AP for one class at one IoU threshold."""
    n_gt = len(gt)
    if n_gt == 0 or len(pred) == 0:
        return 0.0
    M = iou_batch(pred, gt)
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    tp = np.zeros(len(order), dtype=np.float32)
    fp = np.zeros(len(order), dtype=np.float32)
    used_g = set()
    for k, pi in enumerate(order):
        gi = int(np.argmax(M[pi]))
        if M[pi, gi] >= iou_thr and gi not in used_g:
            used_g.add(gi)
            tp[k] = 1.0
        else:
            fp[k] = 1.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    rec = ctp / float(n_gt)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # monotonic precision envelope
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    prev = 0.0
    for i in range(len(rec)):
        if rec[i] > prev:
            ap += (rec[i] - prev) * prec[i]
            prev = rec[i]
    return float(ap)


def coco_style_ap(pred, scores, gt, thresholds=IOU_THRESHOLDS):
    return float(np.mean([ap_all_points(pred, scores, gt, t) for t in thresholds]))


# --------------------------------------------------------------------------
# image degradation (shared with probe0b / probe0_dataprep)
# --------------------------------------------------------------------------
def degrade(img, factor, mode="area"):
    """Downsample by an integer factor, emulating a coarser ground sample
    distance. mode='area' uses INTER_AREA (sensor-integration-like, the honest
    surrogate); 'bicubic' uses INTER_CUBIC (the classic non-faithful surrogate
    that the literature is criticised for)."""
    import cv2
    if factor <= 1:
        return img.copy()
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h / factor))), max(1, int(round(w / factor)))
    itp = cv2.INTER_AREA if mode == "area" else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=itp)


def upsample(img, size_hw, mode="bicubic"):
    import cv2
    itp = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR,
           "nearest": cv2.INTER_NEAREST}[mode]
    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=itp)


# --------------------------------------------------------------------------
# io helper
# --------------------------------------------------------------------------
def save_json(path, obj):
    _d = os.path.dirname(path)
    if _d:
        os.makedirs(_d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    return path


def nan_safe(x, default=0.0):
    return default if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
