# -*- coding: utf-8 -*-
"""为图 2 挑一个"信息量足够"的真实裁窗（确定性，可复现）。

准则（先写死，避免"挑好看"）：
  1. 候选 = 论文评测切片 200 张（与第 3 章同一批）；
  2. 每张按 stride 40 滑 400×400 窗，计窗内 GT 框中心数，**只计短边 ≥ 6 px 的框**
     （更小的在 182 mm 通栏图上不到 2.7 mm，印出来只是像素点）；
  3. 取全池框数最多的前 8 个 (图, 窗)，并列按 (stem, y0, x0) 升序；
  4. 对这 8 个候选跑一次 BSRGAN + YOLOv8n(conf=0.25)，取**检出框数最多**者；
     仍并列则取列表序最小者。
输出 `_cache/fig02/pick.json`。

用法：/home/czy/envs/sr4sod/bin/python pick_fig02_window.py
"""
import io
import json
import os
import sys

import numpy as np

_PIN = [sys.stdout]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
_PIN.append(sys.stdout)

A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
CACHE = os.path.join(A2, "figures", "_cache")
DST = os.path.join(CACHE, "fig02")
os.makedirs(DST, exist_ok=True)
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)

import glob                                              # noqa: E402
import cv2                                               # noqa: E402
import deg_ops as D                                      # noqa: E402
_PIN.append(sys.stdout)
import dcdc_scan as DC                                   # noqa: E402
_PIN.append(sys.stdout)
from probe0_common import read_yolo                      # noqa: E402
_PIN.append(sys.stdout)
from probe0_sr import RealESRGAN                         # noqa: E402
_PIN.append(sys.stdout)

RUN = "/home/czy/data/sr4sod/tv_runs/tv_B2_s2026/weights/best.pt"
CROP, STRIDE = 400, 40
MIN_SIDE = 6.0
BLUR = 2.0
TOPK = 8


def to_u8(a):
    a = np.asarray(a, dtype=np.float32)
    if a.max() <= 1.001:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def main():
    files = sorted(glob.glob(os.path.join(DC.HR_IMG, "*.png")))[:DC.POOL]
    files = files[DC.N_TR:][:200]
    print(f"[pool] {len(files)} 张")

    cands = []
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        g = read_yolo(os.path.join(DC.HR_LBL, stem + ".txt"), DC.BASE, DC.BASE)
        if not g:
            continue
        b = np.array([list(t)[:4] for t in g], dtype=np.float64)
        # ★ 只统计"印出来看得见"的框：短边 ≥ MIN_SIDE。400 px 窗在 182 mm 通栏图上约
        #   0.45 mm/px ⇒ 6 px 目标 ≈ 2.7 mm，肉眼可辨；更小的（如 P0160 的 3 px 点阵）
        #   印出来只是像素噪点，用它挑窗会得到一片"看起来是噪声"的图。
        short = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        b = b[short >= MIN_SIDE]
        if len(b) == 0:
            continue
        cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
        lim = DC.BASE - CROP
        best, bxy = -1, (0, 0)
        for y0 in range(0, lim + 1, STRIDE):
            for x0 in range(0, lim + 1, STRIDE):
                m = (cx >= x0) & (cx < x0 + CROP) & (cy >= y0) & (cy < y0 + CROP)
                if int(m.sum()) > best:
                    best, bxy = int(m.sum()), (x0, y0)
        cands.append((best, stem, bxy[0], bxy[1], p))
    cands.sort(key=lambda t: (-t[0], t[1], t[3], t[2]))
    top = cands[:TOPK]
    print("[候选]")
    for c in top:
        print(f"   n_gt={c[0]:2d}  {c[1]}  窗=({c[2]},{c[3]})")

    eng = RealESRGAN(DC.SCALE, engine="bsrgan")
    from ultralytics import YOLO
    m = YOLO(RUN)
    scored = []
    for n_gt, stem, x0, y0, path in top:
        hr = cv2.imread(path, cv2.IMREAD_COLOR)
        if hr.shape[:2] != (DC.BASE, DC.BASE):
            hr = cv2.resize(hr, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)
        rng = D.arm_rng(stem, "fig2probe")
        lr = DC.degrade_param(hr, DC.SCALE, BLUR, 0.0, None, rng, shot=False, resample="area")
        if lr.shape[:2] != (DC.LR, DC.LR):
            lr = cv2.resize(lr, (DC.LR, DC.LR), interpolation=cv2.INTER_AREA)
        sr = eng.upscale(lr)
        if sr.shape[:2] != (DC.BASE, DC.BASE):
            sr = cv2.resize(sr, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)
        sr8 = to_u8(sr)
        r = m.predict(sr8, imgsz=DC.BASE, conf=0.25, verbose=False)[0]
        bb = r.boxes.xyxy.cpu().numpy()
        cc = r.boxes.conf.cpu().numpy()
        keep = (bb[:, 2] > x0) & (bb[:, 0] < x0 + CROP) & (bb[:, 3] > y0) & (bb[:, 1] < y0 + CROP)
        n_det = int(keep.sum())
        cm = float(cc[keep].mean()) if n_det else 0.0
        print(f"   {stem} ({x0},{y0})  GT={n_gt}  检出={n_det}  conf̄={cm:.2f}")
        scored.append((n_det, cm, n_gt, stem, x0, y0))

    # 先按检出数降序，再按平均置信度降序（仍然并列则保持 top 原序）
    order = sorted(range(len(scored)),
                   key=lambda i: (-scored[i][0], -round(scored[i][1], 2), i))
    best = scored[order[0]]
    out = dict(stem=best[3], x0=best[4], y0=best[5], size=CROP,
               n_gt_in_window=best[2], n_det=int(best[0]), conf_mean=best[1],
               ckpt=RUN, conf_thr=0.25,
               rule="densest 400px window in the 200-image eval slice; "
                    "top-6 candidates by GT count, then most detections at conf 0.25")
    with open(os.path.join(DST, "pick.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("[pick]", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
