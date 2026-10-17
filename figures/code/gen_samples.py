"""生成图 1 的两张实图：训练期退化覆盖 vs 真实部署退化。

★ 用的是论文**同一条管线**（`dcdc_scan.degrade_param` + BSRGAN ×4），不是另画示意图。

样本选取（可复现，不择优）——准则写在 meta.json 里：
  1. 候选池 = 论文评测切片（`sorted(glob)[2103:2303]`，与第 3 章同一批 200 张）；
  2. 按标签框数取**中位数**的那张（代表性，不挑最好看的）；
  3. 裁窗 = 在 800×800 上按 stride 20 滑动，取**框中心数最多**的 240×240 窗，
     并列时取 (y,x) 最小者（确定性）。

两栏唯一差别 = 噪声与 JPEG（模糊 σ 与降采样完全相同）：
  (a) 训练期覆盖：blur=2.0, noise=0.0,  jpeg=None
  (b) 部署场景   ：blur=2.0, noise=0.096, jpeg=40
  ⇒ 视觉差异全部来自「训练时没见过的那两维」。

⚠️ 评测侧 `eval_arm(..., jpeg=None)` ⇒ **JPEG 只在训练期出现**。
   所以 (b) 是 §1.2 的部署场景示意（Y），不是被评测的 16 个条件之一。
   图题必须写清这一点。

用法（WSL，需 GPU）：python gen_samples.py
"""
import io
import json
import os
import sys

import numpy as np

_PIN = [sys.stdout]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
_PIN.append(sys.stdout)          # ★ 钉住自己的 wrapper（否则被下一层 import 顶掉后 GC 会关 buffer）

A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
FIG = os.path.join(A2, "figures")
CACHE = os.path.join(FIG, "_cache")
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)

import glob                                              # noqa: E402
import cv2                                               # noqa: E402
import deg_ops as D                                      # noqa: E402
_PIN.append(sys.stdout)          # deg_ops 在 import 时也替换 sys.stdout
import dcdc_scan as DC                                   # noqa: E402
_PIN.append(sys.stdout)
from probe0_common import read_yolo                      # noqa: E402
_PIN.append(sys.stdout)
from probe0_sr import RealESRGAN                         # noqa: E402
_PIN.append(sys.stdout)

CROP = 400          # SR 空间（800）里的裁窗边长；对应 LR 空间 100 px
STRIDE = 40
BLUR, NOISE, JPEG = 2.0, 0.096, 40


def eval_slice():
    files = sorted(glob.glob(os.path.join(DC.HR_IMG, "*.png")))[:DC.POOL]
    return files[DC.N_TR:][:200]


def box_count(stem):
    g = read_yolo(os.path.join(DC.HR_LBL, stem + ".txt"), DC.BASE, DC.BASE)
    return len(g)


def pick_window(cx, cy):
    """cx/cy：框中心（像素）。取中心数最多的 240×240 窗，并列取 (y,x) 最小。"""
    best, best_xy = -1, (0, 0)
    lim = DC.BASE - CROP
    for y0 in range(0, lim + 1, STRIDE):
        for x0 in range(0, lim + 1, STRIDE):
            m = (cx >= x0) & (cx < x0 + CROP) & (cy >= y0) & (cy < y0 + CROP)
            n = int(m.sum())
            if n > best:
                best, best_xy = n, (x0, y0)
    return best_xy, best


def main():
    files = eval_slice()
    print(f"[pool] 评测切片 {len(files)} 张")
    counts = [(box_count(os.path.splitext(os.path.basename(p))[0]), p) for p in files]
    counts = [(c, p) for c, p in counts if c > 0]
    counts.sort()
    med = counts[len(counts) // 2][0]
    # 中位数那张：框数与中位数之差最小者，并列取路径字典序最小
    cand = sorted(counts, key=lambda t: (abs(t[0] - med), t[1]))
    n_box, path = cand[0]
    stem = os.path.splitext(os.path.basename(path))[0]
    print(f"[pick] {stem}  框数 {n_box}（候选池中位数 {med}）")

    hr = cv2.imread(path, cv2.IMREAD_COLOR)
    if hr.shape[0] != DC.BASE or hr.shape[1] != DC.BASE:
        hr = cv2.resize(hr, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)

    eng = RealESRGAN(DC.SCALE, engine="bsrgan")
    out = {}
    for tag, kw in (("left", dict(noise=0.0, jpeg=None)),
                    ("right", dict(noise=NOISE, jpeg=JPEG))):
        rng = D.arm_rng(stem, "fig1_" + tag)
        lr = DC.degrade_param(hr, DC.SCALE, BLUR, kw["noise"], kw["jpeg"], rng,
                              shot=False, resample="area")
        if lr.shape[0] != DC.LR or lr.shape[1] != DC.LR:
            lr = cv2.resize(lr, (DC.LR, DC.LR), interpolation=cv2.INTER_AREA)
        lr8 = (np.clip(np.asarray(lr, dtype=np.float32) * (255.0 if np.asarray(lr).max() <= 1.001 else 1.0),
                       0, 255)).astype(np.uint8)
        cv2.imwrite(os.path.join(CACHE, tag + "_lr.png"), lr8)
        sr = eng.upscale(lr)
        if sr.shape[0] != DC.BASE or sr.shape[1] != DC.BASE:
            sr = cv2.resize(sr, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)
        sr = np.asarray(sr, dtype=np.float32)
        if sr.max() <= 1.001:
            sr = sr * 255.0
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        out[tag] = sr
        # ★ 不要 cvtColor：管线全程走 cv2（BGR），`tv_sweep.build_view` 也是直接 imwrite。
        cv2.imwrite(os.path.join(CACHE, tag + ".png"), sr)
        print(f"  [{tag}] {sr.shape} 已写")

    # 裁窗：以左图的框位置为准（两图场景相同）。read_yolo 每框是 5 元，[0:4] 为像素 xyxy。
    g = read_yolo(os.path.join(DC.HR_LBL, stem + ".txt"), DC.BASE, DC.BASE)
    b = np.array([list(t)[:4] for t in g], dtype=np.float64)
    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    (x0, y0), hit = pick_window(cx, cy)
    print(f"[crop] 窗 ({x0},{y0}) 240×240，命中 {hit} 个框中心")

    meta = dict(stem=stem, n_box=n_box, pool_median=med, pool_n=len(counts),
                crop=dict(x0=x0, y0=y0, size=CROP, boxes_hit=hit),
                params=dict(blur=BLUR, noise_left=0.0, noise_right=NOISE,
                            jpeg_left=None, jpeg_right=JPEG, scale=DC.SCALE,
                            sr_engine="bsrgan", shot=False, resample="area"),
                note="JPEG appears at TRAINING time only; eval_arm uses jpeg=None")
    with open(os.path.join(CACHE, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print("[meta] 已写 meta.json")


if __name__ == "__main__":
    main()
