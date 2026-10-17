# -*- coding: utf-8 -*-
"""图 2 用的真实图片素材：同一条管线（degrade_param → BSRGAN ×4 → YOLOv8n）。

产出（写入 figures/_cache/fig02/）：
  hr.png          HR 800 上的 400 裁窗（干净真值图）
  lr_clean.png    训练侧 LR（100×100，blur 2.0 / 无噪声）
  sr_clean.png    SR 输出 800 上的同一窗口
  det_clean.png   同窗口 + YOLOv8n 预测框
  lr_noisy.png    部署侧 LR（100×100，blur 2.0 + noise .096 + jpeg 40）
  sr_noisy.png    SR 输出同窗口
  det_noisy.png   同窗口 + 预测框
  det.json        两次推理的框数与置信度摘要

用法（WSL，需 GPU）：/home/czy/envs/sr4sod/bin/python gen_fig02_assets.py
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

import cv2                                               # noqa: E402
import deg_ops as D                                      # noqa: E402
_PIN.append(sys.stdout)
import dcdc_scan as DC                                   # noqa: E402
_PIN.append(sys.stdout)
_PIN.append(sys.stdout)
from probe0_sr import RealESRGAN                         # noqa: E402
_PIN.append(sys.stdout)

RUN = "/home/czy/data/sr4sod/tv_runs/tv_B2_s2026/weights/best.pt"   # 论文推荐配置
CROP = 400          # SR 空间的裁窗边长
STRIDE = 40
BLUR, NOISE, JPEG = 2.0, 0.024, None
CONF = 0.25


def to_u8(a):
    a = np.asarray(a, dtype=np.float32)
    if a.max() <= 1.001:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def main():
    meta = json.load(open(os.path.join(CACHE, "meta.json"), encoding="utf-8"))
    pk = os.path.join(DST, "pick.json")
    if os.path.isfile(pk):
        meta = json.load(open(pk, encoding="utf-8"))
        meta["crop"] = dict(x0=meta["x0"], y0=meta["y0"])
        print("[pick] 用 pick.json 的裁窗（可见目标最多的那一个）")
    stem = meta["stem"]
    x0, y0 = meta["crop"]["x0"], meta["crop"]["y0"]
    print(f"[in] stem={stem} crop=({x0},{y0},{CROP})  ckpt={RUN}")

    hr_path = os.path.join(DC.HR_IMG, stem + ".png")
    hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    if hr.shape[:2] != (DC.BASE, DC.BASE):
        hr = cv2.resize(hr, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(DST, "hr.png"), hr[y0:y0 + CROP, x0:x0 + CROP])
    print("  hr.png 已写")

    eng = RealESRGAN(DC.SCALE, engine="bsrgan")
    from ultralytics import YOLO
    m = YOLO(RUN)
    names = m.names
    det = {}
    for tag, (ns, jq) in (("clean", (0.0, None)), ("noisy", (NOISE, None))):
        rng = D.arm_rng(stem, "fig1_" + ("left" if tag == "clean" else "right"))
        lr = DC.degrade_param(hr, DC.SCALE, BLUR, ns, jq, rng, shot=False, resample="area")
        if lr.shape[:2] != (DC.LR, DC.LR):
            lr = cv2.resize(lr, (DC.LR, DC.LR), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DST, f"lr_{tag}.png"), to_u8(lr))

        sr = eng.upscale(lr)
        if sr.shape[:2] != (DC.BASE, DC.BASE):
            sr = cv2.resize(sr, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)
        sr = to_u8(sr)
        cv2.imwrite(os.path.join(DST, f"sr_{tag}.png"), sr[y0:y0 + CROP, x0:x0 + CROP])

        r = m.predict(sr, imgsz=DC.BASE, conf=CONF, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        # 只保留落在裁窗内的框（裁到窗口边缘）
        keep = (xyxy[:, 2] > x0) & (xyxy[:, 0] < x0 + CROP) & \
               (xyxy[:, 3] > y0) & (xyxy[:, 1] < y0 + CROP)
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

        vis = sr[y0:y0 + CROP, x0:x0 + CROP].copy()
        for b, cf in zip(xyxy, conf):
            p1 = (int(round(b[0] - x0)), int(round(b[1] - y0)))
            p2 = (int(round(b[2] - x0)), int(round(b[3] - y0)))
            cv2.rectangle(vis, p1, p2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.rectangle(vis, p1, p2, (60, 30, 200), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(DST, f"det_{tag}.png"), vis)
        det[tag] = dict(n_boxes=int(len(xyxy)), conf_mean=float(conf.mean()) if len(conf) else 0.0,
                        conf_max=float(conf.max()) if len(conf) else 0.0,
                        top=[f"{names[int(c)]} {float(f):.2f}"
                             for c, f in sorted(zip(cls, conf), key=lambda t: -t[1])[:3]])
        print(f"  [{tag}] 框 {len(xyxy)} 个，均值 conf {det[tag]['conf_mean']:.2f}")

    with open(os.path.join(DST, "det.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(ckpt=RUN, conf_thr=CONF, crop=dict(x0=x0, y0=y0, size=CROP),
                       det=det), fh, ensure_ascii=False, indent=1)
    print("[done] 素材在", DST)


if __name__ == "__main__":
    main()
