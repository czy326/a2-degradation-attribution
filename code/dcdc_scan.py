"""
DCDC 步骤 2 —— 退化强度响应面扫描（冻结检测器 + fp_bg 代理）

设计要点
  * 冻结检测器 = A2 的 `a2_HR_oracle` best.pt（在**无退化 HR** 上训练，
    独立于退化参数空间，全程不更新）⇒ 切断共适应
  * 扫描集取自 **eval 子集**（files[n_tr:]），检测器训练时未见过 ⇒ 无记忆效应
  * 每臂用 `deg_ops.arm_rng(stem, arm_tag)` 逐图固定种子 ⇒ 可复现
  * 粗筛（200 图）→ top-K → 全量复筛（701 图）

输出
  a2/dcdc_scan.json ：每臂的 fp_bg / fp_bg_hi / AP / PSNR，及 λ* 与复筛结果

运行（WSL, sr4sod）
  python dcdc_scan.py --timing            # 只跑 2 臂测速度
  python dcdc_scan.py                     # 全量（27 臂粗筛 + top6 复筛）
"""
import os
import sys
import io
import json
import glob
import time
import argparse
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)
import deg_ops as D                                    # noqa: E402
from probe0_common import (SIZE_EDGES, IOU_THRESHOLDS, bin_label,   # noqa: E402
                           size_bin, box_short_side, read_yolo, iou_batch,
                           ap_all_points, save_json)

HR_IMG = f"{SR4SOD}/_data/val/val/images"
# ★ 标签目录可用环境变量覆盖（用于 AI-TOD v1 vs v2 的稳健性检查）
#   默认仍是 v1（`_yolo/labels/val`），保持历史结论可比。
HR_LBL = os.environ.get("A2_LABEL_DIR", f"{SR4SOD}/_yolo/labels/val")
VIEW_ROOT = os.environ.get("A2_VIEW_ROOT", "/home/czy/data/sr4sod/a2_views")
RUN_ROOT = os.environ.get("A2_RUN_ROOT", "/home/czy/data/sr4sod/a2_runs")
FROZEN = os.path.join(RUN_ROOT, "a2_HR_oracle", "weights", "best.pt")

BASE = 800
SCALE = 4
LR = BASE // SCALE
POOL = 2804
TRAIN_FRAC = 0.75
N_TR = int(round(POOL * TRAIN_FRAC))          # 2103
MAX_DET = 1000
CONF = 0.001
CONF_HI = 0.25

# ---- 退化网格（物理合理范围，参考 BSRGAN / Real-ESRGAN 常用区间）----
BLUR_GRID = [0.5, 1.0, 1.5]        # 高斯模糊 sigma（HR 像素）
NOISE_GRID = [0.002, 0.008, 0.020]  # 传感器噪声 sigma（归一化 [0,1]）
JPEG_GRID = [95, 70, 40]            # JPEG 质量；None 表示不做 JPEG


def arm_tag(b, n, q):
    """(0,0,None) = 纯 INTER_AREA，即 A2 的最优臂 D0 ⇒ 干净锚点。

    没有它，结果无法解释：若 λ* 的 fp_bg 不低于 D0，则说明"最简单就是最好"，
    标定没有增益。有了它才能判断"感知标定是否真的优于不做任何退化处理"。
    """
    if b == 0 and n == 0 and q is None:
        return "D0_anchor"
    return f"b{b:g}_n{n:g}_q{('none' if q is None else q)}"


def _corr_noise(shape, sigma, corr_px, rng):
    """空间相关高斯噪声。

    `corr_px=0` ⇒ 逐像素独立（iid 读出噪声）
    `corr_px>0` ⇒ 先生成 iid，再做高斯低通使噪声在空间上相关，最后
                   重新归一化到目标 sigma（保证与 iid 族**强度可比**）

    用途：模拟**云/大气/气溶胶**这类**空间结构退化**。它与 iid 噪声在频域上
    完全不同（能量集中在低频），但可以让 PSNR 相同 ⇒ 是检验
    「PSNR 不充分」是否在**空间结构维度**同样成立的干净对照，
    且直接对上「感知驱动退化建模」里关心的云遮挡场景。
    """
    import cv2
    if corr_px is None or corr_px <= 0:
        return rng.normal(0, float(sigma), shape).astype(np.float32)
    n = rng.normal(0, 1, shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), float(corr_px), borderType=cv2.BORDER_REFLECT)
    s = float(n.std())
    return (n / max(s, 1e-9) * float(sigma)).astype(np.float32)


def degrade_param(img, scale, blur, noise, jpeg, rng, shot=True, resample="area",
                  noise_corr=0.0):
    """参数化退化：模糊 -> 重采样降采样 -> 噪声 -> JPEG（顺序固定，与 D2 同族）。

    ★ `noise_corr`（2026-10-15 加）
      噪声的空间相关长度（LR 像素）。0 = iid；>0 = 空间相关（模拟云/大气结构）。
      **仅在 `shot=False` 时生效**（shot=True 保留原物理噪声模型）。
      重新归一化到同一 sigma，保证与 iid 族强度可比。

    ★ `resample` 开关（2026-10-15 加）
        "area" = INTER_AREA（传感器像素积分，A2 的 D0）
        "cubic"= INTER_CUBIC（经典非保真代理，A2 的 D1）
        "linear"
      A2 中 D0 与 D1 的 PSNR 差 0.67 dB ⇒ 从未在 PSNR 匹配下比过。

    ★ `shot` 开关（2026-10-15 加）
      原本散粒噪声被 `noise > 0` 门控 ⇒ 常数底噪 σ≈0.053 淹没整个噪声网格
      （实测 noise 0.001→0.128 只让 PSNR 变 0.10 dB）⇒ 受控实验须 `shot=False`。
    """
    import cv2
    _ITP = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC,
            "linear": cv2.INTER_LINEAR}
    x = D._as_float(img)
    if blur and blur > 0:
        x = D._gauss_blur(x, float(blur))
    h, w = x.shape[:2]
    x = cv2.resize(x, (w // scale, h // scale),
                   interpolation=_ITP.get(resample, cv2.INTER_AREA))
    if noise and noise > 0:
        if shot:
            r = rng.normal(0, float(noise), x.shape).astype(np.float32)
            s = (rng.normal(0, 1, x.shape).astype(np.float32)
                 * np.sqrt(D.SHOT_K * np.maximum(x, 1e-6)))
            x = np.clip(x + s + r, 0, 1)
        else:
            x = np.clip(x + _corr_noise(x.shape, noise, noise_corr, rng), 0, 1)
    x = D._to_uint8(x)
    if jpeg:
        x = D.jpeg_roundtrip(x, int(jpeg))
    return x


def eval_arm(model, files, lbl_dir, blur, noise, jpeg, eng, tag_base,
             want_psnr=True, shot=True, resample="area", noise_corr=0.0):
    """在一个 (blur,noise,jpeg[,resample,noise_corr]) 臂上评估：分桶 AP / PSNR。

    `shot=False`     ⇒ 纯加性高斯噪声（见 degrade_param）
    `resample=`      ⇒ 下采样核类型 area/cubic/linear
    `noise_corr=`    ⇒ 噪声空间相关长度（LR 像素），>0 模拟云/大气结构
    """
    import cv2
    nb = len(SIZE_EDGES) - 1
    per_bin = {b: [] for b in range(nb)}
    n_box = np.zeros(nb, dtype=np.int64)
    fp_bg = n_pred = n_gt = 0
    fp_hi = n_pred_hi = 0
    ps = []
    n_img = 0

    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        hr = cv2.imread(p, cv2.IMREAD_COLOR)
        if hr is None:
            continue
        if hr.shape[0] != BASE or hr.shape[1] != BASE:
            hr = cv2.resize(hr, (BASE, BASE), interpolation=cv2.INTER_AREA)
        rng = D.arm_rng(stem, tag_base)
        lr = degrade_param(hr, SCALE, blur, noise, jpeg, rng, shot=shot,
                           resample=resample, noise_corr=noise_corr)
        if lr.shape[0] != LR or lr.shape[1] != LR:
            lr = cv2.resize(lr, (LR, LR), interpolation=cv2.INTER_AREA)
        sr = eng.upscale(lr)
        if sr.shape[0] != BASE or sr.shape[1] != BASE:
            sr = cv2.resize(sr, (BASE, BASE), interpolation=cv2.INTER_AREA)

        gt = read_yolo(os.path.join(lbl_dir, stem + ".txt"), BASE, BASE)
        if not gt:
            continue
        gb = np.array([[g[0], g[1], g[2], g[3]] for g in gt], dtype=np.float32)
        bidx = np.array([size_bin(box_short_side(b)) for b in gb], dtype=np.int64)

        res = model.predict(sr, imgsz=BASE, conf=CONF, iou=0.7,
                            max_det=MAX_DET, verbose=False, device=0)
        pb = res[0].boxes
        if pb is not None and len(pb):
            P = pb.xyxy.cpu().numpy().astype(np.float32)
            S = pb.conf.cpu().numpy().astype(np.float32)
        else:
            P = np.zeros((0, 4), dtype=np.float32)
            S = np.zeros(0, dtype=np.float32)

        n_img += 1
        n_pred += len(P)
        n_gt += len(gb)
        if len(P) and len(gb):
            fp_bg += int((iou_batch(P, gb).max(axis=1) < 0.1).sum())
        elif len(P):
            fp_bg += len(P)
        if len(P):
            hi = S >= CONF_HI
            n_pred_hi += int(hi.sum())
            if len(gb):
                bad = iou_batch(P, gb).max(axis=1) < 0.1
                fp_hi += int((bad & hi).sum())
            else:
                fp_hi += int(hi.sum())
        for b in range(nb):
            m = bidx == b
            if m.sum() == 0:
                continue
            n_box[b] += int(m.sum())
            per_bin[b].append((ap_all_points(P, S, gb[m], 0.50),
                               float(np.mean([ap_all_points(P, S, gb[m], t)
                                              for t in IOU_THRESHOLDS]))))
        if want_psnr:
            ps.append(float(10 * np.log10(255.0 ** 2 / max(np.mean((sr.astype(np.float64)
                                                                   - hr.astype(np.float64)) ** 2), 1e-12))))

    bins = {}
    for b in range(nb):
        v = per_bin[b]
        if n_box[b] < 150 or not v:
            bins[bin_label(b)] = dict(n=int(n_box[b]), a5095=None)
            continue
        arr = np.array(v, dtype=np.float64)
        bins[bin_label(b)] = dict(n=int(n_box[b]), a5095=float(arr[:, 1].mean()))
    vals = [d["a5095"] for d in bins.values() if d.get("a5095") is not None]
    ses = []
    return dict(n_images=n_img, n_gt=int(n_gt), n_pred=int(n_pred),
                fp_bg_rate=fp_bg / max(n_pred, 1),
                fp_bg_rate_hi=fp_hi / max(n_pred_hi, 1),
                psnr=(float(np.mean(ps)) if ps else None),
                mAP5095=(float(np.mean(vals)) if vals else None),
                bins=bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true", help="只跑 2 臂测速度")
    ap.add_argument("--scan-n", type=int, default=200)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--max-arms", type=int, default=0, help="0=全网格")
    args = ap.parse_args()

    if not os.path.isfile(FROZEN):
        raise SystemExit(f"冻结检测器缺失: {FROZEN}")

    files = sorted(glob.glob(os.path.join(HR_IMG, "*.png")))[:POOL]
    if len(files) < POOL:
        print(f"[warn] 只找到 {len(files)} 张（期望 {POOL}）")
    N_TR = int(round(len(files) * TRAIN_FRAC))
    eval_files = files[N_TR:]                 # 检测器训练时未见过
    scan_files = eval_files[:args.scan_n]
    print(f"eval 子集 {len(eval_files)} 张；粗筛用前 {len(scan_files)} 张", flush=True)
    print(f"冻结检测器: {FROZEN}", flush=True)

    from ultralytics import YOLO
    from probe0_sr import RealESRGAN
    model = YOLO(FROZEN)
    eng = RealESRGAN(SCALE, engine="bsrgan")
    print(f"SR 引擎: bsrgan x{SCALE}\n", flush=True)

    # 第 1 臂 = 干净锚点（纯 INTER_AREA，无模糊/噪声/JPEG）
    grid = [(0.0, 0.0, None)] + [(b, n, q) for b in BLUR_GRID
                                 for n in NOISE_GRID for q in JPEG_GRID]
    if args.timing:
        grid = grid[:3]
    if args.max_arms:
        grid = grid[:args.max_arms]

    print(f"粗筛网格: {len(grid)} 臂", flush=True)

    # ---- 断点续跑：每臂完成立即追加一行 JSON（JSON Lines）----
    # 动机：2026-10-15 实测 WSL 会在空闲时被回收重启（/tmp 清空、uptime 归零）。
    # 原实现只在最后写一次 dcdc_scan.json ⇒ 中途重启会丢掉全部 27 臂。
    ckpt = os.path.join(A2, "dcdc_coarse.jsonl")
    done = {}
    if os.path.exists(ckpt):
        for ln in open(ckpt, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                done[r["tag"]] = r
            except Exception:
                pass
        print(f"[resume] 已有 {len(done)} 臂缓存，将跳过", flush=True)

    print(f"{'arm':<26}{'fp_bg':>8}{'fp_hi':>8}{'PSNR':>8}{'mAP':>9}{'sec':>7}", flush=True)
    coarse = []
    t_all = time.time()
    for (b, n, q) in grid:
        tag = arm_tag(b, n, q)
        if tag in done:
            r = done[tag]
            coarse.append(r)
            ps = f"{r['psnr']:.2f}" if r.get("psnr") else "n/a"
            mp = f"{r['mAP5095']:.4f}" if r.get("mAP5095") is not None else "n/a"
            print(f"{tag:<26}{r['fp_bg_rate']:>8.3f}{r['fp_bg_rate_hi']:>8.3f}"
                  f"{ps:>8}{mp:>9}{'cached':>7}", flush=True)
            continue
        t0 = time.time()
        r = eval_arm(model, scan_files, HR_LBL, b, n, q, eng, tag)
        dt = time.time() - t0
        r.update(blur=b, noise=n, jpeg=q, tag=tag)
        coarse.append(r)
        with open(ckpt, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        ps = f"{r['psnr']:.2f}" if r["psnr"] else "n/a"
        mp = f"{r['mAP5095']:.4f}" if r["mAP5095"] is not None else "n/a"
        print(f"{tag:<26}{r['fp_bg_rate']:>8.3f}{r['fp_bg_rate_hi']:>8.3f}{ps:>8}{mp:>9}{dt:>7.1f}",
              flush=True)

    print(f"\n粗筛总耗时 {time.time()-t_all:.0f}s；每臂均值 {(time.time()-t_all)/len(grid):.1f}s",
          flush=True)

    if args.timing:
        est_full = (time.time() - t_all) / len(grid) * 27
        est_ref = (time.time() - t_all) / len(grid) * args.topk * (len(eval_files) / max(len(scan_files), 1))
        print(f"\n[timing] 27 臂粗筛 ≈ {est_full/60:.1f} min；top{args.topk} 全量复筛 ≈ {est_ref/60:.1f} min")
        save_json(os.path.join(A2, "dcdc_timing.json"),
                  dict(per_arm_sec=(time.time() - t_all) / len(grid),
                       est_coarse_min=est_full / 60, est_refine_min=est_ref / 60,
                       scan_n=len(scan_files), eval_n=len(eval_files)))
        return

    # ---- top-K 全量复筛 ----
    cand = sorted([r for r in coarse if r["fp_bg_rate"] is not None],
                  key=lambda r: r["fp_bg_rate"])[:args.topk]
    print(f"\n=== top{len(cand)} 全量复筛（{len(eval_files)} 张）===", flush=True)
    print(f"{'arm':<26}{'fp_bg':>8}{'fp_bg(粗)':>11}{'PSNR':>8}{'mAP':>9}", flush=True)
    ck2 = os.path.join(A2, "dcdc_refined.jsonl")
    done2 = {}
    if os.path.exists(ck2):
        for ln in open(ck2, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r2 = json.loads(ln)
                done2[r2["tag"]] = r2
            except Exception:
                pass
        if done2:
            print(f"[resume] 复筛已有 {len(done2)} 臂缓存", flush=True)
    refined = []
    for r in cand:
        if r["tag"] in done2:
            rr = done2[r["tag"]]
            refined.append(rr)
            print(f"{rr['tag']:<26}{rr['fp_bg_rate']:>8.4f}"
                  f"{rr.get('fp_bg_coarse', float('nan')):>11.4f}"
                  f"{(rr['psnr'] if rr.get('psnr') else float('nan')):>8.2f}"
                  f"{(rr['mAP5095'] if rr.get('mAP5095') is not None else float('nan')):>9.4f}",
                  flush=True)
            continue
        rr = eval_arm(model, eval_files, HR_LBL, r["blur"], r["noise"], r["jpeg"],
                      eng, r["tag"])
        rr.update(blur=r["blur"], noise=r["noise"], jpeg=r["jpeg"], tag=r["tag"],
                  fp_bg_coarse=r["fp_bg_rate"])
        refined.append(rr)
        with open(ck2, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
            fh.flush()
        ps = f"{rr['psnr']:.2f}" if rr["psnr"] else "n/a"
        mp = f"{rr['mAP5095']:.4f}" if rr["mAP5095"] is not None else "n/a"
        print(f"{rr['tag']:<26}{rr['fp_bg_rate']:>8.4f}{rr['fp_bg_coarse']:>11.4f}{ps:>8}{mp:>9}",
              flush=True)

    best = min(refined, key=lambda r: r["fp_bg_rate"])
    print(f"\n★ λ* (argmin fp_bg) = {best['tag']}  blur={best['blur']} noise={best['noise']} "
          f"jpeg={best['jpeg']}  fp_bg={best['fp_bg_rate']:.4f}  mAP={best['mAP5095']}")

    # 相关性：fp_bg 是否仍强于 PSNR
    import itertools
    m = np.array([r["mAP5095"] for r in refined if r["mAP5095"] is not None])
    f = np.array([r["fp_bg_rate"] for r in refined if r["mAP5095"] is not None])
    p = np.array([r["psnr"] for r in refined if r["mAP5095"] is not None and r["psnr"]])
    if len(m) >= 3:
        print(f"  r(fp_bg, mAP) = {np.corrcoef(f, m)[0,1]:+.3f}   "
              f"r(PSNR, mAP) = {np.corrcoef(p, m)[0,1]:+.3f}   (n={len(m)})")

    save_json(os.path.join(A2, "dcdc_scan.json"),
              dict(frozen=FROZEN, scan_n=len(scan_files), eval_n=len(eval_files),
                   grid=dict(blur=BLUR_GRID, noise=NOISE_GRID, jpeg=JPEG_GRID),
                   coarse=coarse, refined=refined,
                   lambda_star=dict(tag=best["tag"], blur=best["blur"],
                                    noise=best["noise"], jpeg=best["jpeg"],
                                    fp_bg=best["fp_bg_rate"], mAP=best["mAP5095"])))
    print(f"\nwrote dcdc_scan.json")


if __name__ == "__main__":
    main()
