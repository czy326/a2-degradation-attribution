"""
A2 探针 —— 退化管线的改进，是否真的改善真实场景的下游性能？

设计 v2：E:\\新方向研究\\a2_design_v2.md（含 §3.1 D5 移除记录）

流水线（每个条件内部再分 train/eval，只报 held-out 数字）
    HR (AI-TOD 800x800, 2804 张)
      -> [deg_ops.degrade(arm, x4)]        <- 唯一变量：退化算子
      -> [BSRGAN x4 重建]                   <- 固定（P1b 证明它伤害最小）
      -> YOLOv8n 训练 + 分桶 AP 评测

七个条件
    D0_interarea / D1_bicubic / D2_highorder / D3_shuffled / D4_physical
    HR_oracle  上界：不退化，直接检测
    LR_nosr    下界：退化后不重建，按自身尺寸检测

预注册判据（v2）
    健全性  mAP(HR_oracle) > mAP(LR_nosr) 且 D0 vs D1 有差异，否则探针无分辨力
    G1  Δ_deg < Δ_eng  -> 退化管线的选择，其影响小于引擎保真度的选择
                          => 综述 §7.1 的建议收益可能被引擎选择淹没
    G2  D4_physical 显著优于 D2/D3  -> 物理动机退化优于启发式，综述方向正确
    G4  Δ_deg 小但 Δ_psnr 大（或臂排序相反）-> 重建质量与下游脱钩（复现 D1 规律）

运行（WSL，sr4sod 环境）
    python a2_pipeline.py --plan
    python a2_pipeline.py           # 全跑；可续跑（每个条件一个 JSON）
"""
import os
import sys
import io
import json
import glob
import time
import shutil
import argparse
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)

# deg_ops 会把 sys.stdout 包成 UTF-8；这里不要再包一次（否则 I/O on closed file）
import deg_ops as D                                    # noqa: E402
from probe0_common import (SIZE_EDGES, IOU_THRESHOLDS, bin_label,   # noqa: E402
                           size_bin, box_short_side, read_yolo, iou_batch,
                           ap_all_points, save_json)

# --------------------------------------------------------------------------
HR_IMG = f"{SR4SOD}/_data/val/val/images"
HR_LBL = f"{SR4SOD}/_yolo/labels/val"          # 2804 份，归一化 => 跨尺度通用
# drvfs 写入只有 ~3 MB/s，视图与训练产物放 WSL 本地盘
VIEW_ROOT = os.environ.get("A2_VIEW_ROOT", "/home/czy/data/sr4sod/a2_views")
RUN_ROOT = os.environ.get("A2_RUN_ROOT", "/home/czy/data/sr4sod/a2_runs")
PARTS = os.path.join(A2, "parts")

BASE = 800
SCALE = 4
LR = BASE // SCALE          # 200

FACTOR = 4
#                 tag            deg arm          do_sr  imgsz
CONDITIONS = [
    ("HR_oracle",    None,              False, BASE),
    ("D0_interarea", "D0_interarea",    True,  BASE),
    ("D1_bicubic",   "D1_bicubic",      True,  BASE),
    ("D2_highorder", "D2_highorder",    True,  BASE),
    ("D3_shuffled",  "D3_shuffled",     True,  BASE),
    ("D4_physical",  "D4_physical",     True,  BASE),
    # imgsz 必须是 32 的倍数，否则 ultralytics 会把 200 静默提到 224 并刷警告；
    # 这里直接写 224，与 P0-B 的 lr_native 实际行为一致。
    ("LR_nosr",      "D0_interarea",    False, 224),
]
ARMS5 = [c[0] for c in CONDITIONS if c[0].startswith("D")]
MIN_N = 150
MAX_DET = 1000


# --------------------------------------------------------------------------
def psnr(a, b):
    """a,b: uint8 HxWx3. 返回 dB；完全相同返回 inf。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = np.mean((a - b) ** 2)
    if m <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / m))


def ssim_gray(a, b, win=11):
    """极简 SSIM（单通道灰度，Gaussian 加权）。够用作相对比较。"""
    import cv2
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(win, 1.5)
    ker = k @ k.T
    mu_a = cv2.filter2D(ga, -1, ker)
    mu_b = cv2.filter2D(gb, -1, ker)
    sa = cv2.filter2D(ga * ga, -1, ker) - mu_a ** 2
    sb = cv2.filter2D(gb * gb, -1, ker) - mu_b ** 2
    sab = cv2.filter2D(ga * gb, -1, ker) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


# --------------------------------------------------------------------------
def build_views(tag, deg_arm, do_sr, pool, train_frac, eng, jobs=8):
    """生成一个条件的 train/val 视图。返回 dict(root, yaml, ...)。"""
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    files = sorted(glob.glob(os.path.join(HR_IMG, "*.png")))
    if pool and pool > 0:
        files = files[:pool]
    n_tr = max(1, int(round(len(files) * train_frac)))
    root = os.path.join(VIEW_ROOT, f"a2_{tag}")
    for part in ("train", "val"):
        os.makedirs(os.path.join(root, "images", part), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", part), exist_ok=True)

    # SR 网络独占 GPU => 单线程；其余 arm 是元数据密集 => 多线程
    n_jobs = 1 if do_sr else max(1, int(jobs))
    stats = {"size_fix": 0, "m1": {}}
    m1_lock = []

    def one(idx_p):
        idx, p = idx_p
        stem = os.path.splitext(os.path.basename(p))[0]
        part = "train" if idx < n_tr else "val"
        hr = cv2.imread(p, cv2.IMREAD_COLOR)
        if hr is None:
            return 0
        if hr.shape[0] != BASE or hr.shape[1] != BASE:
            hr = cv2.resize(hr, (BASE, BASE), interpolation=cv2.INTER_AREA)

        if deg_arm is None:
            out = hr
            lr = None
        else:
            lr = D.degrade(hr, deg_arm, SCALE, D.arm_rng(stem, deg_arm))
            if lr.shape[0] != LR or lr.shape[1] != LR:
                lr = cv2.resize(lr, (LR, LR), interpolation=cv2.INTER_AREA)
                stats["size_fix"] += 1
            if do_sr:
                out = eng.upscale(lr) if eng is not None else lr
                if out.shape[0] != BASE or out.shape[1] != BASE:
                    out = cv2.resize(out, (BASE, BASE), interpolation=cv2.INTER_AREA)
            else:
                out = lr

        di = os.path.join(root, "images", part)
        dl = os.path.join(root, "labels", part)
        cv2.imwrite(os.path.join(di, stem + ".png"), out)
        lp = os.path.join(HR_LBL, stem + ".txt")
        if os.path.isfile(lp):
            shutil.copyfile(lp, os.path.join(dl, stem + ".txt"))
        # M1 描述子只在 eval 部分、只在 LR 上算（配对分析用）
        if part == "val" and lr is not None:
            m = D.fidelity_metrics(lr)
            m1_lock.append((stem, m["spectral_slope"], m["noise_ac_len"]))
        return 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        n_done = sum(ex.map(one, list(enumerate(files))))
    dt = time.time() - t0

    yml = os.path.join(root, "data.yaml")
    with open(yml, "w", encoding="utf-8") as fh:
        fh.write(f"path: {root}\ntrain: images/train\nval: images/val\n"
                 f"nc: 1\nnames: ['object']\n")
    return dict(root=root, yaml=yml, tag=tag, n_train=n_tr,
                n_val=len(files) - n_tr, dt=dt, n_done=n_done,
                size_fix=stats["size_fix"],
                eval_dir=os.path.join(root, "images", "val"),
                m1_eval=m1_lock)


# --------------------------------------------------------------------------
def evaluate(model, img_dir, img_size, min_n=MIN_N, max_det=MAX_DET,
             hr_dir=None):
    """分桶 AP（逐图后平均）+ 背景 FP + M3 重建质量（可选）。"""
    import cv2
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    lbl_dir = os.path.join(os.path.dirname(os.path.dirname(img_dir)),
                           "labels", os.path.basename(img_dir))
    nb = len(SIZE_EDGES) - 1
    per_bin = {b: [] for b in range(nb)}
    n_box = np.zeros(nb, dtype=np.int64)
    fp_bg = n_img = n_pred = n_gt = 0
    fp_hi = n_pred_hi = 0
    CONF_HI = 0.25
    ps, ss = [], []

    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo(os.path.join(lbl_dir, stem + ".txt"), w, h)
        if not gt:
            continue
        gb = np.array([[g[0], g[1], g[2], g[3]] for g in gt], dtype=np.float32)
        bidx = np.array([size_bin(box_short_side(b)) for b in gb], dtype=np.int64)

        res = model.predict(img, imgsz=int(img_size), conf=0.001, iou=0.7,
                            max_det=int(max_det), verbose=False, device=0)
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
            a50 = ap_all_points(P, S, gb[m], 0.50)
            a5095 = float(np.mean([ap_all_points(P, S, gb[m], t)
                                   for t in IOU_THRESHOLDS]))
            per_bin[b].append((a50, a5095))

        # M3：检测器实际看到的输入（统一 resize 到 BASE）与 HR 的差距
        if hr_dir is not None:
            hp = os.path.join(hr_dir, stem + ".png")
            if os.path.isfile(hp):
                hr = cv2.imread(hp, cv2.IMREAD_COLOR)
                if hr is not None and hr.shape[:2] != (BASE, BASE):
                    hr = cv2.resize(hr, (BASE, BASE), interpolation=cv2.INTER_AREA)
                vis = img if img.shape[:2] == (BASE, BASE) else \
                    cv2.resize(img, (BASE, BASE), interpolation=cv2.INTER_CUBIC)
                if hr is not None:
                    ps.append(psnr(vis, hr))
                    ss.append(ssim_gray(vis, hr))

    out = dict(n_images=n_img, n_gt=int(n_gt), n_pred=int(n_pred),
               fp_bg_rate=(fp_bg / max(n_pred, 1)),
               fp_bg_rate_hi=(fp_hi / max(n_pred_hi, 1)),
               img_size=int(img_size), bins={})
    if ps:
        out["psnr_mean"] = float(np.mean(ps))
        out["ssim_mean"] = float(np.mean(ss))
        out["psnr_n"] = int(len(ps))
    for b in range(nb):
        v = per_bin[b]
        if n_box[b] < min_n or not v:
            out["bins"][bin_label(b)] = dict(n=int(n_box[b]), n_img=len(v),
                                             a050=None, a5095=None)
            continue
        arr = np.array(v, dtype=np.float64)
        out["bins"][bin_label(b)] = dict(
            n=int(n_box[b]), n_img=len(v),
            a050=float(arr[:, 0].mean()), a5095=float(arr[:, 1].mean()),
            a050_se=float(arr[:, 0].std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            a5095_se=float(arr[:, 1].std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None)
    return out


# --------------------------------------------------------------------------
def judge(results):
    """预注册判据 G1 / G2 / G4。"""
    by = {r["config"]["tag"]: r for r in results}
    need = ["HR_oracle", "LR_nosr"] + ARMS5
    miss = [t for t in need if t not in by]
    if miss:
        return dict(verdict=f"INCOMPLETE -> missing {miss}", rows=[])

    def mAP(r):
        v = [d["a5095"] for d in r["bins"].values() if d.get("a5095") is not None]
        return float(np.mean(v)) if v else float("nan")

    def mAP50(r):
        v = [d["a050"] for d in r["bins"].values() if d.get("a050") is not None]
        return float(np.mean(v)) if v else float("nan")

    rows = []
    for t in need:
        rows.append(dict(tag=t, map50=mAP50(by[t]), map5095=mAP(by[t]),
                         psnr=by[t].get("psnr_mean"), ssim=by[t].get("ssim_mean"),
                         fp=by[t]["fp_bg_rate"], n_img=by[t]["n_images"]))

    # ---- 健全性 ----
    sanity = (mAP(by["HR_oracle"]) > mAP(by["LR_nosr"]))
    d01 = abs(mAP(by["D0_interarea"]) - mAP(by["D1_bicubic"]))
    se01 = np.hypot(_se(by["D0_interarea"]), _se(by["D1_bicubic"]))
    anchor_ok = d01 > max(se01, 0.0)

    # ---- Δ_deg（五臂极差）与 Δ_psnr ----
    arm_map = [mAP(by[t]) for t in ARMS5]
    arm_psnr = [by[t].get("psnr_mean") for t in ARMS5]
    arm_psnr = [x for x in arm_psnr if x is not None and np.isfinite(x)]
    d_deg = float(max(arm_map) - min(arm_map))
    d_psnr = float(max(arm_psnr) - min(arm_psnr)) if len(arm_psnr) >= 2 else float("nan")

    # ---- Δ_eng：来自 P1b 产物（BSRGAN vs RealESRGAN，f=4，六桶均值差）----
    d_eng = 0.0108        # 实测：f=4 时换引擎带来的六桶均值 ΔAP50-95 改善
    d_eng_f2 = 0.0055     # f=2 的数值，一并报告

    g1 = d_deg < d_eng
    best = ARMS5[int(np.nanargmax(arm_map))]
    g2 = (best == "D4_physical")
    g4 = (np.isfinite(d_psnr) and d_psnr > 0.5 and d_deg < d_eng)

    if not sanity or not anchor_ok:
        v = ("PROBE INVALID -> 健全性未过（HR_oracle<=LR_nosr 或 D0/D1 无差异）。"
             "任务对退化不敏感，结论作废，不是判死。")
    elif g1:
        v = (f"G1 -> 退化管线的收益 Δ_deg={d_deg:.4f} < 引擎收益 Δ_eng={d_eng:.4f}。"
             f"综述 §7.1「构建真实退化模型」的收益可能被引擎选择淹没。")
        if g4:
            v += f" 且 G4 成立（Δ_psnr={d_psnr:.2f} dB 但 Δ_deg 小）= 重建质量与下游脱钩。"
    elif g2:
        v = (f"G2 -> 最优臂是 {best}（物理动机），Δ_deg={d_deg:.4f} >= Δ_eng={d_eng:.4f}，"
             f"综述 §7.1 的建议成立。")
    else:
        v = (f"G2 不成立 -> 最优臂是 {best}，不是 D4_physical；Δ_deg={d_deg:.4f}，"
             f"Δ_eng={d_eng:.4f}。")

    return dict(rows=rows, sanity=bool(sanity), anchor_ok=bool(anchor_ok),
                d01=d01, se01=float(se01), d_deg=d_deg, d_psnr=d_psnr,
                d_eng=d_eng, d_eng_f2=d_eng_f2, best_arm=best,
                g1=bool(g1), g2=bool(g2), g4=bool(g4), verdict=v)


def _se(r):
    """★ 六桶均值 mAP50-95 的标准误。

    BUG 修正（首跑的判据实现错了）：原实现把六个桶的 AP 当独立样本算 `std/sqrt(n)`，
    但桶与桶之间 AP 差异是**真实的**（16-24 px 桶天然高于 2-4 px 桶），不是噪声。
    真正的噪声是**逐图** SE（每个 bin 里的 `a5095_se`）。跨桶合成应取各 bin 逐图 SE 的
    平方和开根 / 桶数。这样才能和"两条件均值之差"同量纲地比较。

    量级校验：逐图 SE 合成 ≈ 0.0042~0.0048（原错误实现给出 0.037，大了近 9 倍，
    把本来 2.1σ 的 D0/D1 差异误判成"无差异"，进而把整个探针误判为 INVALID。**

    """
    ses = [d.get("a5095_se") for d in r["bins"].values()
           if d.get("a5095") is not None and d.get("a5095_se") is not None]
    if len(ses) < 2:
        return float("nan")
    return float(np.sqrt(np.sum(np.square(ses))) / len(ses))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--pool", type=int, default=2804)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=BASE)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑这些条件（默认全部）")
    args = ap.parse_args()

    os.makedirs(VIEW_ROOT, exist_ok=True)
    os.makedirs(RUN_ROOT, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)

    conds = CONDITIONS
    if args.only:
        conds = [c for c in CONDITIONS if c[0] in args.only]
    n_tr = int(round(args.pool * args.train_frac))
    print("=== A2 condition matrix ===", flush=True)
    print(f"{'tag':<14}{'deg_arm':<14}{'SR':>4}{'imgsz':>7}{'train':>7}{'eval':>6}")
    for tag, deg_arm, do_sr, isz in conds:
        print(f"{tag:<14}{str(deg_arm):<14}{str(do_sr):>4}{isz:>7}"
              f"{n_tr:>7}{args.pool - n_tr:>6}")
    print(f"\nHR  {HR_IMG}")
    print(f"LBL {HR_LBL}")
    print(f"VIEW {VIEW_ROOT}\nRUN  {RUN_ROOT}")

    if args.plan:
        save_json(os.path.join(A2, "a2_plan.json"),
                  dict(conditions=[dict(tag=t, deg_arm=a, sr=s, imgsz=i)
                                   for t, a, s, i in conds], args=vars(args)))
        print("\nplan only; wrote a2_plan.json")
        return

    from ultralytics import YOLO
    from probe0_sr import fetch_weights, RealESRGAN

    ws = fetch_weights(scales=(), yolo=(args.model,))
    eng = RealESRGAN(SCALE, engine="bsrgan")
    print(f"\n[sr] engine = bsrgan x{SCALE}  weights {eng.weights_path}", flush=True)
    yolo_w = ws.get(args.model, args.model)

    for tag, deg_arm, do_sr, isz in conds:
        pp = os.path.join(PARTS, f"a2_{tag}.json")
        if os.path.exists(pp):
            print(f"\n### {tag}: already done, skipping", flush=True)
            continue
        print(f"\n### {tag}  (deg={deg_arm}, sr={do_sr}, imgsz={isz})", flush=True)
        v = build_views(tag, deg_arm, do_sr, args.pool, args.train_frac,
                        eng if do_sr else None, jobs=args.jobs)
        print(f"    view {v['tag']}: train={v['n_train']} eval={v['n_val']} "
              f"size_fix={v['size_fix']} ({v['dt']:.1f}s)", flush=True)

        model = YOLO(yolo_w)
        model.train(data=v["yaml"], epochs=args.epochs, imgsz=isz,
                    batch=args.batch, seed=args.seed, device=0,
                    project=RUN_ROOT, name=f"a2_{tag}",
                    exist_ok=True, verbose=False, plots=False)
        ev = evaluate(model, v["eval_dir"], isz, hr_dir=HR_IMG)
        ev.update(config=dict(tag=tag, deg_arm=deg_arm, do_sr=do_sr,
                              imgsz=isz, epochs=args.epochs, pool=args.pool,
                              train_frac=args.train_frac, seed=args.seed))
        ev.update(view=dict(tag=v["tag"], n_train=v["n_train"], n_val=v["n_val"],
                            size_fix=v["size_fix"]))
        ev["m1_eval"] = v["m1_eval"]
        save_json(pp, ev)
        brief = {k: (None if d["a5095"] is None else round(d["a5095"], 3))
                 for k, d in ev["bins"].items()}
        print(f"    saved a2_{tag}.json  AP50-95={brief}", flush=True)
        if "psnr_mean" in ev:
            print(f"    psnr={ev['psnr_mean']:.2f} dB  ssim={ev['ssim_mean']:.4f}",
                  flush=True)

    results = []
    for tag, *_ in CONDITIONS:
        pp = os.path.join(PARTS, f"a2_{tag}.json")
        if os.path.exists(pp):
            try:
                results.append(json.load(open(pp, encoding="utf-8")))
            except Exception as e:
                print(f"[aggregate] unreadable {pp}: {e}", flush=True)
    print(f"\n[aggregate] {len(results)}/{len(CONDITIONS)} conditions", flush=True)
    if len(results) < len(CONDITIONS):
        print("!! 条件不全，判决可能不完整", flush=True)

    vd = judge(results)
    print("\n=== A2 PRE-REGISTERED VERDICT ===")
    print(f"{'tag':<14}{'mAP50':>8}{'mAP50-95':>10}{'PSNR':>8}{'SSIM':>8}{'fp_bg':>8}")
    for r in vd.get("rows", []):
        ps = f"{r['psnr']:.2f}" if r["psnr"] is not None and np.isfinite(r["psnr"]) else "n/a"
        ss = f"{r['ssim']:.4f}" if r["ssim"] is not None else "n/a"
        print(f"{r['tag']:<14}{r['map50']:>8.4f}{r['map5095']:>10.4f}"
              f"{ps:>8}{ss:>8}{r['fp']:>8.3f}")
    if "d_deg" in vd:
        print(f"\nΔ_deg(五臂极差)={vd['d_deg']:.4f}   Δ_eng(换引擎,f=4)={vd['d_eng']:.4f}"
              f"  (f=2: {vd['d_eng_f2']:.4f})")
        print(f"Δ_psnr={vd['d_psnr']:.2f} dB   best_arm={vd['best_arm']}")
        print(f"健全性={vd['sanity']}  D0vsD1锚点={vd['anchor_ok']} "
              f"(d01={vd['d01']:.4f}, se={vd['se01']:.4f})")
    print(f"\n{vd['verdict']}")

    save_json(os.path.join(A2, "a2_result.json"),
              dict(conditions=[dict(tag=t, deg_arm=a, sr=s, imgsz=i)
                               for t, a, s, i in CONDITIONS],
                   results=results, verdict=vd, args=vars(args)))
    print(f"\nwrote a2_result.json")


if __name__ == "__main__":
    main()
