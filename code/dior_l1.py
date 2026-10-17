"""DIOR L1 门控 —— 判断「AI-TOD 的 800/4× 协议」在 DIOR 上还测不测得出主终点。

上游
  `SCALEUP_GATE.md` §3（L1 定义）｜ `L0_VERDICT.md`（L0 通过）｜
  `DIOR_L1_PREREG.md`（★ 判据在结果产生前已写死，本文件只实现，不解释）

做什么（2 臂 × 1 种子 × 16 条件 + 复现对照）
  ① 建两个视图：A = 无退化（blur 0，无噪声/JPEG）；B0 = 窄模糊 U(0.00,0.05) + 噪声 + JPEG
  ② 各训 1 次 YOLOv8n（30 epoch，seed 2026）
  ③ 在**同一批退化图**上评 16 条件（两臂共用 tag ⇒ 逐图配对）
  ④ ref 条件额外换 2 个 tag_base 各评一次 ⇒ 估计**退化实现噪声**（地板）
  ⑤ 按预注册判据出判决

为什么必须复用主实验的函数而不是重写
  * `tv_sweep.degrade_span` / `make_engine` ⇒ 退化与 SR 与主实验**逐字相同**
  * `dcdc_scan.eval_arm`           ⇒ mAP 口径与主实验**逐字相同**（含 float32 短边分桶）
  重写一份必然漂移，而 L1 的全部意义就是"协议是否可移植"。

运行（WSL, 需 GPU）
  python dior_l1.py --prep            # 只建视图（不训练/不评测）
  python dior_l1.py                   # 全流程（可断点续跑）
  python dior_l1.py --verdict-only    # 只用已有 jsonl 出判决
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)

# ★★ stdout 修补（2026-10-16 实测踩到，见 DIOR_L1_PREREG.md §7.5）
#   `deg_ops.py:23` 与 `probe0_sr.py:24` **都**在 import 时执行
#       sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
#   而两处都**不保留旧 wrapper 的引用**。于是第二个被 import 时，
#   第一个 wrapper 的引用计数归零 ⇒ 被 GC ⇒ `TextIOWrapper.__del__` → `close()`
#   → **关掉两者共享的底层 buffer** ⇒ 此后所有 print 报
#       ValueError: I/O operation on closed file
#   实测：不钉住时，"probe0_sr 先"、"deg_ops 先" 两种顺序都炸；钉住后两种都正常。
#   （tv_sweep.py 也有这个隐患，只是它的实跑环境侥幸没触发。）
#   ⇒ 在每个危险 import 前后都把当时的 sys.stdout 钉在一个模块级列表里。
_STDOUT_PINS = [sys.stdout]


def _pin_stdout():
    _STDOUT_PINS.append(sys.stdout)


import importlib

importlib.import_module("probe0_sr")   # 包装 #1（用 importlib 以免 pyflakes 报未使用）
_pin_stdout()

import deg_ops as D       # noqa: E402  （包装 #2）
_pin_stdout()

import dcdc_scan as DC    # noqa: E402
import tv_sweep as TV     # noqa: E402
_pin_stdout()

# ---------------------------------------------------------------- 路径与常量
DIOR = "/home/czy/data/sr4sod/dior"
DIOR_IMG = f"{DIOR}/images/train"      # ★ 只用 train.zip（不依赖 1.93 GB 的 val.zip）
DIOR_LBL = f"{DIOR}/labels/train"
L1_ROOT = "/home/czy/data/sr4sod/dior_l1"
VIEW_ROOT = f"{L1_ROOT}/views"
RUN_ROOT = f"{L1_ROOT}/runs"
CK = os.path.join(A2, "dior_l1.jsonl")

N_IMG = 2000
TRAIN_FRAC = 0.75
SEED = 2026
SCAN_N = 500

# 16 条件：与 tv_sweep.py 的 collect() 逐字一致
BLUR_FAMS = {"blur2": 2.0, "blur02": 0.2}
NOISE_GRID = [0.003, 0.012, 0.024, 0.096]
NOISE_FAMS = {"nhi": 0.096, "nlo": 0.003}
BLUR_GRID = [0.0, 0.6, 1.2, 2.4]

TB_MAIN = "l1X_"
TB_REPS = ["l1Y_", "l1Z_"]             # 地板：ref 条件的另外两个退化实现
REF = (0.0, 0.003)
HEAVY = (2.4, 0.096)

ARMS = {
    "dior_l1_A":  dict(blur=(0.00, 0.00), noise=False, jpeg=False, desc="无模糊 + 无噪声/JPEG"),
    "dior_l1_B0": dict(blur=(0.00, 0.05), noise=True,  jpeg=True,  desc="窄模糊 + 噪声 + JPEG"),
}

# ---- 预注册阈值（DIOR_L1_PREREG.md §3，不许在这里改）----
TH = dict(
    R_min=0.010,          # 动态范围下限
    heavy_floor=0.010,    # 重退化端不许贴地板
    R_over_floor=5.0,     # 两端差 ≥ 5× 实现噪声
    delta_min=0.005,      # 效应量下限
    snr_pass=2.0,         # 放行
    snr_retry=1.0,        # [1,2) ⇒ 加种子；<1 ⇒ 停
    floor_frac=0.20,      # SD_real ≤ 0.20|Δ|
    sd_imported=0.00141,  # AI-TOD 实测 ref 端种子 SD（导入假设，见 prereg §3.2）
    tcrit_df5=2.571,
    n_seeds_ref=6,
)


def say(m):
    print(m, flush=True)


def select_files(root, n):
    """★ 取样规则：按 **md5(文件名) 的确定性乱序** 取前 n 张，而不是"前 n 张排序"。

    为什么必须换（2026-10-16 实测，见 DIOR_L1_PREREG.md §7.4）
      `AIMGroup/DIOR` 的 train.zip **有块结构**：前 1000 张（编号 00001–01000）是
      一个系统性不同的块 —— 每图目标 6.02 个、中位短边 24 px；
      而其余 4,862 张是 5.46 个、29 px。整档 5,862 张平均 5.56 个、28 px。
      "前 N 张排序"因此取到的是**块**，不是随机子集；N=1000 时偏得最厉害。
      md5 乱序与归档顺序无关 ⇒ 子集无偏、完全确定、可审计。
    """
    fs = sorted(glob.glob(os.path.join(root, "*")))
    fs = [p for p in fs if os.path.splitext(p)[1].lower()
          in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")]
    fs = sorted(fs, key=lambda p: hashlib.md5(os.path.basename(p).encode()).hexdigest())
    return fs[:n], fs


def dist_report(sel, tag, lbl_dir):
    """打印子集的尺寸分布，并按 eval_arm 的 n>=150 规则标出哪些桶真的会被算进 mAP。"""
    from probe0_common import read_yolo, SIZE_EDGES
    ss = []
    n_obj = 0
    for p in sel:
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.isfile(lp):
            continue
        gt = read_yolo(lp, DC.BASE, DC.BASE)
        if not gt:
            continue
        A = np.asarray([[g[0], g[1], g[2], g[3]] for g in gt], dtype=np.float32)
        ss += np.minimum(A[:, 2] - A[:, 0], A[:, 3] - A[:, 1]).tolist()
        n_obj += len(gt)
    if not ss:
        say(f"  [{tag}] ★ 没有目标，检查标签目录")
        return dict(n_img=0, n_obj=0)
    b = np.histogram(np.asarray(ss, float), bins=SIZE_EDGES)[0]
    usable = [i for i, x in enumerate(b) if x >= 150]
    LBL = ['0-2', '2-4', '4-6', '6-8', '8-12', '12-16', '16-24', '24-32', '32-64', '64+']
    say(f"  [{tag}] 图 {len(sel)}，目标 {n_obj:,}（{n_obj / len(sel):.2f}/图），"
        f"中位短边 {np.median(ss):.1f} px")
    say("        分桶 " + "/".join(f"{x:>5d}" for x in b))
    say(f"        n>=150 的桶 {len(usable)} 个（mAP5095 只在这些桶上平均）: "
        f"{[LBL[i] for i in usable]}")
    return dict(n_img=len(sel), n_obj=n_obj, per_img=n_obj / len(sel),
                median=float(np.median(ss)), bins=[int(x) for x in b],
                n_usable_bins=len(usable))


def conditions():
    out = []
    for f, b in BLUR_FAMS.items():
        for c in NOISE_GRID:
            out.append((f"{f}__b{b:g}_n{c:g}", b, c))
    for f, c in NOISE_FAMS.items():
        for b in BLUR_GRID:
            out.append((f"{f}__n{c:g}_b{b:g}", b, c))
    return out


def ref_suffix():
    """ref 条件在 tb_main 下的完整 tag（= NOISE_FAMS['nlo'] 族，blur=0.0）。"""
    for tg, b, c in conditions():
        if (b, c) == REF:
            return tg
    raise RuntimeError("ref 条件不在网格里")


def heavy_suffix():
    for tg, b, c in conditions():
        if (b, c) == HEAVY:
            return tg
    raise RuntimeError("heavy 条件不在网格里")


# ---------------------------------------------------------------- 视图
def build_view(tag, cfg, eng, files, n_tr, use_noise, use_jpeg):
    """与 tv_sweep.build_view 同构（退化/SR/健康检查逐字相同），只是图像源可换。"""
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    root = os.path.join(VIEW_ROOT, tag)
    for part in ("train", "val"):
        os.makedirs(os.path.join(root, "images", part), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", part), exist_ok=True)

    def one(idx_p):
        idx, p = idx_p
        stem = os.path.splitext(os.path.basename(p))[0]
        part = "train" if idx < n_tr else "val"
        dst = os.path.join(root, "images", part, stem + ".png")
        if os.path.isfile(dst) and os.path.getsize(dst) >= 5120:
            # ★ 2026-10-17 05:2x 修（AN-4）：中断续跑时「先写图、后拷标签」之间有空隙，
            #   进程若在 imwrite 之后、copyfile 之前被杀，重启会因 dst 已存在而跳过，
            #   标签永久缺失 ⇒ ultralytics 报 `Using 1499 train`（视图 1500 张）。
            #   修法：跳过分支里只在标签确实缺时补拷一次，不重算图像。
            lp0 = os.path.join(DIOR_LBL, stem + ".txt")
            lv0 = os.path.join(root, "labels", part, stem + ".txt")
            if os.path.isfile(lp0) and not os.path.isfile(lv0):
                shutil.copyfile(lp0, lv0)
            return 0
        hr = cv2.imread(p, cv2.IMREAD_COLOR)
        if hr is None:
            return 0
        if hr.shape[0] != DC.BASE or hr.shape[1] != DC.BASE:
            hr = cv2.resize(hr, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)
        lr = TV.degrade_span(hr, DC.SCALE, D.arm_rng(stem, tag), cfg["blur"][0],
                             cfg["blur"][1], use_noise=use_noise, use_jpeg=use_jpeg)
        if lr.shape[0] != DC.LR or lr.shape[1] != DC.LR:
            lr = cv2.resize(lr, (DC.LR, DC.LR), interpolation=cv2.INTER_AREA)

        def _chk(a, b, who):
            tol = 0.25 * max(b, 1.0) + 12.0
            if abs(a - b) > tol:
                raise RuntimeError(f"[view-sanity] {tag}/{stem} {who} 亮度异常："
                                   f"{a:.1f} vs {b:.1f} (容差 {tol:.1f})")
        hm, lm = float(hr.mean()), float(lr.mean())
        _chk(lm, hm, "LR vs HR")
        out = eng.upscale(lr)
        if out.shape[0] != DC.BASE or out.shape[1] != DC.BASE:
            out = cv2.resize(out, (DC.BASE, DC.BASE), interpolation=cv2.INTER_AREA)
        _chk(float(out.mean()), lm, "SR vs LR")
        cv2.imwrite(dst, out)
        lp = os.path.join(DIOR_LBL, stem + ".txt")
        if os.path.isfile(lp):
            shutil.copyfile(lp, os.path.join(root, "labels", part, stem + ".txt"))
        return 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as ex:
        n = sum(ex.map(one, list(enumerate(files))))
    yml = os.path.join(root, "data.yaml")
    with open(yml, "w", encoding="utf-8") as fh:
        fh.write(f"path: {root}\ntrain: images/train\nval: images/val\n")
        fh.write("nc: 1\nnames: ['object']\n")
    say(f"  [view] {tag}: 新写 {n} 张，{time.time() - t0:.0f}s → {root}")
    return yml


# ---------------------------------------------------------------- 缓存
def load_ck():
    done = {}
    if os.path.isfile(CK):
        for ln in open(CK, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                done[(r["detector"], r["tag"])] = r
            except Exception:
                pass
    return done


def append_ck(r):
    with open(CK, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_IMG)
    ap.add_argument("--scan-n", type=int, default=SCAN_N)
    ap.add_argument("--prep", action="store_true", help="只建视图")
    ap.add_argument("--verdict-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(VIEW_ROOT, exist_ok=True)
    os.makedirs(RUN_ROOT, exist_ok=True)
    TV.RUN_ROOT = RUN_ROOT          # ★ train_one 在调用时读模块全局 ⇒ 此赋值有效

    # ---- 选图：只从 DIOR images/train 取（★ md5 确定性乱序，不是"前 N 排序"）----
    files, all_files = select_files(DIOR_IMG, args.n)
    if len(files) < args.n:
        raise SystemExit(f"★ 图像不足：{DIOR_IMG} 只有 {len(all_files)} 张（需要 {args.n}）"
                         f"\n   ⇒ 先跑 dior_prep.py --extract --splits train")
    say(f"[data] 归档 {len(all_files)} 张 ⇒ 取 {len(files)} 张（md5 乱序前 N）")
    n_tr = int(round(len(files) * TRAIN_FRAC))
    scan = files[n_tr:]
    if args.scan_n and len(scan) > args.scan_n:
        scan = scan[:args.scan_n]

    # 子集代表性报告（写进日志，供审计）
    info_pool = dist_report(files, "池(全部)", DIOR_LBL)
    info_tr = dist_report(files[:n_tr], "train", DIOR_LBL)
    info_sc = dist_report(scan, "scan", DIOR_LBL)
    say(f"[data] 扫描集 {len(scan)} 张 / {info_sc.get('n_obj', 0):,} 目标"
        f"  ⇒ {info_sc.get('per_img', 0):.1f} 目标/图")
    if info_sc.get("n_obj", 0) < 1000:
        say("⚠ 扫描集目标数 < 1000，mAP 会很不稳 —— 建议加大 --n / --scan-n")

    rf, hf = ref_suffix(), heavy_suffix()
    say(f"[grid] 16 条件；ref = {REF} → tag '{TB_MAIN}{rf}'；heavy = {HEAVY} → tag '{TB_MAIN}{hf}'")

    done = load_ck()
    say(f"[cache] 已有 {len(done)} 条评测缓存 → {os.path.basename(CK)}")

    # ---- 阶段 1：视图 + 训练 ----
    if not args.verdict_only:
        eng = TV.make_engine("bsrgan")
        say(f"[engine] bsrgan ×{DC.SCALE}")
        ymls = {}
        for tag, cfg in ARMS.items():
            say(f"\n### {tag}  ({cfg['desc']}, blur σ ~ U{cfg['blur']})")
            ymls[tag] = build_view(tag, cfg, eng, files, n_tr,
                                   cfg["noise"], cfg["jpeg"])
        if args.prep:
            say("\n[prep] 视图阶段结束（未训练）")
            return
        for tag in ARMS:
            TV.train_one(tag, ymls[tag], seed=SEED, name=tag)

        # ---- 阶段 2：评测 16 条件 + 地板复现 ----
        from ultralytics import YOLO
        jobs = [(tg, tb, b, c) for (tg, b, c) in conditions() for tb in [TB_MAIN]]
        jobs += [(rf, tb, REF[0], REF[1]) for tb in TB_REPS]
        say(f"\n[eval] {len(ARMS)} 检测器 × {len(jobs)} 条件 = {len(ARMS) * len(jobs)} 次")
        t_all = time.time()
        for tag in ARMS:
            wp = os.path.join(RUN_ROOT, tag, "weights", "best.pt")
            if not os.path.isfile(wp):
                say(f"★ 缺权重 {wp}，跳过")
                continue
            m = YOLO(wp)
            for (sfx, tb, b, c) in jobs:
                key = (tag, tb + sfx)
                if key in done:
                    continue
                t0 = time.time()
                r = DC.eval_arm(m, scan, DIOR_LBL, b, c, None, eng, tb + sfx, shot=False)
                r.update(detector=tag, tag=tb + sfx, blur=b, noise=c,
                         tag_base=tb, shot=False)
                append_ck(r)
                done[key] = r
                say(f"  {tag}  {tb + sfx:<26} mAP5095="
                    f"{('%.4f' % r['mAP5095']) if r['mAP5095'] is not None else 'n/a'}"
                    f"  PSNR={('%.2f' % r['psnr']) if r.get('psnr') else 'n/a'}"
                    f"  {time.time() - t0:.0f}s")
            del m
        say(f"\n[eval] 用时 {time.time() - t_all:.0f}s")

    # ---- 阶段 3：判决 ----
    done = load_ck()
    need = len(ARMS) * (len(conditions()) + len(TB_REPS))
    say(f"[cache] 评测缓存 {len(done)} 条（其中属于本设计的 "
        f"{sum(1 for k in done if k[0] in ARMS)} 条 / 需要 {need} 条）")
    if sum(1 for k in done if k[0] in ARMS) < need:
        say("★ 评测不齐 —— 不写 DIOR_L1_DONE，让监工续跑（防失败被静默掩盖）")
        return
    verdict(done, scan, info_sc.get("n_obj", 0), info_pool, info_tr, info_sc)


def verdict(done, scan, n_obj, info_pool=None, info_tr=None, info_sc=None):
    say("\n" + "=" * 92)
    say("判决（判据来自 DIOR_L1_PREREG.md §3，先写死）")
    say("=" * 92)

    def m(det, tb, bl, no):
        for (s, b, c) in conditions():
            if (b, c) == (bl, no):
                r = done.get((det, tb + s))
                return None if r is None else r.get("mAP5095")
        return None

    MDE_imp = TH["tcrit_df5"] * TH["sd_imported"] / np.sqrt(TH["n_seeds_ref"])
    res = {}
    say(f"{'臂':<12}{'mAP(ref)':>10}{'mAP(heavy)':>12}{'R':>10}"
        f"{'SD_real':>10}{'R/floor':>9}")
    for det in ARMS:
        mref = m(det, TB_MAIN, *REF)
        mhv = m(det, TB_MAIN, *HEAVY)
        reps = [m(det, tb, *REF) for tb in [TB_MAIN] + TB_REPS]
        reps = [x for x in reps if x is not None]
        sd = float(np.std(reps, ddof=1)) if len(reps) >= 2 else float("nan")
        R = (mref - mhv) if (mref is not None and mhv is not None) else None
        ratio = (R / sd) if (R is not None and sd and sd > 0) else float("inf")
        res[det] = dict(mAP_ref=mref, mAP_heavy=mhv, R=R, sd_real=sd,
                        reps=reps, R_over_floor=ratio)
        say(f"{det:<12}{(f'{mref:.4f}' if mref is not None else 'n/a'):>10}"
            f"{(f'{mhv:.4f}' if mhv is not None else 'n/a'):>12}"
            f"{(f'{R:+.4f}' if R is not None else 'n/a'):>10}"
            f"{(f'{sd:.5f}' if sd == sd else 'n/a'):>10}"
            f"{(f'{ratio:.1f}' if ratio != float('inf') else 'inf'):>9}")

    dA = res["dior_l1_A"]["mAP_ref"] if res.get("dior_l1_A") else None
    dB = res["dior_l1_B0"]["mAP_ref"] if res.get("dior_l1_B0") else None
    delta = (dA - dB) if (dA is not None and dB is not None) else None

    say(f"\n主终点  Δ(A − B0) @ ref {REF} = "
        f"{(f'{delta:+.5f}' if delta is not None else 'n/a')}")
    say(f"导入 SD = {TH['sd_imported']:.5f} (AI-TOD 实测) ⇒ MDE_imp = {MDE_imp:.5f}")
    snr = abs(delta) / MDE_imp if delta is not None else None
    say(f"SNR_imp = {(f'{snr:.2f}' if snr is not None else 'n/a')}")

    floors = {k: res[k]["sd_real"] for k in res if res[k]["sd_real"] == res[k]["sd_real"]}
    sd_max = max(floors.values()) if floors else float("nan")

    c1_range = True      # ①a 真·动态范围：R 够大 且 heavy 端不贴地板
    c1_floor = True      # ①b 两端差 ≥ 5× 实现噪声
    for k in ARMS:
        r = res[k]
        if r["R"] is None or r["mAP_heavy"] is None:
            c1_range = c1_floor = False
            continue
        if not (r["R"] >= TH["R_min"] and r["mAP_heavy"] > TH["heavy_floor"]):
            c1_range = False
        if not (r["R"] >= TH["R_over_floor"] * r["sd_real"]):
            c1_floor = False
    c2 = (delta is not None and abs(delta) >= TH["delta_min"] and snr is not None
          and snr >= TH["snr_pass"])
    c3 = (delta is not None and sd_max == sd_max and sd_max <= TH["floor_frac"] * abs(delta))

    say(f"\n①a 动态范围: {'✅' if c1_range else '❌'}  (需两臂 R ≥ {TH['R_min']} 且 "
        f"mAP(heavy) > {TH['heavy_floor']})")
    say(f"①b 实现噪声: {'✅' if c1_floor else '❌'}  (需两臂 R ≥ {TH['R_over_floor']}×SD_real)")
    say(f"②  效应量  : {'✅' if c2 else '❌'}  (需 |Δ| ≥ {TH['delta_min']} 且 SNR ≥ {TH['snr_pass']})")
    say(f"③  地板稳定: {'✅' if c3 else '❌'}  (需 SD_real ≤ {TH['floor_frac']}×|Δ| = "
        f"{TH['floor_frac'] * abs(delta):.5f}, 实测 max {sd_max:.5f})"
        if delta is not None else "③  地板稳定: ❌ (无 Δ)")

    say("")
    # ★ 分支顺序（与 DIOR_L1_PREREG.md §3.3 对应，不许改序）：
    #   先排"不是科学结论"的两类（协议失败 / 仪器噪声），再按效应量判科学结论，
    #   **最后**才用 c3 兜底 —— 因为 Δ 很小时 SD_real/|Δ| 天然变大，
    #   那是"Δ 小"的后果，不是独立的仪器问题，不能抢在"功效不足"之前报。
    if c1_range and c1_floor and c2 and c3:
        v = "★ 放行 L2（协议可移植 + 效应可测）"
    elif not c1_range:
        v = "⛔ 停：协议失败 —— 端点无动态范围（写清这是协议问题，不是现象否证）"
    elif not c1_floor:
        v = (f"⚠ 测量异常：实现噪声相对两端差过大（SD_real max {sd_max:.5f}，"
             f"R < {TH['R_over_floor']}×SD_real）⇒ 先查测量管线，**不得据此下科学结论**")
    elif snr is not None and snr < TH["snr_retry"]:
        v = "⛔ 停：功效不足 —— 主终点在 DIOR 上不可测（写清这是功效问题）"
    elif snr is not None and snr < TH["snr_pass"]:
        v = f"⚠ 未过但不算失败：SNR ∈ [{TH['snr_retry']}, {TH['snr_pass']}) ⇒ 种子加到 12 再判"
    elif delta is not None and abs(delta) < TH["delta_min"]:
        v = "⛔ 停：真否证 —— 效应存在但量级 < 0.005（现象在 DIOR 上确实更弱）"
    elif not c3:
        v = (f"⚠ 地板不稳：SD_real ({sd_max:.5f}) > {TH['floor_frac']}×|Δ| "
             f"⇒ 先查测量管线（pre-reg §3.3 末行）")
    else:
        v = "⚠ 未判死也未判活 —— 需补种子或核测量管线"
    say(f"⇒ {v}")

    # ---- 描述性：16 条件全表 ----
    say("\n" + "-" * 92)
    say("描述性：16 条件 × 两臂（mAP5095）")
    say("-" * 92)
    say(f"{'条件 (blur, noise)':<26}{'A':>10}{'B0':>10}{'A-B0':>10}")
    for (sfx, b, c) in sorted(conditions(), key=lambda t: (t[2], t[1])):
        va = m("dior_l1_A", TB_MAIN, b, c)
        vb = m("dior_l1_B0", TB_MAIN, b, c)
        d = (va - vb) if (va is not None and vb is not None) else None
        say(f"{f'({b}, {c})':<26}{(f'{va:.4f}' if va is not None else 'n/a'):>10}"
            f"{(f'{vb:.4f}' if vb is not None else 'n/a'):>10}"
            f"{(f'{d:+.4f}' if d is not None else 'n/a'):>10}")

    out = dict(prereg="DIOR_L1_PREREG.md", n_scan_img=len(scan), n_gt_obj=n_obj,
               subset=dict(pool=info_pool, train=info_tr, scan=info_sc),
               thresholds=TH, MDE_imported=MDE_imp, delta_ref=delta, snr_imported=snr,
               criteria=dict(dynamic_range=bool(c1_range), realization_floor=bool(c1_floor),
                             effect_size=bool(c2), floor_stable=bool(c3)),
               verdict=v, arms=res)
    with open(os.path.join(A2, "dior_l1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    say("\nwrote dior_l1.json")
    with open(os.path.join(A2, "DIOR_L1_DONE"), "w") as f:
        f.write(v + "\n")
    say("★ 已写 DIOR_L1_DONE")


if __name__ == "__main__":
    main()
