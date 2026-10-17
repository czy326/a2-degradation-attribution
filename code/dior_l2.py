"""DIOR L2 —— 多种子 × 多臂 sweep。预注册 `DIOR_L2_PREREG.md`（判据先写死，本文件只实现）。

与 L1 的关系（**这是本文件的设计核心**）
  L1 的池 / 划分 / 扫描切片 / 16 条件网格 / 退化口径 / SR 引擎 **一个字都不改**，
  只做两件事：**加种子（1 → 6）、加臂（2 → 4）**。
  * 视图：A / B0 的视图 tag **沿用 L1 的 `dior_l1_A` / `dior_l1_B0`**。
    原因：退化的随机数种子是 `deg_ops.arm_rng(stem, tag)` —— **换 tag 就会换 LR 视图**，
    那就不再是"与 L1 逐字节相同"，L1 的单种子点也就不能作为参照并排放。
  * 运行目录 / 检测器名另用 `dior_l2_<臂>_s<种子>`，与视图名解耦（`tv_sweep.train_one`
    本来就支持 `name != tag`）。
  * 直接 `import dior_l1` 复用 `build_view` / `select_files` / `conditions` / 引擎装配，
    **不复制代码** —— 复制必然漂移。

臂的定义与 `tv_sweep.TRAIN_ARMS` 逐字一致：
  A  blur U(0.00,0.00) 无噪声无JPEG   ← 无退化锚点
  B0 blur U(0.00,0.05) 噪声 + JPEG    ← 窄模糊层（L1 主对照）
  B2 blur U(0.20,2.00) 噪声 + JPEG    ← 宽模糊层
  D0 blur U(0.20,2.00) 无噪声无JPEG   ← 宽模糊层（无噪声/JPEG）

本次默认只跑**核心块**（A/B0 × 6 种子，见 prereg §2.2）：
§1 的三个待答问题全部由这两臂回答；B2/D0 是描述性加强，用 `--arms` 显式打开。

运行（WSL, 需 GPU）
  python dior_l2.py --prep                 # 只建/检查视图
  python dior_l2.py --arms A,B0            # 核心块（默认）
  python dior_l2.py --arms A,B0,B2,D0      # 加扩展块（增量，可断点续跑）
  python dior_l2.py --verdict-only         # 只用已有 jsonl 出判决
"""
import argparse
import json
import os
import sys
import time

import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
SR4SOD = "/mnt/e/新方向研究/sr4sod"
sys.path.insert(0, SR4SOD)
sys.path.insert(0, A2)

import dior_l1 as L1          # ★ 复用，不复制
import dcdc_scan as DC
import tv_sweep as TV

L2_ROOT = "/home/czy/data/sr4sod/dior_l2"
RUN_ROOT = f"{L2_ROOT}/runs"
CK = os.path.join(A2, "dior_l2.jsonl")
OUT_J = os.path.join(A2, "dior_l2.json")
DONE = os.path.join(A2, "DIOR_L2_DONE")

SEEDS = [2026, 2027, 2028, 2029, 2030, 2031]          # TTH: prereg §2
CORE_ARMS = ["A", "B0"]                                # prereg §2.2

ARMS = {
    "A":  dict(view="dior_l1_A",  blur=(0.00, 0.00), noise=False, jpeg=False,
               desc="无模糊 + 无噪声/JPEG"),
    "B0": dict(view="dior_l1_B0", blur=(0.00, 0.05), noise=True,  jpeg=True,
               desc="窄模糊 + 噪声 + JPEG"),
    "B2": dict(view="dior_l2_B2", blur=(0.20, 2.00), noise=True,  jpeg=True,
               desc="宽模糊 + 噪声 + JPEG"),
    "D0": dict(view="dior_l2_D0", blur=(0.20, 2.00), noise=False, jpeg=False,
               desc="宽模糊 + 无噪声/JPEG"),
}

REF = L1.REF                    # (0.0, 0.003)
HEAVY = L1.HEAVY                # (2.4, 0.096)
TB_MAIN = L1.TB_MAIN            # "l1X_"
TB_REPS = L1.TB_REPS            # ["l1Y_", "l1Z_"]

# 预注册阈值（`DIOR_L2_PREREG.md` §3，不许在这里改）
TH = dict(tcrit_df5=2.571, n_seeds=6, equiv=0.05, mde_small=0.005,
          sd_seed_aitod=0.00141, sd_real_l1=0.00100)
# AI-TOD 参照（`SCALEUP_GATE.md` §2.2）
AITOD = dict(delta=0.00729, sd_seed=0.00141, snr=4.93)
L1_DELTA = 0.01571           # DIOR L1 单种子点（`dior_l1.json`）
# ★ 数据集名：判决文本里有几处写死了 "DIOR"。`soda_l2.py` 复用本文件的 `verdict()`，
#   只需把本全局改成 "SODA-A" —— 判据与分支表一个字都不动。
DS = "DIOR"


def say(m):
    print(m, flush=True)


def det_name(arm, seed):
    return f"dior_l2_{arm}_s{seed}"


def _cross_out(cross, nonmono=None):
    """把「逐种子的变号区间」整理成可写进 JSON 的形式。

    `cross` 是 [(last_pos_noise, first_nonpos_noise), ...]，每个元素**单看**都是正区间
    （last_pos < first_nonpos）。取交集 = (max(last_pos), min(first_nonpos))；
    交集可能是**空**的，此时 `ci_lo` / `ci_hi` 必须写 None，不许把颠倒的两个数
    当成区间写出去（`DIOR_L2_PREREG.md` §3.3：报区间的并集/交集，不报点估计）。

    `nonmono` 是 {seed: [[noise, 组内全正?], ...]} —— 符号在 noise 轴上变化 ≥2 次的种子。
    它们**不进** `cross`，单独落在 `nonmono_seeds` 里，因为"交叉区间"对它们没有定义。
    """
    nonmono = nonmono or {}
    if not cross:
        return dict(intervals=[], ci_lo=None, ci_hi=None, ci_empty=True,
                    n_seeds=0, nonmono_seeds=nonmono)
    lo = max(x[0] for x in cross)
    hi = min(x[1] for x in cross)
    empty = not (lo < hi)
    return dict(intervals=[list(map(float, x)) for x in cross],
                ci_lo=None if empty else float(lo),
                ci_hi=None if empty else float(hi),
                ci_empty=bool(empty), n_seeds=len(cross),
                nonmono_seeds=nonmono)


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
    with open(CK, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=L1.N_IMG)
    ap.add_argument("--scan-n", type=int, default=L1.SCAN_N)
    ap.add_argument("--arms", default=",".join(CORE_ARMS))
    ap.add_argument("--seeds", default="")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"★ 未知臂 {a}；可选 {list(ARMS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s] or SEEDS

    os.makedirs(RUN_ROOT, exist_ok=True)
    L1.RUN_ROOT = RUN_ROOT
    TV.RUN_ROOT = RUN_ROOT          # ★ train_one 在调用时读模块全局 ⇒ 此赋值有效
    L1.CK = CK                      # 让 L1.load_ck 若被调用也指向 L2 缓存

    # ---- 选图：与 L1 完全相同（同 md5 乱序、同 n、同划分）----
    files, all_files = L1.select_files(L1.DIOR_IMG, args.n)
    if len(files) < args.n:
        raise SystemExit(f"★ 图像不足：{L1.DIOR_IMG} 只有 {len(all_files)} 张（需要 {args.n}）")
    n_tr = int(round(len(files) * L1.TRAIN_FRAC))
    scan = files[n_tr:]
    if args.scan_n and len(scan) > args.scan_n:
        scan = scan[:args.scan_n]
    say(f"[data] 归档 {len(all_files)} ⇒ 取 {len(files)}（md5 乱序前 N）；"
        f"train {n_tr} / scan {len(scan)}")
    done = load_ck()
    say(f"[cache] 已有 {len(done)} 条 → {os.path.basename(CK)}")
    say(f"[scope] 臂 {arms}；种子 {seeds}；"
        f"合计 {len(arms) * len(seeds)} 次训练 × {len(L1.conditions()) + len(TB_REPS)} 次评测")

    # ---- 阶段 1：视图（与种子无关，只建一次；A/B0 复用 L1 的）----
    if not args.verdict_only:
        eng = TV.make_engine("bsrgan")
        say(f"[engine] bsrgan ×{DC.SCALE}")
        ymls = {}
        for a in arms:
            cfg = ARMS[a]
            say(f"\n### 视图 {cfg['view']}  ({a}: {cfg['desc']}, blur σ ~ U{cfg['blur']})")
            ymls[a] = L1.build_view(cfg["view"], cfg, eng, files, n_tr,
                                    cfg["noise"], cfg["jpeg"])
        if args.prep:
            say("\n[prep] 视图阶段结束（未训练）")
            return

        jobs = [(tg, tb, b, c) for (tg, b, c) in L1.conditions() for tb in [TB_MAIN]]
        jobs += [(L1.ref_suffix(), tb, REF[0], REF[1]) for tb in TB_REPS]
        rf = L1.ref_suffix()
        say(f"\n[grid] 16 条件 + {len(TB_REPS)} 地板；ref tag = '{TB_MAIN}{rf}'")

        from ultralytics import YOLO
        # ★ 种子外层循环：每跑完一个种子就得到**完整的一格**（2 臂 × 16 条件），
        #   中途被 WSL 回收也不会丢掉语义完整的中间结果（JSONL 已逐条落盘）。
        for seed in seeds:
            say("\n" + "=" * 92)
            say(f"种子 {seed}")
            say("=" * 92)
            for a in arms:
                TV.train_one(ARMS[a]["view"], ymls[a], seed=seed, name=det_name(a, seed))
            for a in arms:
                name = det_name(a, seed)
                wp = os.path.join(RUN_ROOT, name, "weights", "best.pt")
                if not os.path.isfile(wp):
                    say(f"★ 缺权重 {wp}，跳过该臂")
                    continue
                m = YOLO(wp)
                for (sfx, tb, b, c) in jobs:
                    key = (name, tb + sfx)
                    if key in done:
                        continue
                    t0 = time.time()
                    r = DC.eval_arm(m, scan, L1.DIOR_LBL, b, c, None, eng, tb + sfx, shot=False)
                    r.update(detector=name, tag=tb + sfx, blur=b, noise=c,
                             tag_base=tb, arm=a, seed=seed, shot=False)
                    append_ck(r)
                    done[key] = r
                    say(f"  {name}  {tb + sfx:<26} mAP5095="
                        f"{('%.4f' % r['mAP5095']) if r['mAP5095'] is not None else 'n/a'}"
                        f"  {time.time() - t0:.0f}s")
                del m

    # ---- 阶段 2：判决 ----
    done = load_ck()
    need = len(arms) * len(seeds) * (len(L1.conditions()) + len(TB_REPS))
    have = sum(1 for (d, t) in done if any(d == det_name(a, s) for a in arms for s in seeds))
    say(f"\n[cache] 本设计 {have} 条 / 需要 {need} 条")
    if have < need:
        say("★ 评测不齐 —— 不写 DIOR_L2_DONE，让监工续跑（防失败被静默掩盖）")
        return
    verdict(done, arms, seeds, scan)


# ---------------------------------------------------------------- 判决
def verdict(done, arms, seeds, scan):
    say("\n" + "=" * 92)
    say(f"{DS} L2 判决（判据来自 `DIOR_L2_PREREG.md` §3，先写死）")
    say("=" * 92)

    def m(arm, seed, tb, bl, no):
        for (s, b, c) in L1.conditions():
            if (b, c) == (bl, no):
                r = done.get((det_name(arm, seed), tb + s))
                return None if r is None else r.get("mAP5095")
        return None

    def bins_of(arm, seed, tb, bl, no):
        for (s, b, c) in L1.conditions():
            if (b, c) == (bl, no):
                r = done.get((det_name(arm, seed), tb + s))
                return {} if r is None else (r.get("bins") or {})
        return {}

    # ---- 主终点：Δ(A−B0) @ ref，逐种子 ----
    say(f"\n【主终点】Δ(A − B0) @ ref {REF}   （n = {len(seeds)} 种子）")
    say(f"{'种子':>8}{'mAP_A(ref)':>13}{'mAP_B0(ref)':>14}{'Δ':>11}")
    ds = []
    for s in seeds:
        a = m("A", s, TB_MAIN, *REF)
        b = m("B0", s, TB_MAIN, *REF)
        if a is None or b is None:
            continue
        d = a - b
        ds.append(d)
        say(f"{s:>8}{a:>13.4f}{b:>14.4f}{d:>+11.5f}")
    if len(ds) < 2:
        say("★ 主终点数据不足，无法出 6 种子统计")
        return
    A = np.asarray(ds, dtype=np.float64)
    n = A.size
    mean, sd = float(A.mean()), float(A.std(ddof=1))
    mu = sd / np.sqrt(n)
    tc = TH["tcrit_df5"]
    lo, hi = mean - tc * mu, mean + tc * mu
    t = mean / mu if mu > 0 else float("inf")
    n_pos = int((A > 0).sum())
    mde = tc * mu
    say(f"\n  mean = {mean:+.5f}   SD = {sd:.5f}   SE = {mu:.5f}")
    say(f"  95% CI = [{lo:+.5f}, {hi:+.5f}]（t_crit({n - 1}) = {tc}）")
    say(f"  单样本 t = {t:.2f}   Δ>0 的种子 {n_pos}/{n}   MDE_seed = {mde:.5f}")
    say(f"  对照：L1 单种子 Δ = {L1_DELTA:+.5f}"
        f"（{'∈' if lo <= L1_DELTA <= hi else '∉'} 95% CI）")
    say(f"  对照 AI-TOD：Δ = {AITOD['delta']:+.5f}，SD_seed = {AITOD['sd_seed']:.5f}，"
        f"SNR = {AITOD['snr']:.2f}")
    say(f"  ⇒ 本设计 SNR（用实测 SD）= {abs(mean) / mde if mde > 0 else float('inf'):.2f}")
    say(f"  ⇒ SD_seed 比 AI-TOD {'大' if sd > AITOD['sd_seed'] else '小'} "
        f"{abs(sd / AITOD['sd_seed'] - 1):.1%}")

    c1 = (lo > 0) and (n_pos == n) and (mean > 0)
    c2 = (lo > -TH["equiv"]) and (hi < TH["equiv"])
    say("")
    say(f"  ① 复现（CI 不含 0 ∧ {n}/{n} 同号 ∧ mean>0）: {'✅' if c1 else '❌'}")
    say(f"  ② TOST ±{TH['equiv']}（仅报告，不参与分支）: {'✅ 区间落在等效界内' if c2 else '❌'}")

    if c1:
        v = (f"★ 复现：{DS} 上 Δ(A−B0) 在 6 种子下稳定为正，"
             "主张可升格为跨数据集机制（SCALEUP_GATE 结局 Ⅰ）")
    elif lo <= 0 <= hi and mde < TH["mde_small"]:
        v = (f"⛔ 真否证：CI 含 0 且 MDE_seed = {mde:.5f} < {TH['mde_small']}"
             f" ⇒ 效应确实小于可测下限（写清这是现象否证，不是功效问题）")
    elif lo <= 0 <= hi:
        k = int(np.ceil((tc * sd / TH["mde_small"]) ** 2))
        v = (f"⚠ 功效不足：CI 含 0，但 MDE_seed = {mde:.5f} ≥ {TH['mde_small']}"
             f" ⇒ **不许写成否证**；要到约 {k} 个种子才可测")
    elif mean < 0:
        v = (f"★ 反向：CI 不含 0 且 mean<0 ⇒ 训练覆盖退化在 {DS} 上有净代价，"
             "写「适用边界」（与 L1 §3 的变号发现同源）")
    else:
        v = "⚠ 未判死也未判活 —— 需补种子或核测量管线"
    say(f"\n⇒ {v}")

    # ---- 审计：L1 的 seed-2026 是否被逐位复现 ----
    #   L2 与 L1 用**同一批视图、同一批图、同一划分、同一 seed、deterministic=True**，
    #   所以 seed 2026 两臂的 mAP5095 应当**逐位相同**。不同 ⇒ 管线漂移，必须先查再解读判决。
    l1_path = os.path.join(A2, "dior_l1.jsonl")
    drift = None
    if os.path.isfile(l1_path):
        say("\n【审计】L1 的 seed-2026 复现性（应逐位相同）")
        old = {}
        for ln in open(l1_path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    old[(r["detector"], r["tag"])] = r.get("mAP5095")
                except Exception:
                    pass
        drift = 0
        for a in arms:
            if a not in ("A", "B0"):
                continue
            for (sfx, b, c) in L1.conditions():
                k_old = (f"dior_l1_{a}", TB_MAIN + sfx)
                k_new = (det_name(a, 2026), TB_MAIN + sfx)
                if k_old in old and k_new in done:
                    v0, v1 = old[k_old], done[k_new].get("mAP5095")
                    if v0 is None or v1 is None:
                        continue
                    if abs(v0 - v1) > 1e-12:
                        drift += 1
                        say(f"    ★ 漂移 {a} {sfx}: L1 {v0:.6f} vs L2 {v1:.6f}（差 {v1 - v0:+.6f}）")
        say(f"    ⇒ 16 条件 × 2 臂中，逐位不一致 {drift} 处"
            f"（0 = 管线无漂移；>0 必须先在报告里写清楚再解读判决）")

    # ---- 地板：ref 条件的**退化实现噪声** ----
    # ★ 2026-10-17 01:2x 修正（见 `DIOR_L2_PREREG.md` §6.4）：
    #   原实现是 `reps = [m(a, s, tb_main, *REF) for s in seeds]` 再取 `np.std(..., ddof=1)`
    #   —— 那是 **ref 条件在 6 个种子上的 mAP 种子间 SD**，不是"退化实现噪声"。
    #   `DIOR_L2_PREREG.md` §3.3 第 4 条要的是 L1 的同口径量：**同一 (臂,种子) 下用
    #   3 个 tag_base 各评一次**得到的极差/SD。修正后逐 (臂,种子) 算，再报 max 与 mean。
    say("\n【地板】ref 条件的退化实现噪声（每 (臂,种子) 用 3 个 tag_base 各评一次 ⇒ SD_real）")
    floors = {}
    floor_seed = {}
    for a in arms:
        vals = []
        for s in seeds:
            r3 = [m(a, s, tb, *REF) for tb in [TB_MAIN] + TB_REPS]
            r3 = [x for x in r3 if x is not None]
            if len(r3) >= 2:
                vals.append(float(np.std(r3, ddof=1)))
        if vals:
            floors[a] = max(vals)
            floor_seed[a] = vals
            say(f"  {a}: 逐种子 SD_real = " + ", ".join(f"{v:.5f}" for v in vals)
                + f"   ⇒ max {max(vals):.5f} / mean {float(np.mean(vals)):.5f}")
    floor_max = max(floors.values()) if floors else float("nan")
    if floors:
        say(f"  ⇒ 全场 max SD_real = {floor_max:.5f}"
            f"（L1 报 ≤ {TH['sd_real_l1']:.5f} ⇒ "
            f"{'可复现' if floor_max <= TH['sd_real_l1'] else '★ 更大，须在报告里写清'}）")
        say("  对照：ref 条件下 mAP 的**种子间** SD（与上面不是同一个量）"
            f" A={np.std([m('A', s, TB_MAIN, *REF) for s in seeds], ddof=1):.5f}"
            f" B0={np.std([m('B0', s, TB_MAIN, *REF) for s in seeds], ddof=1):.5f}")

    # ---- 次要 1：16 条件 × 种子平均的 Δ ----
    say("\n【次要】16 条件 × 种子平均的 Δ(A − B0)（描述性，不参与判决）")
    say(f"{'blur':>6}{'noise':>9}{'Δ 均值':>11}{'SD':>10}{'正号种子':>10}")
    curve = {}
    for (sfx, b, c) in sorted(L1.conditions(), key=lambda t: (t[2], t[1])):
        vv = []
        for s in seeds:
            a1, b1 = m("A", s, TB_MAIN, b, c), m("B0", s, TB_MAIN, b, c)
            if a1 is not None and b1 is not None:
                vv.append(a1 - b1)
        if not vv:
            continue
        arr = np.asarray(vv, float)
        curve[f"{b}_{c}"] = dict(n=len(vv), mean=float(arr.mean()),
                                 sd=float(arr.std(ddof=1)) if len(vv) > 1 else None,
                                 n_pos=int((arr > 0).sum()))
        say(f"{b:>6}{c:>9}{arr.mean():>+11.5f}"
            f"{(arr.std(ddof=1) if len(vv) > 1 else float('nan')):>10.5f}"
            f"{int((arr > 0).sum())}/{len(vv):>6}")

    # ---- 次要 2：交叉点（符号变化所在的 noise 区间）----
    say("\n【次要】变号点：按 noise 分组看 Δ 的符号（每个种子给一个区间）")
    bynoise = {}
    for (sfx, b, c) in L1.conditions():
        for s in seeds:
            a1, b1 = m("A", s, TB_MAIN, b, c), m("B0", s, TB_MAIN, b, c)
            if a1 is not None and b1 is not None:
                bynoise.setdefault(s, {}).setdefault(c, []).append(a1 - b1)
    cross = []
    nonmono = {}
    for s in sorted(bynoise):
        ns = sorted(bynoise[s])
        # ★ 每个 noise 组取「组内最小值是否 > 0」作为该组符号（与 L1 同口径）。
        #   区间存在的前提是符号在 noise 轴上**单调**：全部全正组排在全部非正组之前。
        #   若不单调（先负后正），"全正的最大 noise" 与 "转负的最小 noise" 会**倒挂**，
        #   此时绝不能把它当成区间 —— 那是"Δ 在 noise 轴上变号 ≥2 次"，本身是一个发现。
        flags = [(c, min(bynoise[s][c]) > 0) for c in ns]
        idx_t = [i for i, (c, ok) in enumerate(flags) if ok]
        idx_f = [i for i, (c, ok) in enumerate(flags) if not ok]
        if idx_t and idx_f and max(idx_t) < min(idx_f):
            lp, fn = flags[max(idx_t)][0], flags[min(idx_f)][0]
            cross.append((lp, fn))
            say(f"  种子 {s}: 全正的最大 noise = {lp}；转负的最小 noise = {fn}")
        elif not idx_f:
            say(f"  种子 {s}: 全程为正（noise ≤ {ns[-1]} 内未变号）")
        elif not idx_t:
            say(f"  种子 {s}: 全程为非正（noise ≥ {ns[0]} 内未变号）")
        else:
            nonmono[s] = [[c, bool(ok)] for (c, ok) in flags]
            say(f"  种子 {s}: ★ 非单调 —— 逐 noise 的「组内全正?」="
                f"{[(c, ok) for (c, ok) in flags]}"
                f"（{flags[max(idx_t)][0]} 全正却排在 {flags[min(idx_f)][0]} 非正之后）")
    if nonmono:
        say(f"  ★ {len(nonmono)}/{len(bynoise)} 个种子的符号变化**非单调** ⇒ 不得报「交叉区间」，"
            f"只能报逐 noise 的符号表；论文里写「符号随噪声非单调」而不是一个交叉点")
    if cross:
        # ★ 6 个种子的区间取**交集**：`last_pos` 取最大、`first_nonpos` 取最小。
        #   若 max(last_pos) ≥ min(first_nonpos)，交集为**空** —— 此时不能报一个"共同区间"，
        #   否则会打印出形如 (0.096, 0.024) 的颠倒区间（数的顺序看着像区间，语义是反的）。
        c_lo = max(x[0] for x in cross)
        c_hi = min(x[1] for x in cross)
        if c_lo < c_hi:
            say(f"  ⇒ 交叉点落在 ({c_lo}, {c_hi})"
                f"（{len(cross)} 种子区间的**交集**；L1 单种子: (0.024, 0.096)）")
        else:
            say(f"  ⇒ ★ {len(cross)} 个种子的变号区间**无公共交集**"
                f"（max 全正噪声 = {c_lo} ≥ min 转负噪声 = {c_hi}）")
            say("     ⇒ 只能报**逐种子区间**（见上），不能报一个共同区间；"
                "论文里写「交叉位置随种子在 {} ~ {} 之间移动」".format(c_hi, c_lo))

    # ---- 次要 3：尺度分层 ----
    say("\n【次要】逐桶 Δ（种子平均，桶内 n≥150 才计入；描述性）")
    # ★ 标签口径来自 `probe0_common.bin_label`：最后一桶是 '64-inf'，不是 '64+'
    LBL = ['0-2', '2-4', '4-6', '6-8', '8-12', '12-16', '16-24', '24-32', '32-64', '64-inf']
    say(f"{'桶':>7}{'n(框)':>9}{'mAP_A':>10}{'mAP_B0':>10}{'Δ':>11}{'种子里 Δ<0':>11}")
    bin_tab = {}
    for lab in LBL:
        av, bv, nn, nneg = [], [], 0, 0
        for s in seeds:
            ba = bins_of("A", s, TB_MAIN, *HEAVY)
            bb = bins_of("B0", s, TB_MAIN, *HEAVY)
            if lab in ba and ba[lab].get("a5095") is not None and \
               lab in bb and bb[lab].get("a5095") is not None:
                # ★ 这里用重退化端（heavy）：干净端两臂都接近饱和，桶间差异被压缩
                av.append(ba[lab]["a5095"])
                bv.append(bb[lab]["a5095"])
                nn = max(nn, int(ba[lab].get("n") or 0))
                if ba[lab]["a5095"] - bb[lab]["a5095"] < 0:
                    nneg += 1
        if not av:
            say(f"{lab:>7}{'—':>9}{'—':>10}{'—':>10}{'—':>11}")
            continue
        ma, mb = float(np.mean(av)), float(np.mean(bv))
        bin_tab[lab] = dict(n=nn, mAP_A=ma, mAP_B0=mb, delta=ma - mb, n_neg=nneg)
        say(f"{lab:>7}{nn:>9}{ma:>10.4f}{mb:>10.4f}{ma - mb:>+11.5f}{nneg:>8}/{len(av)}")

    # ---- 扩展块：四臂对比（只在 B2/D0 已跑齐时输出；描述性，不改 §3 判据）----
    extra_tab = {}
    need_arms = [a for a in ("B2", "D0") if a in arms]
    if need_arms and all(all(m(a, s, TB_MAIN, *REF) is not None for s in seeds)
                         for a in need_arms):
        say("\n【扩展块】四臂对比（描述性；预注册 §2.2 的扩展块）")

        def pair(a1, a2, tb, bl, no):
            vv = []
            for s in seeds:
                x, y = m(a1, s, tb, bl, no), m(a2, s, tb, bl, no)
                if x is not None and y is not None:
                    vv.append(x - y)
            if len(vv) < 2:
                return None
            arr = np.asarray(vv, float)
            mm, ss_ = float(arr.mean()), float(arr.std(ddof=1))
            se = ss_ / np.sqrt(arr.size)
            return dict(mean=mm, sd=ss_, n=int(arr.size), n_pos=int((arr > 0).sum()),
                        ci=[mm - tc * se, mm + tc * se], ci_excl_0=bool(
                            (mm - tc * se > 0) or (mm + tc * se < 0)))

        say(f"{'对照':>10}{'mean':>11}{'SD':>10}{'正号':>7}{'95% CI':>26}{'CI 不含 0':>11}")
        for (a1, a2) in [("A", "B2"), ("B0", "D0"), ("A", "D0"), ("B2", "D0"),
                         ("A", "B0")]:
            st = pair(a1, a2, TB_MAIN, *REF)
            if st is None:
                continue
            extra_tab[f"{a1}-{a2}"] = st
            lo_s, hi_s = st["ci"]
            ci_txt = "[{:+.5f}, {:+.5f}]".format(lo_s, hi_s)
            say(f"{a1 + '−' + a2:>10}{st['mean']:>+11.5f}{st['sd']:>10.5f}"
                f"{st['n_pos']:>4}/{st['n']:<3}{ci_txt:>26}"
                f"{'✅' if st['ci_excl_0'] else '❌':>11}")
        say("  读法：`A−B0` 与上面【主终点】同一条（此处作一致性复核）；")
        say("        `A−B2` / `B0−D0` / `A−D0` 是把「模糊范围」从窄层换到宽层后的同一对照。")

    out = dict(dataset=DS, prereg="DIOR_L2_PREREG.md", arms=arms, seeds=seeds,
               n_scan_img=len(scan), thresholds=TH, aitod=AITOD, l1_delta=L1_DELTA,
               main=dict(per_seed=ds, mean=mean, sd=sd, se=mu, ci=[lo, hi], t=t,
                         n_pos=n_pos, n=n, mde_seed=mde,
                         snr_measured=(abs(mean) / mde if mde > 0 else None),
                         l1_in_ci=bool(lo <= L1_DELTA <= hi)),
               criteria=dict(replicated=bool(c1), tost=bool(c2)),
               l1_seed2026_drift=drift,
               verdict=v, floors=floors, floor_max=floor_max,
               floor_by_seed=floor_seed, curve=curve,
               crossover=_cross_out(cross, nonmono),
               bins_heavy=bin_tab, extra_contrasts_ref=extra_tab)
    json.dump(out, open(OUT_J, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say(f"\nwrote {os.path.basename(OUT_J)}")
    with open(DONE, "w", encoding="utf-8") as f:
        f.write(v + "\n")
    say("★ 已写 DIOR_L2_DONE")


if __name__ == "__main__":
    main()
