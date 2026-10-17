"""ARCH 判决：复用 `tv_deploy.py` 的全部统计逻辑（不复制），只 patch 数据源与臂名。

  ① patch `tv_deploy.TV`  → `tv_sweep.yolo11n.jsonl`
  ② patch `tv_deploy.ARMS` → `y11_tv_{A,B0,B2,D0}`（检测器名带 y11_ 前缀）
  ③ patch `tv_deploy.A2`  → 临时目录（它的 `main()` 末尾会写 `tv_deploy.json`，
     不能覆盖 YOLOv8n 批的主文件 ⇒ 写完再搬成 `tv_deploy.yolo11n.json`）
  ④ 追加两段 `tv_deploy` 没有的：
     · **判据核对**：按 `ARCH_PREREG.md` §3 的 Q1 / Q2 分支逐条判命
     · **地板**：从 `tv_sweep.yolo11n.floor.jsonl` 算逐 (臂,种子) 的 SD_real

用法：python arch_verdict.py
"""
import collections
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

# ★ 钉住旧 stdout：`io.TextIOWrapper(sys.stdout.buffer, ...)` 在被 GC 时会 close 底层 buffer，
#   若之后还有人再包一层，前一个 wrapper 被回收就会把 buffer 关掉
#   （本项目在 `deg_ops` 上踩过同一个坑，报 "I/O operation on closed file"）。
_PIN = [sys.stdout]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_PIN.append(sys.stdout)

A2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, A2)

import tv_deploy as TD          # noqa: E402  （复用，不复制）

CK = os.path.join(A2, "tv_sweep.yolo11n.jsonl")
CK_FLOOR = os.path.join(A2, "tv_sweep.yolo11n.floor.jsonl")
OUT_JSON = os.path.join(A2, "arch_verdict.json")
# 检测器名 = `{eng-tag}{臂}_s{种子}`，eng-tag="y11" ⇒ `y11tv_A_s2026`
ARMS11 = ["y11tv_A", "y11tv_B0", "y11tv_B2", "y11tv_D0"]
SHORT = {"y11tv_A": "A", "y11tv_B0": "B0", "y11tv_B2": "B2", "y11tv_D0": "D0"}
PAIRS11 = [("y11tv_A", "y11tv_B0"), ("y11tv_A", "y11tv_D0"), ("y11tv_A", "y11tv_B2"),
           ("y11tv_B0", "y11tv_D0"), ("y11tv_B0", "y11tv_B2"), ("y11tv_B2", "y11tv_D0")]
SEEDS = [2026, 2027, 2028, 2029, 2030, 2031]
REF = (0.0, 0.003)
# 预注册 §3 的阈值（一个字都不改）
Q1_SIG_MAX = 2          # 复现：显著 ≤ 2/16
Q1_BOUND_OK = 0.005     # 复现：|Δ| 上界 ≤ 0.005
Q1_SIG_FALSIFY = 6      # 否证：显著 ≥ 6/16
Q1_BOUND_FALSIFY = 0.010
Q2_SIG_MIN = 8          # 复现：显著 ≥ 8/16
Q2_SIG_FALSIFY = 4      # 否证：显著 ≤ 4/16


def build_M(src):
    """→ M[(arm, seed)][(blur, noise)] = mAP5095（口径同 tv_deploy.build）"""
    M = {}
    for ln in open(src, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if r.get("which") == "floor":
            continue
        b, n = TD.tagp(r["tag"])
        if b is None or n is None:
            continue
        arm = r["detector"].rpartition("_s")[0]
        try:
            seed = int(r["detector"].rpartition("_s")[2])
        except ValueError:
            continue
        M.setdefault((arm, seed), {})[(b, n)] = r["mAP5095"]
    return M


def contrast(M, x, y, conds, seeds):
    """逐条件种子水平配对 → [(cond, t, mean)]"""
    out = []
    for c in conds:
        v1 = [M[(x, s)][c] for s in seeds if c in M.get((x, s), {})]
        v2 = [M[(y, s)][c] for s in seeds if c in M.get((y, s), {})]
        if len(v1) == len(v2) and len(v1) == len(seeds):
            t, dm = TD.ptest(v1, v2)
            out.append((c, t, dm))
    return out


def branch_Q1(sig, bound):
    if sig <= Q1_SIG_MAX and bound <= Q1_BOUND_OK:
        return "复现"
    if sig >= Q1_SIG_FALSIFY or bound > Q1_BOUND_FALSIFY:
        return "否证"
    return "未定"


def branch_Q2(sig, light_mean, heavy_mean):
    if sig >= Q2_SIG_MIN and light_mean > 0 and heavy_mean < 0:
        return "复现"
    if sig <= Q2_SIG_FALSIFY or (light_mean < 0 and heavy_mean > 0):
        return "否证"
    return "未定"


def floor_section():
    if not os.path.isfile(CK_FLOOR):
        print("\n[地板] 未找到 " + os.path.basename(CK_FLOOR) + "（地板段可能未跑）")
        return {}
    by = collections.defaultdict(dict)           # name -> tag_base -> mAP
    for ln in open(CK_FLOOR, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        by[r["detector"]][r["tag_base"]] = float(r["mAP5095"])
    print("\n" + "=" * 96)
    print("地板｜ref 条件的退化实现噪声（每 (臂,种子) 用 3 个 tag_base 各评一次 ⇒ SD_real）")
    print("=" * 96)
    out = {}
    for arm in ARMS11:
        vals = []
        for s in SEEDS:
            v3 = [by[f"{arm}_s{s}"][tb] for tb in ["tvX_", "tvY_", "tvZ_"]
                  if tb in by.get(f"{arm}_s{s}", {})]
            if len(v3) >= 2:
                vals.append(float(np.std(v3, ddof=1)))
        if vals:
            out[arm] = dict(per_seed=vals, max=max(vals),
                            mean=float(np.mean(vals)))
            print(f"  {arm:<12} 逐种子 SD_real = "
                  + ", ".join(f"{v:.5f}" for v in vals)
                  + f"   ⇒ max {max(vals):.5f} / mean {float(np.mean(vals)):.5f}")
    if out:
        fam = max(v["max"] for v in out.values())
        print(f"  ⇒ 地板上限 floor_max = {fam:.6f}"
              f"（YOLOv8n 批参照 SD_real ≤ 0.00100）")
    return out


def main():
    if not os.path.isfile(CK):
        raise SystemExit(f"★ 缺 {CK}（主批没跑完？）")
    # ---- ①② patch 数据源、臂名、对照表 ----
    TD.TV = CK
    TD.ARMS = list(ARMS11)
    TD.ARM_SHORT = {"y11tv_A": "A(无覆盖)", "y11tv_B0": "B0(窄+噪声)",
                    "y11tv_B2": "B2(宽+噪声)", "y11tv_D0": "D0(宽+无噪声)"}
    TD.PAIRS = list(PAIRS11)
    TD.CLOSURE_ARMS = ("y11tv_A", "y11tv_B0", "y11tv_D0")
    print(f"[patch] tv_deploy.TV    = {os.path.basename(TD.TV)}")
    print(f"[patch] tv_deploy.ARMS  = {TD.ARMS}")
    print(f"[patch] tv_deploy.PAIRS = {[(SHORT[a], SHORT[b]) for a, b in TD.PAIRS]}")
    print(f"[patch] tv_deploy.CLOSURE_ARMS = {TD.CLOSURE_ARMS}")

    # ---- ③ 跑复用机器，产物搬到临时目录避免覆盖主文件 ----
    tmp = tempfile.mkdtemp(prefix="arch_deploy_")
    A2_ORIG = TD.A2
    TD.A2 = tmp
    try:
        TD.main()
    finally:
        TD.A2 = A2_ORIG
        src = os.path.join(tmp, "tv_deploy.json")
        if os.path.isfile(src):
            shutil.move(src, os.path.join(A2_ORIG, "tv_deploy.yolo11n.json"))
            print(f"→ {os.path.join(A2_ORIG, 'tv_deploy.yolo11n.json')}")
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- ④a 判据核对（预注册 §3）----
    M = build_M(CK)
    seeds = sorted({s for (_, s) in M})
    arms_present = sorted({a for (a, _) in M})
    conds = sorted({c for v in M.values() for c in v})
    tc = TD._tcrit(len(seeds) - 1)
    print("\n" + "=" * 96)
    print(f"判据核对｜种子 {seeds}（n={len(seeds)}，|t|>{tc:.3f} 记显著）"
          f"　条件 {len(conds)} 个　臂 {arms_present}")
    print("=" * 96)
    light = [c for c in conds if c[1] <= 0.024]
    heavy = [c for c in conds if c[1] >= 0.096]
    print(f"  轻端（noise ≤ 0.024）{len(light)} 个，重端（noise = 0.096）{len(heavy)} 个")

    def rep(x, y):
        cc = contrast(M, x, y, conds, seeds)
        sig = sum(1 for _, t, _ in cc if abs(t) > tc)
        dms = np.array([dm for _, _, dm in cc], float)
        bound = float(np.abs(dms).max()) if dms.size else float("nan")
        return cc, sig, dms, bound

    verdict = {}
    print("\n【Q1】训练期模糊范围无效？　A−D0 / B0−B2")
    q1 = []
    for x, y in (("y11tv_A", "y11tv_D0"), ("y11tv_B0", "y11tv_B2")):
        _, sig, dms, bound = rep(x, y)
        br = branch_Q1(sig, bound)
        q1.append((f"{SHORT[x]}−{SHORT[y]}", sig, bound, br))
        print(f"  {SHORT[x]}−{SHORT[y]:<4} 显著 {sig}/{len(conds)}　"
              f"|Δ| 上界 {bound:.5f}　⇒ **{br}**")
    verdict["Q1"] = [dict(pair=a, n_sig=b, abs_bound=c, branch=d) for a, b, c, d in q1]
    verdict["Q1_branch"] = ("复现" if all(v[3] == "复现" for v in q1)
                            else "否证" if any(v[3] == "否证" for v in q1) else "未定")

    print("\n【Q2】噪声/JPEG 覆盖是稳定因子？　A−B0 / A−B2")
    q2 = []
    for x, y in (("y11tv_A", "y11tv_B0"), ("y11tv_A", "y11tv_B2")):
        cc, sig, dms, bound = rep(x, y)
        by = {c: dm for c, _, dm in cc}
        lm = float(np.mean([by[c] for c in light if c in by])) if light else float("nan")
        hm = float(np.mean([by[c] for c in heavy if c in by])) if heavy else float("nan")
        br = branch_Q2(sig, lm, hm)
        q2.append((f"{SHORT[x]}−{SHORT[y]}", sig, lm, hm, br))
        print(f"  {SHORT[x]}−{SHORT[y]:<4} 显著 {sig}/{len(conds)}　"
              f"轻端 Δ 均值 {lm:+.5f}　重端 {hm:+.5f}　⇒ **{br}**")
    verdict["Q2"] = [dict(pair=a, n_sig=b, light_mean=c, heavy_mean=d, branch=e)
                     for a, b, c, d, e in q2]
    verdict["Q2_branch"] = ("复现" if all(v[4] == "复现" for v in q2)
                            else "否证" if any(v[4] == "否证" for v in q2) else "未定")

    # ---- ④b 交叉位置（描述性）----
    print("\n【Q3】A−B0 的交叉位置（描述性，不设阈值）")
    cc, _, _, _ = rep("y11tv_A", "y11tv_B0")
    bynoise = collections.defaultdict(dict)
    for c, _, dm in cc:
        bynoise[c[1]][c[0]] = dm
    n_full = collections.Counter(c[1] for c, _, _ in cc)
    pos_noise, neg_noise = [], []
    for ns in sorted(bynoise):
        vals = bynoise[ns]
        ap = all(v > 0 for v in vals.values())
        an = all(v < 0 for v in vals.values())
        print(f"  noise={ns:<6} blur {len(vals)}/{n_full[ns]}　"
              f"Δ 范围 [{min(vals.values()):+.5f}, {max(vals.values()):+.5f}]　"
              f"{'全为正' if ap else ('全为负' if an else '混合')}")
        if ap:
            pos_noise.append(ns)
        if an:
            neg_noise.append(ns)
    if pos_noise and neg_noise:
        lo, hi = max(pos_noise), min(neg_noise)
        print(f"  ⇒ 全部为正的最大噪声 = {lo}；全部为负的最小噪声 = {hi}　"
              f"交叉落在 ({lo}, {hi}) 内" if lo < hi else
              f"  ⇒ ★ 无公共区间：全正的最大噪声 {lo} ≥ 全负的最小噪声 {hi}")
        verdict["Q3"] = dict(pos_max=lo, neg_min=hi, bracket=(lo, hi), empty=not (lo < hi))
    else:
        verdict["Q3"] = dict(pos_max=max(pos_noise) if pos_noise else None,
                             neg_min=min(neg_noise) if neg_noise else None)
        print("  ⇒ ★ 未同时出现「全正」与「全负」的噪声水平（符号结构异常，需人工看表）")

    verdict["floor"] = floor_section()
    json.dump(verdict, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_JSON}")
    print(f"\n★ 汇总：Q1 = {verdict['Q1_branch']}　Q2 = {verdict['Q2_branch']}")


if __name__ == "__main__":
    main()
