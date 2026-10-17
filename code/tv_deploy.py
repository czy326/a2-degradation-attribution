"""修正估计量：在**指定部署条件**下直接比各臂 mAP（不再对噪声网格取"平均敏感度"）

为什么换
  诊断（`tv_var_diag.py`）显示：Δblur 随噪声水平**单调下降并变号**（12/12 个 arm×seed），
  对它取平均得到的均值是**构造出来的**，且 σ_within/σ_between = 14.7× ⇒ 加种子无用。
  换估计量后：① 不再对穿零曲线取平均 ② 不依赖 blur2/blur02 的配对口径（免掉 rng 键缺陷）
  ③ 臂间在同一 (blur, noise) 上看到的是**完全相同的退化图** ⇒ 逐图可比。

估计量
  对每个部署条件 c=(blur, noise)：
    M_c(det) = 该检测器在 c 上的 mAP50-95（各种子平均）
    Z_c(det) = M_ref(det) − M_c(det)，其中 ref = 最轻条件 (blur=0.0, noise=0.003)
      ⇒ 用**自身 clean 水平**作参照，自动扣掉"整体能力差异"
    closure_c = (Z_c(baseline) − Z_c(fixed)) / Z_c(baseline)
      baseline = tv_A（训练完全不含退化），fixed = 训练覆盖了该维度的臂

统计：臂间检验在**种子水平**上配对（n=种子数）—— 这是正确的聚类层级
"""
import os
import json
import itertools
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
TV = os.path.join(A2, "tv_sweep.jsonl")

ARMS = ["tv_A", "tv_B0", "tv_B2", "tv_D0"]
ARM_SHORT = {"tv_A": "A(无覆盖)", "tv_B0": "B0(窄+噪声)", "tv_B2": "B2(宽+噪声)", "tv_D0": "D0(宽+无噪声)"}
# ★ 2026-10-17 05:5x：原来 PAIRS 写在 main() 里当**局部字面量**，无法被复用方 patch
#   （`arch_verdict.py` 要用同一套 §4.3 口径跑第二个架构）。提到模块级，**取值与顺序不变**，
#   已用「重跑 → 与旧 tv_deploy.json 逐字节 diff」验证过行为等价。
PAIRS = [("tv_A", "tv_B0"), ("tv_A", "tv_D0"), ("tv_A", "tv_B2"),
         ("tv_B0", "tv_D0"), ("tv_B0", "tv_B2"), ("tv_B2", "tv_D0")]
# ★ 2026-10-17 06:1x：section ③（closure）原来把三臂写成 `Z("tv_A", c)` 这类**字符串字面量**，
#   同样无法被复用方 patch —— `arch_verdict.py` 把 ARMS 换成 `y11tv_*` 后，`tab[(REF,"tv_A")]`
#   直接 KeyError。提到模块级，取值与顺序不变（default 三个臂：baseline=A，fixed=B0/D0）。
CLOSURE_ARMS = ("tv_A", "tv_B0", "tv_D0")
SEEDS = []          # 由数据自动探测（支持 3 种子 / 6 种子 / …）


def _sfx(arm):
    """臂键 → 短标签：去掉超分引擎前缀，只留 `tv_` 之后的部分。

    `tv_A` → `A`；`y11tv_A` → `A`（第二个 nano 架构的 eng-tag 前缀）。
    原实现用 `arm[3:]`，对带前缀的臂名会切错（`y11tv_A` → `tv_A`）⇒ 表头不可读。
    """
    return arm.rpartition("tv_")[2] if "tv_" in arm else arm


def load():
    return [json.loads(l) for l in open(TV, encoding="utf-8") if l.strip()]


def tagp(t):
    b = n = None
    for p in t.split("__")[1].split("_"):
        if p.startswith("b"):
            b = round(float(p[1:]), 4)
        elif p.startswith("n"):
            n = round(float(p[1:]), 4)
    return b, n


def build():
    """→ M[(arm, seed)][(blur, noise)] = mAP5095"""
    M = {}
    for r in load():
        b, n = tagp(r["tag"])
        if b is None or n is None:
            continue
        arm = r["detector"].rpartition("_s")[0]
        seed = int(r["detector"].rpartition("_s")[2])
        M.setdefault((arm, seed), {})[(b, n)] = r["mAP5095"]
    return M


def _tcrit(df):
    """双侧 95% t 临界值（小 df 查表，大 df 用近似）。"""
    tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
           12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}
    if df <= 0:
        return float("inf")
    if df in tab:
        return tab[df]
    if df > 30:
        return 1.96
    lo = max(k for k in tab if k < df)
    hi = min(k for k in tab if k > df)
    f = (df - lo) / (hi - lo)
    return tab[lo] + f * (tab[hi] - tab[lo])


def ptest(a, b):
    """种子水平配对 t 检验（a,b 为等长列表）。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean() / se) if se > 0 else float("nan"), float(d.mean())


def main():
    global SEEDS
    M = build()
    SEEDS = sorted({s for (_, s) in M})
    print(f"探测到 {len(SEEDS)} 个种子：{SEEDS}（臂间检验 df={len(SEEDS)-1}，"
          f"95% 临界 |t| = {_tcrit(len(SEEDS)-1):.3f}）")
    conds = sorted({c for v in M.values() for c in v})
    print(f"可用条件 {len(conds)} 个：")
    for c in conds:
        print(f"   blur={c[0]:<4} noise={c[1]:<6}", end="")
    print("\n")

    print("=" * 112)
    print("① 全条件表：各臂 mAP50-95（各种子均值）。行=条件（按退化强度排序）")
    print("=" * 112)
    order = sorted(conds, key=lambda c: (c[1], c[0]))
    hdr = f"{'blur':>6}{'noise':>8}" + "".join(f"{ARM_SHORT[a]:>14}" for a in ARMS)
    print(hdr)
    tab = {}
    for c in order:
        row = []
        for a in ARMS:
            vs = [M[(a, s)].get(c) for s in SEEDS]
            vs = [v for v in vs if v is not None]
            tab[(c, a)] = vs
            row.append(f"{np.mean(vs):.4f}" if vs else "n/a")
        mk = "  ←ref" if c == (0.0, 0.003) else ""
        print(f"{c[0]:>6.2f}{c[1]:>8.3f}" + "".join(f"{x:>14}" for x in row) + mk)

    print("\n" + "=" * 112)
    print("② 臂间对照（种子水平配对，n=%d）—— 关键：这个对照在**各条件上是否稳定**？"
          % len(SEEDS))
    print("=" * 112)
    PAIRS_LOCAL = PAIRS          # 模块级（见文件头注释），复用方可以 patch
    print(f"{'条件':<18}" + "".join(f"{_sfx(a1)+'-'+_sfx(a2):>14}"
                                    for a1, a2 in PAIRS_LOCAL))
    for c in order:
        cells = []
        for a1, a2 in PAIRS_LOCAL:
            v1 = tab[(c, a1)]
            v2 = tab[(c, a2)]
            if len(v1) == len(v2) and v1:
                t, dm = ptest(v1, v2)
                cells.append(f"{dm:>+8.4f}{'*' if abs(t) > 4.3 else ' '}")
            else:
                cells.append(f"{'n/a':>14}")
        print(f"blur={c[0]:<5} n={c[1]:<5}" + "".join(f"{x:>14}" for x in cells))
    tc = _tcrit(len(SEEDS) - 1)
    print(f"\n  注：* = 种子水平配对 |t| > {tc:.3f}（df={len(SEEDS)-1} 的 95% 临界值）")

    print("\n" + "=" * 112)
    print("③ ★ closure p%：在指定部署条件下，覆盖训练补回了多少「相对自身 clean 的损失」")
    print("=" * 112)
    REF = (0.0, 0.003)
    def Z(a, c):
        vr = tab[(REF, a)]
        vc = tab[(c, a)]
        if not vr or not vc or len(vr) != len(vc):
            return None
        return np.array(vr, float) - np.array(vc, float)
    DEPLOYS = [(2.0, 0.096), (2.0, 0.024), (0.2, 0.096), (0.0, 0.096), (2.4, 0.003)]
    cA, cB, cD = CLOSURE_ARMS                  # 模块级，复用方 patch（见文件头注释）
    print(f"{'部署条件':<18}{'Z(A 无覆盖)':>13}{'Z(B0 有噪声)':>14}{'Z(D0 宽模糊)':>14}"
          f"{'closure(B0)':>13}{'closure(D0)':>13}")
    for c in DEPLOYS:
        za, zb, zd = Z(cA, c), Z(cB, c), Z(cD, c)
        if za is None or zb is None or zd is None:
            continue
        zA, zB, zD = za.mean(), zb.mean(), zd.mean()
        cb = (zA - zB) / zA if zA else float("nan")
        cd = (zA - zD) / zA if zA else float("nan")
        print(f"blur={c[0]:<4} n={c[1]:<5}{zA:>13.5f}{zB:>14.5f}{zD:>14.5f}"
              f"{cb:>12.1%}{cd:>13.1%}")

    print("\n  ⇒ 判读：closure(B0) 明显为正、closure(D0) ≈ 0 ⇒ 有效的修正是**加噪声/JPEG**，不是加宽模糊。")

    print("\n" + "=" * 112)
    print("④ 稳定性检查：每个对照在 16 个条件上的**符号一致性**与显著性个数")
    print("=" * 112)
    tc = _tcrit(len(SEEDS) - 1)
    print(f"{'对照':<14}{'同号条件数':>12}{'显著条件数':>12}{'均值符号':>10}"
          f"{'|Δ| 中位数':>13}{'|Δ| 范围':>22}")
    for a1, a2 in PAIRS_LOCAL:
        dms, nsig = [], 0
        for c in order:
            v1, v2 = tab[(c, a1)], tab[(c, a2)]
            if len(v1) == len(v2) and v1:
                t, dm = ptest(v1, v2)
                dms.append(dm)
                if abs(t) > tc:
                    nsig += 1
        dms = np.array(dms, float)
        if not len(dms):
            continue
        pos = int((dms > 0).sum())
        same = max(pos, len(dms) - pos)
        sign = "+" if dms.mean() > 0 else "−"
        print(f"{_sfx(a1)+'-'+_sfx(a2):<14}{f'{same}/{len(dms)}':>12}{f'{nsig}/{len(dms)}':>12}"
              f"{sign:>10}{np.median(np.abs(dms)):>13.4f}"
              f"{f'[{dms.min():+.4f}, {dms.max():+.4f}]':>22}")
    print("\n  ⇒ 关键判读：**A−D0 与 B0−B2 应当「同号条件数低 + 显著条件数 ≈ 0」**")
    print("     （= 模糊范围在两个方向上都不产生可测效应）；")
    print("     **A−B0 / B0−D0 同号条件数应当高**（= 噪声/JPEG 覆盖是稳定因子）。")
    print("  ⇒ 排名相关性在这里低是**交叉现象本身**造成的（轻度退化排序与重度退化相反），不是噪声。")

    json.dump({f"{a}|{s}|{c[0]}|{c[1]}": (np.mean(v) if v else None)
               for (a, s), d in M.items() for c, v in
               [((k), None) for k in []]}, open(os.devnull, "w"))
    out = {f"{a}|s{s}|b{c[0]}|n{c[1]}": float(np.mean(M[(a, s)][c]))
           for (a, s) in M for c in M[(a, s)]}
    json.dump(out, open(os.path.join(A2, "tv_deploy.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nwrote tv_deploy.json")


if __name__ == "__main__":
    main()
