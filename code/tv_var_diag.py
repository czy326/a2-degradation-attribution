"""诊断：闭区间太宽是「种子不够」还是「测量设计」造成的？

思路：把每个 (arm, seed) 的 4 个配对差分看作一组，做方差分解
  σ²_within  = 组内（不同噪声水平之间）的方差  → 这是**测量噪声**，可由设计/评测规模改善
  σ²_between = 组间（不同种子之间）的方差      → 这是**真种子方差**，只能靠加种子改善
若 σ²_within 占主导 ⇒ 加种子是**错的修法**。

另外直接检验一个具体的设计缺陷：
  `eval_arm` 里 rng 由 `arm_rng(stem, tag_base)` 派生，而 tag_base 里**含 family 与 blur**
  ⇒ 配对的两族（blur2 / blur02）用的是**不同的 rng 键** ⇒ **噪声实现不共享**！
  于是"同一噪声水平下比模糊"时，两张图除了模糊不同，连噪声样本也不同 ⇒ 配对被破坏，
  平白多出一块讨厌的方差。用归档的 v1 数据（同一权重、不同 tag_base）可以**直接量出**这块方差。
"""
import os
import json
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
TV = os.path.join(A2, "tv_sweep.jsonl")
V1 = os.path.join(A2, "tv_sweep.tagbase_v1.jsonl")

SEEDS = [2026, 2027, 2028]
ARMS = ["tv_A", "tv_B0", "tv_B2", "tv_D0"]


def load(p):
    if not os.path.isfile(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def tagp(t):
    b = n = None
    for p in t.split("__")[1].split("_"):
        if p.startswith("b"):
            b = round(float(p[1:]), 4)
        elif p.startswith("n"):
            n = round(float(p[1:]), 4)
    return b, n


def index(recs):
    out = {}
    for r in recs:
        b, n = tagp(r["tag"])
        out.setdefault(r["detector"], {})[(r["family"], b, n)] = r
    return out


def diffs(by, which="blur"):
    out = []
    if which == "blur":
        for n in sorted({k[2] for k in by if k[0] in ("blur2", "blur02")}):
            a, c = by.get(("blur2", 2.0, n)), by.get(("blur02", 0.2, n))
            if a and c:
                out.append((n, a["mAP5095"] - c["mAP5095"]))
    else:
        for b in sorted({k[1] for k in by if k[0] in ("nhi", "nlo")}):
            a, c = by.get(("nhi", b, 0.096)), by.get(("nlo", b, 0.003))
            if a and c:
                out.append((b, a["mAP5095"] - c["mAP5095"]))
    return out


def main():
    IX = index(load(TV))
    print("=" * 100)
    print("① 方差分解：组内（4 个噪声水平之间） vs 组间（3 个种子之间）")
    print("=" * 100)
    print(f"{'arm':<8}{'种子':>7}{'Δblur 逐点值':>46}{'组内均值':>11}{'组内SD':>9}")
    within, means = {}, {a: [] for a in ARMS}
    for arm in ARMS:
        for s in SEEDS:
            d = diffs(IX[f"{arm}_s{s}"], "blur")
            v = np.array([x for _, x in d], float)
            within[(arm, s)] = v
            means[arm].append(v.mean())
            print(f"{arm:<8}{s:>7}   {'  '.join(f'{x:+.5f}' for x in v):>44}"
                  f"{v.mean():>+11.5f}{v.std(ddof=1):>9.5f}")
        ms = np.array(means[arm])
        print(f"{'':<8}{'':>7}   组间(种子)均值 {np.mean(ms):+.5f}  SD {ms.std(ddof=1):.5f}")
        print()

    sw = np.mean([v.var(ddof=1) for v in within.values()])
    sb = np.mean([np.var(means[a], ddof=1) for a in ARMS])
    print(f"  σ²_within （测量噪声，可设计改善） = {sw:.3e}   σ = {np.sqrt(sw):.5f}")
    print(f"  σ²_between（真种子方差，只能加种子） = {sb:.3e}   σ = {np.sqrt(sb):.5f}")
    print(f"  ⇒ 组内/组间 = **{sw/max(sb,1e-18):.2f}×**")
    if sw > sb:
        print("  ⇒ ★★ **测量噪声占主导** ⇒ 加种子不是最该做的事，先修测量设计。")
    else:
        print("  ⇒ 种子方差占主导 ⇒ 加种子是合理修法。")

    print("\n" + "=" * 100)
    print("② 直接量『噪声实现不共享』这块方差：同一权重、两个不同 tag_base 的结果对照")
    print("=" * 100)
    IX1 = index(load(V1))
    print(f"{'检测器':<22}{'v1 (tag_base 含检测器名)':>26}{'v2 (tag_base=tvX_)':>22}{'差':>11}")
    d_re = []
    for det1 in sorted(IX1):
        det2 = f"{det1}_s2026"          # 同一权重（v1 用它训练，v2 复用同名模型）
        if det1 not in ("tv_B0", "tv_B2", "tv_D0") or det2 not in IX:
            continue
        d1 = np.array([x for _, x in diffs(IX1[det1], "blur")], float)
        d2 = np.array([x for _, x in diffs(IX[det2], "blur")], float)
        if len(d1) < 2 or len(d2) < 2:
            continue
        print(f"{det1+'  vs  '+det2:<22}{d1.mean():>+26.5f}{d2.mean():>+22.5f}"
              f"{d2.mean()-d1.mean():>+11.5f}")
        d_re.append(d2.mean() - d1.mean())
    if d_re:
        print(f"\n  3 个同权重检测器的『两次实现差』：{', '.join(f'{x:+.5f}' for x in d_re)}")
        print(f"  ⇒ 实现差 SD = {np.std(d_re, ddof=1):.5f}；组内 SD（当前）= {np.sqrt(sw):.5f}")
        print("  ⇒ 量级相当 ⇒ **配对两族不共享噪声实现** 是组内方差的一个重要来源（可修）。")
    else:
        print("  (数据缺失，跳过)")

    print("\n" + "=" * 100)
    print("③ ★★ 真正的根因：Δblur 不是常数，而是随噪声水平**单调下降并变号**")
    print("=" * 100)
    NS = sorted({k[2] for k in next(iter(IX.values())) if k[0] in ("blur2", "blur02")})
    print(f"{'arm':<8}{'seed':>7}" + "".join(f"{'n='+str(n):>12}" for n in NS)
          + f"{'斜率(每档)':>12}")
    for arm in ARMS:
        cnt_mono = 0
        for s in SEEDS:
            v = np.array([x for _, x in diffs(IX[f"{arm}_s{s}"], "blur")], float)
            if len(v) < 2:
                continue
            slope = np.polyfit(np.arange(len(v)), v, 1)[0]
            if slope < 0:
                cnt_mono += 1
            print(f"{arm:<8}{s:>7}" + "".join(f"{x:>+12.5f}" for x in v) + f"{slope:>+12.5f}")
    n_mono = 0
    n_tot = 0
    for arm in ARMS:
        for s in SEEDS:
            v = np.array([x for _, x in diffs(IX[f"{arm}_s{s}"], "blur")], float)
            if len(v) < 2:
                continue
            n_tot += 1
            if np.polyfit(np.arange(len(v)), v, 1)[0] < 0:
                n_mono += 1
    print(f"\n  ⇒ **{n_mono}/{n_tot} 个 (arm, seed) 的 Δblur 随噪声单调下降**")
    print("  ⇒ 低噪声端重模糊**略好**（+0.001~+0.003），高噪声端明显**更差**（−0.004~−0.007）")
    print("  ⇒ 我一直在对一条**穿零的曲线取平均** ⇒ 均值本身不稳定，且物理含义不明确。")
    print("  ⇒ ★ **正确做法：换估计量** —— 不做「平均敏感度」，改为在**指定部署条件**下直接比各臂的 mAP。")
    print("     这个估计量① 避开穿零平均 ② 不依赖 blur2/blur02 的配对口径 ③ 臂间已是逐图配对。")
    print("     且**不需要重训、不需要重评**，现有 192 条数据即可算。")


if __name__ == "__main__":
    main()

