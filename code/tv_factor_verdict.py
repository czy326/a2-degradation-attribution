"""噪声/JPEG 解耦实验的**机械判决**（把 `FACTOR_PREREG.md` §4 的规则写成代码执行）。

为什么单独写一个
  `tv_factor2x2.py` 只做描述性统计 + 一句人话结论。预注册纪律要求**判据由代码执行**，
  不能靠人眼从表里读数 —— 否则"事后改判"无法被排除。
  本脚本严格照抄 §4 的四条互斥规则，命中哪条就报哪条，没命中就报"未判死"。

用法
  python tv_factor_verdict.py
"""
import collections
import json
import os

import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
TV = os.path.join(A2, "tv_sweep.jsonl")
OUT_JSON = os.path.join(A2, "factor_verdict.json")
OUT_MD = os.path.join(A2, "FACTOR_VERDICT.md")

ARMS = {"D0": "tv_D0", "N2": "tv_N2", "J2": "tv_J2", "B2": "tv_B2"}
SEEDS = [2026, 2027, 2028, 2029, 2030, 2031]
TCRIT = 2.571                       # df=5, 95%
REF = (0.0, 0.003)
# ★ §4 的阈值（一个字都不改）
SIG_MIN = 6
SAME_NOISE = 12
SAME_JPEG = 12
SIG_MAX_OTHER = 2
RATIO_LO, RATIO_HI = 1 / 2.5, 2.5


def tagp(t):
    b = n = None
    for p in t.split("__")[1].split("_"):
        if p.startswith("b"):
            b = round(float(p[1:]), 4)
        elif p.startswith("n"):
            n = round(float(p[1:]), 4)
    return b, n


def load():
    """→ M[arm][seed][(b, n)] = mAP5095（检测器名形如 tv_D0_s2026）"""
    M = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(TV, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        det = r["detector"]
        arm = next((a for a in ARMS.values() if det.startswith(a + "_")), None)
        v = r.get("mAP5095")
        if arm is None or v is None:
            continue
        key = next(k for k, a in ARMS.items() if a == arm)          # tv_D0_s2026
        tail = det.rsplit("_", 1)[1]                                 # "s2026"
        M[key][int("".join(ch for ch in tail if ch.isdigit()))][tagp(r["tag"])] = float(v)
    return M


def paired(d):
    d = np.asarray(d, float)
    if d.size < 2:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / np.sqrt(d.size)
    return (float(d.mean() / se) if se > 0 else float("nan")), float(d.mean())


def contrast(M, X, Y):
    """逐条件配对（同种子、同条件）→ {cond: (t, mean)}，条件取两臂都有的"""
    out = {}
    for c in sorted(set(M[X][SEEDS[0]]) & set(M[Y][SEEDS[0]])):
        d = [M[X][s][c] - M[Y][s][c] for s in SEEDS
             if c in M[X][s] and c in M[Y][s]]
        if len(d) == len(SEEDS):
            out[c] = paired(d)
    return out


def summarize(con, positive_is_better=True):
    sig = [c for c, (t, m) in con.items() if abs(t) > TCRIT]
    signs = [np.sign(m) for _, m in con.values()]
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    same = max(pos, neg)
    med = float(np.median([abs(m) for _, m in con.values()]))
    return dict(n=len(con), sig=len(sig), pos=pos, neg=neg, same=same,
                median_abs=med, sign="+" if pos >= neg else "-",
                sig_conds=sig)


def main():
    M = load()
    n_seed = {k: len([s for s in SEEDS if M[k].get(s)]) for k in ARMS}
    print("=" * 88)
    print("噪声 / JPEG 解耦 —— 机械判决（规则来自 FACTOR_PREREG.md §4）")
    print("=" * 88)
    print(f"四格种子覆盖：{n_seed}")
    assert all(v == 6 for v in n_seed.values()), "四格必须各有 6 个种子"

    con = {k: contrast(M, *k.split("_minus_")) for k in
           ("N2_minus_D0", "J2_minus_D0", "B2_minus_D0",
            "B2_minus_N2", "B2_minus_J2")}
    S = {k: summarize(v) for k, v in con.items()}

    print()
    print(f"{'对照':<14}{'条件数':<7}{'显著':<6}{'同号':<6}{'符号':<6}{'|Δ|中位':<11}")
    print("-" * 52)
    for k, s in S.items():
        print(f"{k.replace('_minus_', '−'):<14}{s['n']:<7}{s['sig']:<6}"
              f"{s['same']:<6}{s['sign']:<6}{s['median_abs']:<11.5f}")

    # ---------------- §4 规则 ----------------
    sigN = S["N2_minus_D0"]["sig"]
    sigJ = S["J2_minus_D0"]["sig"]
    sameN = S["N2_minus_D0"]["same"]
    sameJ = S["J2_minus_D0"]["same"]
    medN = S["N2_minus_D0"]["median_abs"]
    medJ = S["J2_minus_D0"]["median_abs"]
    ratio = medN / medJ if medJ > 0 else float("inf")

    rules = []
    rules.append(("H_noise（噪声主导）",
                  sigN >= SIG_MIN and sameN >= SAME_NOISE and sigJ <= SIG_MAX_OTHER,
                  f"sig(N2−D0)={sigN}≥{SIG_MIN} ∧ same(N2−D0)={sameN}≥{SAME_NOISE} "
                  f"∧ sig(J2−D0)={sigJ}≤{SIG_MAX_OTHER}"))
    rules.append(("H_jpeg（JPEG 主导）",
                  sigJ >= SIG_MIN and sameJ >= SAME_JPEG and sigN <= SIG_MAX_OTHER,
                  f"sig(J2−D0)={sigJ}≥{SIG_MIN} ∧ same(J2−D0)={sameJ}≥{SAME_JPEG} "
                  f"∧ sig(N2−D0)={sigN}≤{SIG_MAX_OTHER}"))
    rules.append(("H_joint（两者都实质贡献）",
                  sigN >= SIG_MIN and sigJ >= SIG_MIN
                  and RATIO_LO <= ratio <= RATIO_HI,
                  f"sig(N2−D0)={sigN}≥{SIG_MIN} ∧ sig(J2−D0)={sigJ}≥{SIG_MIN} "
                  f"∧ 中位比 {ratio:.3f} ∈ [{RATIO_LO:.2f},{RATIO_HI:.2f}]"))
    rules.append(("H_null（都无效）",
                  sigN <= SIG_MAX_OTHER and sigJ <= SIG_MAX_OTHER,
                  f"sig(N2−D0)={sigN}≤{SIG_MAX_OTHER} ∧ sig(J2−D0)={sigJ}≤{SIG_MAX_OTHER}"))

    print()
    print("=" * 88)
    print("§4 四条互斥规则逐条核对")
    print("=" * 88)
    hit = []
    for name, ok, why in rules:
        print(f"  [{'✔' if ok else '✘'}] {name}")
        print(f"        {why}")
        if ok:
            hit.append(name)
    print()
    if len(hit) == 1:
        verdict = hit[0]
        print(f"★ 判决：{verdict}（唯一命中，规则互斥性自检通过）")
    elif not hit:
        verdict = "未判死（不落入任何一条）"
        print(f"★ 判决：{verdict} —— 如实说明，不硬套结论")
    else:
        verdict = "★ 规则冲突：" + " / ".join(hit)
        print(f"★ {verdict} —— 互斥性被破坏，需检查实现")

    # ---------------- §6 反向验证 ----------------
    print()
    print("=" * 88)
    print("§6 反向验证（负向对照）")
    print("=" * 88)
    same_conds = []
    for c in con["N2_minus_D0"]:
        diff = abs(M["N2"][SEEDS[0]][c] - M["J2"][SEEDS[0]][c])
        same_conds.append((c, diff))
    mx = max(d for _, d in same_conds)
    print(f"  ① N2 vs J2 逐条件最大差值 = {mx:.6f}  "
          f"{'★ 判为实现错误（<1e-6）' if mx < 1e-6 else '✔ 未触发（两者确实不同）'}")
    refs = {k: float(np.mean([M[k][s][REF] for s in SEEDS])) for k in ARMS}
    print(f"  ② ref 条件 {REF} 上的 mAP5095："
          + "  ".join(f"{k}={v:.4f}" for k, v in refs.items()))
    cost = {k: (refs["D0"] - refs[k]) / refs["D0"] for k in ("N2", "J2", "B2")}
    print(f"     相对 D0 的能力代价：" + "  ".join(f"{k}={v:+.2%}" for k, v in cost.items()))
    flag_ok = all(v > 0.05 for v in cost.values())
    print(f"     {'✔ 三格都有明显代价 ⇒ 噪声/JPEG 开关确实生效' if flag_ok else '★ 代价太小 ⇒ 先查视图开关'}")

    res = dict(verdict=verdict, S=S, ratio=ratio,
               refs={k: v for k, v in refs.items()},
               clean_cost={k: v for k, v in cost.items()},
               rules=[dict(name=n, hit=o, why=w) for n, o, w in rules])
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
