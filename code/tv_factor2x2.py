"""噪声 vs JPEG 解耦：宽模糊层的 2×2 因子分析

设计（四格 blur 范围严格相同 = U(0.20, 2.00)，唯一差别是噪声/JPEG 开关）
    D0 = (噪声 无, JPEG 无)     ← 基线
    N2 = (噪声 有, JPEG 无)
    J2 = (噪声 无, JPEG 有)
    B2 = (噪声 有, JPEG 有)

估计量（沿用 §7 口径 ③：指定部署条件下的绝对 mAP）
    M_c(det) = 检测器在部署条件 c=(blur, noise) 的 mAP50-95（各种子均值）
    Z_c(det) = M_ref(det) − M_c(det)，ref = 最轻条件 (blur=0.0, noise=0.003)
    closure_c(X) = (Z_c(D0) − Z_c(X)) / Z_c(D0)

统计
    逐**种子**配对（同一检测器、同一条件），n=6，df=5，95% 临界 |t| = 2.571。
    2×2 主效应在种子水平上先算格子组合、再跨种子做配对 t：
      E_noise = ((N2+B2)/2) − ((D0+J2)/2)
      E_jpeg  = ((J2+B2)/2) − ((D0+N2)/2)
      E_inter = B2 − N2 − J2 + D0

★ 纪律：**不对逐条件的效应取平均**当作"总效应"（§1.2 已证明这条曲线会随工作点变号，
   取平均是构造出来的）。改为报「同号条件数 + 显著条件数 + |Δ| 中位数」，与 §7/§8 一致。
"""
import os
import json
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
TV = os.path.join(A2, "tv_sweep.jsonl")

ARMS = {"D0": "tv_D0", "N2": "tv_N2", "J2": "tv_J2", "B2": "tv_B2"}
LBL = {"D0": "D0(宽+无)", "N2": "N2(宽+噪声)", "J2": "J2(宽+JPEG)", "B2": "B2(宽+噪声+JPEG)"}
REF = (0.0, 0.003)
DEPLOYS = [(2.0, 0.096), (2.0, 0.024), (0.2, 0.096), (0.0, 0.096), (2.4, 0.003)]


def tagp(t):
    b = n = None
    for p in t.split("__")[1].split("_"):
        if p.startswith("b"):
            b = round(float(p[1:]), 4)
        elif p.startswith("n"):
            n = round(float(p[1:]), 4)
    return b, n


def tcrit(df):
    tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if df <= 0:
        return float("inf")
    return tab.get(df, 1.96 if df > 10 else float("nan"))


def ptest(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return (float(d.mean() / se) if se > 0 else float("nan")), float(d.mean())


def load():
    M = {}
    if not os.path.isfile(TV):
        return M
    for ln in open(TV, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        det = r["detector"]
        arm = None
        for k, v in ARMS.items():
            if det == v or det.startswith(v + "_s"):
                arm = k
                break
        if arm is None:
            continue
        if "_s" not in det:
            continue
        seed = int(det.rpartition("_s")[2])
        b, n = tagp(r["tag"])
        if b is None or n is None:
            continue
        M.setdefault((arm, seed), {}).setdefault((b, n), []).append(r["mAP5095"])
    # 同 (arm,seed,cond) 若有多条（分片重复），取均值
    for k in M:
        for c in M[k]:
            M[k][c] = float(np.mean(M[k][c]))
    return M


def main():
    M = load()
    seeds = sorted({s for (_, s) in M})
    if not seeds:
        print("tv_sweep.jsonl 里还没有 tv_N2 / tv_J2 的任何结果 —— 实验尚未产出。")
        return
    tc = tcrit(len(seeds) - 1)
    conds = sorted({c for v in M.values() for c in v}, key=lambda c: (c[1], c[0]))
    print(f"种子 {seeds}（n={len(seeds)}，df={len(seeds) - 1}，95% 临界 |t| = {tc:.3f}）")
    print(f"可用条件 {len(conds)} 个；四格检测器 "
          f"{sorted({a for (a, _) in M})}")

    # 完整性
    need = {(a, s, c) for a in ARMS for s in seeds for c in conds}
    have = {(a, s, c) for (a, s), v in M.items() for c in v}
    miss = need - have
    if miss:
        print(f"\n⚠ 数据不全：缺 {len(miss)} 格（{len(need)} 总格）⇒ 下列分析可能报 n/a")
        print(f"   例：{sorted(miss)[:6]}")

    def cell(a, s, c):
        return M.get((a, s), {}).get(c)

    # ① clean 端（ref 条件）绝对水平
    print("\n" + "=" * 96)
    print("① clean 端基线（blur=0.0, noise=0.003）—— 训练加噪声/JPEG 的**能力代价**")
    print("=" * 96)
    print(f"{'臂':<20}{'逐种子 mAP5095':<56}{'均值':>10}")
    refm = {}
    for k in ("D0", "N2", "J2", "B2"):
        vs = [cell(k, s, REF) for s in seeds]
        vs = [v for v in vs if v is not None]
        refm[k] = vs
        arr = " ".join(f"{v:.4f}" for v in vs)
        print(f"{LBL[k]:<20}{arr:<56}{(np.mean(vs) if vs else float('nan')):>10.4f}")
    if all(refm.values()):
        d0 = np.mean(refm["D0"])
        for k in ("N2", "J2", "B2"):
            print(f"   ⇒ {k} 相对 D0 的 clean 代价："
                  f"{(np.mean(refm[k]) - d0) / d0:+.1%}")

    # ② 逐条件 2×2 主效应
    print("\n" + "=" * 96)
    print("② 逐部署条件 2×2 主效应（种子水平配对，n=%d，* = |t| > %.3f）" % (len(seeds), tc))
    print("=" * 96)
    print(f"{'条件':<20}{'E_noise':>12}{'t':>8}{'E_jpeg':>12}{'t':>8}{'E_inter':>12}{'t':>8}")
    rows = []
    for c in conds:
        en = [cell("N2", s, c) for s in seeds]
        eb = [cell("B2", s, c) for s in seeds]
        ed = [cell("D0", s, c) for s in seeds]
        ej = [cell("J2", s, c) for s in seeds]
        if any(x is None for x in en + eb + ed + ej):
            continue
        dn = [(a + b) / 2 - (d + j) / 2 for a, b, d, j in zip(en, eb, ed, ej)]
        dj = [(j + b) / 2 - (d + a) / 2 for a, b, d, j in zip(en, eb, ed, ej)]
        di = [b - a - j + d for a, b, d, j in zip(en, eb, ed, ej)]
        tn, mn = ptest(dn)
        tj, mj = ptest(dj)
        ti, mi = ptest(di)
        rows.append((c, mn, tn, mj, tj, mi, ti))
        print(f"{f'blur={c[0]:<4} n={c[1]}':<20}{mn:>+12.5f}{tn:>8.2f}"
              f"{'*' if abs(tn) > tc else ' ':>1}{mj:>+12.5f}{tj:>8.2f}"
              f"{'*' if abs(tj) > tc else ' ':>1}{mi:>+12.5f}{ti:>8.2f}"
              f"{'*' if abs(ti) > tc else ' ':>1}")

    # ③ 简单对照汇总（同号 / 显著 / |Δ|）
    print("\n" + "=" * 96)
    print("③ 简单对照在 %d 个条件上的稳定性" % len(conds))
    print("=" * 96)
    CONTRASTS = [("N2", "D0"), ("J2", "D0"), ("B2", "D0"), ("B2", "N2"), ("B2", "J2")]
    print(f"{'对照':<14}{'同号条件数':>12}{'显著条件数':>12}{'均值符号':>10}"
          f"{'|Δ| 中位数':>13}{'|Δ| 范围':>24}")
    summary = {}
    for a1, a2 in CONTRASTS:
        dms, nsig = [], 0
        for c in conds:
            v1 = [cell(a1, s, c) for s in seeds]
            v2 = [cell(a2, s, c) for s in seeds]
            if any(x is None for x in v1 + v2):
                continue
            t, dm = ptest([x - y for x, y in zip(v1, v2)])
            dms.append(dm)
            if abs(t) > tc:
                nsig += 1
        if not dms:
            continue
        dms = np.array(dms)
        pos = int((dms > 0).sum())
        same = max(pos, len(dms) - pos)
        sign = "+" if dms.mean() > 0 else "−"
        summary[f"{a1}-{a2}"] = dict(n_same=same, n_tot=len(dms), n_sig=nsig,
                                     med=float(np.median(np.abs(dms))),
                                     lo=float(dms.min()), hi=float(dms.max()))
        print(f"{a1 + '-' + a2:<14}{f'{same}/{len(dms)}':>12}{f'{nsig}/{len(dms)}':>12}"
              f"{sign:>10}{np.median(np.abs(dms)):>13.5f}"
              f"{f'[{dms.min():+.5f}, {dms.max():+.5f}]':>24}")

    # ④ closure
    print("\n" + "=" * 96)
    print("④ closure p%：加'仅噪声'/'仅 JPEG'/'两者'分别补回了 D0 的多少损失")
    print("=" * 96)
    def Z(k, c):
        vr = [cell(k, s, REF) for s in seeds]
        vc = [cell(k, s, c) for s in seeds]
        if any(x is None for x in vr + vc):
            return None
        return np.array(vr, float) - np.array(vc, float)
    print(f"{'部署条件':<18}{'Z(D0)':>11}{'Z(N2)':>11}{'Z(J2)':>11}{'Z(B2)':>11}"
          f"{'cl(N2)':>10}{'cl(J2)':>10}{'cl(B2)':>10}")
    for c in DEPLOYS:
        zs = {k: Z(k, c) for k in ("D0", "N2", "J2", "B2")}
        if any(v is None for v in zs.values()):
            continue
        z0 = zs["D0"].mean()
        cells = "".join(f"{zs[k].mean():>11.5f}" for k in ("D0", "N2", "J2", "B2"))
        cls = "".join(f"{(z0 - zs[k].mean()) / z0:>9.1%}" if z0 else f"{'n/a':>10}"
                      for k in ("N2", "J2", "B2"))
        print(f"{f'blur={c[0]:<4} n={c[1]}':<18}{cells}{cls}")

    # ⑤ clean 端剂量-响应斜率（对 blur）：X − D0 随 blur 的斜率
    print("\n" + "=" * 96)
    print("⑤ clean 端斜率（noise=0.003，对 blur 线性拟合）—— 正值 = X 比 D0 更抗模糊")
    print("=" * 96)
    bgrid = sorted({b for (b, n) in conds if n == 0.003})
    print(f"blur 网格 {bgrid}")
    print(f"{'对照':<12}{'逐种子斜率':<58}{'均值':>10}{'t':>8}{'显著':>6}")
    for a1, a2 in [("N2", "D0"), ("J2", "D0"), ("B2", "D0")]:
        ss = []
        okall = True
        for s in seeds:
            xs, ys = [], []
            for b in bgrid:
                v1 = cell(a1, s, (b, 0.003))
                v2 = cell(a2, s, (b, 0.003))
                if v1 is None or v2 is None:
                    okall = False
                    break
                xs.append(b)
                ys.append(v1 - v2)
            if not okall:
                break
            xs = np.asarray(xs, float)
            ys = np.asarray(ys, float)
            den = float(((xs - xs.mean()) ** 2).sum())
            ss.append(float(((xs - xs.mean()) * (ys - ys.mean())).sum() / den) if den else float("nan"))
        if not okall or not ss:
            print(f"{a1 + '-' + a2:<12}  n/a")
            continue
        t, m = ptest(ss)
        arr = " ".join(f"{v:+.5f}" for v in ss)
        print(f"{a1 + '-' + a2:<12}{arr:<58}{m:>+10.5f}{t:>8.2f}{('是' if abs(t) > tc else '否'):>6}")

    # 判决
    print("\n" + "=" * 96)
    print("★ 判决：噪声 / JPEG 谁在起作用")
    print("=" * 96)
    j = summary.get("B2-D0")
    n_only = summary.get("N2-D0")
    j_only = summary.get("J2-D0")
    if j and n_only and j_only:
        def frag(x):
            return f"显著 {x['n_sig']}/{x['n_tot']}，|Δ| 中位 {x['med']:.5f}"
        print(f"  仅噪声 (N2−D0)：{frag(n_only)}")
        print(f"  仅 JPEG (J2−D0)：{frag(j_only)}")
        print(f"  两者  (B2−D0)：{frag(j)}")
        big_n = n_only["med"] > 2.5 * max(j_only["med"], 1e-9)
        big_j = j_only["med"] > 2.5 * max(n_only["med"], 1e-9)
        if big_n:
            print("  ⇒ 效应主要由**噪声**贡献（仅 JPEG 的幅度不足其 1/2.5）")
        elif big_j:
            print("  ⇒ 效应主要由 **JPEG** 贡献（仅噪声的幅度不足其 1/2.5）")
        else:
            print("  ⇒ 两者**都有实质贡献**（幅度同量级）⇒ 视作「噪声+JPEG 联合」更稳妥")

    json.dump({f"{a}|s{s}|b{c[0]}|n{c[1]}": M.get((a, s), {}).get(c)
               for (a, s) in M for c in [k for k in M[(a, s)]]},
              open(os.path.join(A2, "tv_factor2x2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nwrote tv_factor2x2.json")


if __name__ == "__main__":
    main()
