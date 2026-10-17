"""引擎对照补充：A−D0 随 blur 范围的「剂量-响应斜率」，分 noise 水平逐种子配对 t

背景：三引擎判决表（engine_check.py）显示 A−D0 整体 0/16 显著。
      但 bicubic 的 |Δ| 上界（0.00465）是另两个学习式引擎（~0.0009）的 5 倍，
      且集中在 **clean 端（n=0.003）且符号为负** —— 即「训练用宽模糊，clean 部署反而更差」。
      engine_check 的逐条件 t 检验会被 3 种子临界值（4.303）挡住，
      看不出这是个**跨 blur 的单调趋势**。

⇒ 本脚本把「模糊范围效应」的斜率（mAP / blur 单位）逐 noise 水平量化并检验。
   斜率是跨 6 个 blur 点的整体拟合 ⇒ 比单点比较灵敏得多。
"""
import os
import json
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
ENG_FILES = [
    ("bsrgan", os.path.join(A2, "tv_sweep.jsonl")),
    ("realesrgan", os.path.join(A2, "tv_sweep.realesrgan.jsonl")),
    ("bicubic", os.path.join(A2, "tv_sweep.bicubic.jsonl")),
]
ARMS = ("tv_A", "tv_D0")


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


def build(recs):
    M = {}
    for r in recs:
        det = r["detector"]
        # 引擎实验的 detector 带引擎前缀，需剥掉再配对（否则整批静默跳过）
        for pref in ("realesrgan_", "bicubic_", "bsrgan_"):
            if det.startswith(pref):
                det = det[len(pref):]
                break
        arm, _, sdx = det.rpartition("_s")
        if arm not in ARMS or not sdx.isdigit():
            continue
        b, n = tagp(r["tag"])
        if b is None or n is None:
            continue
        M.setdefault((arm, int(sdx)), {})[(b, n)] = r["mAP5095"]
    return M


def tcrit(df):
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(df, 2.0)


def slope(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    xm = xs.mean()
    den = float(((xs - xm) ** 2).sum())
    if den <= 0:
        return float("nan")
    return float(((xs - xm) * (ys - ys.mean())).sum() / den)


out = {}
for eng, path in ENG_FILES:
    M = build(load(path))
    if not M:
        print(f"\n[{eng}] 无数据（还没跑完？）")
        continue
    SEEDS = sorted({s for (_, s) in M})
    tc = tcrit(len(SEEDS) - 1)
    ns = sorted({c[1] for v in M.values() for c in v})
    print("\n" + "=" * 96)
    print(f"引擎 = {eng}   种子 {SEEDS}（n={len(SEEDS)}，临界 |t|={tc:.3f}）")
    print("=" * 96)
    print("A−D0 的斜率（mAP / blur 单位）—— 正值=无模糊训练更好；负值=宽模糊训练更好")
    print(f"{'noise':>9}   {'逐种子斜率':<40}{'均值':>11}{'t':>8}{'sig':>5}")
    res = {}
    for n in ns:
        blurs = sorted({c[0] for v in M.values() for c in v if c[1] == n})
        blurs = [b for b in blurs
                 if all(("tv_A", s) in M and ("tv_D0", s) in M
                        and (b, n) in M[("tv_A", s)] and (b, n) in M[("tv_D0", s)]
                        for s in SEEDS)]
        if len(blurs) < 3:
            continue
        sl = []
        for s in SEEDS:
            ys = [M[("tv_A", s)][(b, n)] - M[("tv_D0", s)][(b, n)] for b in blurs]
            sl.append(slope(blurs, ys))
        sl = np.asarray(sl, float)
        m = float(sl.mean())
        sd = float(sl.std(ddof=1)) if len(sl) > 1 else float("nan")
        se = sd / np.sqrt(len(sl)) if len(sl) > 1 else float("nan")
        t = m / se if se and se > 0 else float("nan")
        sig = "*" if (t == t and abs(t) > tc) else ""
        res[n] = dict(slope=m, sd=sd, t=float(t), sig=bool(sig),
                      blurs=blurs, n_blur=len(blurs))
        cells = " ".join(f"{v:+.5f}" for v in sl)
        print(f"{n:>9}   {cells:<40}{m:>+11.5f}{t:>+8.2f}{sig:>5}")
    out[eng] = res

print("\n" + "=" * 96)
print("★ 三引擎「clean 端（最小 noise）斜率」对比 —— 检验「学习式引擎是否吸收模糊覆盖」")
print("=" * 96)
print(f"{'engine':<14}{'noise':>9}{'斜率':>12}{'t':>9}{'显著':>6}   blur 网格")
rows = {}
for eng, res in out.items():
    if not res:
        continue
    nmin = min(res)
    r = res[nmin]
    rows[eng] = dict(noise=nmin, slope=r["slope"], t=r["t"], sig=r["sig"], n_blur=r["n_blur"])
    print(f"{eng:<14}{nmin:>9}{r['slope']:>+12.5f}{r['t']:>+9.2f}"
          f"{('是' if r['sig'] else '否'):>6}   {r['blurs']}")

if len(rows) >= 2:
    learn = [v["slope"] for k, v in rows.items() if k in ("bsrgan", "realesrgan")]
    print(f"\n  学习式引擎（bsrgan/realesrgan）平均斜率 {np.mean(learn):+.5f}；"
          f"bicubic（无学习超分）{rows.get('bicubic', {}).get('slope', float('nan')):+.5f}")

# ---- clean 端基线：为什么 bicubic 的绝对 mAP 反而更高？（影响斜率能否直接跨引擎比较）----
print("\n" + "=" * 96)
print("★ clean 端基线（blur=0.0, n=0.003）—— 绝对水平 / 背景假阳性 / 输入 PSNR")
print("=" * 96)
print(f"{'engine':<14}{'arm':<8}{'mAP5095':>10}{'fp_bg_rate':>12}{'psnr':>9}{'n_seed':>8}")
base = {}
for eng, path in ENG_FILES:
    M = {}
    for r in load(path):
        det = r["detector"]
        for pref in ("realesrgan_", "bicubic_", "bsrgan_"):
            if det.startswith(pref):
                det = det[len(pref):]
                break
        arm, _, sdx = det.rpartition("_s")
        if arm not in ARMS or not sdx.isdigit():
            continue
        b, n = tagp(r["tag"])
        if b == 0.0 and n == 0.003:
            M.setdefault(arm, []).append((r["mAP5095"], r.get("fp_bg_rate"), r.get("psnr")))
    for arm in ARMS:
        if arm not in M:
            continue
        a = np.asarray([x[0] for x in M[arm]], float)
        f = np.asarray([x[1] for x in M[arm] if x[1] is not None], float)
        p = np.asarray([x[2] for x in M[arm] if x[2] is not None], float)
        print(f"{eng:<14}{arm:<8}{a.mean():>10.4f}"
              f"{(f.mean() if len(f) else float('nan')):>12.4f}"
              f"{(p.mean() if len(p) else float('nan')):>9.3f}{len(a):>8}")
        base[(eng, arm)] = dict(mAP=float(a.mean()),
                                fp=float(f.mean()) if len(f) else None,
                                psnr=float(p.mean()) if len(p) else None)

json.dump({k: {str(n): v for n, v in r.items()} for k, r in out.items()},
          open(os.path.join(A2, "eng_clean_slope.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwrote eng_clean_slope.json")
