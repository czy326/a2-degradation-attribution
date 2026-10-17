"""引擎对照：检验「训练期模糊范围无效」是否 BSRGAN 特有的产物

对每个引擎 e ∈ {bsrgan, realesrgan, bicubic}：
  取两个**只差模糊范围**的臂：tv_A（blur U(0,0)）/ tv_D0（blur U(0.2,2.0)），均无噪声/JPEG
  在 16 个评测条件上，逐条件做种子水平配对检验 A − D0
  ⇒ 若各引擎下 A−D0 都 ≈ 0 且 0/16 显著 ⇒ 「模糊范围无效」不是引擎造成的
  ⇒ 若 bicubic 下 A−D0 显著变大 ⇒ 说明**学习式引擎把模糊吸收掉了**（叙事要改）

估计量沿用修正口径：指定 (blur, noise) 条件下的绝对 mAP + 相对自身 clean 的损失。
"""
import os
import json
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
BSRGAN = os.path.join(A2, "tv_sweep.jsonl")
ENG_FILES = {"realesrgan": os.path.join(A2, "tv_sweep.realesrgan.jsonl"),
             "bicubic": os.path.join(A2, "tv_sweep.bicubic.jsonl")}
ARMS = ["tv_A", "tv_D0"]


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
    """→ M[(arm, seed)][(blur, noise)] = mAP5095（只保留 ARMS）"""
    M = {}
    for r in recs:
        det = r["detector"]
        # ★ 引擎实验的 detector 带引擎前缀（如 realesrgan_tv_A_s2026 / bicubic_tv_D0_s2027），
        #   不剥掉就永远匹配不上 ARMS ⇒ 整批被**静默跳过**
        #   （本轮 realesrgan 96 条曾被误判为"无数据"，差点让判决表出不来）
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
    tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return tab.get(df, 2.0)


def ptest(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return (float(d.mean() / se) if se > 0 else float("nan"), float(d.mean()))


def analyse(name, recs):
    M = build(recs)
    if not M:
        print(f"\n[{name}] 无数据（还没跑完？）")
        return None
    SEEDS = sorted({s for (_, s) in M})
    conds = sorted({c for v in M.values() for c in v})
    tc = tcrit(len(SEEDS) - 1)
    print("\n" + "=" * 104)
    print(f"引擎 = {name}   种子 {SEEDS}（n={len(SEEDS)}，临界 |t|={tc:.3f}）   条件 {len(conds)} 个")
    print("=" * 104)
    REF = (0.0, 0.003)

    def cell(c, arm):
        vs = [M[(arm, s)].get(c) for s in SEEDS]
        return [v for v in vs if v is not None]

    print(f"{'条件':<20}{'A(tv_A)':>10}{'D0(宽模糊)':>12}{'A−D0':>11}{'t':>8}{'sig':>5}")
    dms, nsig = [], 0
    for c in conds:
        a, d = cell(c, "tv_A"), cell(c, "tv_D0")
        if len(a) != len(d) or not a:
            continue
        t, dm = ptest(a, d)
        sig = "*" if abs(t) > tc else ""
        if sig:
            nsig += 1
        dms.append(dm)
        tag = "  ←ref" if c == REF else ""
        print(f"blur={c[0]:<5} n={c[1]:<6}{np.mean(a):>10.4f}{np.mean(d):>12.4f}"
              f"{dm:>+11.5f}{t:>+8.2f}{sig:>5}{tag}")
    dms = np.array(dms, float)
    print(f"\n  ⇒ A−D0：同号 {max(int((dms > 0).sum()), int((dms < 0).sum()))}/{len(dms)}"
          f"，**显著 {nsig}/{len(dms)}**，|Δ| 中位数 {np.median(np.abs(dms)):.5f}"
          f"，范围 [{dms.min():+.5f}, {dms.max():+.5f}]，上界 {np.abs(dms).max():.5f}")

    # closure（部署条件下，覆盖模糊范围能补回多少）
    print(f"  closure(D0 相对 A)：")
    for c in conds:
        if c == REF:
            continue
        aR, dR = cell(REF, "tv_A"), cell(REF, "tv_D0")
        aC, dC = cell(c, "tv_A"), cell(c, "tv_D0")
        if not (aR and dR and aC and dC):
            continue
        zA = np.mean(aR) - np.mean(aC)
        zD = np.mean(dR) - np.mean(dC)
        cl = (zA - zD) / zA if abs(zA) > 1e-9 else float("nan")
        if abs(zA) > 0.002:      # 只在损失足够大的条件上报，避免除以近零
            print(f"     blur={c[0]:<5} n={c[1]:<6} Z(A)={zA:.5f} Z(D0)={zD:.5f}  "
                  f"closure {cl:+.1%}")
    return dict(seeds=SEEDS, nsig=nsig, n=len(dms), median=float(np.median(np.abs(dms))),
                maxabs=float(np.abs(dms).max()), dms=list(map(float, dms)))


def main():
    res = {}
    r = analyse("bsrgan", load(BSRGAN))
    if r:
        res["bsrgan"] = r
    for eng, path in ENG_FILES.items():
        r = analyse(eng, load(path))
        if r:
            res[eng] = r

    if len(res) >= 2:
        print("\n" + "=" * 104)
        print("★ 判决：各引擎下 A−D0（模糊范围效应）的显著条件数")
        print("=" * 104)
        print(f"{'engine':<14}{'显著条件数':>12}{'|Δ| 中位数':>13}{'|Δ| 上界':>11}   判读")
        for e, v in res.items():
            if e == "bsrgan":
                verdict = "（参考）"
            else:
                verdict = ("★ 同样无效 ⇒ 不是引擎产物" if v["nsig"] == 0
                           else f"⚠ 有 {v['nsig']}/{v['n']} 显著 ⇒ 引擎吸收了模糊！")
            sig = f"{v['nsig']}/{v['n']}"
            print(f"{e:<14}{sig:>12}{v['median']:>13.5f}{v['maxabs']:>11.5f}   {verdict}")
        json.dump(res, open(os.path.join(A2, "engine_check.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\nwrote engine_check.json")


if __name__ == "__main__":
    main()
