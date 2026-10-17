"""TV 扫描判决 —— 用正确的配对方式（数据已在 tv_sweep.jsonl，无需重跑 GPU）

上一版为什么失败
  `paired_delta()` 想在**重叠 PSNR 区间**上插值配对。但本管线的 PSNR 由**噪声主导**：
  在固定噪声下把 blur 从 0 扫到 2.4，PSNR 只动 **0.10 dB**（BSRGAN 把模糊基本重构掉了）。
  于是 nhi(19.2 dB) 与 nlo(29.1 dB) 两族**完全没有重叠**⇒ 噪声曲线算不出来。

正确的配对（本脚本）
  ① 模糊效应：**固定噪声**逐点配对  Δblur(n) = mAP(blur2, n) − mAP(blur02, n)
     （因为 blur 几乎不改 PSNR，同噪声即同 PSNR ⇒ 配对天然成立）
  ② 噪声效应：**固定模糊**逐点配对  Δnoise(b) = mAP(nhi, b) − mAP(nlo, b)
     （跨检测器比较同一对比时无需 PSNR 匹配；PSNR 匹配只在"解释该效应独立于 PSNR"时才需要）

参照系修正
  上一版把 tv_B0 拿去比对 D0/D1/HR_oracle 是**错的** —— 那三个的训练管线**没有噪声和 JPEG**，
  而 tv_B0/tv_B2 有。管线相同的参照只有 `D2_highorder`（同结构，仅模糊范围 0.2–1.2 不同）。
"""
import os
import json
import numpy as np

A2 = os.path.dirname(os.path.abspath(__file__))
TV = os.path.join(A2, "tv_sweep.jsonl")
CROSS = os.path.join(A2, "blur_indep_cross.jsonl")

SHARED_N = [0.003, 0.012, 0.024, 0.096]      # tv 用的噪声格（与 D2 的 6 点取交集）
TV_DETS = ["tv_B0", "tv_B2", "D2_highorder"]  # 同管线族：都有 噪声 + JPEG


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def parse_tag(tag):
    """blur2__b2_n0.003  → (2.0, 0.003)   ｜  nhi__n0.096_b0 → (0.0, 0.096)"""
    b = n = None
    for part in tag.split("__")[1].split("_"):
        if part.startswith("b"):
            b = float(part[1:])
        elif part.startswith("n"):
            n = float(part[1:])
    return b, n


def build(records):
    """→ {(det, family, blur, noise): mAP}"""
    M = {}
    for r in records:
        b, n = parse_tag(r["tag"])
        if b is None or n is None:
            continue
        M[(r["detector"], r["family"], round(b, 4), round(n, 4))] = r["mAP5095"]
    return M


def collect_pairs(M, det, fa, fb, axis, keys):
    """axis='noise'：固定每个噪声值 n，取 fa/fb 在该 n 下（任意 blur 但需一一对应）。
    由于记录里 blur/noise 都在 tag 里，这里按 axis 的另一坐标配对。"""
    out = []
    if axis == "noise":
        # 固定 n，配对 (b=2.0, n) 与 (b=0.2, n)
        for n in keys:
            a = M.get((det, "blur2", 2.0, n))
            b = M.get((det, "blur02", 0.2, n))
            if a is not None and b is not None:
                out.append(a - b)
    else:
        # 固定 b，配对 (nhi, b) 与 (nlo, b)
        for b in keys:
            a = M.get((det, "nhi", b, 0.096))
            c = M.get((det, "nlo", b, 0.003))
            if a is not None and c is not None:
                out.append(a - c)
    return np.array(out, float)


def stat(d):
    if len(d) < 2:
        return None
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return dict(mean=m, se=se, t=float(m / se) if se > 0 else float("nan"),
                n=len(d), n_neg=int((d < 0).sum()))


def main():
    tv = load(TV)
    cr = load(CROSS)
    M = build(tv)
    # D2 的参照数据来自 blur_indep_cross.jsonl（family=blur2/blur02，tag 同格式）
    Mc = build(cr)

    print("=" * 100)
    print("① 模糊效应 Δblur（固定噪声逐点配对：mAP[blur2] − mAP[blur02]）")
    print("=" * 100)
    print(f"{'detector':<16}{'PSNR匹配法':>13}{'固定噪声法':>13}{'se':>10}{'t':>8}{'负/总':>8}")
    blur = {}
    base = None
    for det in TV_DETS:
        d = collect_pairs(M, det, "blur2", "blur02", "noise", SHARED_N)
        s = stat(d)
        if s:
            blur[det] = s
            print(f"{det:<16}{'':>13}{s['mean']:>+13.5f}{s['se']:>10.5f}"
                  f"{s['t']:>+8.2f}{str(s['n_neg'])+'/'+str(s['n']):>8}")
        # 若本文件里没有（比如 D2 不在 tv_sweep.jsonl），改从 cross 取
        if s is None:
            d2 = collect_pairs(Mc, det, "blur2", "blur02", "noise", SHARED_N)
            s2 = stat(d2)
            if s2:
                blur[det] = s2
                print(f"{det:<16}{'':>13}{s2['mean']:>+13.5f}{s2['se']:>10.5f}"
                      f"{s2['t']:>+8.2f}{str(s2['n_neg'])+'/'+str(s2['n']):>8}")
            else:
                print(f"{det:<16}  (无数据)")

    print("\n" + "=" * 100)
    print("② 噪声效应 Δnoise（固定模糊逐点配对：mAP[nhi 0.096] − mAP[nlo 0.003]）")
    print("=" * 100)
    BLUR_KEYS = [0.0, 0.6, 1.2, 2.4]
    print(f"{'detector':<16}{'Δnoise':>13}{'se':>10}{'t':>8}{'负/总':>8}")
    noise = {}
    for det in ["tv_B0", "tv_B2"]:
        d = collect_pairs(M, det, "nhi", "nlo", "blur", BLUR_KEYS)
        s = stat(d)
        if s:
            noise[det] = s
            print(f"{det:<16}{s['mean']:>+13.5f}{s['se']:>10.5f}{s['t']:>+8.2f}"
                  f"{str(s['n_neg'])+'/'+str(s['n']):>8}")
        else:
            print(f"{det:<16}  (无数据)")

    print("\n" + "=" * 100)
    print("③ 判决：H_spec（维度特异） vs H_gen（一般鲁棒）")
    print("=" * 100)
    if "tv_B0" in blur and "tv_B2" in blur and "tv_B0" in noise and "tv_B2" in noise:
        ab0, ab2 = abs(blur["tv_B0"]["mean"]), abs(blur["tv_B2"]["mean"])
        an0, an2 = abs(noise["tv_B0"]["mean"]), abs(noise["tv_B2"]["mean"])
        rb, rn = ab2 / ab0, an2 / an0
        print(f"  |Δblur| : tv_B0 {ab0:.5f}  →  tv_B2 {ab2:.5f}   比值 {rb:.2f}")
        print(f"  |Δnoise|: tv_B0 {an0:.5f}  →  tv_B2 {an2:.5f}   比值 {rn:.2f}")
        print()
        if rb < 0.75 and 0.6 < rn < 1.6:
            print("  ⇒ ★★ **H_spec 成立**：加宽训练模糊只钝化『模糊』这一维，噪声敏感性几乎不变。")
            print("     ⇒ 『模型依赖』是**维度特异的不变性**被训练出来的结果 ⇒ 可预测、可控制。")
        elif rb < 0.75 and rn < 0.6:
            print("  ⇒ H_gen：两者同步下降 ⇒ 只是一般鲁棒性。")
        elif rb >= 0.75:
            print(f"  ⇒ 模糊方向未达阈值（比值 {rb:.2f} ≥ 0.75）⇒ 不判活。")
        else:
            print(f"  ⇒ 中间态（rb={rb:.2f}, rn={rn:.2f}）⇒ 需补臂或补种子。")

    print("\n" + "=" * 100)
    print("④ 同管线族的三点剂量-响应（D2 与 tv_B0/tv_B2 结构相同，仅模糊范围不同）")
    print("=" * 100)
    SPAN = {"tv_B0": "U(0, 0.05)", "D2_highorder": "U(0.2, 1.2)", "tv_B2": "U(0.2, 2.0)"}
    for det in ["tv_B0", "D2_highorder", "tv_B2"]:
        if det in blur:
            print(f"  {det:<14}{SPAN[det]:<16}|Δblur| = {abs(blur[det]['mean']):.5f}"
                  f"   (t={blur[det]['t']:+.2f})")
    if all(d in blur for d in SPAN):
        v = [abs(blur[d]["mean"]) for d in ["tv_B0", "D2_highorder", "tv_B2"]]
        print(f"\n  单调递减？ {'★ 是' if v[0] > v[1] > v[2] else '否'}  "
              f"（{v[0]:.5f} > {v[1]:.5f} > {v[2]:.5f}）⇒ 跨度 {v[0]/v[2]:.2f} 倍")

    print("\n" + "=" * 100)
    print("⑤ 副产品：PSNR 对『帧前模糊』几乎失明（解释上一版为何失败）")
    print("=" * 100)
    for det in ["tv_B0", "tv_B2"]:
        rows = []
        for n in SHARED_N:
            v = []
            for fam, b in (("blur02", 0.2), ("blur2", 2.0)):
                rec = [r for r in tv if r["detector"] == det and r["family"] == fam
                       and abs(parse_tag(r["tag"])[1] - n) < 1e-9]
                if rec:
                    v.append(rec[0]["psnr"])
            if len(v) == 2:
                rows.append((n, v[0], v[1], v[1] - v[0]))
        print(f"  {det}:")
        print(f"    {'noise':>8}{'PSNR(blur0.2)':>15}{'PSNR(blur2.0)':>15}{'差':>8}")
        for n, a, b, dd in rows:
            print(f"    {n:>8.3f}{a:>15.2f}{b:>15.2f}{dd:>+8.2f}")
        if rows:
            dd = np.array([r[3] for r in rows])
            print(f"    ⇒ 同噪声下 blur 0.2→2.0 引起的 PSNR 变化："
                  f"均值 {dd.mean():+.2f} dB，幅度 {np.abs(dd).max():.2f} dB")

    json.dump(dict(blur={k: v for k, v in blur.items()},
                   noise={k: v for k, v in noise.items()}),
              open(os.path.join(A2, "tv_verdict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nwrote tv_verdict.json")


if __name__ == "__main__":
    main()
