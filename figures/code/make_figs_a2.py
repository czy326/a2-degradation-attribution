# -*- coding: utf-8 -*-
"""A2 论文新增配图 图2–图7（图8 定性图另出）。

数据来源（全部可溯源，不臆造）：
  图2  纯示意，无数据；臂的退化开关取自 `A2_PAPER_DRAFT.docx` 表 4 / `tv_sweep.TRAIN_ARMS`
  图3  表 6 的 5 个算子（PSNR, mAP）；OLS 与 R² 在本脚本内复算并断言 ≈0.083
  图4  `tv_deploy.json` 4 臂中 A 臂，ref = (blur 0.0, noise 0.003)，Z = ref − mAP（6 种子均值）
  图5  同上，blur = 2.0 固定，噪声轴 0.003/0.012/0.024/0.096，D0 vs B2，配对 t（n=6, |t|=2.571）
  图6  表 8 的 |Δ| 中位数（N2−D0 / J2−D0 / B2−D0，16 条件）
  图7  表 11 的逐种子斜率与 t（三种 SR 前端）

用法（Windows 侧）：
  wsl.exe -e bash -lc 'cd /mnt/e/新方向研究/a2/figures && /home/czy/envs/sr4sod/bin/python make_figs_a2.py'
  ... make_figs_a2.py --only 3,5      # 只重画某几张
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS  # noqa: E402

A2 = "/mnt/e/新方向研究/a2"
OUT = FS.OUT

# ---------------------------------------------------------------- 统一外观
C_A, C_B0, C_B2, C_D0 = "#2166AC", "#E08214", "#B2182B", "#4D4D4D"
C_INK = "#262626"
C_FILL = "#F4F7FB"
LW = 0.55                    # 全项目偏好：细线
plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.linewidth": LW,
    "mathtext.fontset": "stix",     # 数学符号走 Times 风格
    "mathtext.default": "it",
    "xtick.major.width": LW,
    "ytick.major.width": LW,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
})


def frame(ax, lw=LW):
    """四边全封（项目硬要求：含 top/right）。"""
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color(C_INK)


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"  -> {stem}.png / .pdf / .svg")


# ================================================================ 图2 链路框图
def box(ax, x, y, w, h, title, sub="", fill=C_FILL, edge="#8AA4BE", tsize=8.2, ssize=6.8):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.9",
                       linewidth=LW * 1.6, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(p)
    ax.text(x, y + (0.9 if sub else 0), title, ha="center", va="center",
            fontsize=tsize, color=C_INK, zorder=3, fontweight="bold")
    if sub:
        ax.text(x, y - 2.6, sub, ha="center", va="center", fontsize=ssize,
                color="#4D4D4D", zorder=3, linespacing=1.35)
    return p


def arrow(ax, x0, x1, y, dashed=False, color=C_INK):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=LW * 2,
                                 color=color, zorder=4,
                                 linestyle=(0, (2.2, 1.6)) if dashed else "-"))


def fig02():
    """图 2：两阶段链路框图 + 真实图像缩略（同一条管线出的图，非示意图）。"""
    import matplotlib.image as mpimg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    D = os.path.join(FS.__file__.rsplit(os.sep, 1)[0], "_cache", "fig02")
    need = ["hr.png", "lr_clean.png", "sr_clean.png", "det_clean.png",
            "lr_noisy.png", "sr_noisy.png", "det_noisy.png", "det.json"]
    miss = [f for f in need if not os.path.isfile(os.path.join(D, f))]
    if miss:
        raise SystemExit(f"★ 缺素材 {miss}，先跑 gen_fig02_assets.py")
    det = json.load(open(os.path.join(D, "det.json"), encoding="utf-8"))["det"]
    n_c, n_n = det["clean"]["n_boxes"], det["noisy"]["n_boxes"]
    cf_n = det["noisy"]["conf_mean"]

    fig, ax = plt.subplots(figsize=(7.15, 4.86))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis("off")

    S = 15.0
    X1 = [10.0, 30.5, 51.0, 71.5, 92.0]
    X2 = X1[1:]
    Y1, Y2 = 45.0, 14.0
    INK, MUTE = "#2B2B2B", "#6A6A6A"

    # ---------------------------------------------------------- 图标（细线，2.2 单位）
    def _line(xs, ys):
        ax.add_line(Line2D(xs, ys, color=INK, linewidth=1.0, solid_capstyle="round", zorder=7))

    def ico_input(x, y, h=2.2):
        k = h / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                _line([x + sx * k, x + sx * k * 0.45], [y + sy * k, y + sy * k])
                _line([x + sx * k, x + sx * k], [y + sy * k, y + sy * k * 0.45])
        ax.add_patch(Rectangle((x - k * 0.30, y - k * 0.30), k * 0.60, k * 0.60,
                               fill=False, edgecolor=INK, linewidth=1.0, zorder=7))

    def ico_degrade(x, y, h=2.2):
        k = h / 2
        xs = np.linspace(x - k, x + k, 40)
        ax.add_line(Line2D(xs, y + k * 0.45 * np.sin((xs - x) * 3.4), color=INK,
                           linewidth=1.0, zorder=7))
        ax.add_line(Line2D(xs, y - k * 0.10 + k * 0.28 * np.sin((xs - x) * 5.2 + 1.1),
                           color=INK, linewidth=0.9, alpha=0.55, zorder=7))
        rng = np.random.default_rng(3)
        for dx, dy in zip(rng.uniform(-k, k, 7), rng.uniform(-k * 0.9, -k * 0.35, 7)):
            ax.add_patch(Circle((x + dx, y + dy), 0.13, color=INK, zorder=7))

    def ico_sr(x, y, h=2.2):
        k = h / 2
        ax.add_patch(Circle((x - k * 0.15, y + k * 0.15), k * 0.62, fill=False,
                            edgecolor=INK, linewidth=1.0, zorder=7))
        _line([x + k * 0.30, x + k * 0.95], [y - k * 0.30, y - k * 0.95])
        _line([x - k * 0.52, x + k * 0.22], [y + k * 0.15, y + k * 0.15])
        _line([x - k * 0.15, x - k * 0.15], [y - k * 0.22, y + k * 0.52])

    def ico_det(x, y, h=2.2):
        k = h / 2
        ax.add_patch(Rectangle((x - k * 0.80, y - k * 0.80), k * 1.60, k * 1.60,
                               fill=False, edgecolor=INK, linewidth=1.0, zorder=7))
        _line([x - k * 1.05, x - k * 0.80], [y, y])
        _line([x + k * 0.80, x + k * 1.05], [y, y])
        _line([x, x], [y - k * 1.05, y - k * 0.80])
        _line([x, x], [y + k * 0.80, y + k * 1.05])

    def ico_weights(x, y, h=2.2):
        k = h / 2
        pts = [(x - k * 0.85, y + k * 0.55), (x - k * 0.85, y - k * 0.55),
               (x + k * 0.85, y)]
        for a in pts[:2]:
            _line([a[0], pts[2][0]], [a[1], pts[2][1]])
        for px, py in pts:
            ax.add_patch(Circle((px, py), k * 0.22, facecolor="white",
                                edgecolor=INK, linewidth=1.0, zorder=8))

    def ico_boxes(x, y, h=2.2):
        k = h / 2
        for (dx, dy, w, hh) in [(-k * 0.55, -k * 0.10, k * 0.85, k * 0.60),
                                (k * 0.10, -k * 0.70, k * 0.80, k * 0.55),
                                (-k * 0.15, k * 0.35, k * 0.95, k * 0.55)]:
            ax.add_patch(Rectangle((x + dx, y + dy), w, hh, fill=False,
                                   edgecolor=INK, linewidth=1.0, zorder=7))

    def ico_dots(x, y, s=0.16, n=4):
        for dx in (-0.55, -0.18, 0.18, 0.55):
            ax.add_patch(Circle((x + dx, y), s, facecolor=INK, edgecolor="none", zorder=7))

    def ico_grid(x, y, h=0.62):
        for dx in (-0.5, 0.5):
            _line([x + dx * h * 2, x + dx * h * 2], [y - h, y + h])
        for dy in (-0.5, 0.5):
            _line([x - h, x + h], [y + dy * h * 2, y + dy * h * 2])

    # ---------------------------------------------------------- 卡片
    def head(xc, y0, title, icon):
        yy = y0 + S + 3.1
        icon(xc - S / 2 + 1.2, yy)
        ax.text(xc - S / 2 + 3.3, yy, title, ha="left", va="center",
                fontsize=7.1, fontweight="bold", color=INK)

    def img_card(xc, y0, fn, title, icon, badge="", edge=INK):
        im = mpimg.imread(os.path.join(D, fn))
        ax.imshow(im, extent=[xc - S / 2, xc + S / 2, y0, y0 + S], zorder=2)
        ax.add_patch(Rectangle((xc - S / 2, y0), S, S, fill=False,
                               edgecolor=edge, linewidth=LW * 1.5, zorder=4))
        head(xc, y0, title, icon)
        if badge:
            ax.text(xc - S / 2 + 0.6, y0 + 0.6, badge, ha="left", va="bottom",
                    fontsize=5.9, color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#D5D5D5", linewidth=LW, alpha=0.92))

    def txt_card(xc, y0, title, main, sub, icon, fill="#F7F9FC", edge="#9FB3C8"):
        ax.add_patch(FancyBboxPatch((xc - S / 2, y0), S, S,
                                    boxstyle="round,pad=0.012,rounding_size=1.0",
                                    linewidth=LW * 1.5, edgecolor=edge,
                                    facecolor=fill, zorder=2))
        ax.text(xc, y0 + S * 0.60, main, ha="center", va="center", fontsize=12.5,
                color=INK, zorder=3)
        ax.text(xc, y0 + S * 0.31, sub, ha="center", va="center", fontsize=6.2,
                color="#5A5A5A", zorder=3, linespacing=1.35)
        head(xc, y0, title, icon)

    def arrow(x0, x1, y, color=INK):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=LW * 2,
                                     color=color, zorder=3, shrinkA=0, shrinkB=0))

    # ---------------------------------------------------------- 训练链路
    ax.text(1.0, 65.9, "Training pipeline", fontsize=9.2, fontweight="bold", color=INK)
    img_card(X1[0], Y1, "hr.png", "Input", ico_input, "800 $\\times$ 800")
    img_card(X1[1], Y1, "lr_clean.png", "Degradation", ico_degrade, "200 $\\times$ 200")
    img_card(X1[2], Y1, "sr_clean.png", "SR frontend", ico_sr, "BSRGAN")
    img_card(X1[3], Y1, "det_clean.png", "Detector", ico_det, f"{n_c} boxes")
    txt_card(X1[4], Y1, "Weights", "$\\theta$", "3.14 M params\ntrained once", ico_weights)
    for i in range(4):
        arrow(X1[i] + S / 2 + 0.5, X1[i + 1] - S / 2 - 0.5, Y1 + S / 2)

    # ---------------------------------------------------------- 部署链路
    ax.text(1.0, 34.8, "Deployment (inference) pipeline", fontsize=9.2,
            fontweight="bold", color=INK)
    img_card(X2[0], Y2, "lr_noisy.png", "Real LR", ico_input, "200 $\\times$ 200",
             edge=C_B2)
    img_card(X2[1], Y2, "sr_noisy.png", "SR frontend", ico_sr, "BSRGAN")
    img_card(X2[2], Y2, "det_noisy.png", "Detector", ico_det, f"{n_n} boxes")
    txt_card(X2[3], Y2, "Detections", f"{n_n}", f"boxes\nconf$_{{\\mathrm{{mean}}}}$ {cf_n:.2f}",
             ico_boxes, fill="#F6D5D5", edge=C_B2)
    for i in range(3):
        arrow(X2[i] + S / 2 + 0.5, X2[i + 1] - S / 2 - 0.5, Y2 + S / 2)

    # ---------------------------------------------------------- 同一套权重
    for xc in (X1[2], X1[3]):
        ax.plot([xc, xc], [Y1 - 1.2, Y2 + S + 1.2], linestyle=(0, (1.6, 1.8)),
                linewidth=LW, color="#9E9E9E", zorder=1)
    ax.text(61.25, 38.4, "frozen: same graph,\nsame parameters, same data flow",
            fontsize=6.4, color=MUTE, ha="center", va="center",
            style="italic", linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))

    # ---------------------------------------------------------- 唯一改动点
    ax.add_patch(FancyBboxPatch((X1[1] - S / 2 - 0.9, Y1 - 0.9), S + 1.8, S + 1.8,
                                boxstyle="round,pad=0.01,rounding_size=1.1",
                                linewidth=LW * 2.4, edgecolor="#1B7837",
                                facecolor="none", linestyle=(0, (3.2, 1.8)), zorder=5))
    ax.annotate("only this module changes",
                xy=(X1[1], Y1 - 1.1), xytext=(X1[1], 41.6),
                fontsize=6.6, color="#1B7837", ha="center", va="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", linewidth=LW * 2, color="#1B7837",
                                shrinkA=1.0, shrinkB=1.0))

    # ---------------------------------------------------------- 四臂开关（toggle 胶囊）
    ax.text(1.0, 11.2, "Degradation switches of the four arms", fontsize=7.4,
            fontweight="bold", color=INK)
    arms = [("A", "0", 0, 0), ("B0", "U(0, 0.05)", 1, 1),
            ("B2", "U(0.2, 2.0)", 1, 1), ("D0", "U(0.2, 2.0)", 0, 0)]
    xc = [10.6, 33.2, 55.8, 78.4]
    for (a, sig, nz, jp), x in zip(arms, xc):
        ax.add_patch(FancyBboxPatch((x - 10.0, 2.0), 20.0, 7.6,
                                    boxstyle="round,pad=0.012,rounding_size=0.8",
                                    linewidth=LW * 1.4, edgecolor="#C8C8C8",
                                    facecolor="#FBFBFB", zorder=2))
        ax.text(x - 9.0, 7.6, a, fontsize=7.2, fontweight="bold", color=INK,
                va="center", ha="left", zorder=3)
        ax.text(x - 5.2, 7.6, "blur $\\sigma$: " + sig, fontsize=5.9,
                color="#3A3A3A", va="center", ha="left", zorder=3)
        for k, (lab, on) in enumerate((("noise", nz), ("JPEG", jp))):
            px = x - 9.2 + k * 8.4
            ax.add_patch(FancyBboxPatch((px, 3.2), 7.8, 2.8,
                                        boxstyle="round,pad=0.02,rounding_size=1.2",
                                        linewidth=LW * 1.3,
                                        edgecolor="#1B7837" if on else "#D9D9D9",
                                        facecolor="#EDF6EE" if on else "#F6F6F6",
                                        zorder=3))
            ax.add_patch(Circle((px + 1.1, 4.6), 0.46,
                                facecolor="#1B7837" if on else "white",
                                edgecolor="#1B7837" if on else "#BDBDBD",
                                linewidth=0.9, zorder=4))
            ax.text(px + 2.05, 4.6, lab, fontsize=5.5,
                    color="#20501F" if on else "#8A8A8A",
                    va="center", ha="left", zorder=4)

    save(fig, "fig02_pipeline")


# ================================================================ 图3 mAP–PSNR
def fig03():
    pts = [("INTER_AREA\n(resampling only)", 30.71, 0.0816),
           ("bicubic down-\nsampling", 30.04, 0.0681),
           ("high-order deg.", 29.83, 0.0573),
           ("shuffled deg.", 29.50, 0.0578),
           ("physical MTF\n+ noise", 26.35, 0.0653)]
    x = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])
    A = np.vstack([x, np.ones_like(x)]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ np.array([k, b])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  [fig03] OLS mAP = {k:+.5f}·PSNR {b:+.5f}  R2 = {r2:.4f}")
    assert abs(r2 - 0.083) < 0.002, "R² 与正文 0.083 不符，先核对表 6"

    fig, ax = plt.subplots(figsize=(4.05, 3.15))     # ~103 mm
    xs = np.linspace(25.6, 31.3, 50)
    ax.plot(xs, k * xs + b, color="#8C8C8C", linewidth=LW * 2,
            linestyle=(0, (4, 2)), zorder=1)

    # 残差竖线
    for xi, yi, pi in zip(x, y, pred):
        ax.plot([xi, xi], [yi, pi], color="#C9C9C9", linewidth=LW * 1.4, zorder=1)
    ax.scatter(x, y, s=34, facecolor=C_B2, edgecolor=C_INK, linewidth=LW * 1.6, zorder=3)

    offs = [(-9, 8), (-9, 11), (-9, -22), (8, -13), (9, 5)]
    ha = ["right", "right", "right", "left", "left"]
    for (lab, xi, yi), (ox, oy), h in zip(pts, offs, ha):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(ox, oy),
                    fontsize=6.3, color=C_INK, ha=h, va="center", linespacing=1.25)

    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("mAP$_{50\\text{-}95}$")
    ax.set_xlim(25.5, 31.9)
    ax.set_ylim(0.0500, 0.0880)
    ax.grid(True, color="#EAEAEA", linewidth=LW * 1.2, zorder=0)
    ax.set_axisbelow(True)
    frame(ax)
    ax.text(0.035, 0.945,
            f"mAP $= {k:.4f}\\,\\mathrm{{PSNR}} {b:+.4f}$\n$R^2 = {r2:.3f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
            color=C_INK, bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                                   edgecolor="#DDDDDD", linewidth=LW))
    save(fig, "fig03_map_vs_psnr")


# ================================================================ 图4 Z 热图
def _z_grid():
    """返回 (Z, ref, BL, NO)：Z[i,j] = ref − mAP(A, blur=BL[j], noise=NO[i])，6 种子均值。"""
    d = json.load(open(os.path.join(A2, "tv_deploy.json")))
    seeds = [2026, 2027, 2028, 2029, 2030, 2031]
    BL = [0.0, 0.2, 0.6, 1.2, 2.0, 2.4]
    NO = [0.003, 0.012, 0.024, 0.096]

    def mean_map(b, n):
        vs = []
        for s in seeds:
            k = f"tv_A|s{s}|b{b:.1f}|n{n:.3f}"
            if k not in d:
                return None
            v = d[k]
            vs.append(v["mAP5095"] if isinstance(v, dict) else float(v))
        return float(np.mean(vs))

    ref = mean_map(0.0, 0.003)
    Z = np.full((len(NO), len(BL)), np.nan)
    for i, n in enumerate(NO):
        for j, b in enumerate(BL):
            m = mean_map(b, n)
            if m is not None:
                Z[i, j] = ref - m
    return Z, ref, BL, NO


def fig04():
    """图 4（剖面线版）：16 个条件全部用上，零空格。

    (a) 固定模糊看噪声：σ = 0.2 / 2.0 两条完整曲线（4 点）
    (b) 固定噪声看模糊：η = 0.003 / 0.096 两条完整曲线（6 点）
    """
    Z, ref, BL, NO = _z_grid()
    print(f"  [fig04] ref(A) = {ref:.5f}，已测 {int(np.isfinite(Z).sum())}/24 格，"
          f"Z 范围 [{np.nanmin(Z):+.5f}, {np.nanmax(Z):+.5f}]")
    assert abs(Z[NO.index(0.096), BL.index(2.0)] - 0.02802) < 5e-5, "与表 5 的 blur=2.0 格不符"
    ipk = np.unravel_index(np.nanargmax(Z), Z.shape)
    print(f"  [fig04] 全网格峰值 {np.nanmax(Z):.5f} @ blur={BL[ipk[1]]:g}, noise={NO[ipk[0]]:g}")

    C_LIGHT, C_DARK = "#7FB3D5", "#1F4E79"      # σ = 0.2 / 2.0
    C_WARM_L, C_WARM_D = "#E8A87C", "#8C2D04"   # η = 0.003 / 0.096

    fig, axes = plt.subplots(1, 2, figsize=(6.10, 2.55),
                             gridspec_kw=dict(width_ratios=[1.0, 1.22], wspace=0.26))
    ax = axes[0]
    xs = np.arange(len(NO))
    for b, c, lab in ((0.2, C_LIGHT, "$\\sigma$ = 0.2"), (2.0, C_DARK, "$\\sigma$ = 2.0")):
        ys = Z[:, BL.index(b)] * 1000
        ax.plot(xs, ys, "-o", color=c, linewidth=LW * 2.2, markersize=4.0,
                markerfacecolor="white", markeredgewidth=LW * 1.8, label=lab, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n:g}" for n in NO])
    ax.set_xlabel("Noise level $\\eta$")
    ax.set_ylabel("$Z(A)$  ($\\times10^{-3}$)")
    ax.set_xlim(-0.3, len(NO) - 0.7)
    ax.set_ylim(-1, 31)
    ax.grid(True, color="#EDEDED", linewidth=LW * 1.2, zorder=0)
    ax.set_axisbelow(True)
    frame(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.7, fontsize=7.0)
    ax.text(-0.02, 1.045, "(a)", transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", color=C_INK, va="bottom")

    ax = axes[1]
    xs = np.arange(len(BL))
    for n, c, lab in ((0.003, C_WARM_L, "$\\eta$ = 0.003"), (0.096, C_WARM_D, "$\\eta$ = 0.096")):
        ys = Z[NO.index(n), :] * 1000
        ax.plot(xs, ys, "-s", color=c, linewidth=LW * 2.2, markersize=3.8,
                markerfacecolor="white", markeredgewidth=LW * 1.8, label=lab, zorder=3)
    ax.annotate(f"{Z[NO.index(0.096), BL.index(2.4)]*1000:.1f}",
                xy=(BL.index(2.4), Z[NO.index(0.096), BL.index(2.4)] * 1000),
                textcoords="offset points", xytext=(-7, -11), fontsize=6.4,
                color=C_WARM_D, ha="right", va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b:g}" for b in BL])
    ax.set_xlabel("Blur $\\sigma$")
    ax.set_ylabel("$Z(A)$  ($\\times10^{-3}$)")
    ax.set_xlim(-0.3, len(BL) - 0.7)
    ax.set_ylim(-1, 31)
    ax.grid(True, color="#EDEDED", linewidth=LW * 1.2, zorder=0)
    ax.set_axisbelow(True)
    frame(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.7, fontsize=7.0)
    ax.text(-0.02, 1.045, "(b)", transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", color=C_INK, va="bottom")

    save(fig, "fig04_z_profiles")


def fig04_heat():
    """图 4（热图版，保留备选）：6×4 网格，8 格未采样以斜纹标出。"""
    Z, ref, BL, NO = _z_grid()
    cmap = LinearSegmentedColormap.from_list("zr", ["#FFFFFF", "#FDD0A2", "#E6550D", "#7F0000"])
    cmap.set_bad("#EDEDED")
    fig, ax = plt.subplots(figsize=(4.55, 3.05))
    Zm = np.ma.masked_invalid(Z)
    im = ax.imshow(Zm, cmap=cmap, aspect="auto", origin="lower",
                   vmin=0, vmax=0.030, interpolation="nearest")
    from matplotlib.patches import Rectangle
    for i in range(len(NO)):
        for j in range(len(BL)):
            if np.isfinite(Z[i, j]):
                v = Z[i, j]
                lab = "0.0" if abs(v) < 5e-5 else f"{v*1000:+.1f}".replace("-", "\u2212")
                ax.text(j, i, lab, ha="center", va="center", fontsize=6.4,
                        color="white" if v > 0.017 else C_INK)
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F7F7F7",
                                       edgecolor="#BDBDBD", hatch="//",
                                       linewidth=LW, zorder=2))
    ax.set_xticks(range(len(BL)))
    ax.set_xticklabels([f"{b:g}" for b in BL])
    ax.set_yticks(range(len(NO)))
    ax.set_yticklabels([f"{n:g}" for n in NO])
    ax.set_xlabel("Blur $\\sigma$")
    ax.set_ylabel("Noise level $\\eta$")
    ax.set_xticks(np.arange(-.5, len(BL), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(NO), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", length=0)
    for j in range(1, len(BL)):
        ax.axvline(j - 0.5, color="white", linewidth=0.9, zorder=4)
    for i in range(1, len(NO)):
        ax.axhline(i - 0.5, color="white", linewidth=0.9, zorder=4)
    frame(ax)
    cb = fig.colorbar(im, ax=ax, pad=0.025, fraction=0.046)
    cb.set_label("$Z(A)$   (mAP$_{50\\text{-}95}$ loss, $\\times10^{-3}$)")
    cb.ax.set_yticks([0, 0.01, 0.02, 0.03])
    cb.ax.set_yticklabels(["0", "10", "20", "30"])
    cb.outline.set_linewidth(LW)
    cb.ax.tick_params(width=LW, length=2.2)
    ax.text(0.0, -0.235, "cell labels and colour bar are $Z(A)\\times10^{3}$. "
                         "Hatched cells: condition outside the pre-registered 16-point grid.",
            transform=ax.transAxes, fontsize=6.2, color="#6A6A6A", va="top")
    save(fig, "fig04_z_heatmap")


# ================================================================ 图5 交叉曲线
def fig05():
    d = json.load(open(os.path.join(A2, "tv_deploy.json")))
    seeds = [2026, 2027, 2028, 2029, 2030, 2031]
    NO = [0.003, 0.012, 0.024, 0.096]

    def per_seed(arm, n):
        out = []
        for s in seeds:
            k = f"tv_{arm}|s{s}|b2.0|n{n:.3f}"
            v = d[k]
            out.append(v["mAP5095"] if isinstance(v, dict) else float(v))
        return np.array(out)

    D0 = np.array([per_seed("D0", n) for n in NO])
    B2 = np.array([per_seed("B2", n) for n in NO])
    d0, b2 = D0.mean(1), B2.mean(1)
    dif = D0 - B2
    md = dif.mean(1)
    t = md / (dif.std(1, ddof=1) / np.sqrt(dif.shape[1]))
    print("  [fig05] D0−B2 = " + ", ".join(f"{v:+.4f}(t={tt:.2f})" for v, tt in zip(md, t)))
    for v, tt in zip(md, t):
        if abs(tt) > 2.571:
            assert v > 0, "意料之外的显著反转，先核对"

    fig, ax = plt.subplots(figsize=(4.35, 3.35))
    x = np.arange(len(NO))
    ax.fill_between([2, 3], 0.010, 0.048, color="#FDE8C8", alpha=0.55, zorder=0, linewidth=0)
    ax.plot(x, d0, "-o", color=C_D0, linewidth=LW * 2.2, markersize=4.2,
            markerfacecolor="white", markeredgewidth=LW * 2, label="D0  (no noise/JPEG in training)",
            zorder=3)
    ax.plot(x, b2, "-s", color=C_B2, linewidth=LW * 2.2, markersize=4.0,
            markerfacecolor=C_B2, markeredgewidth=LW * 2, label="B2  (noise + JPEG in training)",
            zorder=3)

    for xi, (a, b, tt) in enumerate(zip(d0, b2, t)):
        sig = abs(tt) > 2.571
        if xi == 3:
            ax.annotate(f"{md[xi]:+.4f}{'*' if sig else ' (n.s.)'}",
                        xy=(2.97, (a + b) / 2), xytext=(2.12, 0.0137),
                        textcoords="data", ha="right", fontsize=6.4,
                        color=C_INK if sig else "#7A7A7A",
                        arrowprops=dict(arrowstyle="-", linewidth=LW, color="#9E9E9E",
                                        shrinkA=2, shrinkB=2))
            continue
        ax.annotate(f"{md[xi]:+.4f}{'*' if sig else ' (n.s.)'}",
                    (xi, max(a, b)), textcoords="offset points",
                    xytext=(-2, 11) if xi == 2 else (0, 10), ha="center", fontsize=6.4,
                    color=C_INK if sig else "#7A7A7A")
    ax.annotate("sign reversal\ncrossing in (0.024, 0.096)",
                xy=(2.52, 0.0228), xytext=(2.30, 0.0408),
                fontsize=6.6, color="#B26B00", ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", linewidth=LW * 1.8, color="#B26B00"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:g}" for n in NO])
    ax.set_xlabel("Deployment noise level $\\eta$")
    ax.set_ylabel("mAP$_{50\\text{-}95}$")
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(0.010, 0.049)
    ax.grid(True, axis="y", color="#EAEAEA", linewidth=LW * 1.2, zorder=0)
    ax.set_axisbelow(True)
    frame(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.235), ncol=2, frameon=False,
              handlelength=1.8, columnspacing=1.4, handletextpad=0.6)
    save(fig, "fig05_crossing")


# ================================================================ 图6 次可加
def fig06():
    # 表 8 的 |Δ| 中位数（16 条件，vs D0）
    labs = ["D0", "J2", "N2", "B2"]
    vals = [0.0, 0.00424, 0.00637, 0.00500]
    cols = [C_D0, "#762A83", "#1B7837", C_B2]
    fig, ax = plt.subplots(figsize=(3.55, 3.00))
    x = np.arange(4)
    ax.bar(x, [v * 1000 for v in vals], width=0.58,
           color=cols, edgecolor=C_INK, linewidth=LW * 1.6, zorder=2)
    add = (vals[1] + vals[2]) * 1000
    ax.axhline(add, color="#B26B00", linewidth=LW * 2, linestyle=(0, (4, 2)), zorder=3)
    ax.text(3.55, add + 0.28, f"J2 + N2 = {add:.2f}", fontsize=6.6, color="#B26B00",
            ha="right", fontweight="bold")
    xa = 3.38
    ax.annotate("", xy=(xa, add), xytext=(xa, vals[3] * 1000),
                arrowprops=dict(arrowstyle="<|-|>", linewidth=LW * 1.6, color="#B26B00"))
    ax.text(xa + 0.08, (add + vals[3] * 1000) / 2, "sub-additive\ngap", fontsize=6.4,
            color="#B26B00", ha="left", va="center", linespacing=1.25)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1000 + 0.18, f"{v*1000:.2f}" if v else "0", ha="center", fontsize=6.5,
                color=C_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("$|\\Delta|$ vs D0  (mAP$_{50\\text{-}95}$, $\\times10^{-3}$)")
    ax.set_ylim(0, add * 1.22)
    ax.set_xlim(-0.6, 4.05)
    ax.grid(True, axis="y", color="#EAEAEA", linewidth=LW * 1.2, zorder=0)
    ax.set_axisbelow(True)
    frame(ax)
    save(fig, "fig06_subadditive")


# ================================================================ 图7 斜率对比
def fig07():
    # 表 11
    eng = ["BSRGAN", "Real-ESRGAN", "bicubic"]
    per = [[0.00023, -0.00047, -0.00077, -0.00110, 0.00053, -0.00159],
           [-0.00010, -0.00031, -0.00071, -0.00147, -0.00031, 0.00017],
           [-0.00270, -0.00277, -0.00250, -0.00315, -0.00180, -0.00327]]
    ts = [-1.62, -1.94, -12.61]
    cols = ["#2166AC", "#1B7837", C_B2]
    fig, ax = plt.subplots(figsize=(3.75, 3.05))
    x = np.arange(3)
    means = [float(np.mean(p)) for p in per]
    ax.bar(x, [-m * 1000 for m in means], width=0.55, color=cols,
           edgecolor=C_INK, linewidth=LW * 1.6, zorder=2)
    rng = np.random.default_rng(7)
    for xi, p in zip(x, per):
        ax.scatter(xi + rng.uniform(-0.13, 0.13, len(p)), [-v * 1000 for v in p],
                   s=11, facecolor="white", edgecolor=C_INK, linewidth=LW * 1.4, zorder=4)
    ns_mark = "$^{\\mathrm{ns}}$"
    for xi, (m, tt, p) in enumerate(zip(means, ts, per)):
        sig = abs(tt) > 2.571
        mark = "*" if sig else ns_mark
        top = max(-v * 1000 for v in p)
        tstr = f"{tt:.2f}".replace("-", "\u2212")
        ax.text(xi, top + 0.22, f"{abs(m):.5f}\n$t$ = {tstr}{mark}",
                ha="center", va="bottom", fontsize=6.5,
                color=C_INK if sig else "#6A6A6A",
                fontweight="bold" if sig else "normal", linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(eng, fontsize=7.6)
    ax.set_ylabel("|dose-response slope|  (mAP $50$-95 / blur unit, $\\times10^{-3}$)")
    ax.set_ylim(0, 4.35)
    ax.grid(True, axis="y", color="#EAEAEA", linewidth=LW * 1.2, zorder=0)
    ax.set_axisbelow(True)
    frame(ax)
    ax.text(0.02, 0.975, "clean end ($\\eta$ = 0.003), 6 seeds", transform=ax.transAxes,
            fontsize=6.4, color="#6A6A6A", ha="left", va="top")
    save(fig, "fig07_slope")


FIGS = {"2": fig02, "3": fig03, "4": fig04, "4b": fig04_heat, "5": fig05, "6": fig06, "7": fig07}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(FIGS))
    a = ap.parse_args()
    for k in [s.strip() for s in a.only.split(",") if s.strip()]:
        print(f"[fig{k}]")
        FIGS[k]()
    print("[done]")


if __name__ == "__main__":
    main()
