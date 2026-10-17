# -*- coding: utf-8 -*-
"""Shared style for the S2DS paper figures.

Design goals (Q2-journal look):
  * one palette, one font, one set of sizes -- no per-figure ad-hoc colours
  * every panel carries a bold (a)(b)(c) letter, journals require it
  * statistical panels: no top/right spines, light y-grid, 300 dpi
  * image panels: 1 px frame, no ticks, equal aspect
  * an isometric-box helper for the architecture figure (pseudo-3D, the way
    Remote Sensing / ISPRS journal figures are usually drawn)
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch  # noqa: E402

# --------------------------------------------------------------------------- palette
C_HARD = "#B2182B"      # HARD endpoint (primary)
C_VALID = "#2166AC"     # valid endpoint (control)
C_ACCENT = "#1B7837"    # green, used for thresholds / good news
C_NEUTRAL = "#4D4D4D"
C_GRID = "#D8D8D8"
C_LIGHT = "#F2F2F2"

# qualitative fills for the architecture figure (pastel + darker edge)
FILL_INPUT = "#DCEBF7"
FILL_ENC = "#CFE3F3"
FILL_COND = "#FDE8C8"
FILL_ATT = "#FAD9C1"
FILL_FUSE = "#DDEEDD"
FILL_HEAD = "#E6DCF2"
FILL_OUT = "#F6D5D5"

SEQ = [C_HARD, C_VALID, C_ACCENT, "#762A83", "#E08214", "#4D4D4D"]

# --------------------------------------------------------------------------- rcParams
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#BBBBBB",
    "mathtext.fontset": "dejavusans",
})

OUT = "/mnt/e/新方向研究/a2/figures/out"
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- helpers
def panel_label(ax, s, x=-0.13, y=1.06, size=10.5):
    """Bold (a)(b)(c) letter, in axes coordinates, outside the axes box."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, fontweight="bold",
            ha="left", va="bottom", color="#111111")


def panel_tag(ax, s, x=-0.01, y=1.035, size=9.0):
    """Bold (a)(b)(c) letter OUTSIDE the axes, above-left.

    Never drawn inside the image, where it would cover content.
    """
    ax.text(x, y, "(%s)" % s, transform=ax.transAxes, fontsize=size,
            fontweight="bold", color="#111111", ha="left", va="bottom",
            zorder=40)


def panel_cap(ax, title, sub=None, y=-0.055, size=7.2):
    """Per-panel caption BELOW the axes, so it cannot collide with the letter."""
    t = title if sub is None else title + "\n" + sub
    return ax.text(0.5, y, t, transform=ax.transAxes, fontsize=size,
                   color="#222222", ha="center", va="top", linespacing=1.40,
                   zorder=40)


def show(ax, vis, cmap=None, vmin=None, vmax=None, interp="nearest"):
    """imshow with the paper's image conventions.

    `nearest` on purpose: the panels are 192 x 192 and are drawn at roughly
    2.5x that in print, so bilinear/cubic resampling visibly softens every
    edge.  Nearest keeps the pixels crisp.
    """
    if cmap is not None:
        ax.imshow(vis, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    else:
        ax.imshow(np.clip(np.asarray(vis), 0, 1), interpolation=interp)
    frame_image(ax, color="#FFFFFF", lw=0.9)


def style_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, color=C_GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.7)


def frame_image(ax, color="#FFFFFF", lw=1.0):
    """Thin frame around an imshow panel; hides ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_edgecolor(color)


def _bbox_overlap(a, b):
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def audit_text(fig, name, verbose=True):
    """Report text that overlaps other text, or text that sits on top of a
    different Axes (i.e. is occluded by a panel).

    Runs automatically after every save, so a broken layout is caught without
    having to look at the render.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = []
    for t in fig.findobj(match=matplotlib.text.Text):
        s = str(t.get_text())
        if not s.strip() or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        texts.append((s, bb, t))

    notes = []

    # ---- text vs text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            s1, b1, _ = texts[i]
            s2, b2, _ = texts[j]
            inter = _bbox_overlap(b1, b2)
            if inter <= 1.0:
                continue
            small = min((b1.x1 - b1.x0) * (b1.y1 - b1.y0),
                        (b2.x1 - b2.x0) * (b2.y1 - b2.y0))
            if small <= 0:
                continue
            frac = inter / small
            if frac >= 0.08:
                notes.append("TEXT/TEXT  %5.0f px2 (%3.0f%%)  %r  <->  %r"
                             % (inter, 100 * frac, s1[:34], s2[:34]))

    # ---- text sitting on a different Axes (would be covered by that panel)
    axes = [a for a in fig.axes if a.get_visible()]
    for s, bb, t in texts:
        own = getattr(t, "axes", None)
        for a in axes:
            if a is own:
                continue
            try:
                ab = a.get_window_extent(r)
            except Exception:
                continue
            # inset axes live inside their parent; skip those
            if own is not None and a in getattr(own, "child_axes", []):
                continue
            inter = _bbox_overlap(bb, ab)
            if inter <= 1.0:
                continue
            area = (bb.x1 - bb.x0) * (bb.y1 - bb.y0)
            if area > 0 and inter / area >= 0.25:
                notes.append("TEXT/AXES  %5.0f px2 (%3.0f%%)  %r  on axes %s"
                             % (inter, 100 * inter / area, s[:34],
                                a.get_title() or a.get_subplotspec() or "?"))
                break

    # ---- anything spilling outside the figure area
    fw, fh = fig.canvas.get_width_height()
    for s, bb, t in texts[:0]:          # informational only, kept off by default
        pass

    if verbose:
        if notes:
            print("  !! %s: %d layout warning(s)" % (name, len(notes)))
            for n in notes[:25]:
                print("     " + n)
        else:
            print("  ok %s: no text overlap" % name)
    return notes


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    audit_text(fig, name)
    plt.close(fig)
    print("saved", p)
    return p


# ------------------------------------------------------------- isometric 3D helpers
UX = (1.00, 0.00)        # width direction
UY = (0.46, 0.30)        # depth direction (up and to the right)
UZ = (0.00, 1.00)        # height


def _add(a, b, k=1.0):
    return (a[0] + b[0] * k, a[1] + b[1] * k)


def iso_box(ax, origin, w, d, h, color, alpha=1.0, lw=0.7, edge="#3A3A3A",
            shade=(1.0, 0.86, 0.70), zorder=10, shadow=True):
    """Draw a pseudo-3D box.  `origin` is the front-bottom-left corner.

    shade = (top, front, side) multipliers applied to `color`.
    """
    from matplotlib.colors import to_rgb
    base = to_rgb(color)
    o = origin
    top = [o, _add(_add(o, UX, w), UZ, h), _add(_add(_add(o, UX, w), UY, d), UZ, h),
           _add(_add(o, UY, d), UZ, h)]
    front = [o, _add(o, UX, w), _add(_add(o, UX, w), UZ, h), _add(o, UZ, h)]
    side = [_add(o, UX, w), _add(_add(o, UX, w), UY, d),
            _add(_add(_add(o, UX, w), UY, d), UZ, h), _add(_add(o, UX, w), UZ, h)]
    if shadow:
        sh = [_add(p, (0.045, -0.045)) for p in front]
        ax.add_patch(Polygon(sh, closed=True, facecolor="#000000", alpha=0.10,
                             edgecolor="none", zorder=zorder - 1))
    ax.add_patch(Polygon(side, closed=True, facecolor=tuple(c * shade[2] for c in base),
                         edgecolor=edge, lw=lw, alpha=alpha, zorder=zorder))
    ax.add_patch(Polygon(front, closed=True, facecolor=tuple(c * shade[1] for c in base),
                         edgecolor=edge, lw=lw, alpha=alpha, zorder=zorder + 1))
    ax.add_patch(Polygon(top, closed=True, facecolor=tuple(c * shade[0] for c in base),
                         edgecolor=edge, lw=lw, alpha=alpha, zorder=zorder + 2))
    return o


def iso_plane_stack(ax, origin, w, d, h, n, color, gap=0.10, **kw):
    """A stack of n thin isometric slabs, receding along UY (feature-map stack look)."""
    for i in range(n):
        off = _add(origin, UY, i * (d + gap))
        a = 0.35 + 0.65 * (i + 1) / n
        iso_box(ax, off, w, d * 0.55, h, color, alpha=a, lw=0.6, shadow=False,
                zorder=10 + i, **kw)
    return _add(origin, UY, (n - 1) * (d + gap))


def arrow(ax, p0, p1, color="#3A3A3A", lw=1.1, style="-|>", rad=0.0, ls="-",
          zorder=30, shrink=2.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls, zorder=zorder,
                                 shrinkA=shrink, shrinkB=shrink,
                                 connectionstyle="arc3,rad=%.2f" % rad))
