"""Concept figures for the manuscript: Fig. 1 (falling-cat model, from the authors' earlier manuscript) and
Fig. 2 (FalCO mechanism, stability and scale invariance)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
OUT = paths.OUTPUT
MEDIA = paths.FIGDATA
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False})
BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#e34948", "#6250d6"
LEGS = ["#0b4f9c", "#2a78d6", "#6fa8e8", "#a9c9ee", "#d6e4f5"]


def flat(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    a = np.asarray(bg.convert("RGB"))
    mask = (a < 245).any(2)
    ys, xs = np.nonzero(mask)
    pad = 8
    return a[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def two_rod(ax):
    """Two-rod model of the falling cat after Essen and Nordmark (2018), redrawn."""
    teal, rod, ball = "#1b8a7a", "#9fd4c8", "#7fa9c9"
    ax.annotate("", xy=(10.6, 0), xytext=(-0.4, 0), arrowprops=dict(arrowstyle="-|>", color=teal, lw=2.2))
    ax.text(10.7, -0.05, "$z$", fontsize=10, va="center")
    G1, G2 = np.array([2.6, 0.0]), np.array([7.0, 0.0])
    for G, ang, (lm, lM) in [(G1, 42, (2.6, 1.0)), (G2, 75, (3.1, 2.4))]:
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        pm, pM = G + lm * d, G - lM * d
        ax.plot(*np.c_[pM, pm], color=rod, lw=4, solid_capstyle="round", zorder=2)
        ax.scatter(*pm, s=170, color=ball, edgecolor="#4f7d9e", lw=0.8, zorder=4)
        ax.scatter(*pM, s=560, color=ball, edgecolor="#4f7d9e", lw=0.8, zorder=4)
        ax.text(*(pm + [0.35, 0.25]), "$m$", fontsize=9)
        ax.text(*(pM + [-1.05, -0.15]), "$M$", fontsize=9, ha="right", va="center")
    for P, lab in [(G1, "$G_1$"), ((G1 + G2) / 2, "$G$"), (G2, "$G_2$")]:
        ax.scatter(*P, s=40, color="#1f1f1f", zorder=5)
        ax.text(P[0] + 0.15, P[1] - 0.55, lab, fontsize=9)
    ax.text(3.85, 0.18, "$L/2$", fontsize=8, ha="center", color="#52514e")
    ax.text(6.0, 0.18, "$L/2$", fontsize=8, ha="center", color="#52514e")
    ax.text(0.2, 1.3, "front body", fontsize=7.5, color="#52514e", va="center")
    ax.text(8.45, 1.6, "rear body", fontsize=7.5, color="#52514e", va="center")
    ax.set_xlim(-0.5, 11.2); ax.set_ylim(-3.4, 3.7); ax.set_aspect("equal"); ax.set_axis_off()


def fig1():
    a, c = (flat(os.path.join(MEDIA, f)) for f in ["image3.png", "image5.png"])
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.3], height_ratios=[1.0, 0.75], wspace=0.12, hspace=0.12)
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    for ax, im, lab in zip(axes, [a, None, c], "abc"):
        if im is None:
            two_rod(ax)
        else:
            ax.imshow(im)
            ax.set_axis_off()
        ax.set_anchor("N" if lab == "a" else "NW")
        ax.text(-0.02, 1.0, lab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
    fig.savefig(os.path.join(OUT, "fig_concept.png"), dpi=300, bbox_inches="tight")


def landscape(X, Y):
    Z = (X - 1.2) ** 2 + (Y + 0.8) ** 2
    return Z + 2.5 * (2 - np.cos(2.2 * (X - 1.2)) - np.cos(2.2 * (Y + 0.8)))


def snapshot():
    rs = np.random.RandomState(11)
    N, K = 100, 5
    m = N // K
    X = rs.uniform(-5, 5, (N, 2))
    F = landscape(X[:, 0], X[:, 1])
    o = np.argsort(F)
    X, F = X[o], F[o]
    wk = F[:m].max() - F[:m] + 0.5
    MC = (wk[:, None] * X[:m]).sum(0) / wk.sum()
    w = 0.832 - 0.792 * np.arange(N) / (N - 1)
    return X, MC, w, m


def fig2():
    st = json.load(open(os.path.join(HERE, "analysis", "stab.json")))
    sc = json.load(open(os.path.join(HERE, "analysis", "scale.json")))
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.32)
    # (a) mechanism
    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(-5, 5, 300)
    GX, GY = np.meshgrid(g, g)
    ax.contourf(GX, GY, landscape(GX, GY), levels=18, cmap="Greys", alpha=0.35)
    X, MC, w, m = snapshot()
    for k in range(5):
        pts = X[k * m:(k + 1) * m]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=LEGS[k], edgecolor="#0b0b0b" if k == 0 else "none",
                   linewidth=0.3, zorder=3 + (4 - k) * 0.01, label=f"$G_{k + 1}$" + (" (elite)" if k == 0 else ""))
    ax.scatter(*X[0], marker="*", s=110, color=ORANGE, edgecolor="k", linewidth=0.5, zorder=5, label="$x_{best}$")
    ax.scatter(*MC, marker="P", s=70, color=GREEN, edgecolor="k", linewidth=0.5, zorder=5, label="$MC$")
    d = np.linalg.norm(X - MC, axis=1)
    hi = int(np.argmax(np.where(np.arange(len(X)) < 20, d, -1) * (np.arange(len(X)) > 0)))
    lo = 80 + int(np.argmin(np.abs(d[80:] - 3.2)))
    for idx, dx, ha in [(hi, (9, 2), "left"), (lo, (-2, 9), "left")]:
        xi = X[idx]
        for tgt, col, ls in [(MC, GREEN, "-"), (X[0], ORANGE, (0, (2, 1.5)))]:
            ax.add_patch(FancyArrowPatch(xi, tgt, arrowstyle="-|>", mutation_scale=7, color=col, lw=0.9, ls=ls,
                                         shrinkB=4, zorder=4))
        step = w[idx] * (2.5 * (MC - xi) + 0.5 * (X[0] - xi))
        ax.add_patch(FancyArrowPatch(xi, xi + step, arrowstyle="-|>", mutation_scale=9, color="#0b0b0b", lw=1.6,
                                     zorder=6))
        ax.scatter(*xi, s=26, facecolor="white", edgecolor="#0b0b0b", lw=1.0, zorder=7)
        ax.annotate(f"$p_i$ = {idx}, $w_i$ = {w[idx]:.2f}", xi, xytext=dx, textcoords="offset points", fontsize=6.5, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=8)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=GREEN, lw=0.9), Line2D([], [], color=ORANGE, lw=0.9, ls=(0, (2, 1.5))),
          Line2D([], [], color="#0b0b0b", lw=1.6)]
    l += ["to $MC$", "to $x_{best}$", "expected step"]
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6.3, ncol=4, handletextpad=0.3,
              columnspacing=0.8, handlelength=1.4)
    ax.set_title("a   Groups, elite centre and attractions", loc="left")
    # (b) inertia vs rank
    ax = fig.add_subplot(gs[0, 1])
    p = np.arange(100)
    ws = 0.832 - 0.792 * p / 99
    wst = (7 - 33 ** 0.5) / 2
    ax.axhspan(wst, 0.9, color=RED, alpha=0.08, lw=0)
    ax.axhline(wst, color=RED, lw=0.8, ls="--")
    ax.plot(p, ws, color=BLUE, lw=2)
    for k in range(1, 5):
        ax.axvline(k * 20 - 0.5, color="#d6d5d0", lw=0.6)
    for k in range(5):
        ax.text(k * 20 + 10, 0.02, f"$G_{k + 1}$", ha="center", fontsize=7, color=GREY)
    ax.text(60, wst + 0.08, "variance-unstable for fixed attractors\n($w_i > w^* = 0.628$, Eq. 2)", fontsize=6.5,
            color=RED, ha="center")
    ax.annotate("26 best ranks", (12.5, wst), xytext=(30, 0.33), fontsize=6.5,
                arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.6))
    ax.set_xlim(0, 99); ax.set_ylim(0, 0.9)
    ax.set_xlabel("Rank $p_i$ (0 = best)"); ax.set_ylabel("Inertia weight $w_i$")
    ax.set_title("b   Rank-dependent inertia", loc="left")
    # (c) convergence: rank-dependent vs fixed weights
    ax = fig.add_subplot(gs[1, 0])
    fe = np.arange(1, 151) * 100
    for lab, key, col, ls in [("FalCO (rank-dependent $w_i$)", "FC", BLUE, "-"),
                              ("fixed $w$ = 0.436 (stable)", "w=0.44 (all stable)", GREEN, "-"),
                              ("fixed $w$ = 0.70 (unstable)", "w=0.70 (all unstable)", ORANGE, "-"),
                              ("fixed $w$ = 0.832 (unstable)", "w=0.832 (all unstable)", RED, "--")]:
        c = np.maximum(np.array(st[f"sphere|{key}"]["curve_median"]), 1e-12)
        ax.plot(fe[:len(c)], c, color=col, lw=2 if key == "FC" else 1.2, ls=ls, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("Function evaluations"); ax.set_ylabel("Median error (shifted sphere)")
    ax.set_xticks([0, 5000, 10000, 15000]); ax.set_xticklabels(["0", "5k", "10k", "15k"])
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    ax.grid(True, color="#e4e3df", lw=0.5)
    ax.set_title("c   Self-stabilization", loc="left")
    # (d) scale invariance
    ax = fig.add_subplot(gs[1, 1])
    s = [0.01, 1, 100, 1000, 10000]
    for fn, mk in [("sphere", "o"), ("rastrigin", "s")]:
        for alg, col, lab in [("new", BLUE, "FalCO"), ("old", GREY, "earlier version")]:
            v = [np.mean(sc[f"{fn}|{x:g}|{alg}"]) for x in s]
            ax.plot(s, v, marker=mk, ms=4, color=col, lw=1.6 if alg == "new" else 1.1,
                    ls="-" if fn == "sphere" else "--", label=f"{lab}, {fn.capitalize()}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Scale factor $s$ of all variables"); ax.set_ylabel("Mean final error")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    ax.grid(True, color="#e4e3df", lw=0.5)
    ax.set_title("d   Scale invariance", loc="left")
    fig.savefig(os.path.join(OUT, "fig_mechanism.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    fig1()
    fig2()
