"""Summarise results: tables (as markdown for pandoc) and figures."""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "analysis")
os.makedirs(OUT, exist_ok=True)

MAIN = ["FC", "CMA-ES", "L-SHADE", "PSO", "GWO", "WOA", "HHO", "SSA", "SCA", "JAYA", "DE", "GA"]
CLASSIC = [f"F{i}" for i in range(1, 24)]
SHIFTED = [f"sF{i}" for i in range(1, 14) if i != 8]
ALLF = CLASSIC + SHIFTED
FMIN = {f"F{i}": 0.0 for i in range(1, 14)}
FMIN.update({"F8": -12569.4866, "F14": 0.998004, "F15": 0.000307486, "F16": -1.0316285, "F17": 0.397887,
             "F18": 3.0, "F19": -3.86278, "F20": -3.32237, "F21": -10.1532, "F22": -10.4029,
             "F23": -10.5364})
for s in SHIFTED:
    FMIN[s] = FMIN[s[1:]]
# label used in the paper: FC = corrected implementation, FC-orig = released code
LABEL = ({"FC3-v": "FC", "FC2-K5": "FC-v2", "FC-c": "FC-v1", "FC": "FC-orig"} if os.environ.get("FC_FINAL") == "3"
         else {"FC2-K5": "FC", "FC-c": "FC-v1", "FC": "FC-orig"})


def load():
    data = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(p))
        a = LABEL.get(d["algo"], d["algo"])
        data[(a, d["func"])] = d["runs"]
    return data


D = load()


def best(a, f):
    return np.array([r["best"] for r in D[(a, f)]])


def fmt(x):
    if not np.isfinite(x):
        return "–"
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-2 <= ax < 1e4:
        return f"{x:.3g}" if ax < 100 else f"{x:.4g}" if ax < 1e4 else f"{x:.3g}"
    m, e = f"{x:.2e}".split("e")
    return f"{m}E{int(e)}"


def sign(a, f, ref="FC", alpha=0.05):
    x, y = best(ref, f), best(a, f)
    if np.allclose(x, y, rtol=0, atol=1e-12) or (np.ptp(np.r_[x, y]) == 0):
        return "="
    p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
    if p >= alpha:
        return "="
    return "+" if np.median(x) < np.median(y) or (np.median(x) == np.median(y) and x.mean() < y.mean()) else "−"


def have(algos, funcs):
    return all((a, f) in D for a in algos for f in funcs)


def results_table(funcs, algos, title):
    """Rows: function × {Mean, Std}; columns: algorithms. Best mean in bold. Wilcoxon sign vs FC on mean row."""
    head = "| F | | " + " | ".join(algos) + " |"
    sep = "|:--|:--|" + "|".join([":-:"] * len(algos)) + "|"
    lines = [f"Table: {title}", "", head, sep]
    for f in funcs:
        means = {a: best(a, f).mean() for a in algos}
        stds = {a: best(a, f).std(ddof=1) for a in algos}
        bmin = min(means.values())
        row_m, row_s = [], []
        for a in algos:
            s = fmt(means[a])
            if np.isclose(means[a], bmin, rtol=1e-9, atol=1e-300):
                s = f"**{s}**"
            if a != "FC":
                s += f" {sign(a, f)}" if True else ""
            row_m.append(s)
            row_s.append(fmt(stds[a]))
        lines.append(f"| {f} | Mean | " + " | ".join(row_m) + " |")
        lines.append(f"| | Std | " + " | ".join(row_s) + " |")
    return "\n".join(lines)


def wtl(funcs, algos, ref="FC"):
    out = {}
    for a in algos:
        if a == ref:
            continue
        s = [sign(a, f, ref) for f in funcs]
        out[a] = (s.count("+"), s.count("="), s.count("−"))
    return out


def friedman(funcs, algos):
    M = np.array([[best(a, f).mean() for a in algos] for f in funcs])
    R = np.array([stats.rankdata(row) for row in M])
    mean_rank = R.mean(axis=0)
    chi = stats.friedmanchisquare(*M.T)
    k, n = len(algos), len(funcs)
    se = math.sqrt(k * (k + 1) / (6 * n))
    i0 = algos.index("FC")
    ps = {}
    for j, a in enumerate(algos):
        if j == i0:
            continue
        z = (mean_rank[j] - mean_rank[i0]) / se
        ps[a] = 2 * (1 - stats.norm.cdf(abs(z)))
    # Holm
    order = sorted(ps, key=ps.get)
    m = len(order)
    adj, prev = {}, 0
    for i, a in enumerate(order):
        v = min(1.0, max(prev, (m - i) * ps[a]))
        adj[a] = v
        prev = v
    return dict(zip(algos, mean_rank)), chi, adj


def summary_table(algos):
    groups = [("Classical F1–F23", CLASSIC), ("Shifted sF1–sF13", SHIFTED)]
    lines = []
    w = {g: wtl(fs, algos) for g, fs in groups}
    fr_all = friedman(ALLF, algos)
    fr_c = friedman(CLASSIC, algos)
    fr_s = friedman(SHIFTED, algos)
    lines += ["Table: Table 8. Wilcoxon rank-sum outcomes of FC against each algorithm (+/=/−: FC significantly "
              "better / no significant difference / significantly worse, α = 0.05) and Friedman mean ranks "
              "(lower is better)", "",
              "| Algorithm | W/T/L classical | W/T/L shifted | Rank classical | Rank shifted | Rank all | Holm-adjusted p (all) |",
              "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    order = sorted(algos, key=lambda a: fr_all[0][a])
    for a in order:
        if a == "FC":
            c = s = "–"
            p = "–"
        else:
            c = "/".join(map(str, w[groups[0][0]][a]))
            s = "/".join(map(str, w[groups[1][0]][a]))
            p = f"{fr_all[2][a]:.3g}" if fr_all[2][a] >= 1e-4 else "<1E-4"
        name = f"**{a}**" if a == "FC" else a
        lines.append(f"| {name} | {c} | {s} | {fr_c[0][a]:.2f} | {fr_s[0][a]:.2f} | {fr_all[0][a]:.2f} | {p} |")
    info = {"chi_all": fr_all[1].statistic, "p_all": fr_all[1].pvalue,
            "chi_c": fr_c[1].statistic, "p_c": fr_c[1].pvalue,
            "chi_s": fr_s[1].statistic, "p_s": fr_s[1].pvalue,
            "ranks_all": fr_all[0], "ranks_c": fr_c[0], "ranks_s": fr_s[0], "holm": fr_all[2], "wtl": w}
    return "\n".join(lines), info


def pair_table(funcs, cols, title, labels=None):
    labels = labels or cols
    lines = [f"Table: {title}", "", "| F | " + " | ".join(labels) + " |",
             "|:--|" + "|".join([":-:"] * len(cols)) + "|"]
    for f in funcs:
        ms = [best(a, f).mean() for a in cols]
        bmin = min(ms)
        cells = []
        for a, m in zip(cols, ms):
            s = fmt(m)
            if np.isclose(m, bmin, rtol=1e-9, atol=1e-300):
                s = f"**{s}**"
            if a != "FC":
                s += " " + sign(a, f)
            cells.append(s)
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cpu_table(algos):
    rows = []
    for g, fs in [("F1–F13", CLASSIC[:13]), ("F14–F23", CLASSIC[13:]), ("sF1–sF13", SHIFTED)]:
        rows.append((g, [np.mean([np.mean([r["time"] for r in D[(a, f)]]) for f in fs]) for a in algos]))
    lines = ["Table: Table 11. Mean CPU time per run (s) for a budget of 15,000 evaluations", "",
             "| Functions | " + " | ".join(algos) + " |", "|:--|" + "|".join([":-:"] * len(algos)) + "|"]
    for g, v in rows:
        lines.append(f"| {g} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(len(D), "cells loaded")
