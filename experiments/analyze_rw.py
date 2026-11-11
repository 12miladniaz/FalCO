"""CEC2020 RW analysis: FC vs SASS, COLSHADE, sCMAgES (published per-run results)."""
import numpy as np
from scipy import stats

import cecscore as C

TOP = ["SASS", "COLSHADE", "sCMAgES"]
ALL8 = ["SASS", "COLSHADE", "sCMAgES", "BiPop", "EnMODE", "VMCH", "DEQL", "FCHA"]
CATS = [("Industrial chemical processes", 1, 7), ("Process synthesis and design", 8, 14),
        ("Mechanical engineering", 15, 33), ("Power systems", 34, 44), ("Power electronics", 45, 50),
        ("Livestock feed ration", 51, 57)]


def data():
    d = {a: C.load_xlsx(a) for a in ALL8}
    d["FC"] = C.load_fc()
    return d


def keys(F, V, k):
    return [(0, F[r, k]) if V[r, k] <= 0 else (1, V[r, k]) for r in range(F.shape[0])]


def has(d, a, k):
    """False if the published results of algorithm a contain no values for problem k (DEQL omits 14 problems)."""
    return not np.isnan(d[a][0][:, k]).any()


def compare(d, a, k, ref="FC", alpha=0.05):
    """Wilcoxon rank-sum on the competition ordering (feasible by f, then infeasible by v)."""
    if not has(d, a, k):
        return "n/a"
    ka, kr = keys(*d[a], k), keys(*d[ref], k)
    allk = sorted(set(ka + kr))
    rank = {x: i for i, x in enumerate(allk)}
    x = [rank[t] for t in kr]
    y = [rank[t] for t in ka]
    if x == y or len(set(x + y)) == 1:
        return "="
    p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
    if p >= alpha:
        return "="
    return "+" if np.median(x) < np.median(y) or (np.median(x) == np.median(y) and np.mean(x) < np.mean(y)) else "−"


def fr(d, a, k):
    return float((d[a][1][:, k] <= 0).mean())


def stat_cell(d, a, k, which="mean"):
    F, V = d[a]
    feas = V[:, k] <= 0
    if not feas.any():
        return None
    if which == "best":
        return F[0, k]
    if which == "median":
        return F[12, k] if V[12, k] <= 0 else None
    return F[feas, k].mean()


def fmt(x):
    if x is None or not np.isfinite(x):
        return "–"
    ax = abs(x)
    if ax == 0:
        return "0"
    if 1e-3 <= ax < 1e5:
        return f"{x:.5g}"
    m, e = f"{x:.3e}".split("e")
    return f"{m}E{int(e)}"


def detail_table(d, title):
    algs = ["FC"] + TOP
    lines = [f"Table: {title}", "",
             "| Problem | D | " + " | ".join(f"{a} best" for a in algs[:1]) + " | " +
             " | ".join(f"{a} mean (FR)" for a in algs) + " |",
             "|:--|:-:|" + "|".join([":-:"] * (1 + len(algs))) + "|"]
    for k in range(57):
        cells = []
        for a in algs:
            m = stat_cell(d, a, k, "mean")
            s = f"{fmt(m)} ({fr(d, a, k):.2f})" if m is not None else f"inf. (0.00)"
            if a != "FC":
                s += " " + compare(d, a, k)
            cells.append(s)
        b = stat_cell(d, "FC", k, "best")
        lines.append(f"| RC{k + 1:02d} | {C.DIMS[k]} | {fmt(b) if b is not None else 'inf.'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary(d):
    sc = C.score({a: d[a] for a in ["FC"] + TOP})
    sc8 = C.score({a: d[a] for a in ["FC"] + ALL8})
    out = {"score4": sc, "score9": sc8, "wtl": {}, "wtl_cat": {}, "fr": {}}
    for a in TOP + ALL8:
        s = [compare(d, a, k) for k in range(57)]
        out["wtl"][a] = (s.count("+"), s.count("="), s.count("−"))
        for name, lo, hi in CATS:
            ss = s[lo - 1:hi]
            out["wtl_cat"][(a, name)] = (ss.count("+"), ss.count("="), ss.count("−"))
    for a in ["FC"] + ALL8:
        frs = [fr(d, a, k) for k in range(57) if has(d, a, k)]
        out["fr"][a] = (np.mean(frs), sum(f == 1 for f in frs), sum(f == 0 for f in frs))
    return out


if __name__ == "__main__":
    d = data()
    print("FC problems loaded:", int((~np.isnan(d["FC"][0][0])).sum()))
