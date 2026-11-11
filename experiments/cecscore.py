"""Port of the official CEC2020 RW scoring script (Score.m) and loaders for published results."""
import json
import os

import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
XL = paths.CEC20_XL
DIMS = [9, 11, 7, 6, 9, 38, 48, 2, 3, 3, 7, 7, 5, 10, 7, 14, 3, 4, 4, 2, 5, 9, 5, 7, 4, 22, 10, 10, 4, 3, 4, 5,
        30, 118, 153, 158, 126, 126, 126, 76, 74, 86, 86, 30, 25, 25, 25, 30, 30, 30, 59, 59, 59, 59, 64, 64, 64]


def weights():
    w = np.array([1 if d <= 10 else 2 if d <= 30 else 3 if d <= 50 else 4 if d <= 150 else 5 for d in DIMS], float)
    return w / w.sum()


def load_xlsx(name):
    wb = openpyxl.load_workbook(os.path.join(XL, name + ".xlsx"), read_only=True)
    F = np.array([list(r) for r in wb.worksheets[0].iter_rows(values_only=True)], float)
    V = np.array([list(r) for r in wb.worksheets[1].iter_rows(values_only=True)], float)
    return F, V


def deb_sort(F, V):
    """Sort runs per problem: feasible first by f, then infeasible by v (competition rule)."""
    Fs, Vs = np.empty_like(F), np.empty_like(V)
    for k in range(F.shape[1]):
        idx = sorted(range(F.shape[0]), key=lambda r: (0, F[r, k]) if V[r, k] <= 0 else (1, V[r, k]))
        Fs[:, k], Vs[:, k] = F[idx, k], V[idx, k]
    return Fs, Vs


def load_fc(path=os.path.join(HERE, "results_rw")):
    F = np.full((25, 57), np.nan)
    V = np.full((25, 57), np.nan)
    for k in range(1, 58):
        p = os.path.join(path, f"FC__RC{k:02d}.json")
        if not os.path.exists(p):
            continue
        runs = json.load(open(p))["runs"]
        F[:, k - 1] = [r["f"] for r in runs]
        V[:, k - 1] = [r["v"] for r in runs]
    return deb_sort(F, V)


def _component(vals1, vals2):
    """vals1, vals2: (n_alg, 57) objective and violation of one statistic. Returns normalised matrix."""
    n = vals1.shape[0]
    out = np.zeros_like(vals1)
    feas = vals2 == 0
    for k in range(vals1.shape[1]):
        if feas[:, k].sum() == 0:
            out[:, k] = vals2[:, k]
        else:
            M = np.nanmax(vals1[feas[:, k], k])
            out[:, k] = np.where(feas[:, k], vals1[:, k], M + vals2[:, k])
    mn, mx = np.nanmin(out, 0), np.nanmax(out, 0)
    rng = mx - mn
    rng[rng == 0] = 1e-8
    out = (out - mn) / rng
    out[np.isnan(out)] = 1
    return out


def score(algs):
    """algs: dict name -> (F, V) each (25, 57), runs sorted by the competition rule."""
    names = list(algs)
    w = weights()
    best1 = np.array([algs[a][0][0] for a in names]); best2 = np.array([algs[a][1][0] for a in names])
    mean1 = np.array([algs[a][0].mean(0) for a in names]); mean2 = np.array([algs[a][1].mean(0) for a in names])
    med1 = np.array([algs[a][0][12] for a in names]); med2 = np.array([algs[a][1][12] for a in names])
    sb, sm, sd = (_component(best1, best2) @ w, _component(mean1, mean2) @ w, _component(med1, med2) @ w)
    total = 0.5 * sb + 0.3 * sm + 0.2 * sd
    return {a: (sb[i], sm[i], sd[i], total[i]) for i, a in enumerate(names)}


if __name__ == "__main__":
    ALL = ["VMCH", "sCMAgES", "SASS", "EnMODE", "COLSHADE", "BiPop", "FCHA", "DEQL"]
    algs = {a: load_xlsx(a) for a in ALL}
    for a, s in sorted(score(algs).items(), key=lambda t: t[1][3]):
        print(f"{a:10s} {s[0]:.4f} {s[1]:.4f} {s[2]:.4f} {s[3]:.4f}")
