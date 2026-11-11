"""Rank-based self-stabilisation of FC.

For v' = w[v + eta1 r1 (MC - x) + eta2 r2 (xb - x)], the second-order (variance) stability condition of
Poli (2009) for a PSO with inertia w and acceleration coefficients c1 = eta1 w, c2 = eta2 w reads
(eta1 + eta2) w < 24 (1 - w^2) / (7 - 5 w), i.e. w < w* = (7 - sqrt(33)) / 2 = 0.628 for eta1 + eta2 = 6.
With w_i from 0.832 to 0.04, the top ~26 % of the ranks lie in the unstable region.

This script measures (i) how long an agent stays in the unstable ranks, and (ii) what happens if every agent
gets the same w (all stable, all unstable), on shifted sphere and shifted Rastrigin (D = 30, 15,000 FE).
"""
import json
import os

import numpy as np

D, N, K, FE, RUNS = 30, 100, 5, 15000, 20
WSTAR = (7 - 33 ** 0.5) / 2
rng0 = np.random.default_rng(7)
SH = {"sphere": rng0.uniform(-50, 50, D), "rastrigin": rng0.uniform(-2.56, 2.56, D)}
BND = {"sphere": 100.0, "rastrigin": 5.12}


def fobj(name, X):
    Z = X - SH[name]
    if name == "sphere":
        return (Z ** 2).sum(1)
    return (Z ** 2 - 10 * np.cos(2 * np.pi * Z) + 10).sum(1)


def fc(name, mode, seed, wfix=None):
    rs = np.random.RandomState(seed)
    b = BND[name]
    lb, ub = -b, b
    X = lb + rs.rand(N, D) * (ub - lb)
    V = np.zeros((N, D))
    m = N // K
    vmax = 0.2 * (ub - lb)
    w = 0.832 - (0.832 - 0.04) * np.arange(N) / (N - 1)
    if mode == "fixed":
        w = np.full(N, wfix)
    ids = np.arange(N)                 # identity of the agent stored in each row
    unst_prev = np.zeros(N, bool)      # per agent id: was in an unstable rank last iteration
    run_len = np.zeros(N, int)
    stays, best = [], np.inf
    curve = []
    fe = 0
    while fe + N <= FE:
        np.clip(X, lb, ub, out=X)
        F = fobj(name, X); fe += N
        best = min(best, F.min())
        curve.append(float(best))
        o = np.argsort(F, kind="stable")
        X, V, F, ids = X[o], V[o], F[o], ids[o]
        unst = np.zeros(N, bool)
        unst[ids[w > WSTAR]] = True
        ended = unst_prev & ~unst
        stays.extend(run_len[ended].tolist())
        run_len = np.where(unst, run_len + 1, 0)
        unst_prev = unst
        f1 = F[:m]
        wk = f1.max() - f1 + rs.rand()
        MC = (wk[:, None] * X[:m]).sum(0) / wk.sum()
        Vn = w[:, None] * (V + 5 * rs.rand(N, D) * (MC - X) + 1 * rs.rand(N, D) * (X[0] - X))
        Vn = np.clip(Vn, -vmax, vmax)
        Vn[0] = V[0]
        Xn = X + Vn; Xn[0] = X[0]
        X, V = Xn, Vn
    return best, stays, curve


if __name__ == "__main__":
    out = {}
    for name in ["sphere", "rastrigin"]:
        for label, mode, wf in [("FC", "rank", None), ("w=0.44 (all stable)", "fixed", 0.436),
                                ("w=0.70 (all unstable)", "fixed", 0.70), ("w=0.832 (all unstable)", "fixed", 0.832)]:
            bs, st, cv = [], [], []
            for r in range(RUNS):
                b_, s_, c_ = fc(name, mode, 100 + r, wf)
                bs.append(float(b_)); st += s_; cv.append(c_)
            out[f"{name}|{label}"] = {"best": bs, "stay_mean": float(np.mean(st)) if st else None,
                                      "stay_median": float(np.median(st)) if st else None,
                                      "curve_median": np.median(np.array(cv), 0).tolist()}
            print(f"{name:10s} {label:24s} mean best {np.mean(bs):.3e}  median {np.median(bs):.3e}  "
                  f"stay in unstable ranks: mean {np.mean(st) if st else float('nan'):.2f} it., "
                  f"median {np.median(st) if st else float('nan'):.0f}", flush=True)
    print("w* =", WSTAR)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "stab.json"), "w"))
