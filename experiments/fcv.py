"""Vectorised FC, numerically the same update rules as FC9_corrected.py (only the random-number stream differs).

Used for the CEC2020 real-world problems, where budgets reach 10^6 evaluations and dimensions reach 158.
"""
import numpy as np


def FCv(objf, lb, ub, dim, N=100, iters=10**9, K=5, eta1=5.0, eta2=1.0, eta3=0.3, a=0.8, b=0.008, p0=4, rank=True,
        norm=False, wmax=0.832, wmin=0.04):
    lb = np.asarray(lb if isinstance(lb, (list, tuple, np.ndarray)) else [lb] * dim, float)
    ub = np.asarray(ub if isinstance(ub, (list, tuple, np.ndarray)) else [ub] * dim, float)
    X = lb + np.random.rand(N, dim) * (ub - lb)
    V = np.zeros((N, dim))
    m = N // K
    vmax = dim / np.maximum(np.minimum(5.0, ub - lb), 3.0)
    p = np.arange(K * m)
    c = (p - p0).astype(float)                       # rank coefficient, Eq. (3)
    if not rank:
        c[:] = N / 2.0 - p0                          # ablation: same coefficient for all agents
    w = a - b * c                                    # inertia weight, Eq. (3)
    if norm:                                         # population-size independent version
        w = wmax - (wmax - wmin) * p / max(N - 1, 1)
        if not rank:
            w[:] = (wmax + wmin) / 2
    leg = p // m                                     # 0-based leg of each sorted position
    coef = (np.arange(K)[None, :] - K) * c[:, None] / 1000.0   # beta_{r,i}, Eq. (5) (0-based r)
    coef[np.arange(K * m), leg] = 0.0               # exclude own leg
    for _ in range(iters):
        np.clip(X, lb, ub, out=X)
        F = np.array([objf(x) for x in X])
        order = np.argsort(F, kind="stable")
        Xs, Fs = X[order[:K * m]], F[order[:K * m]]
        MC = np.empty((K, dim))
        for k in range(K):
            fk = Fs[k * m:(k + 1) * m]
            wk = fk.max() - fk + np.random.rand()
            MC[k] = (wk[:, None] * Xs[k * m:(k + 1) * m]).sum(0) / wk.sum()
        best = Xs[0].copy()
        r1 = np.random.rand(K * m, dim)
        r2 = np.random.rand(K * m, dim)
        R = coef @ MC - coef.sum(1)[:, None] * Xs       # sum_r beta (MC_r - x)
        Vs = V[order[:K * m]]
        Vn = w[:, None] * (Vs + eta1 * r1 * (MC[0] - Xs) + eta2 * r2 * (best - Xs)) + eta3 * R
        Vn = np.clip(Vn, -vmax, vmax)
        Vn[0] = Vs[0]                                  # best agent is not moved (elitism)
        Xn = Xs + Vn
        Xn[0] = Xs[0]
        X[order[:K * m]] = Xn
        V[order[:K * m]] = Vn
