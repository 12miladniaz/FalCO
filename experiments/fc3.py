"""FC with candidate fixes of the mathematical model (for the diagnostic study).

vfrac   : velocity limit as a fraction of each variable's range (None = original D/max(min(5,range),3))
wts     : 'fitness' (original: max f - f + U(0,1)) or 'rank' (log-rank weights, invariant to f scaling)
restart : re-initialise all agents except the best when the population has collapsed
"""
import numpy as np


def FC3(objf, lb, ub, dim, N=100, K=5, eta1=5.0, eta2=1.0, wmax=0.832, wmin=0.04,
        vfrac=None, wts="fitness", restart=False, collapse=1e-8, iters=10**9, rank=True):
    lb = np.asarray(lb if np.ndim(lb) else [lb] * dim, float)
    ub = np.asarray(ub if np.ndim(ub) else [ub] * dim, float)
    rngw = ub - lb
    X = lb + np.random.rand(N, dim) * rngw
    V = np.zeros((N, dim))
    m = N // K
    vmax = vfrac * rngw if vfrac else dim / np.maximum(np.minimum(5.0, rngw), 3.0)
    p = np.arange(K * m)
    w = wmax - (wmax - wmin) * p / max(N - 1, 1)
    if not rank:
        w = np.full_like(w, w.mean())          # ablation: same inertia for every agent
    lw = np.log(m + 0.5) - np.log(np.arange(1, m + 1)); lw /= lw.sum()
    for _ in range(iters):
        np.clip(X, lb, ub, out=X)
        F = np.array([objf(x) for x in X])
        order = np.argsort(F, kind="stable")
        Xs, Fs = X[order[:K * m]], F[order[:K * m]]
        if wts == "rank":
            MC = lw @ Xs[:m]
        else:
            f1 = Fs[:m]; wk = f1.max() - f1 + np.random.rand()
            MC = (wk[:, None] * Xs[:m]).sum(0) / wk.sum()
        best = Xs[0].copy()
        if restart and np.max(np.abs(Xs - MC) / rngw) < collapse:
            Xn = lb + np.random.rand(K * m, dim) * rngw
            Xn[0] = best
            X[order[:K * m]] = Xn
            V[:] = 0
            continue
        r1 = np.random.rand(K * m, dim); r2 = np.random.rand(K * m, dim)
        Vs = V[order[:K * m]]
        Vn = w[:, None] * (Vs + eta1 * r1 * (MC - Xs) + eta2 * r2 * (best - Xs))
        Vn = np.clip(Vn, -vmax, vmax)
        Vn[0] = Vs[0]
        Xn = Xs + Vn; Xn[0] = Xs[0]
        X[order[:K * m]] = Xn
        V[order[:K * m]] = Vn
