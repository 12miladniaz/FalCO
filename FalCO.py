"""Falling Cat Optimizer (FalCO): reference implementation (NumPy only).

Velocity update (Eq 1 of the paper): v <- w_i * [v + eta1*r1*(MC - x) + eta2*r2*(x_best - x)], x <- x + v,
where MC is the fitness-weighted centre of the elite group (the best m = N // K agents, Eq 2),
w_i = wmax - (wmax - wmin) * p_i / (N - 1) decreases linearly with the rank p_i of the agent,
and each velocity component is limited to vfrac * (ub - lb), which makes the method
invariant to a rescaling of the search space.
"""
import numpy as np


def FalCO(objf, lb, ub, dim, N=100, max_fe=15000, K=5, eta1=5.0, eta2=1.0,
       wmax=0.832, wmin=0.04, vfrac=0.2, seed=None):
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(np.asarray(lb, float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, float), (dim,)).copy()
    X = lb + rng.random((N, dim)) * (ub - lb)
    V = np.zeros((N, dim))
    m = N // K
    vmax = vfrac * (ub - lb)
    w = wmax - (wmax - wmin) * np.arange(K * m) / (N - 1)      # rank-dependent inertia (Eq 3)
    fe, best_x, best_f, curve = 0, None, np.inf, []
    while fe + N <= max_fe:
        np.clip(X, lb, ub, out=X)
        F = np.array([objf(x) for x in X]); fe += N
        order = np.argsort(F, kind="stable")
        if F[order[0]] < best_f:
            best_f, best_x = float(F[order[0]]), X[order[0]].copy()
        curve.append(best_f)
        Xs, Fs = X[order[:K * m]], F[order[:K * m]]
        f1 = Fs[:m]
        wk = f1.max() - f1 + rng.random()                       # fitness weights of the elite group (Eq 2)
        MC1 = (wk[:, None] * Xs[:m]).sum(0) / wk.sum()
        xb = Xs[0].copy()
        r1, r2 = rng.random((K * m, dim)), rng.random((K * m, dim))
        Vs = V[order[:K * m]]
        Vn = w[:, None] * (Vs + eta1 * r1 * (MC1 - Xs) + eta2 * r2 * (xb - Xs))
        Vn = np.clip(Vn, -vmax, vmax)
        Vn[0] = Vs[0]                                            # elitism: best agent is not moved
        Xn = Xs + Vn; Xn[0] = Xs[0]
        X[order[:K * m]], V[order[:K * m]] = Xn, Vn
    return best_x, best_f, curve


if __name__ == "__main__":
    sphere = lambda x: float(np.sum((x - 30.0) ** 2))
    x, f, _ = FalCO(sphere, -100, 100, 30, seed=1)
    print("shifted sphere, 15000 evaluations:", f)


FC = FalCO  # name used in the experiment scripts
