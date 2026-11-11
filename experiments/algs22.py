"""Vectorised algorithms for the CEC2022 study. Every algorithm has the signature
alg(P) where P is a c22.Problem: P(X) evaluates a batch (rows) and raises BudgetExhausted at the budget.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
sys.path.insert(0, paths.PYCMA)


# ------------------------------------------------------------------ FC (final) and its rank-PSO ablation
def FC(P, N=100, K=5, eta1=5.0, eta2=1.0, wmax=0.832, wmin=0.04, vfrac=0.2):
    lb, ub, D = P.lb, P.ub, P.D
    rg = ub - lb
    X = lb + np.random.rand(N, D) * rg
    V = np.zeros((N, D))
    m = N // K
    vmax = vfrac * rg
    w = wmax - (wmax - wmin) * np.arange(K * m) / (N - 1)
    while True:
        np.clip(X, lb, ub, out=X)
        F = P(X)
        o = np.argsort(F, kind="stable")[:K * m]
        Xs, Fs = X[o], F[o]
        f1 = Fs[:m]
        wk = f1.max() - f1 + np.random.rand()
        MC = (wk[:, None] * Xs[:m]).sum(0) / wk.sum()
        xb = Xs[0].copy()
        Vs = V[o]
        Vn = w[:, None] * (Vs + eta1 * np.random.rand(K * m, D) * (MC - Xs)
                           + eta2 * np.random.rand(K * m, D) * (xb - Xs))
        Vn = np.clip(Vn, -vmax, vmax)
        Vn[0] = Vs[0]
        Xn = Xs + Vn
        Xn[0] = Xs[0]
        X[o], V[o] = Xn, Vn


def FC_noMC(P):
    """Ablation: rank-dependent inertia with attraction to the best agent only (eta1 = 0)."""
    return FC(P, eta1=0.0)


# ------------------------------------------------------------------ standard PSO (Shi & Eberhart)
def PSO(P, N=40, wmax=0.9, wmin=0.4, c1=2.0, c2=2.0, vfrac=0.2):
    lb, ub, D = P.lb, P.ub, P.D
    rg = ub - lb
    X = lb + np.random.rand(N, D) * rg
    V = np.zeros((N, D))
    vmax = vfrac * rg
    F = P(X)
    pb, pf = X.copy(), F.copy()
    while True:
        g = pb[np.argmin(pf)]
        w = wmax - (wmax - wmin) * P.fe / P.maxfe
        V = w * V + c1 * np.random.rand(N, D) * (pb - X) + c2 * np.random.rand(N, D) * (g - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lb, ub)
        F = P(X)
        imp = F < pf
        pb[imp], pf[imp] = X[imp], F[imp]


# ------------------------------------------------------------------ SPSO2011 (Clerc 2011; Zambrano-Bigiarini et al. 2013)
def SPSO2011(P, N=40, K=3):
    lb, ub, D = P.lb, P.ub, P.D
    rg = ub - lb
    w = 1.0 / (2.0 * math.log(2.0))
    c = 0.5 + math.log(2.0)
    X = lb + np.random.rand(N, D) * rg
    V = (lb - X) + np.random.rand(N, D) * rg          # U(lb - x, ub - x)
    F = P(X)
    pb, pf = X.copy(), F.copy()

    def topology():
        L = np.random.rand(N, N) < 1 - (1 - 1.0 / N) ** K   # each particle informs K random others
        np.fill_diagonal(L, True)
        return L

    links = topology()
    gbest = pf.min()
    while True:
        # best informant of each particle (links[j, i]: j informs i)
        fmat = np.where(links, pf[:, None], np.inf)          # rows: informant j, cols: receiver i
        lbest = pb[np.argmin(fmat, axis=0)]
        same = np.all(lbest == pb, axis=1)
        p_ = X + c * (pb - X)
        l_ = X + c * (lbest - X)
        G = np.where(same[:, None], (X + p_) / 2.0, (X + p_ + l_) / 3.0)
        r = np.linalg.norm(G - X, axis=1)
        dirn = np.random.randn(N, D)
        dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
        rad = r * np.random.rand(N) ** (1.0 / D)
        Xp = G + dirn * rad[:, None]
        V = w * V + Xp - X
        Xn = X + V
        low, high = Xn < lb, Xn > ub
        Xn = np.clip(Xn, lb, ub)
        V[low | high] *= -0.5
        X = Xn
        F = P(X)
        imp = F < pf
        pb[imp], pf[imp] = X[imp], F[imp]
        if pf.min() >= gbest:
            links = topology()
        gbest = min(gbest, pf.min())


# ------------------------------------------------------------------ CLPSO (Liang et al. 2006)
def CLPSO(P, N=40, wmax=0.9, wmin=0.4, c=1.49445, gap=7, vfrac=0.2):
    lb, ub, D = P.lb, P.ub, P.D
    rg = ub - lb
    vmax = vfrac * rg
    X = lb + np.random.rand(N, D) * rg
    V = -vmax + 2 * np.random.rand(N, D) * vmax
    F = P(X)
    pb, pf = X.copy(), F.copy()
    Pc = 0.05 + 0.45 * (np.exp(10.0 * np.arange(N) / (N - 1)) - 1) / (math.exp(10.0) - 1)
    stall = np.zeros(N, int)
    ex = np.tile(np.arange(N)[:, None], (1, D))       # exemplar index per particle and dimension

    def assign(i):
        e = np.full(D, i)
        use = np.random.rand(D) < Pc[i]
        if not use.any():
            use[np.random.randint(D)] = True
        for d in np.nonzero(use)[0]:
            a, b = np.random.randint(N, size=2)
            e[d] = a if pf[a] < pf[b] else b
        if np.all(e == i):
            d = np.random.randint(D)
            e[d] = (i + 1 + np.random.randint(N - 1)) % N
        ex[i] = e

    for i in range(N):
        assign(i)
    while True:
        w = wmax - (wmax - wmin) * P.fe / P.maxfe
        E = pb[ex, np.arange(D)[None, :]]
        V = w * V + c * np.random.rand(N, D) * (E - X)
        V = np.clip(V, -vmax, vmax)
        X = X + V
        inb = np.all((X >= lb) & (X <= ub), axis=1)
        F = P(np.clip(X, lb, ub))
        imp = (F < pf) & inb
        pb[imp], pf[imp] = X[imp], F[imp]
        stall = np.where(imp, 0, stall + 1)
        for i in np.nonzero(stall >= gap)[0]:
            assign(i)
            stall[i] = 0


# ------------------------------------------------------------------ GWO (Mirjalili et al. 2014)
def GWO(P, N=50):
    lb, ub, D = P.lb, P.ub, P.D
    X = lb + np.random.rand(N, D) * (ub - lb)
    lead_f = np.full(3, np.inf)
    lead_x = np.zeros((3, D))
    while True:
        X = np.clip(X, lb, ub)
        F = P(X)
        allx = np.vstack([lead_x, X])
        allf = np.concatenate([lead_f, F])
        o = np.argsort(allf)[:3]
        lead_x, lead_f = allx[o].copy(), allf[o].copy()
        a = 2.0 - 2.0 * P.fe / P.maxfe
        Xn = np.zeros_like(X)
        for k in range(3):
            A = 2 * a * np.random.rand(N, D) - a
            C = 2 * np.random.rand(N, D)
            Xn += lead_x[k] - A * np.abs(C * lead_x[k] - X)
        X = Xn / 3.0


# ------------------------------------------------------------------ CMA-ES with IPOP restarts (pycma)
def CMAES(P):
    import cma
    lb, ub, D = P.lb, P.ub, P.D
    popsize = None
    while True:
        x0 = lb + np.random.rand(D) * (ub - lb)
        opts = {"bounds": [list(lb), list(ub)], "verbose": -9, "seed": int(np.random.randint(1, 2**31 - 1)),
                "maxfevals": 10**12, "tolfun": 1e-12, "tolx": 1e-12}
        if popsize:
            opts["popsize"] = popsize
        es = cma.CMAEvolutionStrategy(list(x0), 0.3 * float(np.max(ub - lb)), opts)
        while not es.stop():
            Xs = es.ask()
            es.tell(Xs, list(P(np.array(Xs))))
        popsize = 2 * es.popsize


# ------------------------------------------------------------------ L-SHADE (Tanabe & Fukunaga 2014)
def LSHADE(P, H=6, p=0.11, arc_rate=2.6, rNinit=18):
    lb, ub, D = P.lb, P.ub, P.D
    maxfe = P.maxfe
    NPinit, NPmin = max(rNinit * D, 30), 4
    NP = NPinit
    X = lb + np.random.rand(NP, D) * (ub - lb)
    F = P(X)
    MF, MCR, k = np.full(H, 0.5), np.full(H, 0.5), 0
    A = np.empty((0, D))
    while True:
        r = np.random.randint(0, H, NP)
        CR = np.where(MCR[r] < 0, 0.0, np.clip(np.random.normal(MCR[r], 0.1), 0, 1))
        Fs = np.empty(NP)
        for i in range(NP):
            f = -1
            while f <= 0:
                f = MF[r[i]] + 0.1 * math.tan(math.pi * (np.random.rand() - 0.5))
            Fs[i] = min(f, 1.0)
        order = np.argsort(F)
        pnum = max(2, int(round(p * NP)))
        pbest = X[order[np.random.randint(pnum, size=NP)]]
        r1 = np.array([np.random.choice([j for j in range(NP) if j != i]) for i in range(NP)])
        union = np.vstack([X, A]) if len(A) else X
        r2 = np.empty(NP, int)
        for i in range(NP):
            while True:
                t = np.random.randint(len(union))
                if t != i and t != r1[i]:
                    break
            r2[i] = t
        Vm = X + Fs[:, None] * (pbest - X) + Fs[:, None] * (X[r1] - union[r2])
        Vm = np.where(Vm < lb, (lb + X) / 2, Vm)
        Vm = np.where(Vm > ub, (ub + X) / 2, Vm)
        mask = np.random.rand(NP, D) <= CR[:, None]
        mask[np.arange(NP), np.random.randint(D, size=NP)] = True
        U = np.where(mask, Vm, X)
        FU = P(U)
        s = FU < F
        if s.any():
            dF = np.abs(F[s] - FU[s])
            wgt = dF / dF.sum()
            sF, sCR = Fs[s], CR[s]
            MF[k] = np.sum(wgt * sF ** 2) / max(np.sum(wgt * sF), 1e-30)
            MCR[k] = -1 if (MCR[k] < 0 or sCR.max() == 0) else np.sum(wgt * sCR ** 2) / max(np.sum(wgt * sCR), 1e-30)
            k = (k + 1) % H
            A = np.vstack([A, X[s]]) if len(A) else X[s].copy()
        b = FU <= F
        X[b], F[b] = U[b], FU[b]
        newNP = max(NPmin, int(round((NPmin - NPinit) / maxfe * P.fe + NPinit)))
        if newNP < NP:
            keep = np.argsort(F)[:newNP]
            X, F, NP = X[keep], F[keep], newNP
        amax = int(round(arc_rate * NP))
        if len(A) > amax:
            A = A[np.random.choice(len(A), amax, replace=False)]


def FC_noBest(P):
    """Ablation: attraction to the elite centre only (eta2 = 0)."""
    return FC(P, eta2=0.0)


ALGS = {"FC": FC, "FC-noMC": FC_noMC, "FC-noBest": FC_noBest, "PSO": PSO, "SPSO2011": SPSO2011, "CLPSO": CLPSO,
        "GWO": GWO, "CMA-ES": CMAES, "L-SHADE": LSHADE}
