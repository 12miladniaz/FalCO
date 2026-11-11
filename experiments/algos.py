"""Algorithm registry for the FC re-evaluation.

All algorithms are called as f(objf, lb, ub, dim, pop, iters).
The objective passed in is a budget-limited wrapper (see run.py) that raises
BudgetExhausted once MAXFE evaluations are spent, so every algorithm gets the
same number of function evaluations regardless of how many it uses per iteration.
"""
import os
import sys
import io
import contextlib
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
sys.path.insert(0, paths.EVOLOPY)
sys.path.insert(0, paths.PYCMA)
sys.path.insert(0, os.path.join(paths.EVOLOPY, "EvoloPy"))  # provides `solution` for FC9.py

from EvoloPy.optimizers import (PSO, GWO, WOA, HHO, DE, SSA, MFO, MVO, SCA, JAYA,
                                GA, CS, BAT, FFA)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Authors' published implementation, unchanged.
FC_orig_mod = _load(os.path.join(HERE, "legacy", "FC9.py"), "FC9")
# Same file with one change: the bracket in the inter-leg term (see FC9_corrected.py).
FC_corr_mod = _load(os.path.join(HERE, "FC9_corrected.py"), "FC9_corrected")


def _variant(name, **env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return _load(os.path.join(HERE, "FC9_variant.py"), name)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


FC_VARIANTS = {
    "FC-noW": _variant("v_noW", FC_USE_W="0"),
    "FC-noRank": _variant("v_noRank", FC_RANK="0"),
    "FC-K3": _variant("v_K3", FC_K="3"),
    "FC-K10": _variant("v_K10", FC_K="10"),
}


def _quiet(fn):
    def wrapped(*a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)
    return wrapped


def cmaes(objf, lb, ub, dim, pop, iters):
    import cma
    lb = np.array(lb if isinstance(lb, list) else [lb] * dim, float)
    ub = np.array(ub if isinstance(ub, list) else [ub] * dim, float)
    x0 = lb + np.random.rand(dim) * (ub - lb)
    sigma0 = 0.3 * float(np.max(ub - lb))
    opts = {"bounds": [list(lb), list(ub)], "verbose": -9,
            "seed": int(np.random.randint(1, 2**31 - 1)),
            "maxfevals": 10**9, "tolfun": 0, "tolx": 0, "tolfunhist": 0,
            "tolstagnation": 10**9}
    # IPOP-style restarts until the budget wrapper stops us
    popsize = None
    while True:
        o = dict(opts)
        if popsize:
            o["popsize"] = popsize
        es = cma.CMAEvolutionStrategy(list(x0), sigma0, o)
        while not es.stop():
            X = es.ask()
            es.tell(X, [objf(np.asarray(x)) for x in X])
        popsize = 2 * es.popsize
        x0 = lb + np.random.rand(dim) * (ub - lb)


def lshade(objf, lb, ub, dim, pop, iters, maxfe=None):
    """L-SHADE (Tanabe & Fukunaga, CEC 2014) with linear population size reduction."""
    lb = np.array(lb if isinstance(lb, list) else [lb] * dim, float)
    ub = np.array(ub if isinstance(ub, list) else [ub] * dim, float)
    maxfe = maxfe or pop * iters
    NPinit, NPmin, H, p, arc_rate = max(18 * dim, 30), 4, 6, 0.11, 2.6
    NP = NPinit
    X = lb + np.random.rand(NP, dim) * (ub - lb)
    F = np.array([objf(x) for x in X])
    fe = NP
    MF, MCR, k = np.full(H, 0.5), np.full(H, 0.5), 0
    A = np.empty((0, dim))
    while True:
        r = np.random.randint(0, H, NP)
        CR = np.where(MCR[r] < 0, 0.0, np.clip(np.random.normal(MCR[r], 0.1), 0, 1))
        Fs = np.empty(NP)
        for i in range(NP):
            f = -1
            while f <= 0:
                f = MF[r[i]] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            Fs[i] = min(f, 1.0)
        order = np.argsort(F)
        pnum = max(2, int(round(p * NP)))
        U = np.empty_like(X)
        union = np.vstack([X, A]) if len(A) else X
        for i in range(NP):
            pb = X[order[np.random.randint(pnum)]]
            r1 = i
            while r1 == i:
                r1 = np.random.randint(NP)
            r2 = r1
            while r2 == r1 or r2 == i:
                r2 = np.random.randint(len(union))
            v = X[i] + Fs[i] * (pb - X[i]) + Fs[i] * (X[r1] - union[r2])
            v = np.where(v < lb, (lb + X[i]) / 2, v)
            v = np.where(v > ub, (ub + X[i]) / 2, v)
            jr = np.random.randint(dim)
            mask = np.random.rand(dim) <= CR[i]
            mask[jr] = True
            U[i] = np.where(mask, v, X[i])
        FU = np.array([objf(u) for u in U])
        fe += NP
        better = FU < F
        strictly = FU < F
        if strictly.any():
            dF = np.abs(F[strictly] - FU[strictly])
            w = dF / dF.sum()
            sF, sCR = Fs[strictly], CR[strictly]
            MF[k] = np.sum(w * sF**2) / max(np.sum(w * sF), 1e-30)
            MCR[k] = -1 if (MCR[k] < 0 or sCR.max() == 0) else np.sum(w * sCR**2) / max(np.sum(w * sCR), 1e-30)
            k = (k + 1) % H
            A = np.vstack([A, X[strictly]]) if len(A) else X[strictly].copy()
        better = FU <= F
        X[better], F[better] = U[better], FU[better]
        newNP = int(round((NPmin - NPinit) / maxfe * fe + NPinit))
        newNP = max(NPmin, newNP)
        if newNP < NP:
            keep = np.argsort(F)[:newNP]
            X, F, NP = X[keep], F[keep], newNP
        amax = int(round(arc_rate * NP))
        if len(A) > amax:
            A = A[np.random.choice(len(A), amax, replace=False)]


ALGOS = {
    "FC": _quiet(FC_orig_mod.FC),
    "FC-c": _quiet(FC_corr_mod.FC),
    "PSO": PSO.PSO, "GWO": GWO.GWO, "WOA": WOA.WOA, "HHO": HHO.HHO,
    "DE": DE.DE, "SSA": SSA.SSA, "MFO": MFO.MFO, "MVO": MVO.MVO,
    "SCA": SCA.SCA, "JAYA": JAYA.JAYA, "GA": GA.GA, "CS": CS.CS,
    "BAT": BAT.BAT, "FFA": FFA.FFA,
    "CMA-ES": cmaes, "L-SHADE": lshade,
}
for _k, _m in FC_VARIANTS.items():
    ALGOS[_k] = _m.FC
for _k in list(ALGOS):
    ALGOS[_k] = _quiet(ALGOS[_k])


# ---- redefined FC (no inter-leg term), vectorised; variants for ablation/sensitivity
from fcv import FCv as _FCv


def _fc2(**kw):
    def f(objf, lb, ub, dim, pop, iters):
        return _FCv(objf, lb, ub, dim, N=pop, **kw)
    return f


ALGOS.update({
    "FC2-K3": _fc2(K=3, eta3=0.0),
    "FC2-K5": _fc2(K=5, eta3=0.0),
    "FC2-K2": _fc2(K=2, eta3=0.0),
    "FC2-K10": _fc2(K=10, eta3=0.0),
})

ALGOS.update({
    "FC2-noRank": _fc2(K=5, eta3=0.0, rank=False),
    "FC2-noMC": _fc2(K=5, eta3=0.0, eta1=0.0),
    "FC2-noBest": _fc2(K=5, eta3=0.0, eta2=0.0),
    "FC2-W": _fc2(K=5, eta3=0.3),
})

# population-size study: current (absolute rank) vs normalised rank coefficients
for _n in (30, 50, 100, 200):
    ALGOS[f"FC-abs-N{_n}"] = (lambda n: (lambda o, l, u, d, pop, it: _FCv(o, l, u, d, N=n, K=5, eta3=0.0)))(_n)
    ALGOS[f"FC-norm-N{_n}"] = (lambda n: (lambda o, l, u, d, pop, it: _FCv(o, l, u, d, N=n, K=5, eta3=0.0, norm=True)))(_n)

# ---- diagnostic study of the mathematical model
from fc3 import FC3 as _FC3
for _name, _kw in {"FC3-v": dict(vfrac=0.2), "FC3-vw": dict(vfrac=0.2, wts="rank"),
                   "FC3-vwr": dict(vfrac=0.2, wts="rank", restart=True),
                   "FC3-w": dict(wts="rank"), "FC3-r": dict(restart=True)}.items():
    ALGOS[_name] = (lambda kw: (lambda o, l, u, d, pop, it: _FC3(o, l, u, d, N=pop, **kw)))(_kw)

# ---- ablation and sensitivity of the final FC (FC3-v)
for _name, _kw in {"FCv-noRank": dict(rank=False), "FCv-noMC": dict(eta1=0.0), "FCv-noBest": dict(eta2=0.0),
                   "FCv-K2": dict(K=2), "FCv-K3": dict(K=3), "FCv-K10": dict(K=10)}.items():
    ALGOS[_name] = (lambda kw: (lambda o, l, u, d, pop, it: _FC3(o, l, u, d, N=pop, vfrac=0.2, **kw)))(_kw)
