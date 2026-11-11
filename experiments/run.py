"""Run one (algorithm, function) cell for all runs and save results.

Usage: python3 -I run.py ALGO FUNC [RUNS]
FUNC is F1..F23 or sF1..sF13 (shifted versions).
"""
import os
import sys
import json
import time
import random
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from algos import ALGOS
from EvoloPy import benchmarks as B

POP = 100
MAXFE = 15000
NCHK = 100  # convergence checkpoints


class BudgetExhausted(Exception):
    pass


def get_func(name):
    if name == "BLDC":
        import bldc
        return bldc.BLDC, list(bldc.LB), list(bldc.UB), len(bldc.LB)
    shifted = name.startswith("s")
    base = name[1:] if shifted else name
    fname, lb, ub, dim = B.getFunctionDetails(base)
    import fixbench
    f = getattr(fixbench, base, None) or getattr(B, base)
    if shifted:
        rng = np.random.RandomState(zlib.crc32(base.encode()))
        o = rng.uniform(0.5 * lb, 0.5 * ub, dim)
        g = lambda x, f=f, o=o: f(np.asarray(x) - o)
    else:
        g = f
    return g, lb, ub, dim


class Budget:
    def __init__(self, f):
        self.f, self.fe, self.best = f, 0, np.inf
        self.curve = np.full(NCHK, np.nan)
        self.step = MAXFE // NCHK

    def __call__(self, x):
        if self.fe >= MAXFE:
            raise BudgetExhausted
        v = float(np.asarray(self.f(np.asarray(x, float))).reshape(-1)[0])
        self.fe += 1
        if v < self.best:
            self.best = v
            self.bestx = [float(t) for t in np.asarray(x, float)]
        if self.fe % self.step == 0:
            self.curve[self.fe // self.step - 1] = self.best
        return v


def main():
    algo, fname = sys.argv[1], sys.argv[2]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    out = os.path.join(HERE, os.environ.get("RUN_OUT", "results"), f"{algo}__{fname}.json")
    if os.path.exists(out):
        return
    f, lb, ub, dim = get_func(fname)
    iters = MAXFE // POP + 5
    recs = []
    for r in range(runs):
        seed = zlib.crc32(f"{algo}-{fname}-{r}".encode()) % (2**31)
        np.random.seed(seed)
        random.seed(seed)
        bud = Budget(f)
        bud.__name__ = fname
        t0 = time.process_time()
        try:
            ALGOS[algo](bud, lb, ub, dim, POP, iters)
        except BudgetExhausted:
            pass
        t = time.process_time() - t0
        c = bud.curve
        # forward-fill any checkpoint the run did not reach (should not happen)
        for i in range(1, NCHK):
            if np.isnan(c[i]):
                c[i] = c[i - 1]
        recs.append({"run": r, "best": bud.best, "fe": bud.fe, "time": t,
                     "x": getattr(bud, "bestx", None),
                     "curve": [float(v) for v in c]})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump({"algo": algo, "func": fname, "pop": POP, "maxfe": MAXFE,
                   "runs": recs}, fh)
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
