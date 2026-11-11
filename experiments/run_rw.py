"""Run FalCO on one CEC2020 real-world constrained problem: python3 run_rw.py K [RUNS]"""
import json
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rw
from fcv import FCv
from fc3 import FC3


class BudgetExhausted(Exception):
    pass


class Wrapper:
    def __init__(self, prob):
        self.p = prob
        self.fe = 0
        self.best = (np.inf, np.inf)  # (v, f)
        self.bestx = None
        self.chk = []
        self.step = prob.maxfes // 10
        self.eps0 = None
        self.init_v = []

    # epsilon-constrained ranking (Takahama & Sakai): violations below eps(t) are treated as feasible.
    EPS = os.environ.get("FC_EPS", "0") == "1"
    THETA, TC, CP = 0.2, 0.2, 5.0

    def eps(self):
        if not self.EPS:
            return 0.0
        if self.eps0 is None:
            return float("inf")
        t = self.fe / self.p.maxfes
        return self.eps0 * (1 - t / self.TC) ** self.CP if t < self.TC else 0.0

    def __call__(self, x):
        if self.fe >= self.p.maxfes:
            raise BudgetExhausted
        f, v = self.p.evaluate(x)
        self.fe += 1
        if self.EPS and self.eps0 is None:
            self.init_v.append(v if np.isfinite(v) else np.inf)
            if len(self.init_v) == 100:
                vs = np.sort(self.init_v)
                self.eps0 = float(vs[int(self.THETA * len(vs))])
        if not (np.isfinite(f) and np.isfinite(v)):
            # undefined objective or constraints (e.g. log of a non-positive number): worst possible value
            if self.fe % self.step == 0:
                self.chk.append([self.best[1], self.best[0]])
            return 1e15
        key = (v, f) if v > 0 else (0.0, f)
        if key < self.best:
            self.best = key
            self.bestx = [float(t) for t in x]
        if self.fe % self.step == 0:
            self.chk.append([self.best[1], self.best[0]])
        e = self.eps()
        return rw.scalar(f, 0.0 if v <= e else v)


def main():
    k = int(sys.argv[1])
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    out = os.path.join(HERE, os.environ.get("RW_OUT", "results_rw"), f"FC__RC{k:02d}.json")
    if os.path.exists(out):
        return
    prob = rw.Problem(k)
    recs = []
    for r in range(runs):
        seed = zlib.crc32(f"FC-RC{k}-{r}".encode()) % (2**31)
        np.random.seed(seed)
        w = Wrapper(prob)
        t0 = time.process_time()
        try:
            if os.environ.get("FC_LEGACY") == "1":
                # earlier version of the method (repulsion term eta3, absolute velocity limit)
                FCv(w, prob.lb, prob.ub, prob.d, N=100, eta3=float(os.environ.get("FC_ETA3", "0.3")))
            else:
                # FalCO as defined in the paper (results_rw4/)
                FC3(w, prob.lb, prob.ub, prob.d, N=100, vfrac=float(os.environ.get("FC_VFRAC", "0.2")))
        except BudgetExhausted:
            pass
        recs.append({"run": r, "f": w.best[1], "v": w.best[0], "fe": w.fe,
                     "time": time.process_time() - t0, "chk": w.chk, "x": w.bestx})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump({"problem": k, "D": prob.d, "maxfes": prob.maxfes, "runs": recs}, fh)
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
