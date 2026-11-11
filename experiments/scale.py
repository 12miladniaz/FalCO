"""Scale-invariance test: f_s(x) = f(x / s) on [s*lb, s*ub]; an invariant method gives the same errors for every s."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fcv import FCv
from fc3 import FC3

D, N, FE, RUNS = 30, 100, 15000, 20
rng0 = np.random.default_rng(12345)
SHIFT = rng0.uniform(-50, 50, D)


def sphere(z):
    return float(np.sum(z ** 2))


def rastrigin(z):
    return float(np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10))


class Stop(Exception):
    pass


def run(alg, base, s, seed):
    np.random.seed(seed)
    cnt, best = [0], [np.inf]

    def f(x):
        if cnt[0] >= FE:
            raise Stop
        cnt[0] += 1
        v = base(x / s - SHIFT * (5.12 / 100 if base is rastrigin else 1))
        best[0] = min(best[0], v)
        return v
    lo, hi = (-5.12, 5.12) if base is rastrigin else (-100.0, 100.0)
    try:
        if alg == "old":
            FCv(f, lo * s, hi * s, D, N=N, K=5, eta3=0.0, norm=True)
        else:
            FC3(f, lo * s, hi * s, D, N=N, vfrac=0.2)
    except Stop:
        pass
    return best[0]


if __name__ == "__main__":
    out = {}
    for name, base in [("sphere", sphere), ("rastrigin", rastrigin)]:
        for s in [1e-2, 1, 1e2, 1e3, 1e4]:
            for alg in ["old", "new"]:
                v = [run(alg, base, s, 1000 + r) for r in range(RUNS)]
                out[f"{name}|{s:g}|{alg}"] = v
                print(name, s, alg, f"{np.mean(v):.3e} ± {np.std(v):.1e}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "scale.json"), "w"))
