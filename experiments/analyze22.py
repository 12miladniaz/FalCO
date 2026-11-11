"""Analysis of the CEC2022 study: mean/std error, Wilcoxon W/T/L of FC vs each, Friedman ranks."""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, os.environ.get("C22_OUT", "results_c22"))
ALG = ["FC", "FC-noMC", "PSO", "SPSO2011", "CLPSO", "GWO", "L-SHADE", "CMA-ES"]
TOL = 1e-8


def load(a, D, f):
    p = os.path.join(RD, f"{a}__D{D}_F{f}.json")
    if not os.path.exists(p):
        return None
    e = np.array([r["err"] for r in json.load(open(p))["runs"]])
    return np.where(e < TOL, 0.0, e)


def main(D):
    algs = [a for a in ALG if all(load(a, D, f) is not None for f in range(1, 13))]
    print(f"\n=== D={D}  complete algorithms: {algs}")
    E = {a: [load(a, D, f) for f in range(1, 13)] for a in algs}
    print("F   " + "".join(f"{a:>20s}" for a in algs))
    for f in range(12):
        print(f"F{f+1:<3d}" + "".join(f"{E[a][f].mean():11.3e}±{E[a][f].std():.1e}" for a in algs))
    # Wilcoxon rank-sum FC vs others
    print("\nFC vs (+ FC better / = / -):")
    for a in algs:
        if a == "FC":
            continue
        w = t = l = 0
        for f in range(12):
            x, y = E["FC"][f], E[a][f]
            p = 1.0 if np.allclose(x, y) else stats.ranksums(x, y).pvalue
            if p >= 0.05:
                t += 1
            elif np.median(x) < np.median(y) or (np.median(x) == np.median(y) and x.mean() < y.mean()):
                w += 1
            else:
                l += 1
        print(f"  {a:10s} {w}/{t}/{l}")
    # Friedman on mean errors per function
    M = np.array([[E[a][f].mean() for a in algs] for f in range(12)])
    R = np.array([stats.rankdata(r) for r in M])
    ch = stats.friedmanchisquare(*M.T)
    print("\nFriedman mean ranks:", ", ".join(f"{a} {r:.2f}" for a, r in sorted(zip(algs, R.mean(0)), key=lambda z: z[1])),
          f"  p={ch.pvalue:.2e}")


if __name__ == "__main__":
    for D in (10, 20):
        main(D)
