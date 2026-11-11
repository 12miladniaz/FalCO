"""python3 -I run22.py ALGO D FNUM [RUNS]  -> results_c22/ALGO__D{D}_F{FNUM}.json"""
import json
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import c22
from algs22 import ALGS


def main():
    algo, D, fn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    outdir = os.path.join(HERE, os.environ.get("C22_OUT", "results_c22"))
    out = os.path.join(outdir, f"{algo}__D{D}_F{fn}.json")
    if os.path.exists(out):
        return
    recs = []
    for r in range(runs):
        seed = zlib.crc32(f"{algo}-{D}-{fn}-{r}".encode()) % (2**31)
        np.random.seed(seed)
        P = c22.Problem(fn, D)
        t0 = time.process_time()
        try:
            ALGS[algo](P)
        except c22.BudgetExhausted:
            pass
        recs.append({"run": r, "best": P.best, "err": P.best - c22.BIAS[fn - 1], "fe": P.fe,
                     "time": time.process_time() - t0, "curve": P.curve,
                     "hit_fe": P.hit_fe if P.hit_fe is not None else P.maxfe})
    os.makedirs(outdir, exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump({"algo": algo, "D": D, "f": fn, "maxfe": c22.maxfes(D), "runs": recs}, fh)
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
