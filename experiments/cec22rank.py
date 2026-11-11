"""CEC2022 competition ranking (U-score, port of the organizers' ranking.m) with FalCO added.

For each pair of algorithms and each function, all 30x30 pairs of runs are compared: the lower final error wins;
equal errors are broken by the number of evaluations used to reach the target (lower wins); full ties give 0.5.
An algorithm's score in one dimension is the sum of its wins over all opponents and functions.
"""
import json
import os

import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402
COMP = paths.CEC22_COMP
ENTRIES = ["Co-PPSO", "EA4eigN100_10", "IMPML-SHADE", "IUMOEAII", "jSObinexpEig", "MTT_SHADE", "NL-SHADE-LBC",
           "NL-SHADE-RSP-MID", "OMCSOMA", "S_LSHADE_DP", "SPHH_Ensemble", "ZOCMAES", "NLSOMACLP"]
NAMES = {"EA4eigN100_10": "EA4eig", "S_LSHADE_DP": "S-LSHADE-DP"}
TOL = 1e-8


def load_entry(name, D):
    f = os.path.join(COMP, f"{name}_{D}.xlsx") if os.path.exists(os.path.join(COMP, f"{name}_{D}.xlsx")) \
        else os.path.join(COMP, f"{name}_{D}_{D}.xlsx")
    if name == "EA4eigN100_10":
        f = os.path.join(COMP, f"EA4eigN100_10_{D}.xlsx")
    wb = openpyxl.load_workbook(f, data_only=True)
    E = np.array([[c for c in r[:12]] for r in wb.worksheets[0].iter_rows(min_row=1, max_row=30, values_only=True)],
                 float)
    T = np.array([[c for c in r[:12]] for r in wb.worksheets[1].iter_rows(min_row=1, max_row=30, values_only=True)],
                 float)
    return E, T


def load_ours(algo, D, rd="results_c22_hit"):
    E, T = np.zeros((30, 12)), np.zeros((30, 12))
    for f in range(1, 13):
        runs = json.load(open(os.path.join(HERE, rd, f"{algo}__D{D}_F{f}.json")))["runs"]
        e = np.array([r["err"] for r in runs])
        E[:, f - 1] = np.where(e < TOL, 0.0, e)
        T[:, f - 1] = [r.get("hit_fe", 200000 if D == 10 else 1000000) for r in runs]
    return E, T


def uscore(data):
    names = list(data)
    S = {a: 0.0 for a in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A1, A2 = data[a]
            B1, B2 = data[b]
            for p in range(12):
                x1, x2 = A1[:, p][:, None], A2[:, p][:, None]
                y1, y2 = B1[:, p][None, :], B2[:, p][None, :]
                win = (x1 < y1) | ((x1 == y1) & (x2 < y2))
                los = (x1 > y1) | ((x1 == y1) & (x2 > y2))
                tie = (x1 == y1) & (x2 == y2)
                S[a] += win.sum() + 0.5 * tie.sum()
                S[b] += los.sum() + 0.5 * tie.sum()
    return S


if __name__ == "__main__":
    tot = {}
    for D in (10, 20):
        data = {e: load_entry(e, D) for e in ENTRIES}
        S = uscore(data)
        print(f"D={D} reproduced official scores:", {NAMES.get(k, k): v for k, v in S.items()})
        data["FalCO"] = load_ours("FC", D)
        S2 = uscore(data)
        for k, v in S2.items():
            tot[k] = tot.get(k, 0) + v
        print(f"D={D} with FalCO:", sorted(((NAMES.get(k, k), v) for k, v in S2.items()), key=lambda z: -z[1]))
    rank = sorted(tot.items(), key=lambda z: -z[1])
    print("TOTAL rank:")
    for i, (k, v) in enumerate(rank):
        print(i + 1, NAMES.get(k, k), v)
