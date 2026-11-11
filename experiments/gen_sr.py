"""Tables and figures for the Scientific Reports manuscript (final FC = FC3-v)."""
import json
import os
import sys

os.environ["FC_FINAL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import analyze as A
import analyze22 as A22
import analyze_rw as R
import cecscore as C
import bldc

import paths  # noqa: E402
OUT = paths.OUTPUT
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False})


def w(name, text):
    import re
    text = text.replace("FC-old", "earlier version")
    text = re.sub(r"\bFC\b", "FalCO", text)
    open(os.path.join(OUT, name), "w").write(text.strip() + "\n")


def e(x, d=2):
    if x == 0:
        return "0"
    m, ex = f"{x:.{d - 1}e}".split("e")
    ex = int(ex)
    if -2 <= ex <= 3:
        return f"{x:.{max(0, d - 1 - ex)}f}"
    return f"{m}×10^{ex}^"


# ------------------------------------------------------------------ Table 1
def table1():
    st = json.load(open(os.path.join(HERE, "analysis", "stab.json")))
    sc = json.load(open(os.path.join(HERE, "analysis", "scale.json")))
    L = ["Table: **Supplementary Table S10.** Rank-based self-stabilization and scale invariance of FC (shifted sphere and Rastrigin "
         "functions, $D=30$, 15,000 evaluations, 20 runs). (a) Mean error of FC and of two controls in which all agents "
         "share one inertia weight, and mean number of consecutive iterations an agent spends in the ranks with "
         "$w_i>w^*=0.628$. (b) Mean error when every variable and its bounds are multiplied by $s$; FC-old uses the "
         "absolute velocity limit of an earlier version.", "",
         "| (a) Inertia weight | Stable by Eq. (2) | Sphere | Rastrigin | Residence in unstable ranks (sphere / Rastrigin) |",
         "|:--|:-:|:-:|:-:|:-:|"]
    rows = [("FC", "Rank-dependent, 0.832 to 0.04", "top 26 ranks: no"),
            ("w=0.44 (all stable)", "Fixed, 0.436 (mean of FC)", "yes"),
            ("w=0.70 (all unstable)", "Fixed, 0.70", "no"),
            ("w=0.832 (all unstable)", "Fixed, 0.832", "no")]
    for k, lab, stab in rows:
        s, r = st[f"sphere|{k}"], st[f"rastrigin|{k}"]
        res = f"{s['stay_mean']:.2f} / {r['stay_mean']:.2f} iterations" if s["stay_mean"] else ("none" if "stable" in k and "un" not in k
                                                                            else "permanent")
        L.append(f"| {lab} | {stab} | {e(np.mean(s['best']))} | {e(np.mean(r['best']), 3)} | {res} |")
    L += ["", "| (b) Scale factor $s$ | Sphere, FC | Sphere, FC-old | Rastrigin, FC | Rastrigin, FC-old |",
          "|:--|:-:|:-:|:-:|:-:|"]
    for s in ["0.01", "1", "100", "1000", "10000"]:
        L.append(f"| {s} | " + " | ".join(e(np.mean(sc[f"{fn}|{s}|{a}"]), 2 if fn == "sphere" else 3)
                                         for fn in ["sphere", "rastrigin"] for a in ["new", "old"]) + " |")
    w("S10.md", "\n".join(L))


# ------------------------------------------------------------------ Table 2 and supplementary benchmark tables
def table2():
    s, info = A.summary_table(A.MAIN)
    s = s.replace("Table: Table 8. Wilcoxon", "Table: **Table 1.** Results on the 23 classical and 12 shifted "
                  "functions (15,000 evaluations, 30 runs). Wilcoxon")
    w("T2.md", s)
    w("S2.md", A.results_table(A.CLASSIC[:13], A.MAIN, "**Supplementary Table S2.** Results on F1–F13 ($D=30$): mean "
      "and standard deviation of the best value over 30 runs. The best mean is in bold; +/=/− after a mean: FC is "
      "significantly better / not significantly different / significantly worse (Wilcoxon rank-sum, α = 0.05)"))
    w("S3.md", A.results_table(A.CLASSIC[13:], A.MAIN, "**Supplementary Table S3.** Results on the fixed-dimension "
      "functions F14–F23 (notation as in Table S2)"))
    w("S4.md", A.results_table(A.SHIFTED, A.MAIN, "**Supplementary Table S4.** Results on the shifted functions "
      "sF1–sF13 ($D=30$; notation as in Table S2)"))
    w("S5.md", A.cpu_table(A.MAIN).replace("Table: Table 11.", "Table: **Supplementary Table S5.**"))
    return info


# ------------------------------------------------------------------ Table 3 ablation
ABL = [("FCv-noRank", "Same inertia weight for all agents (0.436)"), ("FCv-noMC", "No attraction to $MC$ ($\\eta_1=0$)"),
       ("FCv-noBest", "No attraction to $x_{best}$ ($\\eta_2=0$)"), ("FCv-K2", "$K=2$ (elite group of 50)"),
       ("FCv-K3", "$K=3$ (elite group of 33)"), ("FCv-K10", "$K=10$ (elite group of 10)")]


def table3():
    var = ["FC"] + [v for v, _ in ABL]
    if not A.have(var, A.ALLF):
        print("ablation incomplete")
        return None
    ranks, _, _ = A.friedman(A.ALLF, var)
    c22 = {}
    for v, v22 in [("FCv-noMC", "FC-noMC"), ("FCv-noBest", "FC-noBest")]:
        c22[v] = {D: wtl22([(A22.load("FC", D, f), A22.load(v22, D, f)) for f in range(1, 13)]) for D in (10, 20)}
    L = ["Table: **Table 2.** Ablation and sensitivity analysis. W/T/L: number of functions on which FC (all components, "
         "$K=5$) is significantly better / not different / significantly worse than the variant (Wilcoxon rank-sum, "
         "α = 0.05); mean Friedman rank of the seven versions over the 35 benchmark functions. The last column "
         "compares FC with two of the variants on the 12 CEC2022 functions.", "",
         "| Variant | W/T/L classical (23) | W/T/L shifted (12) | Mean rank (35) | W/T/L CEC2022 $D=10$ / $D=20$ |",
         "|:--|:-:|:-:|:-:|:-:|", f"| **FC** | – | – | {ranks['FC']:.2f} | – |"]
    out = {}
    for v, lab in ABL:
        c = A.wtl(A.CLASSIC, [v])[v]
        sh = A.wtl(A.SHIFTED, [v])[v]
        out[v] = (c, sh, ranks[v])
        cc = f"{'/'.join(map(str, c22[v][10]))} ; {'/'.join(map(str, c22[v][20]))}" if v in c22 else "–"
        L.append(f"| {lab} | {'/'.join(map(str, c))} | {'/'.join(map(str, sh))} | {ranks[v]:.2f} | {cc} |")
    w("T3.md", "\n".join(L))
    out["ranks"] = ranks
    out["c22"] = c22
    return out


def sign22(x, y):
    if np.allclose(x, y):
        return "="
    p = stats.ranksums(x, y).pvalue
    if p >= 0.05:
        return "="
    if np.median(x) < np.median(y) or (np.median(x) == np.median(y) and x.mean() < y.mean()):
        return "+"
    return "−"


def wtl22(pairs):
    s = [sign22(x, y) for x, y in pairs]
    return s.count("+"), s.count("="), s.count("−")


# ------------------------------------------------------------------ Table 4 CEC2022
C22 = ["FC", "CMA-ES", "L-SHADE", "CLPSO", "SPSO2011", "PSO", "GWO", "FC-noMC"]
WIN = [("EA4eig", "EA4eigN100_10"), ("NL-SHADE-LBC", "NL-SHADE-LBC"), ("NL-SHADE-RSP-MID", "NL-SHADE-RSP-MID")]
WIN_MARK = {"EA4eig": " (1st)", "NL-SHADE-LBC": " (2nd)", "NL-SHADE-RSP-MID": " (3rd)"}
CAT = [("basic", range(1, 6)), ("hybrid", range(6, 9)), ("composition", range(9, 13))]


def table4():
    if not all(A22.load(a, D, f) is not None for a in C22 for D in (10, 20) for f in range(1, 13)):
        print("CEC2022 incomplete")
        return None
    res = {}
    L = ["Table: **Table 3.** Results on the 12 CEC2022 functions (30 runs; 200,000 evaluations for $D=10$ and 1,000,000 "
         "for $D=20$). W/T/L: number of functions on which FC is significantly better / not different / significantly "
         "worse (Wilcoxon rank-sum, α = 0.05), over all functions and over the hybrid and composition functions "
         "F6–F12; Friedman mean rank of the eleven algorithms (lower is better). The three best entries of the CEC2022 "
         "competition (final rank in parentheses) are compared through their published per-run results.", "",
         "| Algorithm | $D=10$: W/T/L (F1–F12) | W/T/L (F6–F12) | Rank | $D=20$: W/T/L (F1–F12) | W/T/L (F6–F12) | Rank |",
         "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    E = {(a, D): [A22.load(a, D, f) for f in range(1, 13)] for a in C22 for D in (10, 20)}
    import cec22rank as CR
    for name, key in WIN:
        for D in (10, 20):
            Ew, _ = CR.load_entry(key, D)
            E[(name, D)] = [np.where(Ew[:, f] < 1e-8, 0.0, Ew[:, f]) for f in range(12)]
    ALL = C22 + [n for n, _ in WIN]
    rk, holm = {}, {}
    for D in (10, 20):
        M = np.array([[E[(a, D)][f].mean() for a in ALL] for f in range(12)])
        Rk = np.array([stats.rankdata(r) for r in M]).mean(0)
        rk[D] = dict(zip(ALL, Rk))
        res[f"chi{D}"] = stats.friedmanchisquare(*M.T)
    order = sorted(ALL, key=lambda a: rk[10][a] + rk[20][a])
    for a in order:
        cells = []
        for D in (10, 20):
            if a == "FC":
                cells += ["–", "–", f"**{rk[D][a]:.2f}**"]
                continue
            tot = wtl22([(E[("FC", D)][f], E[(a, D)][f]) for f in range(12)])
            cat = [wtl22([(E[("FC", D)][f - 1], E[(a, D)][f - 1]) for f in fs]) for _, fs in CAT]
            res[(a, D)] = tot, cat
            hc = wtl22([(E[("FC", D)][f], E[(a, D)][f]) for f in range(5, 12)])
            cells += ["/".join(map(str, tot)), "/".join(map(str, hc)), f"{rk[D][a]:.2f}"]
        lab = '**FC**' if a == 'FC' else ('FC without $MC$' if a == 'FC-noMC' else a)
        if a in dict(WIN):
            lab += WIN_MARK[a]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    w("T4.md", "\n".join(L))
    res["rk"] = rk
    res["E"] = E
    # supplementary per-function tables
    for D, tag in [(10, "S7"), (20, "S8")]:
        L = [f"Table: **Supplementary Table {tag}.** Mean error (standard deviation) on CEC2022, $D={D}$, 30 runs; "
             "errors below $10^{-8}$ are reported as 0. The best mean is in bold; +/=/−: Wilcoxon outcome of FC "
             "against the algorithm", "",
             "| F | " + " | ".join("FC without MC" if a == "FC-noMC" else a for a in ALL) + " |",
             "|:--|" + "|".join([":-:"] * len(ALL)) + "|"]
        for f in range(12):
            ms = [E[(a, D)][f].mean() for a in ALL]
            mn = min(ms)
            cells = []
            for a, m in zip(ALL, ms):
                s = f"{A.fmt(m)} ({A.fmt(E[(a, D)][f].std(ddof=1))})"
                if np.isclose(m, mn, rtol=1e-9, atol=1e-300):
                    s = f"**{A.fmt(m)}** ({A.fmt(E[(a, D)][f].std(ddof=1))})"
                if a != "FC":
                    s += " " + sign22(E[("FC", D)][f], E[(a, D)][f])
                cells.append(s)
            L.append(f"| F{f + 1} | " + " | ".join(cells) + " |")
        w(f"{tag}.md", "\n".join(L))
    res["rk"] = rk
    fig2(E)
    return res


def fig2(E):
    import glob
    sel = [1, 4, 6, 7, 10, 12]
    cols = {"FC": "#2a78d6", "CMA-ES": "#eb6834", "L-SHADE": "#1baf7a", "CLPSO": "#6250d6", "SPSO2011": "#e87ba4",
            "PSO": "#eda100", "GWO": "#8a8984", "FC-noMC": "#2a78d6"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    D = 20
    for ax, f in zip(axes.flat, sel):
        for a in C22:
            d = json.load(open(os.path.join(A22.RD, f"{a}__D{D}_F{f}.json")))
            c = np.maximum(np.array([r["curve"] for r in d["runs"]]) - [300, 400, 600, 800, 900, 1800, 2000, 2200,
                                                                       2300, 2400, 2600, 2700][f - 1], 1e-8)
            fe = np.arange(1, c.shape[1] + 1) * (1_000_000 // 100)
            ax.plot(fe, np.median(c, 0), color=cols[a], lw=2.0 if a == "FC" else 1.1,
                    ls=":" if a == "FC-noMC" else "-", label="FalCO without MC" if a == "FC-noMC" else ("FalCO" if a == "FC" else a),
                    zorder=3 if a == "FC" else 2)
        ax.set_yscale("log")
        ax.set_title(f"F{f}", fontsize=9)
        ax.set_xticks([0, 500000, 1000000])
        ax.set_xticklabels(["0", "0.5M", "1M"])
    for ax in axes[-1]:
        ax.set_xlabel("Function evaluations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median error")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=7, bbox_to_anchor=(0.5, -0.005), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_cec2022.png"), dpi=300)


# ------------------------------------------------------------------ Figure 1
def fig1():
    sel = ["F5", "F9", "F10", "F21", "sF1", "sF5", "sF9", "sF10", "sF12"]
    series = [("FC", "#2a78d6"), ("CMA-ES", "#eb6834"), ("L-SHADE", "#1baf7a"), ("PSO", "#eda100"),
              ("GWO", "#8a8984"), ("HHO", "#008300"), ("WOA", "#6250d6"), ("DE", "#e34948")]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.6))
    fe = np.arange(1, 101) * 150
    for ax, f in zip(axes.flat, sel):
        for a, col in series:
            c = np.maximum(np.array([r["curve"] for r in A.D[(a, f)]]) - A.FMIN[f], 1e-16)
            ax.plot(fe, np.median(c, 0), color=col, lw=2.0 if a == "FC" else 1.1, label="FalCO" if a == "FC" else a,
                    zorder=3 if a == "FC" else 2)
        ax.set_yscale("log")
        ax.set_title(f, fontsize=9)
        ax.set_xlim(0, 15000)
        ax.set_xticks([0, 5000, 10000, 15000])
        ax.set_xticklabels(["0", "5k", "10k", "15k"])
    for ax in axes[-1]:
        ax.set_xlabel("Function evaluations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median error")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=7.5, bbox_to_anchor=(0.5, -0.005), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "figS1_convergence.png"), dpi=300)


# ------------------------------------------------------------------ Table 5 CEC2020, S9
def table5():
    d = {a: C.load_xlsx(a) for a in R.ALL8}
    d["FC"] = C.load_fc(os.path.join(HERE, "results_rw4"))
    s = R.summary(d)
    sc9 = {a: v[3] for a, v in s["score9"].items()}
    rank9 = {a: i + 1 for i, a in enumerate(sorted(sc9, key=sc9.get))}
    L = ["Table: **Table 4.** Results on the 57 problems of the CEC2020 real-world constrained suite. Score: competition "
         "score (lower is better) computed with the organizers' procedure against the three best entries and against all "
         "eight entries; FR: feasibility rate (fraction of runs ending with a feasible solution); W/T/L: Wilcoxon "
         "outcomes of FC against each algorithm.", "",
         "| Algorithm | Score (FC + top 3) | Score (FC + 8 entries) | Rank among 9 | Mean FR | Problems with FR = 1 | FR = 0 | W/T/L of FC |",
         "|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for a in ["FC"] + R.ALL8:
        s4 = f"{s['score4'][a][3]:.3f}" if a in s["score4"] else "–"
        fr = s["fr"][a]
        wt = "–" if a == "FC" else "/".join(map(str, s["wtl"][a]))
        L.append(f"| {'**FC**' if a == 'FC' else a} | {s4} | {sc9[a]:.3f} | {rank9[a]} | {fr[0]:.2f} | {fr[1]} | {fr[2]} | {wt} |")
    w("T5.md", "\n".join(L))
    t = R.detail_table(d, "**Supplementary Table S9.** CEC2020 results per problem: best value of FC, and mean of the "
                       "feasible runs with the feasibility rate in parentheses for FC and the three best entries; "
                       "+/=/−: Wilcoxon outcome of FC; inf.: no feasible run")
    w("S9.md", t)


# ------------------------------------------------------------------ Table 6 BLDC, S6
def table6():
    fb = 2.5600511969
    L = ["Table: **Table 5.** Results on the BLDC motor design problem (30 runs, 15,000 evaluations). +/=/−: Wilcoxon "
         "outcome of FC; Time: mean CPU time per run; $FE_{10^{-4}}$: median number of evaluations needed to reach a "
         "value within a relative $10^{-4}$ of the best known design (– if fewer than half of the runs reached it).", "",
         "| Algorithm | Best | Mean | Std | Wilcoxon | Time (s) | $FE_{10^{-4}}$ |",
         "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for a in A.MAIN:
        runs = A.D[(a, "BLDC")]
        b = np.array([r["best"] for r in runs])
        t = np.mean([r["time"] for r in runs])
        fes = []
        for r in runs:
            c = np.array(r["curve"])
            i = np.nonzero(c <= fb * (1 + 1e-4))[0]
            fes.append((i[0] + 1) * 150 if len(i) else np.inf)
        med = np.median(fes)
        L.append(f"| {'**FC**' if a == 'FC' else a} | {b.min():.7f} | {b.mean():.7f} | "
                 f"{A.fmt(b.std(ddof=1))} | {'–' if a == 'FC' else A.sign(a, 'BLDC')} | {t:.2f} | "
                 f"{'–' if not np.isfinite(med) else f'{int(med):,}'} |")
    w("T6.md", "\n".join(L))
    sel = ["FC", "CMA-ES", "L-SHADE", "DE", "PSO", "GWO"]
    X = {a: min(A.D[(a, "BLDC")], key=lambda r: r["best"])["x"] for a in sel}
    names = ["$\\beta$", "$l_m$", "$l_y$", "$l_w$", "$l_g$", "$r_r$", "$l_s$", "$A_c$", "$J_{cu}$"]
    L = ["Table: **Supplementary Table S6.** Best BLDC design found by selected algorithms ($P=5$) and the components "
         "of the objective", "", "| Variable | " + " | ".join(sel) + " |", "|:--|" + "|".join([":-:"] * len(sel)) + "|"]
    for j, n in enumerate(names):
        L.append(f"| {n} | " + " | ".join(f"{X[a][j]:.6g}" for a in sel) + " |")
    for key, lab in [("cost", "Cost"), ("loss", "Losses (W)"), ("volume", "Volume (m³)"), ("F", "Objective F")]:
        L.append(f"| {lab} | " + " | ".join(f"{bldc.parts(X[a])[key]:.7g}" for a in sel) + " |")
    w("S6.md", "\n".join(L))


if __name__ == "__main__":
    table1()
    info = table2()
    fig1()
    table5()
    table6()
    a = table3()
    c = table4()
    print("done", a is not None, c is not None)
