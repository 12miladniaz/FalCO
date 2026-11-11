# Falling Cat Optimizer (FalCO)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24141760.svg)](https://doi.org/10.5281/zenodo.24141760)

This repository contains the reference implementation of the Falling Cat Optimizer (FalCO), the scripts of all
experiments and the per-run results reported in the article

> Niaz Azari M, Ghanbari A, Pirgazi J, Kermani A. Falling Cat Optimizer (FalCO): a particle swarm with elite-centre
> attraction and rank-dependent inertia. (submitted)

FalCO is a particle swarm method for continuous black-box optimization. The population is sorted by objective value
in every iteration; each agent is attracted to the fitness-weighted centre of the best fifth of the population and,
more weakly, to the best agent, and its inertia weight decreases linearly with its rank. The velocity limit is a
fixed fraction of the search range, so the method is invariant to a rescaling of the variables.

## Quick start

`FalCO.py` is self-contained and needs only NumPy.

```python
import numpy as np
from FalCO import FalCO

def sphere(x):
    return float(np.sum((x - 30.0) ** 2))

best_x, best_f, curve = FalCO(sphere, lb=-100, ub=100, dim=30, max_fe=15000, seed=1)
print(best_f)
```

`lb` and `ub` may be scalars or arrays of length `dim`. The function returns the best position, its objective
value and the best value after every iteration. Default parameters (used in all experiments of the paper):

| Parameter | Meaning | Default |
|:--|:--|:-:|
| `N` | population size | 100 |
| `K` | number of rank groups (the elite group has `N // K` agents) | 5 |
| `eta1` | attraction to the fitness-weighted elite centre | 5 |
| `eta2` | attraction to the best agent | 1 |
| `wmax`, `wmin` | inertia weight of the best and of the worst agent | 0.832, 0.04 |
| `vfrac` | velocity limit as a fraction of `ub - lb` | 0.2 |
| `max_fe` | budget of objective evaluations | 15000 |

Running `python FalCO.py` minimizes a shifted 30-dimensional sphere function as a test.

## Repository layout

```
FalCO.py                  reference implementation (NumPy only)
experiments/              experiment scripts and all per-run results
  paths.py                locations of third-party code (see below)
  run.py                  classical/shifted functions and the BLDC motor problem
  run22.py, c22.py        CEC2022 benchmark (official C code through ctypes)
  run_rw.py, rw.py        CEC2020 real-world constrained suite
  algos.py, algs22.py     all compared algorithms and the FalCO ablation variants
  fc3.py                  FalCO as used by the scripts (fc3.FC3 with vfrac=0.2; same update as FalCO.py,
                          but it draws random numbers from NumPy's global generator)
  fcv.py                  earlier version of the method (repulsion term, absolute velocity limit)
  fixbench.py             corrected F6 and Shekel F21-F23 (see the paper)
  bldc.py                 BLDC motor design objective
  stab.py, scale.py       stability and scale-invariance experiments
  analyze*.py, cecscore.py, cec22rank.py   statistics, CEC2020 scoring, CEC2022 competition ranking
  gen_sr.py, figs_sr.py   generate the tables and figures of the paper
  legacy/FC9.py           the first published version of the algorithm (for the comparisons with the earlier version)
  figdata/                the two drawings used in Fig 1
  results/                classical, shifted and BLDC runs
  results_c22/            CEC2022 runs
  results_c22_hit/        CEC2022 runs of FalCO that also record when the target error was reached
  results_rw4/            CEC2020 runs of FalCO
  analysis/               stab.json, scale.json
```

## Result files

Every JSON file holds one algorithm on one problem, with one record per run (best value, the best solution where
stored, the number of evaluations, the CPU time and a convergence record of 100 points).

| Folder | Experiment | File name | Algorithm names |
|:--|:--|:--|:--|
| `results/` | 23 classical functions (`F1`–`F23`), 12 shifted functions (`sF1`–`sF13`, no `sF8`), BLDC motor (`BLDC`); 30 runs, 15,000 evaluations | `ALGO__FUNC.json`, field `best` | **`FC3-v` is FalCO**; `FCv-noRank`, `FCv-noMC`, `FCv-noBest`, `FCv-K2`, `FCv-K3`, `FCv-K10` are the ablation variants of Table 8 |
| `results_c22/` | CEC2022, D = 10 and 20, 30 runs | `ALGO__D{D}_F{f}.json`, field `err` = f(x) − f* | **`FC` is FalCO**; `FC-noMC`, `FC-noBest` are ablation variants |
| `results_c22_hit/` | CEC2022, FalCO rerun with identical seeds (identical errors) | as above, extra field `hit_fe` | `FC` |
| `results_rw4/` | CEC2020 real-world constrained suite, 25 runs | `FC__RCxx.json`, fields `f` (objective) and `v` (mean constraint violation) | `FC` is FalCO |
| `analysis/` | stability and scale-invariance experiments | `stab.json`, `scale.json` | |

`FC` is the internal name of the method in the scripts. The published per-run results of the competition entries
(CEC2022 winners, CEC2020 entries) are not redistributed here; they are read from the organizers' repositories.

## Mapping between the paper and the scripts

| Paper | Produced by | Output file in `experiments/output/` |
|:--|:--|:--|
| Fig 1, Fig 2 | `figs_sr.py` | `fig_concept.png`, `fig_mechanism.png` |
| Fig 3, Fig 4 | `gen_sr.py` | `figS1_convergence.png`, `fig_cec2022.png` |
| Table 2 | `stab.py`, `scale.py`, then `gen_sr.py` | `S10.md` |
| Tables 3–7 | `gen_sr.py` | `T2.md`, `S2.md`, `S3.md`, `S4.md`, `S5.md` |
| Table 8 | `gen_sr.py` | `T3.md` |
| Tables 9–11 | `gen_sr.py` | `T4.md`, `S7.md`, `S8.md` |
| Tables 12–13 | `gen_sr.py` | `T5.md`, `S9.md` |
| Tables 14–15 | `gen_sr.py` | `T6.md`, `S6.md` |
| CEC2022 competition ranking | `cec22rank.py` | printed |

The output file names follow an earlier numbering of the tables; the table titles inside the files may also use
the internal name FC.

## Requirements

Python 3.10 or later, and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The experiment scripts also need third-party code that is not included in this repository. Put it in a folder
`third_party/` next to `experiments/` (or set the environment variable `FALCO_THIRD_PARTY` to another folder):

```
mkdir third_party && cd third_party

# compared algorithms and classical benchmark functions
git clone https://github.com/7ossam81/EvoloPy
git -C EvoloPy checkout ce97ecf57b2b7399d3fe5519eb792bd7353fa1bb
git clone https://github.com/CMA-ES/pycma
git -C pycma checkout f1142b2ba90f01926b82a946f87b7ca1a4a237fe

# CEC2022: official C code and the competition results
git clone https://github.com/P-N-Suganthan/2022-SO-BO
unzip 2022-SO-BO/CEC2022.zip -d cec22src
unzip 2022-SO-BO/2022-Full-Download.zip "2022-SO-BO-main/Results_and_Ranking/*" -d .
mv 2022-SO-BO-main/Results_and_Ranking 2022-SO-BO/
mkdir cec22c
cp -r cec22src/CEC2022/C-Code/input_data cec22src/CEC2022/C-Code/cec22_test_func.cpp cec22c/
sed -i 's/#include <WINDOWS.H>/#include <stdlib.h>/' cec22c/cec22_test_func.cpp
cp ../experiments/cec22_wrap.cpp cec22c/
g++ -O2 -shared -fPIC -o cec22c/libcec22.so cec22c/cec22_wrap.cpp

# CEC2020: C++ port of the suite (needs Eigen 3.4) and the organizers' scoring files
git clone https://github.com/VladimirStanovov/CEC-2020-RWC-C
git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen
g++ -std=c++14 -O3 -fPIC -shared -I eigen -o CEC-2020-RWC-C/cec2020rwc.so CEC-2020-RWC-C/cec2020rwc.cpp
git clone https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation
unzip "2020-RW-Constrained-Optimisation/Code used For Ranking.zip" -d rank20
mv "rank20/Code used For Ranking" cec2020-ranking
```

The only change to the official CEC2022 code is the replacement of a Windows header, which is not used by the
functions. The commits given above are the versions used for the paper. The folder `cec2020-ranking` must contain
`SASS.xlsx`, `COLSHADE.xlsx` and the other result files of the entries.

## Reproducing the experiments

Run all commands from `experiments/`. Every run uses a fixed seed derived from the algorithm, problem and run
index, so the results are reproducible on the same platform. A script does nothing if its output file already
exists, so the result files of the paper are never overwritten. To repeat a run, direct the output to a new folder
with the environment variables `RUN_OUT`, `C22_OUT` or `RW_OUT`:

```
# classical and shifted functions, BLDC motor: ALGO FUNC RUNS
RUN_OUT=new_results python run.py FC3-v F1 30
RUN_OUT=new_results python run.py CMA-ES sF9 30
RUN_OUT=new_results python run.py FC3-v BLDC 30

# CEC2022: ALGO D FUNCTION RUNS
C22_OUT=new_results_c22 python run22.py FC 20 10 30

# CEC2020: PROBLEM RUNS
RW_OUT=new_results_rw python run_rw.py 15 25

# stability and scale invariance
python stab.py
python scale.py

# CEC2022 competition ranking with FalCO added
python cec22rank.py

# all tables and Figs 3-4, then Figs 1-2
python gen_sr.py
python figs_sr.py
```

The analysis scripts (`gen_sr.py`, `cec22rank.py`) read the result folders of the paper. With the seeds above,
`run.py FC3-v F1 2` reproduces the first two runs of `results/FC3-v__F1.json` exactly on the platform we used
(Linux, NumPy 2.5); other platforms may differ in the last digits.

## Notes on the benchmarks

* The EvoloPy implementation of the step function F6 computes sum((x_i + 0.5)^2) instead of
  sum(floor(x_i + 0.5)^2), and its Shekel functions F21–F23 use an element-wise product, so that they depend on
  x_1 only. `fixbench.py` contains the corrected versions used in the paper.
* The CEC2022 functions of the `opfunu` library (version 1.0.4) differ from the official C code on 11 of the 12
  functions. All CEC2022 results here use the official C code.
* In the BLDC objective, the windage loss uses the laminar-flow formula for 500 <= Re < 10^4. The original MATLAB
  code used 500 < Re < 10^4, which created an artificial optimum at Re = 500.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). Third-party code keeps its own
licence: EvoloPy (Apache 2.0), pycma (BSD 3-clause), the CEC2020 C++ port (GPL 3.0) and the CEC benchmark code of
the competition organizers.

## Citation

If you use FalCO, please cite the article above. The code and results are archived at Zenodo, doi: [10.5281/zenodo.24141760](https://doi.org/10.5281/zenodo.24141760) (see also `CITATION.cff`).
