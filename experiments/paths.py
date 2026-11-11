"""Locations of third-party code and data (not included in this repository; see README.md).

Set FALCO_THIRD_PARTY to use another folder than ../third_party."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TP = os.environ.get("FALCO_THIRD_PARTY", os.path.join(ROOT, "third_party"))

PYCMA = os.path.join(TP, "pycma")                          # git clone https://github.com/CMA-ES/pycma
EVOLOPY = os.path.join(TP, "EvoloPy")                       # git clone https://github.com/7ossam81/EvoloPy
CEC22C = os.path.join(TP, "cec22c")                         # libcec22.so and input_data/ of the official C code
CEC22_COMP = os.path.join(TP, "2022-SO-BO", "Results_and_Ranking", "Comparison")   # competition results
CEC20_LIB = os.path.join(TP, "CEC-2020-RWC-C", "cec2020rwc.so")                     # C++ port of the CEC2020 suite
CEC20_XL = os.path.join(TP, "cec2020-ranking")             # Score.m and the xlsx result files of the entries
OUTPUT = os.path.join(HERE, "output")                      # tables and figures written by gen_sr.py / figs_sr.py
FIGDATA = os.path.join(HERE, "figdata")
