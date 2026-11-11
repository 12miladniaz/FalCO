"""CEC2022 benchmark through the official C implementation (batch evaluation) and a budget wrapper."""
import ctypes
import os

import numpy as np

import paths  # noqa: E402
CDIR = paths.CEC22C           # holds libcec22.so and input_data/ (read with relative paths)
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        os.chdir(CDIR)
        _LIB = ctypes.CDLL(os.path.join(CDIR, "libcec22.so"))
        _LIB.cec22.restype = None
    return _LIB


def maxfes(D):
    return 200000 if D == 10 else 1000000


BIAS = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]


class BudgetExhausted(Exception):
    pass


class Problem:
    """Batch objective with an evaluation budget; records the best value and checkpoints."""

    def __init__(self, fnum, D, nchk=100):
        self.fnum, self.D = fnum, D
        self.maxfe = maxfes(D)
        self.lb, self.ub = np.full(D, -100.0), np.full(D, 100.0)
        self.fe, self.best = 0, np.inf
        self.hit_fe = None                       # first evaluation with error <= 1e-8 (CEC2022 tie-break)
        self.step = self.maxfe // nchk
        self.curve = []
        self.lib = _lib()

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        rem = self.maxfe - self.fe
        if rem <= 0:
            raise BudgetExhausted
        Xe = np.ascontiguousarray(X[:rem])
        n = len(Xe)
        F = np.zeros(n)
        self.lib.cec22(Xe.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       F.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       self.D, n, self.fnum)
        # record best-so-far at every checkpoint crossed
        cm = np.minimum.accumulate(np.minimum(F, self.best))
        idx = np.arange(self.fe + 1, self.fe + n + 1)
        for k in np.nonzero(idx % self.step == 0)[0]:
            self.curve.append(float(cm[k]))
        if self.hit_fe is None:
            ok = np.nonzero(F - BIAS[self.fnum - 1] <= 1e-8)[0]
            if len(ok):
                self.hit_fe = self.fe + int(ok[0]) + 1
        self.best = float(cm[-1])
        self.fe += n
        if n < len(X):
            raise BudgetExhausted
        return F

    def one(self, x):
        return float(self(np.asarray(x)[None, :])[0])
