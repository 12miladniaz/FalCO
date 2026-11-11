"""CEC2020 real-world constrained suite (C++ port by V. Stanovov) with the official violation measure."""
import ctypes
import os

import numpy as np

import paths  # noqa: E402
LIB = ctypes.CDLL(paths.CEC20_LIB)
D = [9, 11, 7, 6, 9, 38, 48, 2, 3, 3, 7, 7, 5, 10, 7, 14, 3, 4, 4, 2, 5, 9, 5, 7, 4, 22, 10, 10, 4, 3, 4, 5, 30,
     118, 153, 158, 126, 126, 126, 76, 74, 86, 86, 30, 25, 25, 25, 30, 30, 30, 59, 59, 59, 59, 64, 64, 64]
GN = [0, 0, 14, 1, 2, 0, 0, 2, 1, 3, 4, 9, 3, 10, 11, 15, 4, 4, 5, 3, 8, 10, 8, 7, 7, 86, 3, 9, 1, 8, 0, 6, 30, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 24, 24, 24, 29, 29, 29, 14, 14, 14, 14, 0, 0, 0]
HN = [8, 9, 0, 4, 4, 32, 38, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108,
      148, 148, 116, 116, 116, 76, 74, 76, 76, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6]
LIB.cec20_func.restype = None
LIB.get_bounds.restype = None
PENALTY_OFFSET = 1e9


def maxfes(d):
    return 100000 if d <= 10 else 200000 if d <= 30 else 400000 if d <= 50 else 800000 if d <= 150 else 1000000


class Problem:
    def __init__(self, k):
        self.k, self.d, self.ng, self.nh = k, D[k - 1], GN[k - 1], HN[k - 1]
        self._x = (ctypes.c_double * self.d)()
        self._f = (ctypes.c_double * 1)()
        self._g = (ctypes.c_double * max(self.ng, 1))()
        self._h = (ctypes.c_double * max(self.nh, 1))()
        lb = (ctypes.c_double * self.d)()
        ub = (ctypes.c_double * self.d)()
        LIB.get_bounds(ctypes.c_int(k), lb, ub)
        self.lb, self.ub = list(lb), list(ub)
        self.maxfes = maxfes(self.d)
        self._xn = np.frombuffer(self._x, dtype=np.float64)
        self._gn = np.frombuffer(self._g, dtype=np.float64)[:self.ng]
        self._hn = np.frombuffer(self._h, dtype=np.float64)[:self.nh]
        self._k = ctypes.c_int(k)
        self.m = self.ng + self.nh

    def evaluate(self, x):
        self._xn[:] = x
        LIB.cec20_func(self._x, self._k, self._f, self._g, self._h)
        if not self.m:
            return self._f[0], 0.0
        g, h = self._gn, self._hn
        s = 0.0
        if self.ng:
            s += g[g > 0].sum()
        if self.nh:
            ah = np.abs(h)
            s += ah[ah > 1e-4].sum()
        return self._f[0], float(s / self.m)


def scalar(f, v):
    """Feasibility-first scalarisation (Deb's rules): any feasible point beats any infeasible one."""
    return f if v <= 0 else PENALTY_OFFSET + v


if __name__ == "__main__":
    p = Problem(15)
    print(p.d, p.lb, p.ub)
    x = [3.5, 0.7, 17, 7.3, 7.7153199, 3.3502147, 5.2866545]
    print(p.evaluate(x))
