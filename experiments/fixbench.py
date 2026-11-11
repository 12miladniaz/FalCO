"""Corrections to the EvoloPy benchmark definitions used here.

F6: classical step function sum(floor(x_i + 0.5)^2) (EvoloPy has sum((x_i + 0.5)^2)).
F21-F23: Shekel functions; the EvoloPy version multiplies element-wise and returns a function of x_1 only.
"""
import numpy as np

_A = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9],
               [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]], float)
_C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def F6(x):
    return float(np.sum(np.floor(np.asarray(x) + 0.5) ** 2))


def _shekel(x, m):
    d = np.asarray(x, float)[None, :] - _A[:m]
    return float(-np.sum(1.0 / (np.sum(d * d, axis=1) + _C[:m])))


def F21(x):
    return _shekel(x, 5)


def F22(x):
    return _shekel(x, 7)


def F23(x):
    return _shekel(x, 10)
