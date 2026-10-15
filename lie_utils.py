"""Shared, numerically stable helpers for the SO(3) / S3 / SE(3) / SE(2) modules.

All the closed-form coefficients used by Rodrigues' formula, the left/right Jacobians
and the SE(3) Q-matrix suffer from catastrophic cancellation for small angles
(e.g. (1 - cos t)/t, (t - sin t)/t^3, ...).  Below ``SERIES_THRESHOLD`` radians they are
evaluated with their Taylor series instead, which is accurate to machine precision.
Every coefficient is an even function of theta, so only |theta| matters.
"""
import math
from fractions import Fraction

import numpy as np

SERIES_THRESHOLD = 0.5
_N = 10  # number of Taylor terms (more than enough for |theta| < 0.5)


def vec_hat(v):
    """3-vector -> 3x3 skew-symmetric matrix [v]_x."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


def vec_vee(W):
    """3x3 skew-symmetric matrix -> 3-vector."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def _poly(coeffs, x):
    r = 0.0
    for c in reversed(coeffs):
        r = r * x + c
    return r


# Taylor coefficients in powers of theta^2
_B_SER = [(-1) ** k / math.factorial(2 * k + 2) for k in range(_N)]                      # (1-cos t)/t^2
_C_SER = [(-1) ** k / math.factorial(2 * k + 3) for k in range(_N)]                      # (t-sin t)/t^3
_C2_SER = [(-1) ** k / math.factorial(2 * k + 4) for k in range(_N)]                     # (t^2+2cos t-2)/(2 t^4)
_C3_SER = [(-1) ** k * (1. / math.factorial(2 * k + 4) - 3. / math.factorial(2 * k + 5)) / 2.
           for k in range(_N)]                                                           # (2t-3sin t+t cos t)/(2 t^5)
_BERN = [Fraction(1, 6), Fraction(-1, 30), Fraction(1, 42), Fraction(-1, 30), Fraction(5, 66),
         Fraction(-691, 2730), Fraction(7, 6), Fraction(-3617, 510), Fraction(43867, 798),
         Fraction(-174611, 330)]                                                         # B_2 ... B_20
_D_SER = [float(-(-1) ** n * _BERN[n - 1] / math.factorial(2 * n)) for n in range(1, _N + 1)]  # (1-(t/2)cot(t/2))/t^2


def coeff_A(theta):
    """sin(t)/t"""
    return float(np.sinc(abs(theta) / np.pi))


def coeff_B(theta):
    """(1 - cos t)/t^2"""
    t = abs(theta)
    if t < SERIES_THRESHOLD:
        return _poly(_B_SER, t * t)
    return 2. * math.sin(t / 2.) ** 2 / (t * t)


def coeff_C(theta):
    """(t - sin t)/t^3"""
    t = abs(theta)
    if t < SERIES_THRESHOLD:
        return _poly(_C_SER, t * t)
    return (t - math.sin(t)) / t ** 3


def coeff_D(theta):
    """1/t^2 - (1 + cos t)/(2 t sin t)   (SO(3) inverse-Jacobian coefficient; stable also at t = pi)"""
    t = abs(theta)
    if t < SERIES_THRESHOLD:
        return _poly(_D_SER, t * t)
    return 1. / (t * t) - 1. / (2. * t * math.tan(t / 2.))


def coeff_c2(theta):
    """(t^2 + 2cos t - 2)/(2 t^4)"""
    t = abs(theta)
    if t < SERIES_THRESHOLD:
        return _poly(_C2_SER, t * t)
    return (t * t + 2. * math.cos(t) - 2.) / (2. * t ** 4)


def coeff_c3(theta):
    """(2t - 3 sin t + t cos t)/(2 t^5)"""
    t = abs(theta)
    if t < SERIES_THRESHOLD:
        return _poly(_C3_SER, t * t)
    return (2. * t - 3. * math.sin(t) + t * math.cos(t)) / (2. * t ** 5)
