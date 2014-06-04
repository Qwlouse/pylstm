#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


def get_min_err(errors):
    min_epoch = np.argmin(errors)
    return min_epoch, errors[min_epoch]


def conjugate_gradient(gradient, v, f_hessp, maxiter=300):
    ## set parameters for our conjgrad version
    miniter = 1
    inext = 5
    imult = 1.3
    tolerance = 5e-6
    gap_ratio = 0.1
    min_gap = 10
    max_testgap = np.maximum(np.ceil(maxiter * gap_ratio), min_gap) + 1
    vals = np.zeros(max_testgap)
    r = f_hessp(v) - gradient  # residual
    p = -r  # current step
    rsold = r.T.dot(r)
    allvecs = [v.copy()]

    for i in range(maxiter):
        Ap = f_hessp(p)
        curv = (p.T.dot(Ap))
        if curv < 3 * np.finfo(np.float64).eps:
            break  # curvature is negative or zero
        alpha = rsold / curv
        v += alpha * p
        allvecs.append(v.copy())
        r = r + alpha * Ap   # updated residual
        rsnew = r.T.dot(r)
        if np.sqrt(rsnew) < 1e-10:
            break  # found solution
        p = -r + rsnew / rsold * p
        rsold = rsnew

        val = 0.5*(np.dot((-gradient+r), v))
        vals[np.mod(i, max_testgap)] = val
        test_gap = np.maximum(np.ceil(gap_ratio * i), min_gap)
        prev_val = vals[np.mod((i - test_gap), max_testgap)]

        if i == np.ceil(inext):
            allvecs.append(v)
            inext *= imult

        if (i > test_gap and (val - prev_val) / val < (tolerance * test_gap)
                and i >= miniter):
            break

    return allvecs
