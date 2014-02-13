#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from .data_iterators import Minibatches


def conjgrad(gradient, v, f_hessp, maxiter=300):
    r = gradient - f_hessp(v)  # residual
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

    return allvecs

def conjgrad2(gradient, v, f_hessp, maxiter=300):
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

    return allvecs

def conjgrad3(gradient, v, f_hessp, maxiter=300):

    ## set parameters for our conjgrad version
    miniter = 1
    inext = 5
    imult = 1.3

    tolerance = 5e-6
    gapRatio = 0.1
    minGap = 10
    maxTestgap = np.maximum(np.ceil(maxiter * gapRatio), minGap) + 1

    vals = np.zeros(maxTestgap)

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
        vals[np.mod(i, maxTestgap)] = val

        testGap = np.maximum(np.ceil(gapRatio * i), minGap)
        prevVal = vals[np.mod((i - testGap), maxTestgap)]

        if i == np.ceil(inext):
            allvecs.append(v)
            inext *= imult
            saved = True

        if (i > testGap and (val - prevVal)/val < (tolerance * testGap) and i >= miniter):
            #print("BREAK AT ITER: %d" %i  )
            #print("pAp: %f" %curv)
            break

    return allvecs
