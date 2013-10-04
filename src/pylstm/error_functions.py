#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from .training.data_iterators import Online
from .wrapper import ctcpp


def MeanSquaredError(Y, T, M=None):
    assert Y.shape == T.shape, "Shape mismatch Y%s != T%s" % (Y.shape, T.shape)
    diff = Y - T
    norm = Y.shape[1]  # normalize by number of sequences
    if M is not None:
        diff *= M
    error = 0.5 * np.sum(diff ** 2) / norm
    deltas = diff / norm
    return error, deltas


def CrossEntropyError(Y, T, M=None):
    assert Y.shape == T.shape, "Shape mismatch Y%s != T%s" % (Y.shape, T.shape)
    Y = Y.copy()  # do not modify original Y
    Y[Y < 1e-6] = 1e-6
    Y[Y > 1 - 1e-6] = 1 - 1e-6
    cee = T * np.log(Y) + (1 - T) * np.log(1 - Y)
    ceed = (T - Y) / (Y * (Y - 1))
    norm = Y.shape[1]  # normalize by number of sequences
    if M is not None:
        cee *= M
        ceed *= M
    error = - np.sum(cee) / norm
    deltas = ceed / norm
    return error, deltas


def MultiClassCrossEntropyError(Y, T, M=None):
    assert Y.shape == T.shape, "Shape mismatch Y%s != T%s" % (Y.shape, T.shape)
    Y = Y.copy()  # do not modify original Y
    Y[Y < 1e-6] = 1e-6
    cee = T * np.log(Y)
    quot = T / Y
    norm = Y.shape[1]  # normalize by number of sequences
    if M is not None:
        cee *= M
        quot *= M
    error = - np.sum(cee) / norm
    deltas = - quot / norm
    return error, deltas


def CTC(Y, T, M=None):
    N, batch_size, label_count = Y.shape
    deltas = np.zeros((N, batch_size, label_count))
    deltas[:] = float('-inf')
    errors = []
    for b, (y, t, m) in enumerate(Online(Y, T, M, verbose=False)()):
        err, delt = ctcpp(y, list(t[0]))
        errors.append(err)
        deltas[:, b:b+1, :] = delt.as_array()

    return np.mean(errors), -deltas / batch_size



def ctc_best_path_decoding(Y):
    assert Y.shape[1] == 1
    Y_win = Y.argmax(2).reshape(Y.shape[0])
    t = []
    blank = True
    for y in Y_win:
        if blank is True and y != 0:
            t.append(y)
            blank = False
        elif blank is False:
            if y == 0:
                blank = True
            elif y != t[-1]:
                t.append(y)
    return t
