#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.utils import get_sequence_lengths
from .training.data_iterators import Online
from .wrapper import ctcpp


def _NotImplemented(Y, T, M):
    raise NotImplementedError('This combination is not implemented yet!')


def _IllegalCombination(Y, T, M):
    raise RuntimeError('Illegal combination of targets and error function!')


def _FramewiseMSE(Y, T, M):
    diff = Y - T
    if M is not None:
        diff *= M
    norm = Y.shape[1]  # normalize by number of sequences
    error = 0.5 * np.sum(diff ** 2) / norm
    return error, (diff / norm)


def _SequencewiseBinarizingMSE(Y, T, M):
    diff = Y.copy()
    for b in range(Y.shape[1]):
        diff[:, b, T[b]] -= 1
    if M is not None:
        diff *= M
    norm = Y.shape[1]  # normalize by number of sequences
    error = 0.5 * np.sum(diff ** 2) / norm
    return error, (diff / norm)

MSE_implementations = {
    ('F', False): _FramewiseMSE,
    ('F', True): _NotImplemented,
    ('L', False): _IllegalCombination,
    ('L', True): _IllegalCombination,
    ('C', False): _FramewiseMSE,  # should work smoothly through broadcasting
    ('C', True): _SequencewiseBinarizingMSE
}


def MeanSquaredError(Y, T, M=None):
    assert T.validate_for_output_shape(*Y.shape)
    return MSE_implementations[T.targets_type](Y, T, M)


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


def CrossEntropyError_class(Y, T, M=None):
    _, batch_size, nr_outputs = Y.shape
    #assert isinstance(T, list)
    assert len(T) == batch_size
    classes = np.max(T) + 1  # assume 0 based indexing
    assert nr_outputs >= classes
    y_m = Y.copy()  # do not modify original Y
    y_m[y_m < 1e-6] = 1e-6
    y_m[y_m > 1 - 1e-6] = 1 - 1e-6
    cee = np.log(1 - y_m)
    ceed = 1. / (y_m - 1)

    for b in range(batch_size):
        cee[:, b, T[b]] = np.log(y_m[:, b, T[b]])
        ceed[:, b, T[b]] = 1. / y_m[:, b, T[b]]
    norm = y_m.shape[1]  # normalize by number of sequences
    if M is None:
        # only inject error at end of sequence
        cee[:-1, :, :] = 0
        ceed[:-1, :, :] = 0
    else:
        cee *= M
        ceed *= M

    error = -np.sum(cee) / norm
    deltas = -ceed / norm
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
            t.append(y - 1)
            blank = False
        elif blank is False:
            if y == 0:
                blank = True
            elif y - 1 != t[-1]:
                t.append(y - 1)
    return t
