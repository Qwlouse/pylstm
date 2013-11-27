#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from .training.data_iterators import Online
from .wrapper import ctcpp


def _not_implemented(*_):
    raise NotImplementedError('This combination is not implemented yet!')


def _illegal_combination(*_):
    raise RuntimeError('Illegal combination of targets and error function!')


################################################################################

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
    ('F', True): _not_implemented,
    ('L', False): _illegal_combination,
    ('L', True): _illegal_combination,
    ('C', False): _FramewiseMSE,  # should work smoothly through broadcasting
    ('C', True): _SequencewiseBinarizingMSE
}


def MeanSquaredError(Y, T, M=None):
    assert T.validate_for_output_shape(*Y.shape)
    return MSE_implementations[T.targets_type](Y, T.data, M)


################################################################################

def _FramewiseCEE(y_m, T, M):
    cee = T * np.log(y_m) + (1 - T) * np.log(1 - y_m)
    ceed = (T - y_m) / (y_m * (y_m - 1))
    if M is not None:
        cee *= M
        ceed *= M
    norm = y_m.shape[1]  # normalize by number of sequences
    return (-np.sum(cee) / norm), (ceed / norm)


def _SequencewiseBinarizingCEE(y_m, T, M):
    cee = np.log(1 - y_m)
    ceed = 1. / (y_m - 1)
    for b in range(y_m.shape[1]):
        cee[:, b, T[b]] = np.log(y_m[:, b, T[b]])
        ceed[:, b, T[b]] = 1. / y_m[:, b, T[b]]
    norm = y_m.shape[1]  # normalize by number of sequences
    if M is not None:
        cee *= M
        ceed *= M
    return (-np.sum(cee) / norm), (-ceed / norm)


CEE_implementations = {
    ('F', False): _FramewiseCEE,
    ('F', True): _not_implemented,
    ('L', False): _illegal_combination,
    ('L', True): _illegal_combination,
    ('C', False): _not_implemented,
    ('C', True): _SequencewiseBinarizingCEE
}


def CrossEntropyError(Y, T, M=None):
    assert T.validate_for_output_shape(*Y.shape)
    y_m = np.clip(Y, 1e-6, 1.0-1e-6)  # do not modify original Y
    return CEE_implementations[T.targets_type](y_m, T.data, M)


################################################################################

def _FramewiseMCCEE(Y, T, M):
    cee = T * np.log(Y)
    quot = T / Y
    norm = Y.shape[1]  # normalize by number of sequences
    if M is not None:
        cee *= M
        quot *= M
    return (- np.sum(cee) / norm), (- quot / norm)


MCCEE_implementations = {
    ('F', False): _FramewiseMCCEE,
    ('F', True): _not_implemented,
    ('L', False): _illegal_combination,
    ('L', True): _illegal_combination,
    ('C', False): _not_implemented,
    ('C', True): _not_implemented
}


def MultiClassCrossEntropyError(Y, T, M=None):
    assert T.validate_for_output_shape(*Y.shape)
    y_m = np.clip(Y, 1e-6, 1.0)  # do not modify original Y
    return MCCEE_implementations[T.targets_type](y_m, T.data, M)


################################################################################

def _LabelingBinarizingCTC(Y, T, M):
    time_size, batch_size, label_count = Y.shape
    deltas = np.zeros((time_size, batch_size, label_count))
    deltas[:] = float('-inf')
    errors = np.zeros(batch_size)
    for b, (y, t, m) in enumerate(Online(Y, T, M, verbose=False)()):
        err, delt = ctcpp(y, list(t.data[0]))
        errors[b] = err
        deltas[:, b:b+1, :] = delt.as_array()

    return np.mean(errors), -deltas / batch_size


CTC_implementations = {
    ('F', False): _illegal_combination,
    ('F', True): _illegal_combination,
    ('L', False): _not_implemented,
    ('L', True): _LabelingBinarizingCTC,
    ('C', False): _illegal_combination,
    ('C', True): _illegal_combination
}


def CTC(Y, T, M=None):
    assert T.validate_for_output_shape(*Y.shape)
    return CTC_implementations[T.targets_type](Y, T, M)


################################################################################

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
