#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm import binarize_array
from .training.data_iterators import Online
from .wrapper import ctcpp


def _not_implemented(*_):
    raise NotImplementedError('This combination is not implemented yet!')


def _illegal_combination(*_):
    raise RuntimeError('Illegal combination of targets and error function!')


################################################################################
# Mean Squared Error Implementations
# (Gaussian Cross Entropy)

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
        diff[:, b, T[b, 0]] -= 1
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
    ('S', False): _FramewiseMSE,  # should work smoothly through broadcasting
    ('S', True): _SequencewiseBinarizingMSE
}


def MeanSquaredError(Y, T, M=None):
    T.validate_for_output_shape(*Y.shape)
    return MSE_implementations[T.targets_type](Y, T.data, M)


################################################################################
# Cross Entropy Error Implementations
# (Independent Binomial Cross Entropy)

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
        cee[:, b, T[b, 0]] = np.log(y_m[:, b, T[b, 0]])
        ceed[:, b, T[b, 0]] = 1. / y_m[:, b, T[b, 0]]
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
    ('S', False): _not_implemented,
    ('S', True): _SequencewiseBinarizingCEE
}


def CrossEntropyError(Y, T, M=None):
    T.validate_for_output_shape(*Y.shape)
    y_m = np.clip(Y, 1e-6, 1.0-1e-6)  # do not modify original Y
    return CEE_implementations[T.targets_type](y_m, T.data, M)


################################################################################
# Multi-Class (Multi-Label) Cross Entropy Error Implementations
# (Multinomial/softmax cross entropy)

def _FramewiseMCCEE(y_m, T, M):
    cee = T * np.log(y_m)
    quot = T / y_m
    norm = y_m.shape[1]  # normalize by number of sequences
    if M is not None:
        cee *= M
        quot *= M
    return (- np.sum(cee) / norm), (- quot / norm)


def _FramewiseBinarizingMCCEE(y_m, T, M):
    T_b = binarize_array(T, range(y_m.shape[2]))

    return _FramewiseMCCEE(y_m, T_b, M)


def _SequencewiseBinarizingMCCEE(y_m, T, M):
    cee = np.zeros_like(y_m)
    quot = np.zeros_like(y_m)
    for b in range(y_m.shape[1]):
        cee[:, b, T[b, 0]] = np.log(y_m[:, b, T[b, 0]])
        quot[:, b, T[b, 0]] = 1.0 / y_m[:, b, T[b, 0]]

    norm = y_m.shape[1]  # normalize by number of sequences
    if M is not None:
        cee *= M
        quot *= M
    return (- np.sum(cee) / norm), (- quot / norm)


MCCEE_implementations = {
    ('F', False): _FramewiseMCCEE,
    ('F', True): _FramewiseBinarizingMCCEE,
    ('L', False): _illegal_combination,
    ('L', True): _illegal_combination,
    ('S', False): _not_implemented,
    ('S', True): _SequencewiseBinarizingMCCEE
}


def MultiClassCrossEntropyError(Y, T, M=None):
    T.validate_for_output_shape(*Y.shape)
    y_m = np.clip(Y, 1e-6, 1.0)  # do not modify original Y
    return MCCEE_implementations[T.targets_type](y_m, T.data, M)


################################################################################
# CTC error implementations for labellings

def _LabelingBinarizingCTC(Y, T, M):
    time_size, batch_size, label_count = Y.shape
    deltas = np.zeros((time_size, batch_size, label_count))
    deltas[:] = float('-inf')
    errors = np.zeros(batch_size)
    for b, (y, t, m) in enumerate(Online(Y, T, M, verbose=False)()):
        err, delt = ctcpp(y, list(t.data[0]))
        errors[b] = err
        deltas[:y.shape[0], b:b+1, :] = delt.as_array()

    return np.mean(errors), -deltas / batch_size


CTC_implementations = {
    ('F', False): _illegal_combination,
    ('F', True): _illegal_combination,
    ('L', False): _not_implemented,
    ('L', True): _LabelingBinarizingCTC,
    ('S', False): _illegal_combination,
    ('S', True): _illegal_combination
}


def CTC(Y, T, M=None):
    T.validate_for_output_shape(*Y.shape)
    return CTC_implementations[T.targets_type](Y, T, M)


################################################################################
# Classification Error for monitoring

ClassificationError_implementations = {
    ('F', False): _not_implemented,
    ('F', True): _not_implemented,
    ('L', False): _illegal_combination,
    ('L', True): _illegal_combination,
    ('S', False): _not_implemented,
    ('S', True): _not_implemented
}


def ClassificationError(Y, T, M=None):
    T.validate_for_output_shape(*Y.shape)
    return ClassificationError_implementations[T.targets_type](Y, T, M)
