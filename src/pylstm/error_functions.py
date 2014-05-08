#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.datasets.preprocessing import binarize_array
from pylstm.targets import create_targets_object
from .training.data_iterators import Online
from .wrapper import ctcpp


def _not_implemented(*_):
    raise NotImplementedError('This combination is not implemented yet!')


def _illegal_combination(*_):
    raise RuntimeError('Illegal combination of targets and error function!')


################################################################################
# Mean Squared Error Implementations
# (Gaussian Cross Entropy)

def _FramewiseMSE(outputs, targets, mask):
    diff = outputs - targets
    if mask is not None:
        diff *= mask
    norm = outputs.shape[1]  # normalize by number of sequences
    error = 0.5 * np.sum(diff ** 2) / norm
    return error, (diff / norm)


def _SequencewiseBinarizingMSE(outputs, targets, mask):
    # TODO change behavior for mask = None to only inject at last timestep
    diff = outputs.copy()
    for b in range(outputs.shape[1]):
        diff[:, b, int(targets[b, 0])] -= 1
    if mask is not None:
        diff *= mask
    norm = outputs.shape[1]  # normalize by number of sequences
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


def MeanSquaredError(outputs, targets):
    targets.validate_for_output_shape(*outputs.shape)
    return MSE_implementations[targets.targets_type](outputs, targets.data,
                                                     targets.mask)


################################################################################
# Cross Entropy Error Implementations
# (Independent Binomial Cross Entropy)

def _FramewiseCEE(clipped_outputs, targets, mask):
    cee = targets * np.log(clipped_outputs) + \
        (1 - targets) * np.log(1 - clipped_outputs)
    ceed = (targets - clipped_outputs) /\
           (clipped_outputs * (clipped_outputs - 1))
    if mask is not None:
        cee *= mask
        ceed *= mask
    norm = clipped_outputs.shape[1]  # normalize by number of sequences
    return (-np.sum(cee) / norm), (ceed / norm)


def _SequencewiseBinarizingCEE(clipped_outputs, targets, mask):
    # TODO change behavior for mask = None to only inject at last timestep
    cee = np.log(1 - clipped_outputs)
    ceed = 1. / (clipped_outputs - 1)
    for b in range(clipped_outputs.shape[1]):
        cee[:, b, targets[b, 0]] = np.log(clipped_outputs[:, b, targets[b, 0]])
        ceed[:, b, targets[b, 0]] = 1. / clipped_outputs[:, b, targets[b, 0]]
    norm = clipped_outputs.shape[1]  # normalize by number of sequences
    if mask is not None:
        cee *= mask
        ceed *= mask
    return (-np.sum(cee) / norm), (-ceed / norm)


CEE_implementations = {
    ('F', False): _FramewiseCEE,
    ('F', True): _not_implemented,
    ('L', False): _illegal_combination,
    ('L', True): _illegal_combination,
    ('S', False): _not_implemented,
    ('S', True): _SequencewiseBinarizingCEE
}


def CrossEntropyError(outputs, targets):
    targets.validate_for_output_shape(*outputs.shape)
    clipped_outputs = np.clip(outputs, 1e-6, 1.0-1e-6)  # do not modify original Y
    return CEE_implementations[targets.targets_type](clipped_outputs,
                                                     targets.data, targets.mask)


################################################################################
# Multi-Class (Multi-Label) Cross Entropy Error Implementations
# (Multinomial/softmax cross entropy)

def _FramewiseMCCEE(clipped_outputs, targets, mask):
    cee = targets * np.log(clipped_outputs)
    quot = targets / clipped_outputs
    norm = clipped_outputs.shape[1]  # normalize by number of sequences
    if mask is not None:
        cee *= mask
        quot *= mask
    return (- np.sum(cee) / norm), (- quot / norm)


def _FramewiseBinarizingMCCEE(clipped_outputs, targets, mask):
    T_b = binarize_array(targets, range(clipped_outputs.shape[2]))

    return _FramewiseMCCEE(clipped_outputs, T_b, mask)


def _SequencewiseBinarizingMCCEE(clipped_outputs, targets, mask):
    # TODO change behavior for mask = None to only inject at last timestep
    cee = np.zeros_like(clipped_outputs)
    quot = np.zeros_like(clipped_outputs)
    for b in range(clipped_outputs.shape[1]):
        cee[:, b, targets[b, 0]] = np.log(clipped_outputs[:, b, targets[b, 0]])
        quot[:, b, targets[b, 0]] = 1.0 / clipped_outputs[:, b, targets[b, 0]]

    norm = clipped_outputs.shape[1]  # normalize by number of sequences
    if mask is not None:
        cee *= mask
        quot *= mask
    return (- np.sum(cee) / norm), (- quot / norm)


MCCEE_implementations = {
    ('F', False): _FramewiseMCCEE,
    ('F', True): _FramewiseBinarizingMCCEE,
    ('L', False): _illegal_combination,
    ('L', True): _illegal_combination,
    ('S', False): _not_implemented,
    ('S', True): _SequencewiseBinarizingMCCEE
}


def MultiClassCrossEntropyError(outputs, targets):
    targets.validate_for_output_shape(*outputs.shape)
    clipped_outputs = np.clip(outputs, 1e-6, 1.0)  # do not modify original Y
    return MCCEE_implementations[targets.targets_type](clipped_outputs,
                                                       targets.data,
                                                       targets.mask)


################################################################################
# CTC error implementations for labellings

def _LabelingBinarizingCTC(outputs, targets, mask):
    # TODO: use mask to mask deltas
    time_size, batch_size, label_count = outputs.shape
    deltas = np.zeros((time_size, batch_size, label_count))
    deltas[:] = float('-inf')
    errors = np.zeros(batch_size)
    targets = create_targets_object(targets)
    for b, (y, t) in enumerate(Online(outputs, targets, verbose=False)()):
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


def CTC(outputs, targets):
    targets.validate_for_output_shape(*outputs.shape)
    return CTC_implementations[targets.targets_type](outputs, targets,
                                                     targets.mask)


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


def ClassificationError(outputs, targets):
    targets.validate_for_output_shape(*outputs.shape)
    return ClassificationError_implementations[targets.targets_type](
        outputs, targets, targets.mask)


def get_error_function_by_name(name):
    error_functions = [
        MeanSquaredError, CrossEntropyError, MultiClassCrossEntropyError,
        CTC, ClassificationError
    ]
    for e in error_functions:
        if e.__name__ == name:
            return e

    raise ValueError('Error Function "%s" not found!' % name)
