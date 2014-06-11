#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
from pylstm import Describable

from pylstm.wrapper import ctcpp
from pylstm.datasets.preprocessing import binarize_array
from pylstm.targets import create_targets_object
from pylstm.utils import get_sequence_lengths, ctc_best_path_decoding, \
    levenshtein
from pylstm.datasets.data_iterators import Online


def _not_implemented(*_):
    raise NotImplementedError('This combination is not implemented yet!')


def _illegal_combination(*_):
    raise RuntimeError('Illegal combination of targets and error function!')


class ErrorFunction(Describable):
    @staticmethod
    def aggregate(errors):
        return np.mean(errors)


################################################################################
# Mean Squared Error Implementations
# (Gaussian Cross Entropy)

class MeanSquaredError(ErrorFunction):
    @staticmethod
    def _framewise(outputs, targets, mask):
        diff = outputs - targets
        if mask is not None:
            diff *= mask
        norm = outputs.shape[1]  # normalize by number of sequences
        error = 0.5 * np.sum(diff ** 2) / norm
        return error, (diff / norm)

    @staticmethod
    def _sequencewise_binarizing(outputs, targets, mask):
        # TODO change behavior for mask = None to only inject at last timestep
        diff = outputs.copy()
        for b in range(outputs.shape[1]):
            diff[:, b, int(targets[b, 0])] -= 1
        if mask is not None:
            diff *= mask
        norm = outputs.shape[1]  # normalize by number of sequences
        error = 0.5 * np.sum(diff ** 2) / norm
        return error, (diff / norm)

    def __call__(self, outputs, targets):
        implementations = {
            ('F', False): self._framewise,
            ('F', True): _not_implemented,
            ('L', False): _illegal_combination,
            ('L', True): _illegal_combination,
            ('S', False): self._framewise,  # should work smoothly through broadcasting
            ('S', True): self._sequencewise_binarizing
        }
        targets.validate_for_output_shape(*outputs.shape)
        impl = implementations[targets.targets_type]
        return impl(outputs, targets.data, targets.mask)


MeanSquaredError = MeanSquaredError()
MSE = MeanSquaredError


################################################################################
# Cross Entropy Error Implementations
# (Independent Binomial Cross Entropy)

class CrossEntropyError(ErrorFunction):
    @staticmethod
    def _framewise(clipped_outputs, targets, mask):
        cee = targets * np.log(clipped_outputs) + \
            (1 - targets) * np.log(1 - clipped_outputs)
        ceed = (targets - clipped_outputs) /\
               (clipped_outputs * (clipped_outputs - 1))
        if mask is not None:
            cee *= mask
            ceed *= mask
        norm = clipped_outputs.shape[1]  # normalize by number of sequences
        return (-np.sum(cee) / norm), (ceed / norm)

    @staticmethod
    def _sequencewise_binarizing(clipped_outputs, targets, mask):
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

    def __call__(self, outputs, targets):
        implementations = {
            ('F', False): self._framewise,
            ('F', True): _not_implemented,
            ('L', False): _illegal_combination,
            ('L', True): _illegal_combination,
            ('S', False): _not_implemented,
            ('S', True): self._sequencewise_binarizing
        }
        targets.validate_for_output_shape(*outputs.shape)
        # do not modify original outputs
        clipped_outputs = np.clip(outputs, 1e-6, 1.0-1e-6)
        impl = implementations[targets.targets_type]
        return impl(clipped_outputs, targets.data, targets.mask)


CrossEntropyError = CrossEntropyError()


################################################################################
# Multi-Class (Multi-Label) Cross Entropy Error Implementations
# (Multinomial/softmax cross entropy)

class MultiClassCrossEntropyError(ErrorFunction):

    @staticmethod
    def _framewise(clipped_outputs, targets, mask):
        cee = targets * np.log(clipped_outputs)
        quot = targets / clipped_outputs
        norm = clipped_outputs.shape[1]  # normalize by number of sequences
        if mask is not None:
            cee *= mask
            quot *= mask
        return (- np.sum(cee) / norm), (- quot / norm)

    @staticmethod
    def _framewise_binarizing(clipped_outputs, targets, mask):
        # TODO: More efficient implementation
        binarized_targets = binarize_array(targets,
                                           range(clipped_outputs.shape[2]))
        return MultiClassCrossEntropyError._framewise(
            clipped_outputs, binarized_targets, mask)

    @staticmethod
    def _sequencewise_binarizing(clipped_outputs, targets, mask):
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

    def __call__(self, outputs, targets):
        implementations = {
            ('F', False): self._framewise,
            ('F', True): self._framewise_binarizing,
            ('L', False): _illegal_combination,
            ('L', True): _illegal_combination,
            ('S', False): _not_implemented,
            ('S', True): self._sequencewise_binarizing
        }
        targets.validate_for_output_shape(*outputs.shape)
        # do not modify original outputs
        clipped_outputs = np.clip(outputs, 1e-6, 1.0)
        impl = implementations[targets.targets_type]
        return impl(clipped_outputs, targets.data, targets.mask)


MultiClassCrossEntropyError = MultiClassCrossEntropyError()


################################################################################
# CTC error implementations for labellings

class ConnectionistTemporalClassificationError(ErrorFunction):
    @staticmethod
    def _labeling_binarizing(outputs, targets):
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

    def __call__(self, outputs, targets):
        implementations = {
            ('F', False): _illegal_combination,
            ('F', True): _illegal_combination,
            ('L', False): _not_implemented,
            ('L', True): self._labeling_binarizing,
            ('S', False): _illegal_combination,
            ('S', True): _illegal_combination
        }
        targets.validate_for_output_shape(*outputs.shape)
        impl = implementations[targets.targets_type]
        return impl(outputs, targets)


CTC = ConnectionistTemporalClassificationError()
ConnectionistTemporalClassificationError = CTC


################################################################################
# Classification Error for monitoring

class ClassificationError(ErrorFunction):
    @staticmethod
    def _framewise(outputs, targets, mask):
        y_win = outputs.argmax(2)
        t_win = targets.argmax(2)
        if mask is not None:
            errors = np.sum((y_win != t_win) * mask[:, :, 0])
            total = np.sum(mask)
        else:
            errors = np.sum((y_win != t_win))
            total = targets.shape[0] * targets.shape[1]
        return (errors, total), None

    @staticmethod
    def _framewise_binarizing(outputs, targets, mask):
        y_win = outputs.argmax(2)
        t_win = targets[:, :, 0]
        if mask is not None:
            errors = np.sum((y_win != t_win) * mask[:, :, 0])
            total = np.sum(mask)
        else:
            errors = np.sum((y_win != t_win))
            total = targets.shape[0] * targets.shape[1]
        return (errors, total), None

    @staticmethod
    def _sequencewise(outputs, targets, mask):
        if mask is None:
            y_win = outputs[-1, :, :].argmax(1)
            t_win = targets[:, :].argmax(1)
            errors = np.sum((y_win != t_win))
        else:
            errors = 0
            for b, t in enumerate(get_sequence_lengths(mask)):
                if outputs[t, b, :].argmax() != targets[b, :].argmax():
                    errors += 1

        return (errors, outputs.shape[1]), None

    @staticmethod
    def _sequencewise_binarizing(outputs, targets, mask):
        if mask is None:
            y_win = outputs[-1, :, :].argmax(1)
            t_win = targets[:, 0]
            errors = np.sum((y_win != t_win))
        else:
            errors = 0
            for b, t in enumerate(get_sequence_lengths(mask)):
                if outputs[t, b, :].argmax() != targets[b, 0]:
                    errors += 1

        return (errors, outputs.shape[1]), None

    def __call__(self, outputs, targets):
        implementations = {
            ('F', False): self._framewise,
            ('F', True): self._framewise_binarizing,
            ('L', False): _illegal_combination,
            ('L', True): _illegal_combination,
            ('S', False): self._sequencewise,
            ('S', True): self._sequencewise_binarizing
        }
        targets.validate_for_output_shape(*outputs.shape)
        impl = implementations[targets.targets_type]
        return impl(outputs, targets.data, targets.mask)

    @staticmethod
    def aggregate(errors):
        e = np.sum(errors, axis=0)
        return np.round(e[0] * 100. / e[1], 2)

ClassificationError = ClassificationError()


################################################################################
# Label Error for monitoring

class LabelingError(ErrorFunction):
    @staticmethod
    def _labeling_binarizing(outputs, targets):
        errors = 0
        total_length = 0
        for y, t in Online(outputs, targets, verbose=False)():
            lab = ctc_best_path_decoding(y)
            errors += levenshtein(lab, t.data[0])
            total_length += len(t.data[0])

        return (errors, total_length), None

    def __call__(self, outputs, targets):
        implementations = {
            ('F', False): _illegal_combination,
            ('F', True): _illegal_combination,
            ('L', False): _not_implemented,
            ('L', True): self._labeling_binarizing,
            ('S', False): _illegal_combination,
            ('S', True): _illegal_combination
        }
        targets.validate_for_output_shape(*outputs.shape)
        impl = implementations[targets.targets_type]
        return impl(outputs, targets)

    @staticmethod
    def aggregate(errors):
        e = np.sum(errors, axis=0)
        return np.round(e[0] * 100. / e[1], 2)

LabelingError = LabelingError()