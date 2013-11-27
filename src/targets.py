#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


def create_targets_object(T):
    if isinstance(T, Targets):
        return T

    if isinstance(T, np.ndarray):
        if len(T.shape) == 3:
            return FramewiseTargets(T)
        if len(T.shape) == 2:
            return SequencewiseTargets(T)

    elif isinstance(T, list):
        return LabelingTargets(T)

    raise ValueError('Not a valid format for auto-conversion to targets')
        

class Targets(object):
    def __getitem__(self, item):
        raise NotImplementedError()

    def crop_time(self, end):
        raise NotImplementedError()

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        raise NotImplementedError()


class FramewiseTargets(Targets):
    def __init__(self, T):
        assert len(T.shape) == 3
        self.T = T

    def __getitem__(self, item):
        return FramewiseTargets(self.T[:, item, :])

    def crop_time(self, end):
        return FramewiseTargets(self.T[:end, :, :])

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        return self.T.shape == (timesteps, batchsize, out_size)


class FramewiseClassificationTargets(FramewiseTargets):
    def __init__(self, T, nr_classes):
        assert len(T.shape) == 2 or T.shape[2] == 1
        T = T.reshape(T.shape[0], T.shape[1], 1)
        super(FramewiseClassificationTargets, self).__init__(T)
        self.nr_classes = nr_classes

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        return self.T.shape == (timesteps, batchsize, 1) and \
            self.nr_classes == out_size


class LabelingTargets(Targets):
    def __init__(self, L):
        self.L = L

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            return LabelingTargets(self.L[item])
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            assert len(item) == len(self.L)
            return FramewiseTargets([self.L[i] for i in item])
        else:
            raise ValueError("Indexing with type '%s' unsupported" % type(item))

    def crop_time(self, end):
        return self
        

class SequencewiseTargets(Targets):
    def __init__(self, C):
        assert len(C.shape) == 2
        self.C = C

    def __getitem__(self, item):
        return SequencewiseTargets(self.C[item])

    def crop_time(self, end):
        return self

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        return self.C.shape == (batchsize, out_size)


class SequencewiseClassificationTargets(SequencewiseTargets):
    def __init__(self, C, nr_classes):
        assert len(C.shape) == 1 or C.shape[1] == 1
        C = C.reshape(C.shape[0], 1)
        super(SequencewiseClassificationTargets, self).__init__(C)
        self.nr_classes = nr_classes

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        return self.C.shape == (batchsize, 1) and self.nr_classes == out_size