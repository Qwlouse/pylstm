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
    def __init__(self, targets_type):
        self.targets_type = targets_type

    def __getitem__(self, item):
        raise NotImplementedError()

    def crop_time(self, end):
        raise NotImplementedError()

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        raise NotImplementedError()


class FramewiseTargets(Targets):
    def __init__(self, T, binarize_to=None):
        super(FramewiseTargets, self).__init__(('F', binarize_to is not None))
        dim = len(T.shape)
        assert dim == 3 or (binarize_to and dim == 2)
        self.binarize_to = binarize_to
        self.T = T.reshape(T.shape[0], T.shape[1], -1)

    def __getitem__(self, item):
        return FramewiseTargets(self.T[:, item, :])

    def crop_time(self, end):
        return FramewiseTargets(self.T[:end, :, :])

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        if self.binarize_to:
            return self.T.shape == (timesteps, batchsize, 1) and \
                self.binarize_to == out_size
        else:
            return self.T.shape == (timesteps, batchsize, out_size)


class LabelingTargets(Targets):
    def __init__(self, L, binarize_to=None):
        super(LabelingTargets, self).__init__(('L', binarize_to is not None))
        self.L = L
        self.binarize_to = binarize_to

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
    def __init__(self, C, binarize_to=None):
        super(SequencewiseTargets, self).__init__(('C',
                                                   binarize_to is not None))
        dim = len(C.shape)
        assert dim == 2 or (binarize_to and dim == 1)
        self.binarize_to = binarize_to
        self.C = C.reshape(C.shape[0], -1)

    def __getitem__(self, item):
        return SequencewiseTargets(self.C[item])

    def crop_time(self, end):
        return self

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        if self.binarize_to:
            return self.C.shape == (batchsize, 1) and \
                self.binarize_to == out_size
        else:
            return self.C.shape == (batchsize, out_size)
