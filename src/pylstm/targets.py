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

    def __str__(self):
        return "<%s-Targets>" % self.targets_type


class FramewiseTargets(Targets):
    def __init__(self, T, binarize_to=None):
        super(FramewiseTargets, self).__init__(('F', binarize_to is not None))
        dim = len(T.shape)
        assert dim == 3 or (binarize_to and dim == 2)
        self.binarize_to = binarize_to
        self.data = T.reshape(T.shape[0], T.shape[1], -1)

    def __getitem__(self, item):
        return FramewiseTargets(self.data[:, item, :],
                                binarize_to=self.binarize_to)

    def crop_time(self, end):
        return FramewiseTargets(self.data[:end, :, :],
                                binarize_to=self.binarize_to)

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        if self.binarize_to:
            return self.data.shape == (timesteps, batchsize, 1) and \
                self.binarize_to == out_size
        else:
            return self.data.shape == (timesteps, batchsize, out_size)

    def __str__(self):
        return "<FramewiseTargets dim=%s>" % self.data.shape


class LabelingTargets(Targets):
    def __init__(self, L, binarize_to=None):
        super(LabelingTargets, self).__init__(('L', binarize_to is not None))
        self.data = L
        self.binarize_to = binarize_to

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            return LabelingTargets(self.data[item], binarize_to=self.binarize_to)
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            assert len(item) == len(self.data)
            return LabelingTargets([self.data[i] for i in item],
                                   binarize_to=self.binarize_to)
        else:
            raise ValueError("Indexing with type '%s' unsupported" % type(item))

    def crop_time(self, end):
        return self

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        return len(self.data) == batchsize

    def __str__(self):
        return "<LabelingTargets len=%d>" % len(self.data)


class SequencewiseTargets(Targets):
    def __init__(self, C, binarize_to=None):
        super(SequencewiseTargets, self).__init__(('C',
                                                   binarize_to is not None))
        dim = len(C.shape)
        assert dim == 2 or (binarize_to and dim == 1)
        self.binarize_to = binarize_to
        self.data = C.reshape(C.shape[0], -1)

    def __getitem__(self, item):
        return SequencewiseTargets(self.data[item], binarize_to=self.binarize_to)

    def crop_time(self, end):
        return self

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        if self.binarize_to:
            return self.data.shape == (batchsize, 1) and \
                self.binarize_to == out_size
        else:
            return self.data.shape == (batchsize, out_size)

    def __str__(self):
        return "<SequencewiseTargets dim=%s>" % self.data.shape
