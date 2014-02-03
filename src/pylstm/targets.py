#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


def create_targets_object(targets, mask=None):
    if isinstance(targets, Targets):
        return targets

    if isinstance(targets, np.ndarray):
        if targets.ndim == 3:
            return FramewiseTargets(targets, mask)
        if targets.ndim == 2:
            return SequencewiseTargets(targets, mask)

    elif isinstance(targets, list):
        return LabelingTargets(targets, mask)

    raise ValueError('Not a valid format for auto-conversion to targets')
        

class Targets(object):
    """
    Baseclass for all targets objects. A targets object holds all the desired
    outputs and the corresponding mask (if any).
    """
    def __init__(self, targets_type, binarizing):
        self.targets_type = (targets_type, binarizing)

    def __getitem__(self, item):
        raise NotImplementedError()

    def crop_time(self, end):
        raise NotImplementedError()

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        raise NotImplementedError()

    def __str__(self):
        return "<%s-Targets>" % str(self.targets_type)


def assert_shape_equals(s1, s2):
    assert s1 == s2, "targets shape error: %s != %s" % (str(s1), str(s2))


class FramewiseTargets(Targets):
    def __init__(self, targets, mask=None, binarize_to=None):
        super(FramewiseTargets, self).__init__('F', binarize_to is not None)
        assert (targets.ndim == 3) or (binarize_to and targets.ndim == 2)
        self.binarize_to = binarize_to
        self.data = targets.reshape(targets.shape[0], targets.shape[1], -1)
        if binarize_to:
            self.data = np.array(self.data, dtype=np.int)
        self.mask = None
        if mask is not None:
            assert (mask.ndim == 3 and mask.shape[2] == 1) or (mask.ndim == 2)
            self.mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            assert_shape_equals(self.data, self.mask)

    def __getitem__(self, item):
        t = self.data[:, item, :]
        m = self.mask[:, item, :] if self.mask is not None else None
        return FramewiseTargets(t, m, binarize_to=self.binarize_to)

    def crop_time(self, end):
        t = self.data[:end, :, :]
        m = self.mask[:end, :, :] if self.mask is not None else None
        return FramewiseTargets(t, m, binarize_to=self.binarize_to)

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        if self.binarize_to:
            assert_shape_equals(self.data.shape, (timesteps, batchsize, 1))
            assert self.binarize_to == out_size
        else:
            assert_shape_equals(self.data.shape,
                                (timesteps, batchsize, out_size))

    def __str__(self):
        return "<FramewiseTargets dim=%s>" % str(self.data.shape)


class LabelingTargets(Targets):
    def __init__(self, labels, mask=None, binarize_to=None):
        super(LabelingTargets, self).__init__('L', binarize_to is not None)
        self.data = labels
        self.binarize_to = binarize_to
        self.mask = None
        if mask is not None:
            assert (mask.ndim == 3 and mask.shape[2] == 1) or (mask.ndim == 2)
            self.mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            l = self.data[item]
            m = self.mask[:, item, :] if self.mask is not None else None
            return LabelingTargets(l, m, binarize_to=self.binarize_to)
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            l = [self.data[i] for i in item]
            m = self.mask[:, item, :] if self.mask is not None else None
            return LabelingTargets(l, m, binarize_to=self.binarize_to)
        else:
            raise ValueError("Indexing with type '%s' unsupported" % type(item))

    def crop_time(self, end):
        t = self.data
        m = self.mask[:end, :, :] if self.mask is not None else None
        return LabelingTargets(t, m, binarize_to=self.binarize_to)

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        assert len(self.data) == batchsize
        if self.mask is not None:
            assert_shape_equals(self.mask.shape, (timesteps, batchsize, 1))
        if self.binarize_to is not None:
            assert self.binarize_to == out_size

    def __str__(self):
        return "<LabelingTargets len=%d>" % len(self.data)


class SequencewiseTargets(Targets):
    def __init__(self, sequence_targets, mask=None, binarize_to=None):
        super(SequencewiseTargets, self).__init__('S', binarize_to is not None)
        assert sequence_targets.ndim == 2 or \
            (binarize_to and sequence_targets.ndim == 1)
        self.binarize_to = binarize_to
        self.data = sequence_targets.reshape(sequence_targets.shape[0], -1)
        self.mask = None
        if mask is not None:
            assert (mask.ndim == 3 and mask.shape[2] == 1) or (mask.ndim == 2)
            assert sequence_targets.shape[1] == mask.shape[1]
            self.mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    def __getitem__(self, item):
        s = self.data[item]
        m = self.mask[:, item, :] if self.mask is not None else None
        return SequencewiseTargets(s, m, binarize_to=self.binarize_to)

    def crop_time(self, end):
        s = self.data
        m = self.mask[:end, :, :] if self.mask is not None else None
        return SequencewiseTargets(s, m, self.binarize_to)

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        if self.binarize_to:
            assert_shape_equals(self.data.shape, (batchsize, 1))
            assert self.binarize_to == out_size
        else:
            assert_shape_equals(self.data.shape, (batchsize, out_size))

        if self.mask is not None:
            assert_shape_equals(self.mask.shape, (timesteps, batchsize, 1))

    def __str__(self):
        return "<SequencewiseTargets dim=%s>" % str(self.data.shape)
