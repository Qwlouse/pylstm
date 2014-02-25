#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


def create_targets_object(targets, mask=None):
    """
    Try to create a suitable targets object from given targets and optionally
    mask. If targets is already a targets object then do nothing.
    If the dimensionality of the targets is 3 assume framewise targets.
    If the dimensionality of the targets is 2 assume sequencewise targets.
    If targets is a list assume labelling targets.
    """
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
    def __init__(self, targets_type, binarizing, mask):
        self.targets_type = (targets_type, binarizing)
        self.mask = None
        if mask is not None:
            assert (mask.ndim == 3 and mask.shape[2] == 1) or (mask.ndim == 2)
            self.mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    def __getitem__(self, item):
        raise NotImplementedError()

    def trim(self):
        """
        Cut all the timesteps from the end of the sequence that have a mask
        value of 0.
        @return: The number of trimmed timesteps
        @rtype: int
        """
        if self.mask is None:
            return 0
        # get trim amount
        k = np.min(self.mask[::-1, :, 0].argmax(axis=0))
        self.mask = self.mask[:-k, :, :]
        return k

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        raise NotImplementedError()

    def __str__(self):
        return "<%s-Targets>" % str(self.targets_type)


def assert_shape_equals(s1, s2):
    assert s1 == s2, "targets shape error: %s != %s" % (str(s1), str(s2))


class FramewiseTargets(Targets):
    """
    Provide a target value for every point in time. Although some timesteps
    might be masked out.
    """
    def __init__(self, targets, mask=None, binarize_to=None):
        super(FramewiseTargets, self).__init__('F', binarize_to is not None,
                                               mask)
        assert (targets.ndim == 3) or (binarize_to and targets.ndim == 2)
        self.binarize_to = binarize_to
        self.data = targets.reshape(targets.shape[0], targets.shape[1], -1)
        if binarize_to:
            self.data = np.array(self.data, dtype=np.int)
        if self.mask is not None:
            assert self.mask.shape[0] == self.data.shape[0]
            assert self.mask.shape[1] == self.data.shape[1]

    def __getitem__(self, item):
        t = self.data[:, item, :]
        m = self.mask[:, item, :] if self.mask is not None else None
        return FramewiseTargets(t, m, binarize_to=self.binarize_to)

    def trim(self):
        k = Targets.trim(self)
        self.data = self.data[:-k, :, :]
        return k

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
    """
    Provide a list of labels for the sequence. If a mask is given, then the
    resulting deltas will be masked.
    """
    def __init__(self, labels, mask=None, binarize_to=None):
        super(LabelingTargets, self).__init__('L', binarize_to is not None,
                                              mask)
        self.data = labels
        self.binarize_to = binarize_to

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

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        assert len(self.data) == batchsize
        if self.mask is not None:
            assert_shape_equals(self.mask.shape, (timesteps, batchsize, 1))
        if self.binarize_to is not None:
            assert self.binarize_to == out_size

    def __str__(self):
        return "<LabelingTargets len=%d>" % len(self.data)


class SequencewiseTargets(Targets):
    """
    Provide one target per sequence. If no mask is given then only the last
    timestep will receive deltas. If a mask is given, then all the masked in
    timesteps will receive deltas.
    """
    def __init__(self, sequence_targets, mask=None, binarize_to=None):
        super(SequencewiseTargets, self).__init__('S', binarize_to is not None,
                                                  mask)
        assert sequence_targets.ndim == 2 or \
            (binarize_to and sequence_targets.ndim == 1)
        self.binarize_to = binarize_to
        self.data = sequence_targets.reshape(sequence_targets.shape[0], -1)

    def __getitem__(self, item):
        s = self.data[item]
        m = self.mask[:, item, :] if self.mask is not None else None
        return SequencewiseTargets(s, m, binarize_to=self.binarize_to)

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
