#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


def create_targets_object(targets, mask=None, targets_type=None):
    """
    Try to create a suitable targets object from given targets and optionally
    mask. If targets is already a targets object then do nothing.
    If the dimensionality of the targets is 3 assume framewise targets.
    If the dimensionality of the targets is 2 assume sequencewise targets.
    If targets is a list assume labelling targets.
    """
    if targets_type is not None:
        if targets_type == 'F':
            return FramewiseTargets(targets, mask)
        elif targets_type == 'L':
            return LabelingTargets(targets, mask)
        elif targets_type == 'S':
            return SequencewiseTargets(targets, mask)
        else:
            raise ValueError('Not a valid targets_type: "%s"' % targets_type)

    if isinstance(targets, Targets):
        assert mask is None, "Can't combine Targets object and mask"
        return targets

    elif isinstance(targets, np.ndarray):
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
    def __init__(self, targets_type, binarize_to, mask):
        self.targets_type = (targets_type, binarize_to is not None)
        self.binarize_to = binarize_to
        self.mask = None
        self.data = None
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
        if k > 0:
            self.mask = self.mask[:-k, :, :]
        return k

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        if self.mask is not None:
            assert_shape_equals(self.mask.shape, (timesteps, batchsize, 1))
        if self.binarize_to is not None:
            assert self.binarize_to == out_size

    def __str__(self):
        return "<%s-Targets>" % str(self.targets_type)

    def is_framewise(self):
        return self.targets_type[0] == 'F'

    def is_labeling(self):
        return self.targets_type[0] == 'L'

    def pad_time(self, target_length):
        if self.mask:
            t, b, _ = self.mask.shape
            assert target_length >= t
            padding = np.zeros((target_length - t, b, 1))
            self.mask = np.vstack((self.mask, padding))
        else:  # now it should have a mask
            self.mask = np.ones((target_length, self.get_sequence_cnt(), 1))
            if self.get_duration():
                self.mask[self.get_duration():, :, 0] = 0

    def get_duration(self):
        if self.mask:
            return self.mask.shape[0]
        else:
            return None

    def get_sequence_cnt(self):
        if self.mask:
            return self.mask.shape[1]

    def _get_joint_mask(self, other):
        # make them the same duration
        if self.get_duration() != other.get_duration():
            max_duration = max(self.get_duration(), other.get_duration)
            if max_duration:
                self.pad_time(max_duration)
                other.pad_time(max_duration)

        if self.mask and other.mask:
            new_mask = np.hstack((self.mask, other.mask))
        elif self.mask is None and other.mask is None:
            new_mask = None
        else:
            max_duration = max(self.get_duration(), other.get_duration)
            self.pad_time(max_duration)
            other.pad_time(max_duration)
            new_mask = np.hstack((self.mask, other.mask))


def assert_shape_equals(s1, s2):
    assert s1 == s2, "targets shape error: %s != %s" % (str(s1), str(s2))


class FramewiseTargets(Targets):
    """
    Provide a target value for every point in time. Although some timesteps
    might be masked out.
    """
    def __init__(self, targets, mask=None, binarize_to=None):
        super(FramewiseTargets, self).__init__('F', binarize_to, mask)
        assert (targets.ndim == 3) or (binarize_to and targets.ndim == 2)
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

    def index_time(self, item):
        t = self.data[item]
        m = self.mask[item] if self.mask is not None else None
        return FramewiseTargets(t, m, binarize_to=self.binarize_to)

    def trim(self):
        k = Targets.trim(self)
        self.data = self.data[:self.data.shape[0] - k, :, :]
        return k

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        Targets.validate_for_output_shape(self, timesteps, batchsize, out_size)
        if self.binarize_to:
            assert_shape_equals(self.data.shape, (timesteps, batchsize, 1))
        else:
            assert_shape_equals(self.data.shape,
                                (timesteps, batchsize, out_size))

    def __str__(self):
        return "<FramewiseTargets dim=%s>" % str(self.data.shape)

    def pad_time(self, target_length):
        super(FramewiseTargets, self).pad_time(target_length)
        t, b, f = self.data.shape
        assert target_length >= t
        padding = np.zeros((target_length - t, b, f))
        self.data = np.vstack((self.data, padding))

    def get_duration(self):
        return self.data.shape[0]

    def get_sequence_cnt(self):
        return self.data.shape[1]

    def join_with(self, other):
        new_mask = self._get_joint_mask(other)
        assert self.binarize_to == other.binarize_to
        assert self.targets_type == other.targets_type
        t1, b1, f1 = self.data.shape
        t2, b2, f2 = other.data.shape
        assert f1 == f2, 'Feature size has to match: %d != %d' % (f1, f2)
        new_t = max(t1, t2)
        pad1 = np.zeros((new_t - t1, b1, f1))
        pad2 = np.zeros((new_t - t2, b2, f2))
        padded_data1 = np.vstack((self.data, pad1))
        padded_data2 = np.vstack((other.data, pad2))
        new_data = np.hstack((padded_data1, padded_data2))
        return FramewiseTargets(new_data, new_mask, self.binarize_to)


class LabelingTargets(Targets):
    """
    Provide a list of labels for the sequence. If a mask is given, then the
    resulting deltas will be masked.
    """
    def __init__(self, labels, mask=None, binarize_to=None):
        super(LabelingTargets, self).__init__('L', binarize_to, mask)
        assert isinstance(labels, list)
        self.data = labels

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
        if self.mask is not None:
            assert_shape_equals(self.mask.shape, (timesteps, batchsize, 1))
        assert len(self.data) == batchsize

    def index_time(self, item):
        t = self.data
        m = self.mask[item] if self.mask is not None else None
        return LabelingTargets(t, m, binarize_to=self.binarize_to)

    def __str__(self):
        return "<LabelingTargets len=%d>" % len(self.data)

    def get_sequence_cnt(self):
        return len(self.data)

    def join_with(self, other):
        assert self.binarize_to == other.binarize_to
        assert self.targets_type == other.targets_type
        new_mask = self._get_joint_mask(other)
        new_data = self.data + other.data
        return FramewiseTargets(new_data, new_mask, self.binarize_to)


class SequencewiseTargets(Targets):
    """
    Provide one target per sequence. If no mask is given then only the last
    timestep will receive deltas. If a mask is given, then all the masked in
    timesteps will receive deltas.
    """
    def __init__(self, sequence_targets, mask=None, binarize_to=None):
        super(SequencewiseTargets, self).__init__('S', binarize_to, mask)
        assert sequence_targets.ndim == 2 or \
            (binarize_to and sequence_targets.ndim == 1)
        self.data = sequence_targets.reshape(sequence_targets.shape[0], -1)

    def __getitem__(self, item):
        s = self.data[item]
        m = self.mask[:, item, :] if self.mask is not None else None
        return SequencewiseTargets(s, m, binarize_to=self.binarize_to)

    def validate_for_output_shape(self, timesteps, batchsize, out_size):
        Targets.validate_for_output_shape(self, timesteps, batchsize, out_size)
        if self.binarize_to:
            assert_shape_equals(self.data.shape, (batchsize, 1))
        else:
            assert_shape_equals(self.data.shape, (batchsize, out_size))

    def index_time(self, item):
        t = self.data
        m = self.mask[item] if self.mask is not None else None
        return SequencewiseTargets(t, m, binarize_to=self.binarize_to)

    def get_sequence_cnt(self):
        return self.data.shape[1]

    def __str__(self):
        return "<SequencewiseTargets dim=%s>" % str(self.data.shape)

    def join_with(self, other):
        assert self.binarize_to == other.binarize_to
        assert self.targets_type == other.targets_type
        assert self.data.shape[1] == other.data.shape[1]
        new_mask = self._get_joint_mask(other)
        new_data = np.vstack((self.data, other.data))
        return SequencewiseTargets(new_data, new_mask, self.binarize_to)
