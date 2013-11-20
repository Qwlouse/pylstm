#!/usr/bin/python
# coding=utf-8
"""
Data iterators take the data X, the targets T and optionally the mask M and
provide a unified way of iterating through them. They can divide the data into
Minibatches, single samples or leave it as one block, and they can shuffle it.

There are also modificators which can be stacked with a data iterator and which
modify or augment the data.
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
from pylstm import shuffle_data


class Undivided(object):
    """
     Iterates through the data in one block (only one iteration). But it can
     shuffle the data.
     :param X: Data: Batch of sequences. shape = (time, sample, feature)
     :param T: Targets: Batch of sequences[shape = (time, sample, targets)]
                        or list of labels
     :param M: Masks: Batch of sequences. shape = (time, sample, 1).
                      Can be None(default).
     :param shuffle: if this is true(default) then the data will be shuffled.
    """
    def __init__(self, X, T, M=None, shuffle=True):
        self.X = X
        self.T = T
        self.M = M
        self.shuffle = shuffle

    def __call__(self):
        X, T, M, _ = shuffle_data(self.X, self.T, self.M) if self.shuffle else \
                     self.X, self.T, self.M, None
        yield X, T, M


class Minibatches(object):
    def __init__(self, X, T, M=None, batch_size=1, shuffle=True, verbose=True):
        self.X = X
        self.T = T
        self.M = M
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

    def __call__(self):
        if self.verbose:
            _update_progress(0)
        total_batches = self.X.shape[1]

        X, T, M, _ = shuffle_data(self.X, self.T, self.M) if self.shuffle else \
                     self.X, self.T, self.M, None

        for i in range(0, total_batches, self.batch_size):
            j = min(i + self.batch_size, total_batches)
            x = X[:, i:j, :]
            t = T[i:j] if isinstance(T, list) else T[:, i:j, :]
            m = None if M is None else M[:, i:j, :]
            yield x, t, m
            if self.verbose:
                _update_progress(i/total_batches)


class Online(object):
    def __init__(self, X, T, M=None, shuffle=True, verbose=True):
        self.X = X
        self.T = T
        self.M = M
        self.shuffle = shuffle
        self.verbose = verbose

    def __call__(self):
        if self.verbose:
            _update_progress(0)
        total_batches = self.X.shape[1]
        indices = np.arange(total_batches)
        if self.shuffle:
            np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            x = self.X[:, idx:idx+1, :]
            t = self.T[idx:idx+1] if isinstance(self.T, list) else \
                self.T[:, idx:idx+1, :]
            m = None
            if self.M is not None:
                m = self.M[:, idx:idx+1, :]
                for k in range(m.shape[0] - 1, -1, -1):
                    if m[k, 0, 0] != 0:
                        x = x[:k + 1, :, :]
                        t = t[:k + 1, :, :] if not isinstance(t, list) else t
                        m = m[:k + 1, :, :]
                        break
            yield x, t, m
            if self.verbose:
                _update_progress((i+1)/total_batches)


def _update_progress(progress):
    barLength = 50  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}{1}] {2}% {3}".format("#" * block,
                                                  "-" * (barLength-block),
                                                  round(progress*100, 2),
                                                  status)
    sys.stdout.write(text)
    sys.stdout.flush()


################## Dataset Modifications ######################
class Noisy(object):
    def __init__(self, data_iter, std=0.1, rnd=np.random.RandomState()):
        self.f = data_iter
        self.rnd = rnd
        self.std = std

    def __call__(self):
        for x, t, m in self.f():
            x_noisy = x + self.rnd.randn(*x.shape) * self.std
            yield x_noisy, t, m


class FrameDrop(object):
    """
    Retains only a fraction of time frames of the time fraction.
    Thus, the lengths of sequences are changed.
    Uses a fixed seed by default to keep reproducibility

    NOTE: Designed for use with the Online iterator only.
    If you use it with other iterators, keep in mind that the dropped frames
    will be the same ones for each sequence in your batches.
    """
    def __init__(self, data_iter, keep_fraction=0.9, rnd=np.random.RandomState(42), drop_targets=True):
        self.data_iter = data_iter
        self.fraction = keep_fraction
        self.rnd = rnd
        self.drop_targets = drop_targets

    def __call__(self):
        for x, t, m in self.data_iter():
            mask = np.random.random_sample(x.shape[0]) < self.fraction
            x_drop = x[mask, :, :]
            if self.drop_targets:
                t_drop = t[mask, :, :]
            else:
                t_drop = t
            m_drop = m[mask, :, :]
            yield x_drop, t_drop, m_drop
