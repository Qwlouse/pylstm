#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import sys


class Undivided(object):
    def __init__(self, X, T, M=None, shuffle=True):
        self.X = X
        self.T = T
        self.M = M
        self.shuffle = shuffle

    def __call__(self):
        total_batches = self.X.shape[1]
        if self.shuffle:
            indices = np.arange(total_batches)
            np.random.shuffle(indices)
            self.X = self.X[:, indices, :]
            self.T = self.T[:, indices, :]
            self.M = None if self.M is None else self.M[:, indices, :]
        yield self.X, self.T, self.M


class Minibatches(object):
    def __init__(self, X, T, M=None, shuffle=True, batch_size=1):
        self.X = X
        self.T = T
        self.M = M
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __call__(self):
        i = 0
        _update_progress(0)
        total_batches = self.X.shape[1]
        if self.shuffle:
            indices = np.arange(total_batches)
            np.random.shuffle(indices)
            self.X = self.X[:, indices, :]
            self.T = self.T[:, indices, :]
            self.M = None if self.M is None else self.M[:, indices, :]
        while i < total_batches:
            j = min(i + self.batch_size, total_batches)
            x = self.X[:, i:j, :]
            t = self.T[i:j] if isinstance(self.T, list) else self.T[:, i:j, :]
            m = None if self.M is None else self.M[:, i:j, :]
            yield x, t, m
            i += self.batch_size
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
