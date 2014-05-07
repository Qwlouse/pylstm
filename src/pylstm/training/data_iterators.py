#!/usr/bin/python
# coding=utf-8
"""
Data iterators take the data X, the targets T and provide a unified way of
iterating through them. They can divide the data into Minibatches, single
samples or leave it as one block, and they can shuffle it.

There are also modificators which can be stacked with a data iterator and which
modify or augment the data.
"""

from __future__ import division, print_function, unicode_literals
import math
import datetime

import numpy as np
from pylstm.randomness import Seedable

from pylstm.datasets.preprocessing import shuffle_data
from pylstm.targets import Targets


class ProgressBar(object):
    def __init__(self):
        self.start_time = datetime.datetime.utcnow()
        self.progress = 0
        self.progress_string = \
            "====1====2====3====4====5====6====7====8====9====0"
        self.prefix = '['
        self.suffix = '] Took: {0}'
        print(self.prefix, end='')

    def update_progress(self, fraction):
        assert 0.0 <= fraction <= 1.0
        new_progress = math.trunc(fraction * len(self.progress_string))
        if new_progress > self.progress:
            print(self.progress_string[self.progress:new_progress], end='')
            self.progress = new_progress
        if new_progress == len(self.progress_string):
            elapsed = datetime.datetime.utcnow() - self.start_time
            elapsed_str = str(elapsed)[:-5]
            print(self.suffix.format(elapsed_str), end='')


class Undivided(Seedable):
    """
    Processes the data in one block (only one iteration).
    It can shuffle the data.
    """

    def __init__(self, input_data, targets, shuffle=True, seed=None):
        """
        @param input_data: Batch of sequences. shape = (time, sample, feature)
        @type input_data: ndarray
        @param targets: Batch of sequences[shape = (time, sample, targets)]
           or list of labels
        @type targets: pylstm.targets.Target
        @param shuffle: if this is true(default) then the data will be shuffled.
        @type shuffle: bool
        @param seed: if set this drives the randomness of the shuffle.
        @type seed: int | None
        """
        super(Undivided, self).__init__(seed=seed, category='data_iterator')
        assert isinstance(targets, Targets)
        self.input_data = input_data
        self.targets = targets
        self.shuffle = shuffle

    def __call__(self):
        if self.shuffle:
            input_data, targets, _ = shuffle_data(self.input_data, self.targets,
                                                  seed=self.rnd.generate_seed())
        else:
            input_data, targets = self.input_data, self.targets
        yield input_data, targets


class Minibatches(Seedable):
    """
    Minibatch (batch_size samples at a time) iterator for inputs and targets.
    Argument verbose=True enables a progress bar.
    """
    def __init__(self, input_data, targets, batch_size=4, shuffle=True,
                 verbose=False, seed=None):
        super(Minibatches, self).__init__(seed=seed, category='data_iterator')
        self.input_data = input_data
        assert isinstance(targets, Targets)
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

    def __call__(self):
        p = ProgressBar() if self.verbose else None
        nr_sequences = self.input_data.shape[1]

        if self.shuffle:
            input_data, targets, _ = shuffle_data(self.input_data, self.targets,
                                                  seed=self.rnd.generate_seed())
        else:
            input_data, targets = self.input_data, self.targets

        for i in range(0, nr_sequences, self.batch_size):
            j = min(i + self.batch_size, nr_sequences)
            t = targets[i:j]
            new_end = input_data.shape[0] - t.trim()
            x = input_data[:new_end, i:j, :]
            yield x, t
            if self.verbose:
                p.update_progress((j+1)/nr_sequences)


class Online(Seedable):
    """
    Online (one sample at a time) iterator for inputs, targets and masks.
    Argument verbose=True enables a progress bar.
    """

    def __init__(self, input_data, targets, shuffle=True, verbose=False,
                 seed=None):
        super(Online, self).__init__(seed=seed, category='data_iterator')
        self.input_data = input_data
        assert isinstance(targets, Targets)
        self.targets = targets
        self.shuffle = shuffle
        self.verbose = verbose

    def __call__(self):
        p = ProgressBar() if self.verbose else None
        nr_sequences = self.input_data.shape[1]
        indices = np.arange(nr_sequences)
        if self.shuffle:
            self.rnd.shuffle(indices)
        for i, idx in enumerate(indices):
            targets = self.targets[idx:idx+1]
            new_end = self.input_data.shape[0] - targets.trim()
            input_data = self.input_data[:new_end, idx:idx+1, :]
            yield input_data, targets
            if self.verbose:
                p.update_progress((i+1)/nr_sequences)


################## Dataset Modifications ######################
class Noisy(Seedable):
    """
    Adds noise to each input sample.
    Can be applied to any iterator (Online, Minibatches or Undivided).
    """

    def __init__(self, data_iter, std=0.1, seed=None):
        super(Noisy, self).__init__(seed=seed, category='data_iterator')
        self.f = data_iter
        self.std = std

    def __call__(self):
        for x, t in self.f():
            x_noisy = x + self.rnd.randn(*x.shape) * self.std
            yield x_noisy, t


class FrameDrop(Seedable):
    """
    Retains only a fraction of time frames of the time fraction.
    Thus, the lengths of sequences are changed.

    NOTE: Designed for use with the Online iterator only.
    If you use it with other iterators, keep in mind that the dropped frames
    will be the same ones for each sequence in your batches.
    """

    def __init__(self, data_iter, keep_fraction=0.9, seed=None):
        super(FrameDrop, self).__init__(seed=seed, category='data_iterator')
        self.data_iter = data_iter
        self.fraction = keep_fraction

    def __call__(self):
        for x, t in self.data_iter():
            drop_mask = self.rnd.random_sample(x.shape[0]) < self.fraction
            x_drop = x[drop_mask, :, :]
            if t.is_framewise():
                t_drop = t.index_time(drop_mask)
            else:
                t_drop = t
            yield x_drop, t_drop
