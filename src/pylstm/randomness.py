#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import deepcopy, copy
import numpy as np


class HierarchicalRandomState(np.random.RandomState):
    """
    An extension of the numpy RandomState that saves it's own seed and allows to
    create sub-RandomStates like this:
    >> sub_rnd = rnd['sub1']

    The sub-RandomStates depend on the seed of their creator, but are otherwise
    as independent as possible. That means that the seed of the sub-RandomState
    is only dependent on the seed of the creator and on it's name, not on the
    random state of the creator.
    """
    def __init__(self, seed=None, seed_range=(0, 1000000000)):
        if seed is None:
            seed = np.random.randint(*seed_range)
        super(HierarchicalRandomState, self).__init__(seed)
        self._seed = seed
        self._default_seed_range = seed_range
        self.categories = dict()

    def seed(self, seed=None):
        """
        This method is kept for compatibility with the numpy RandomState. But
        for better readability you are encouraged to use the set_seed() method
        instead.

        :param seed: the seed to reseed this random state with.
        :type seed: int
        """
        super(HierarchicalRandomState, self).seed(seed)
        self._seed = seed
        self.categories = dict()

    def get_seed(self):
        return self._seed

    def set_seed(self, seed):
        self.seed(seed)

    def generate_seed(self, seed_range=None):
        if seed_range is None:
            seed_range = self._default_seed_range
        return self.randint(*seed_range)

    def get_new_random_state(self, seed=None):
        if seed is None:
            seed = self.generate_seed()

        return HierarchicalRandomState(seed)

    def __getitem__(self, item):
        if item not in self.categories:
            seed = abs(hash(str(self._seed) + '$' + str(item)))
            self.categories[item] = HierarchicalRandomState(seed)
        return self.categories[item]


class Seedable(object):
    def __init__(self, seed=None):
        self.rnd = HierarchicalRandomState(seed)
        self.seed = self.rnd.get_seed()

    def set_seed(self, seed):
        self.rnd.set_seed(seed)
        self.seed = seed


def reseeding_deepcopy(values, seed):
    r = deepcopy(values)
    if isinstance(r, (HierarchicalRandomState, Seedable)):
        r.set_seed(seed)
    return r


def reseeding_copy(values, seed):
    r = copy(values)
    if isinstance(r, (HierarchicalRandomState, Seedable)):
        r.set_seed(seed)
    return r

### used categories:
# - preprocessing
# - datasets
# * initializers
# * weight_constraints
# - network
global_rnd = HierarchicalRandomState(np.random.randint(0, 1000000000))

set_global_seed = global_rnd.set_seed




