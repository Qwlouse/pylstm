#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import deepcopy
import numpy as np


class HierarchicalRandomState(np.random.RandomState):
    def __init__(self, seed=None, seed_range=(0, 1000000000)):
        if seed is None:
            seed = np.random.randint(seed_range)
        super(HierarchicalRandomState, self).__init__(seed)
        self._seed = seed
        self._default_seed_range = seed_range
        self.categories = dict()

    def seed(self, seed=None):
        super(HierarchicalRandomState, self).seed(seed)
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
            seed = abs(hash(str(self.seed) + '$' + str(item)))
            self.categories[item] = HierarchicalRandomState(seed)
        return self.categories[item]


class Seedable(object):
    def __init__(self, seed=None):
        self.rnd = None
        self.set_seed = HierarchicalRandomState(seed)

    def set_seed(self, seed):
        self.rnd.set_seed(seed)


def reseeding_deepcopy(values, seed):
    r = deepcopy(values)
    if isinstance(r, HierarchicalRandomState):
        r.set_seed(seed)
    return r


### used categories:
# - preprocessing
# - datasets
# * initializers
# * weight_constraints
# - network
GLOBAL_RND = HierarchicalRandomState(np.random.randint(0, 1000000000))

set_global_seed = GLOBAL_RND.set_seed




