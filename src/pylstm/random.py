#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import deepcopy
import numpy as np


class SeedGenerator(object):
    def __init__(self, seed, seed_range=(0, 1000000000)):
        self.seed = seed
        self.seed_range = seed_range
        self.categories = None
        self.rnd = None
        self.reset()

    def reset(self, seed=None):
        self.categories = dict()
        if seed is not None:
            self.seed = seed
        self.rnd = self._get_or_create_random_state_for_category("self")

    def get_seed(self, category):
        rnd = self._get_or_create_random_state_for_category(category)
        return rnd.randint(*self.seed_range)

    def get_seeder(self, category, seed=None):
        if seed is None:
            seed = self.get_seed(category)

        return SeedGenerator(seed)

    def _get_or_create_random_state_for_category(self, category):
        if category not in self.categories:
            rnd = np.random.RandomState(abs(hash(str(self.seed) + '$' +
                                                 str(category))))
            self.categories[category] = rnd
        return self.categories[category]


class Seedable(object):
    def __init__(self, seed=None):
        self.seeder = None
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seeder = get_seeder_for('initializers', seed=seed)


def reseeding_deepcopy(values, seed):
    r = deepcopy(values)
    if isinstance(r, Seedable):
        r.set_seed(seed)
    return r


### used categories:
# - preprocessing
# - datasets
# * initializers
# * weight_constraints
# - network
GLOBAL_SEEDER = SeedGenerator(np.random.randint(0, 1000000000))

set_global_seed = GLOBAL_SEEDER.reset

get_seeder_for = GLOBAL_SEEDER.get_seeder





