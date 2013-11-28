#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import deepcopy
import numpy as np

SEED_RANGE = 0, 1000000000


class SeedGenerator(object):
    def __init__(self, seed):
        self.seed = seed
        self.categories = None
        self.reset()

    def reset(self, initializer=None):
        self.categories = dict()
        if initializer is not None:
            self.seed = initializer

    def get_seed(self, category):
        if category not in self.categories:
            self.categories[category] = np.random.RandomState(
                abs(hash(str(self.seed) + '$' + str(category))))
        return self.categories[category].randint(*SEED_RANGE)


### used categories:
# - preprocessing
# - datasets
# * initializers
# * weight_constraints
# - network
GLOBAL_SEEDER = SeedGenerator(np.random.randint(*SEED_RANGE))


def set_global_seed(seed):
    GLOBAL_SEEDER.reset(seed)


def get_next_seed_for(category, seed=None):
    if seed is not None:
        return seed

    return GLOBAL_SEEDER.get_seed(category)


def get_random_state_for(category, seed=None):
    if seed is None:
        seed = get_next_seed_for(category)

    return np.random.RandomState(hash(seed))


def get_seeder_for(category, seed=None):
    if seed is None:
        seed = get_next_seed_for(category)

    return seed, SeedGenerator(seed)


class Seedable(object):
    def __init__(self, seed=None):
        self.seed = None
        self.rnd = None
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = get_next_seed_for('initializers', seed=seed)
        self.rnd = get_random_state_for('initializers', self.seed)


def reseeding_deepcopy(values, seed):
    r = deepcopy(values)
    if isinstance(r, Seedable):
        r.set_seed(seed)
    return r


