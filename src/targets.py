#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import copy
import numpy as np


def create_targets_object(T):
    if isinstance(T, Targets):
        return T

    if isinstance(T, np.ndarray):
        if len(T.shape) == 3:
            return FramewiseTargets(T)
        if len(T.shape) == 1:
            return ClassificationTargets(T)
    
    elif isinstance(T, list):
        return LabelingTargets(T)
        

class Targets(object):
    def __getitem__(self, item):
        raise NotImplementedError()

    def crop_time(self, end):
        raise NotImplementedError()
        
        
class FramewiseTargets(Targets):
    def __init__(self, T):
        self.T = T

    def __getitem__(self, item):
        return FramewiseTargets(self.T[:, item, :])

    def crop_time(self, end):
        return FramewiseTargets(self.T[:end, :, :])


class LabelingTargets(Targets):
    def __init__(self, L):
        self.L = L

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            return LabelingTargets(self.L[item])
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            assert len(item) == len(self.L)
            return FramewiseTargets([self.L[i] for i in item])
        else:
            raise ValueError("Indexing with type '%s' unsupported" % type(item))

    def crop_time(self, end):
        return self
        

class ClassificationTargets(Targets):
    def __init__(self, C):
        self.C = C

    def __getitem__(self, item):
        return ClassificationTargets(self.C[item])

    def crop_time(self, end):
        return self
