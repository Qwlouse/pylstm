#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')

class L1(object):
    def __init__(self, reg_coeff=0.01):
        self.reg_coeff = 0.01
        
    def __call__(self, view):
        return self.reg_coeff*np.sign(view)
        

class L2(object):
    def __init__(self, reg_coeff=0.01):
        self.reg_coeff = 0.01
        
    def __call__(self, view):
        return self.reg_coeff*view
        