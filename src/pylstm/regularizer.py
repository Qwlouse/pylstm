#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
sys.path.append('.')
sys.path.append('..')

class WeightRegularizer(object):
    def __init__(self, net = None, lambda_L2 = 0, lambda_L1 = 0, norm = 0):
        if net is None:
            return
        self.full_L1 = (lambda_L1 > 0)
        self.full_L2 = (lambda_L2 > 0)
        self.full_norm = (norm > 0)
        self.selective_L1 = False
        self.selective_L2 = False
        
        self.lambda_full_L1 = lambda_L1
        self.lambda_full_L2 = lambda_L2
        self.norm_full = norm
        
        self.selected_set = {}
        self.mask = np.ones(net.get_param_size())

    def add_selected_layer(self, layer_name, layer_params, regularizer_type, layer_lambda):
        pass
    
    def create_mask(self):
        self.mask = np.zeros(net.get_param_size())