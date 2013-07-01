#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')

class WeightRegularizer(object):
        # Use either L1/L2 or max-norm at a time
    def __init__(self, type, net = None, lambda_L2 = 0, lambda_L1 = 0, max_norm = 0):
        # Type is 'elastic' or 'max-norm' in general.
        # If you provide L1 or L2 lambdas or norm, they are used for all weights
        # and type is changed to reflect this.
        if net is None:
            return
        self.full_L1 = (lambda_L1 > 0)
        self.full_L2 = (lambda_L2 > 0)
        self.full_norm = (max_norm > 0)
        self.selective_L1 = False
        self.selective_L2 = False
        self.type = type
        
        self.lambda_full_L1 = lambda_L1
        self.lambda_full_L2 = lambda_L2
        self.norm_full = max_norm
        
        if (self.full_L1 or self.full_L2) and self.full_norm:
            raise Exception("Don't use elastic and max-norm regularization together!")
        if self.full_L1 or self.full_L2:
            self.type = 'full elastic'
            self.L1_mask = np.ones(net.get_param_size())
            self.L2_mask = np.ones(net.get_param_size())
        elif self.full_norm:
            self.type = 'full max-norm'
            
        self.selected_set = {}
        
    def add_specific_weights(self, layer_name, layer_params, regularizer_type, layer_lambda):
        pass
    
    def create_mask(self):
        self.mask = np.zeros(net.get_param_size())
        # Create mask from selected weight sets
        
    def get_regularization_grad(self, net):
        if self.type.lower() == 'full elastic' or self.type.lower() == 'elastic':
            return self.lambda_full_L1 * np.sign(net.param_buffer) * self.L1_mask + self.lambda_full_L2 * net.param_buffer * self.L2_mask
        elif self.type.lower() == 'full max-norm' or self.type.lower() == 'max-norm':
            return np.zeros(net.get_param_size())
        raise Exception("Regularizer type not recognized for gradient.")
    
    def apply_max_norm(self, net):
        if (self.type.lower() != 'full max-norm') and (self.type.lower() != 'max-norm'):
            return
        else:
            raise NotImplementedError