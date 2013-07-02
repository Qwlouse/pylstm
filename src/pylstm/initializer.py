#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')

class WeightInitializer(object):
    """
    Neural Network Weight Initializer
    Usage:
    wi = WeightInitializer(network)
    wi.add_init({'RnnLayer': {'HR': 'gaussian', 'H_bias': ['uniform', 0, 1]},
                'LstmLayer': {'FH': 'uniform'}})
    wi.initialize()
    
    Supported initializations: 'zeros', 'gaussian', 'uniform'.
    You can specify mean and variance for gaussian as ['gaussian', mean, variance].
    You can specify lower and upper bound for uniform as ['uniform', lowb, upb].
    You can use set_default() to set the initialization for all params which are not added by you.
    """
                
    def __init__(self, net):
        self.network = net
        self.init_set = {}
        self.default_init = 'zeros'
        for layer_name in net.layers.keys():
            if layer_name.lower() != 'input' and layer_name.lower() != 'output':
                self.init_set[layer_name] = {}
                for param_name in net.get_param_view_for(layer_name).keys():
                    self.init_set[layer_name][param_name] = 'default'
        
    def add_init(self, layer_init_dict):
        for layer_name in layer_init_dict.keys():
            if layer_name in self.init_set:
                for param_name in layer_init_dict[layer_name].keys():
                    if param_name in self.init_set[layer_name]:
                        self.init_set[layer_name][param_name] = layer_init_dict[layer_name][param_name]
                    else:
                        raise KeyError("Parameter " + param_name + " does not exist in layer " + layer_name)   
            else:
                raise KeyError("Layer " + layer_name + " does not exist.")
    
    def initialize(self):
        for layer_name in self.init_set.keys():
            for param_name in self.init_set[layer_name].keys():
                self.initialize_param(layer_name, param_name, self.init_set[layer_name][param_name])
                
    def initialize_param(self, layer_name, param_name, param_init):
        if param_init == 'default':
            init_type = self.default_init
        param_shape = self.network.get_param_view_for(layer_name)[param_name].shape
        param_size  = self.network.get_param_view_for(layer_name)[param_name].size
        init_type = param_init[0] if (type(param_init) is list) else param_init
        if init_type.lower() == 'zeros':
            self.network.get_param_view_for(layer_name)[param_name][:] = 0
        elif init_type.lower() == 'gaussian':
            if type(param_init) is list:
                mu = param_init[1]
                sigma = param_init[2]
            else:
                mu    = 0
                sigma = 0.1
            self.network.get_param_view_for(layer_name)[param_name][:] = (np.random.randn(param_size).reshape(param_shape) * sigma) + mu
            print(self.network.get_param_view_for(layer_name)[param_name])
            
        elif init_type.lower() == 'uniform':
            if type(param_init) is list:
                lowb = param_init[1]
                upb  = param_init[2]
            else:
                lowb = -0.1
                upb  =  0.1
            self.network.get_param_view_for(layer_name)[param_name][:] = ((upb - lowb) * np.random.rand(param_size).reshape(param_shape)) + lowb
        else:
            raise NotImplementedError
        
    def set_default(self, default_init):
        self.default_init = default_init
        