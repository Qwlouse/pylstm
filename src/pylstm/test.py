#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import py_matrix as pm
import py_lstm_layer as pl
t = 1 # time 
b = 1 # batches
n = 5 # input size
m = 3 # output size

l = pl.LstmLayer(n, m)
X = pm.MatrixCPU(t, b, n)
Y = pm.MatrixCPU(t, b, m)
wm = pm.MatrixCPU(1, 1, l.get_param_size())
W = l.create_param_view(wm)
im = pm.MatrixCPU(t, b, l.get_internal_state_size(t, b))
I = l.create_internal_view(im)
l.forward(W, I, X, Y)
Y.print_me()

