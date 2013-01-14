#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pylstm_wrapper as pw

t = 1 # time 
b = 1 # batches
n = 5 # input size
m = 3 # output size

l = pw.LstmLayer(n, m)
X = pw.MatrixCPU(t, b, n)
Y = pw.MatrixCPU(t, b, m)
wm = pw.MatrixCPU(1, 1, l.get_param_size())
W = l.create_param_view(wm)
im = pw.MatrixCPU(t, b, l.get_internal_state_size(t, b))
I = l.create_internal_view(im)
l.forward(W, I, X, Y)
Y.print_me()

