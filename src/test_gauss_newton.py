#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from pylstm import NetworkBuilder, RegularLayer, RnnLayer, LstmLayer, ReverseLayer
import numpy as np


def estimate_jacobian(net, X=None, nr_timesteps=4, nr_batches=1, rnd=np.random.RandomState()):
    out_size = net.get_output_size()
    in_size = net.get_input_size()
    if X is None:
        X = rnd.randn(nr_timesteps, nr_batches, in_size)
    nr_timesteps, nr_batches, _ = X.shape

    out = net.forward_pass(X).copy()
    J = np.zeros((out_size * nr_timesteps, net.get_param_size()))

    for i in range(out_size * nr_timesteps):
        out_masked = np.zeros_like(out).flatten()
        out_masked[i] = 1.0
        out_masked = out_masked.reshape(*out.shape)

        deltas = net.pure_backpass(out_masked)
        J[i, :] = net.calc_gradient().flatten()
    return J






rnd = np.random.RandomState(2893749283)

in_size = 2
out_size = 2
nr_batches = 1
nr_timesteps = 4

netb = NetworkBuilder()
#netb.input(in_size) >> LstmLayer(out_size, act_func='linear') >> netb.output
netb.input(in_size) >>  RegularLayer(out_size, act_func='linear') >> netb.output
net = netb.build()


net.param_buffer = rnd.randn(net.get_param_size())
X1 = rnd.randn(nr_timesteps, nr_batches, in_size)
X2 = rnd.randn(nr_timesteps, nr_batches, in_size)


J1 = estimate_jacobian(net, X1, rnd=rnd)
J2 = estimate_jacobian(net, X2, rnd=rnd)

G1 = J1.T.dot(J1)
G2 = J2.T.dot(J2)
print(G1)
v = np.arange(net.get_param_size())

o1 = net.hessian_pass(X1, v).flatten()
p1 = G1.dot(v).flatten()

print('hessian_pass')
print(o1)
print('jacobi approx')
print(p1)
print('difference')
print(o1 - p1)

print('Sum of squares error', np.sum(np.abs(o1-p1)))
print('\n\n')

o2 = net.hessian_pass(X2, v).flatten()
p2 = G2.dot(v).flatten()

print('hessian_pass')
print(o2)
print('jacobi approx')
print(p2)
print('difference')
print(o2 - p2)

print('Sum of squares error', np.sum(np.abs(o2-p2)))
print('\n\n')

print('added Hessians')








# print('\n\n')
# net.param_buffer = o-p
# for l in net.layers.keys()[1:-1]:
#     print(l)
#     w = net.get_param_view_for(l)
#     for v in w.keys():
#         print("  ", v)
#         print(w[v])
#



