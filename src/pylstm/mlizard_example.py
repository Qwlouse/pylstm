#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import mlizard
from netbuilder import NetworkBuilder
from layers import LstmLayer, FwdLayer
from datasets import generate_memo_problem
from trainer import SgdTrainer
import matplotlib.pyplot as plt
conf = """
# Control the randomness or comment out to leave it to
seed=290882

# Configuration for some of the parameters
timesteps = 30
learning_rate=0.001
epochs=1000
"""
ex = mlizard.createExperiment("LSTM 5Bit Problem", config_string=conf)


@ex.stage
def create_network(rnd):
    netb = NetworkBuilder()
    netb.input(2) >> LstmLayer(50) >> FwdLayer(2) >> netb.output
    net = netb.build()
    net.set_param_buffer(rnd.randn(net.get_param_size()))
    return net


@ex.stage
def create_5bit_problem(timesteps):
    X, T = generate_memo_problem(5,  2, 32, timesteps)
    return X, T


@ex.stage
def train(net, X, T, learning_rate, epochs, logger):
    t = SgdTrainer(learning_rate=learning_rate)
    t.train(net, X, T, epochs=epochs,
            callback=lambda e, err: logger.append_result(error=err))


@ex.live_plot
def plot_error():
    fig, ax = plt.subplots()
    yield fig
    while True:
        results = yield
        ax.clear()
        ax.plot(results['error'], '-r')


@ex.main
def main():
    net = create_network()
    X, T = create_5bit_problem()
    train(net, X, T)





