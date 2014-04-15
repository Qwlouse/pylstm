# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import print_function, division, unicode_literals
from pylstm import *
import numpy as np
set_global_seed(4242)

# Prepare data

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist.data.shape = (1,) + mnist.data.shape
mnist.target.shape = (1,) + mnist.target.shape + (1,)

# shuffle first since data is ordered
rnd = np.random.RandomState(42)
shuffling = rnd.permutation(arange(60000))
train_valid_inputs = mnist.data[:, shuffling, :]/255.0
train_valid_targets = binarize_array(mnist.target[:, shuffling, :])

train_inputs = train_valid_inputs[:, 0:50000, :]
train_targets = train_valid_targets[:, 0:50000, :]
valid_inputs = train_valid_inputs[:, 50000:60000, :]
valid_targets = train_valid_targets[:, 50000:60000, :]

test_inputs = mnist.data[:, 60000:, :]/255.0
test_targets = binarize_array(mnist.target[:, 60000:, :])

# <codecell>

print("Shapes of training, validation and test set inputs")
print(train_inputs.shape, valid_inputs.shape, test_inputs.shape)
print("Shapes of training, validation and test set targets")
print(train_targets.shape, valid_targets.shape, test_targets.shape)

# <codecell>

# Build Network
# DropoutLayer(drop_prob=0.5) >>
network = build_net(InputLayer(784) >>
                    ForwardLayer(100, act_func="relu", name="h1") >>
                    ForwardLayer(10, act_func="softmax", name="out"))
network.error_func = MultiClassCrossEntropyError
network.initialize({"default": Gaussian(0.0, 0.03)}, seed=25)
network.set_constraints({"h1": {"HX": LimitIncomingWeightsSquared(5)},
                         "out": {"HX": LimitIncomingWeightsSquared(5)},
                         })

# Build trainer

trainer = Trainer(stepper=MomentumStep(learning_rate=ExponentialSchedule(initial_value=0.1, factor=0.98, minimum=0.001, interval=3000),
                                       momentum=LinearSchedule(initial_value=0.5, final_value=0.95, num_changes=10, interval=3000)))
#trainer.stopping_criteria.append(ValidationErrorRises(delay=10))
trainer.stopping_criteria.append(MaxEpochsSeen(max_epochs=10))
trainer.monitor["validation error"] = MonitorClassificationError(data_iter=
                                                                 Minibatches(valid_inputs,
                                                                             FramewiseTargets(valid_targets), 
                                                                             batch_size=20, verbose=False, shuffle=False),
                                                                 name='Validation Set', timescale='epoch', interval=1)
trainer.monitor["test error"] = MonitorClassificationError(data_iter=
                                                           Minibatches(test_inputs,
                                                                       FramewiseTargets(test_targets), 
                                                                       batch_size=20, verbose=False, shuffle=False),
                                                           name='Test Set', timescale='epoch', interval=1)
for layer_name in network.layers.keys():
    if layer_name != 'InputLayer':
        trainer.monitor[layer_name + " monitor"] = MonitorLayerProperties(network, layer_name)
trainer.monitor["printing"] = PrintError()

train_getter = Minibatches(train_inputs, FramewiseTargets(train_targets), batch_size=20, verbose=False)
valid_getter = Minibatches(valid_inputs, FramewiseTargets(valid_targets), batch_size=20, verbose=False)

# <codecell>

# Train

trainer.train(network, train_getter, valid_getter)

# <codecell>


