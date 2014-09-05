.. _quickstart:

**********
Quickstart
**********

.. note::

    - It is suggested that you execute the code blocks in the tutorial one by one in an IPython Notebook to familiarize yourself with the library.
    - Names of modules/functions etc. are highlighted in **bold**. Names of Brainstorm specific concepts are highlighted in *italics*.

.. _getdata:

Getting MNIST data
==================
As per the usual custom from which we dare not deviate,
we will demonstrate the use of feedforward networks using a single experiment
on the MNIST dataset.
To fetch the dataset, we will use the **sklearn.datasets**
module from scikit-learn. To install
scikit-learn, (in a Terminal) do::

    [sudo] pip install scikit-learn

The following Python code fetches the MNIST dataset and converts it to a format which can be used
by Brainstorm. It uses the **binarize_array** utility provided in Brainstorm to convert the class
labels to one-hot vectors::

    from __future__ import print_function, division, unicode_literals
    from sklearn.datasets import fetch_mldata
    import numpy as np
    from pylstm import *

    mnist = fetch_mldata('MNIST original')
    mnist.data.shape = (1,) + mnist.data.shape
    mnist.target.shape = (1,) + mnist.target.shape + (1,)

    # Shuffle before splitting into training and validation sets since data is ordered
    # Divide pixel values by 255 to convert to range [0, 1]
    rnd = np.random.RandomState(42)
    shuffling = rnd.permutation(arange(60000))
    train_valid_inputs = mnist.data[:, shuffling, :]/255.0
    train_valid_targets = binarize_array(mnist.target[:, shuffling, :])

    # Split training examples into training and validation sets
    train_inputs = train_valid_inputs[:, 0:50000, :]
    train_targets = train_valid_targets[:, 0:50000, :]
    valid_inputs = train_valid_inputs[:, 50000:60000, :]
    valid_targets = train_valid_targets[:, 50000:60000, :]

    test_inputs = mnist.data[:, 60000:, :]/255.0
    test_targets = binarize_array(mnist.target[:, 60000:, :])

    print("Shapes of training, validation and test set inputs")
    print(train_inputs.shape, valid_inputs.shape, test_inputs.shape)
    print("Shapes of training, validation and test set targets")
    print(train_targets.shape, valid_targets.shape, test_targets.shape)

You should see the following output::

    Shapes of training, validation and test set inputs
    (1, 50000, 784) (1, 10000, 784) (1, 10000, 784)
    Shapes of training, validation and test set targets
    (1, 50000, 10) (1, 10000, 10) (1, 10000, 10)

Our dataset is now ready for experimentation.

.. _logistic:

Logistic regression example
===========================
A multinomial logistic regression model for classifying MNIST digits can be implemented as simple
neural network with a softmax output layer and zero hidden layers.
Here is how you can train such a model.
You can run this example without worrying about what each line does. Just note the main steps::


    # Build Network
    network = build_net(InputLayer(784) >> ForwardLayer(10, act_func="softmax", name="out"))
    network.initialize({"default": Gaussian(0.03)}, seed=25)

    # Build trainer and iterators
    network.error_func = MultiClassCrossEntropyError
    trainer = Trainer(stepper=MomentumStep(learning_rate=0.1, momentum=0.5))

    train_iter = Minibatches(train_inputs, FramewiseTargets(train_targets), batch_size=20, verbose=False)
    valid_iter = Minibatches(valid_inputs, FramewiseTargets(valid_targets), batch_size=20, verbose=False, shuffle=False)

    # Add monitoring
    trainer.add_monitor(MaxEpochsSeen(max_epochs=5))
    trainer.add_monitor(MonitorClassificationError(data_name='valid_iter', name='validation_error', timescale='epoch', interval=1))

    # Train
    trainer.train(network, train_iter, valid_iter=valid_iter)

This will produce output like the following::

     - - - - - - - - - - - - - - -  Pretraining  - - - - - - - - - - - - - - -
    validation_error                        : 89.32


     - - - - - - - - - - - - - - -  Epoch 1  - - - - - - - - - - - - - - -
    training_errors                         : 0.397256399552
    validation_error                        : 9.13


     - - - - - - - - - - - - - - -  Epoch 2  - - - - - - - - - - - - - - -
    training_errors                         : 0.309033808259
    validation_error                        : 8.75


     - - - - - - - - - - - - - - -  Epoch 3  - - - - - - - - - - - - - - -
    training_errors                         : 0.293025995252
    validation_error                        : 8.33


     - - - - - - - - - - - - - - -  Epoch 4  - - - - - - - - - - - - - - -
    training_errors                         : 0.284061853941
    validation_error                        : 8.43


     - - - - - - - - - - - - - - -  Epoch 5  - - - - - - - - - - - - - - -
    training_errors                         : 0.278669556951
    >> Stopping because: Max epochs reached.
    validation_error                        : 7.99

The output shows that the validation error monitor was first activated before training to record its
value for the untrained network during the *Pretraining* phase. Thereafter,
the monitored validation error is evaluated and printed after each epoch, in addition to
the running average of the loss function which is computed during training.

Here is a high-level overview of what we did above:

-
    We first built our network using the **build_net** function and initialized
    its weights using a Gaussian distribution. The network is ready for performing forward passes on
    some data at this point.
-
    For training our network, we assigned a suitable loss function to the network and build a **Trainer**
    which uses gradient descent with momentum. We also setup two *data iterators* which will iterate through our
    training and validation sets.
-
    We attached a couple of *monitors* to our trainer. The first one monitors the number of epochs of
    training, and is actually of the *stopper* type -- it stops training after 5 epochs. The second one monitors the
    classification error obtained using the model after each epoch of training on the validation set.
-
    Finally, we trained the network using the trainer and the data iterators we prepared.

.. note::
    -   Following common convention, one *epoch* refers to one pass through the training set.

    -   The *training_error* printed and logged by the trainer is not the true training loss produced by
        the model on the training data. Instead, it is the mean of the training losses
        computed for each batch during the epoch. Since the model is updated after every batch,
        this value of the loss is only indicative of the learning progress,
        but does not correspond to any single model.

All monitored values are recorded by the trainer in the **log** dictionary and are available
for examination after training::

    print(trainer.logs['training_errors']) # default monitor
    print(trainer.logs['validation_error']) # named monitor

The performance of the trained model on the test set can be evaluated as follows::

    from pylstm.tools import evaluate
    print('%.2f%% error on test set' % pylstm.tools.evaluate(network, test_inputs, FramewiseTargets(test_targets), ClassificationError))

This will produce output like the following::

    [====1====2====3====4====5====6====7====8====9====0] Took: 0:00:01.1
    8.00% error on test set

We obtained 8% classification error on the test set using logistic regression.
The above example includes some core ingredients of many neural network experiments.
At this point, you can tinker with the experiment by changing the values of different parameters
like the learning rate, momentum, batch sizes etc. and see their effect on performance.

.. _add_hidden:

Adding Hidden Layers
====================

Adding hidden layers to our simple network is easy, requiring a simple addition to the argument of the
**build_net** function. The following code builds, trains and evaluates a
feedforward network with a single hidden layer of Rectified Linear Units (ReLUs)::

    # Build Network
    network = build_net(InputLayer(784) >>
                        ForwardLayer(500, act_func='relu', name='h1') >>
                        ForwardLayer(10, act_func="softmax", name="out"))
    network.error_func = MultiClassCrossEntropyError
    network.initialize({"default": Gaussian(0.03)}, seed=25)

    # Build trainer and iterators
    trainer = Trainer(stepper=MomentumStep(learning_rate=0.1, momentum=0.5))

    train_iter = Minibatches(train_inputs, FramewiseTargets(train_targets), batch_size=20, verbose=False)
    valid_iter = Minibatches(valid_inputs, FramewiseTargets(valid_targets), batch_size=20, verbose=False, shuffle=False)

    # Add monitoring
    trainer.add_monitor(MaxEpochsSeen(max_epochs=5))
    trainer.add_monitor(MonitorClassificationError(data_name='valid_iter', name='validation_error', timescale='epoch', interval=1))

    # Train
    trainer.train(network, train_iter, valid_iter=valid_iter)

    # Evaluate
    print('%.2f%% error on test set' % pylstm.tools.evaluate(network, test_inputs, FramewiseTargets(test_targets), ClassificationError))

This produces the following output::

     - - - - - - - - - - - - - - -  Pretraining  - - - - - - - - - - - - - - -
    validation_error                        : 90.69


     - - - - - - - - - - - - - - -  Epoch 1  - - - - - - - - - - - - - - -
    training_errors                         : 0.288767719818
    validation_error                        : 4.52


     - - - - - - - - - - - - - - -  Epoch 2  - - - - - - - - - - - - - - -
    training_errors                         : 0.120742993624
    validation_error                        : 3.23


     - - - - - - - - - - - - - - -  Epoch 3  - - - - - - - - - - - - - - -
    training_errors                         : 0.0811216923832
    validation_error                        : 2.74


     - - - - - - - - - - - - - - -  Epoch 4  - - - - - - - - - - - - - - -
    training_errors                         : 0.0604931253936
    validation_error                        : 2.84


     - - - - - - - - - - - - - - -  Epoch 5  - - - - - - - - - - - - - - -
    training_errors                         : 0.0460170904523
    >> Stopping because: Max epochs reached.
    validation_error                        : 2.46
    [====1====2====3====4====5====6====7====8====9====0] Took: 0:00:06.7
    2.33% error on test set

This should familiarize you with the very basics of Brainstorm. You can now move on
to understanding the library features in detail.