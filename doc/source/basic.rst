.. _basic:

******
Basics
******

The most important and basic features of Brainstorm are explained in this section.
After going through it, you should be able to setup and run a large variety of
neural network experiments.

-------------------------------------------------------------------------------

.. _data_format:

Data handling
=============

.. _raw_data:

Preparing raw inputs and targets
--------------------------------

You might have noticed the 3-dimensional arrays we prepared for MNIST inputs and targets for our
first experiments (:ref:`getdata`). Since Brainstorm is designed to work with sequential data,
time is the first dimension for all data objects. The other two dimensions are samples
and features. Thus, an array of shape (10, 100, 32) contains 100 32-dimensional sequences of length 10.

This is why the shape of the MNIST training set was (1, 50000, 784) -- each sequence was of
length 1 (static images), there were 50000 images, and each image was represented using 784 pixel values
(for 28x28 images).


.. _targets:

Targets
-------

Targets are the first thing you need to prepare if you want to train
neural networks.
Sequential learning problems can be as simple as having a target output for each time step
(e.g. framewise classification, next-step prediction), but
there it is also possible to have a single target output for each sequence
(sequence classification) or more structured outputs (e.g. a short sequence of phoneme labels
as targets for a long input audio sequence). To cover such cases explicitly, Brainstorm
requires that raw targets be converted into a **Targets** object of one of the following types:

-   **FramewiseTargets**
-   **SequencewiseTargets**
-   **LabelingTargets**

Additionally, it is common to have sequences of varying lengths,
and possible to have learning tasks for which you have targets for selected
time steps. This is why the **Targets** classes support *masks*. A *mask* is
a simple binary array.
When the error is computed using the targets and network outputs during
training, the mask is multiplied element-wise with the errors.
This way, you can specify (for each sequence) the time steps for
which no error should be backpropagated during training. Clearly,
the shape of the *masks* array needs to be
(number of time steps, number of samples, 1).

The last little tidbit you should know about is target binarization.
For convenience, you can binarize class labels (convert them into
one-hot vectors) when constructing a **Targets** object. This can
save a lot of memory if you have a large number of classes and sequences,
because the conversion to one hot is implicitly done when calculating
errors. For this to work correctly, make raw targets of shape
(number of time steps, number of samples, 1) with each entry being
the 0-based index of the target class.

.. note::
    It is highly recommended that you always you 3 dimensional
    arrays as your raw data with which you build **Targets** objects,
    even if some dimensions are not needed.
    You can see we did this for the MNIST dataset (:ref:`getdata`).
    For example, for
    sequencewise classification, you will have targets for each
    sequence. In this case, the *masks* should be of shape
    (number of time steps, number of samples, 1)
    -- since the sequences can be of different lengths --
    and the targets should be of shape
    (1, number of samples, 1).

Once you have the raw targets and the masks, you can build
a **Targets** object by simply doing something like::

    targets = FramewiseTargets(raw_targets, mask=raw_masks, binarize_to=True)

This will create a **FramewiseTargets** object assuming that you have already
binarized the class labels, or your loss function does not need
them to be binarized.

The MNIST targets were shaped (1, 50000, 10) and already binarized
to one-hot vectors using the **binarize_array** utility.
As you might expect, the 2nd dimension of targets and masks
have to match the inputs, and the last dimension is
size 10 since our targets are 10-dimensional one-hot vectors.

The primary advantage of specifying the type of targets explicitly is that
other components of the library (like loss functions) can provide custom
implementations for the particular kind of task based on the target type.

.. _data_iterators:

Data Iterators
--------------

Brainstrom uses *data iterators* to cycle through the input-target pairs
in a dataset. Three kinds of *data iterators* are available, covering most
use cases:

-   **Online**
-   **Minibatches**
-   **Undivided**

The names are self-explanatory -- **Online** iterates through samples
one at a time, **Minibatches** iterates through them in fixed size
batches, and **Undivided** returns all the data as a whole.
The dataset is shuffled by default after each full pass.
An additional parameter **verbose** can be used to print progress bars
as the iterator goes through the data.

Some examples::

    train_iter = Online(train_inputs, train_targets, shuffle=True, verbose=True)
    train_iter = Minibatches(train_inputs, train_targets, batch_size=10)
    train_iter = Undivided(train_inputs, train_targets)

-------------------------------------------------------------------------------

Networks
========

.. _build_network:

Construction
------------

Now that you know how to prepare a dataset for a Brainstorm experiment,
we can now get into constructing some networks. First,
let us construct some layers::

    l1 = InputLayer(10)
    l2 = ForwardLayer(100, act_func='sigmoid', name='h1')
    l3 = LstmLayer(20, name='h2')
    l4 = ForwardLayer(10, act_func='softmax', name='output')

The syntax for layer construction should be clear from the above examples.
The first argument is size of the layer -- meaning the size of the activation
vectors this layer will produce. You can also choose an activation function
('sigmoid', 'tanh', 'relu', 'softmax' etc.), and name the layer to identify
it. If you do not provide a size, it is assumed that it produces the same
number of features as its total number of input features. If you do
not name the layer, a name is generated for the layer.

To build a network, we need to first specify how the layers are connected.
This is very simple using the **>>** operator in Brainstorm. To build
a very simple network from the layer above, you can do::

    layers = l1 >> l2 >> l3 >> l4

Finally, you can construct the network using the **build_net** function::

    network = build_net(layers)

The above steps can be combined for many simple networks::

    network = build_net(InputLayer(10) >>
                        ForwardLayer(100, act_func='sigmoid', name='h1') >>
                        LstmLayer(20, name='h2') >>
                        ForwardLayer(10, act_func='softmax', name='output'))

You can connect one layer to multiple layers, or multiple layers to one layer.
This allows you to build any directed acyclic graph using the **>>** operator
and the **build_net** function.

.. note::
    The argument to **build_net** can be any layer which has already been
    connected to the other layers. Note that applying the **>>** operator
    returns a layer, which is why the above example works.
    This is useful to know for building complicated networks.


Brainstorm provides the **ReverseLayer** for building bidirectional
recurrent networks and the **NoOpLayer** for building other complicated
structures. Here is an example of building a network with
two bidirectional fully recurrent layers::

    input_layer = InputLayer(100)
    output_layer = ForwardLayer(10, act_func='softmax', name='Output')
    noop_layer = NoOpLayer()

    input_layer >> RnnLayer(40, name='ForwardRec-1', act_func='tanh') >> noop_layer
    input_layer >> ReverseLayer() >> RnnLayer(40, name='BackwardRec-1', act_func='tanh') >> ReverseLayer() >> noop_layer

    noop_layer >> RnnLayer(20, name='ForwardRec-2', act_func='tanh') >> output_layer
    noop_layer >> ReverseLayer() >> RnnLayer(20, name='BackwardRec-2', act_func='tanh') >> ReverseLayer() >> output_layer

    net = build_net(output_layer)

.. _init_network:

Initialization
--------------

initialize()

.. _constrain_regularization:

Constraints & regularization
----------------------------

gradient_modifiers and set_constraints()

.. _usage_network:

Usage
-----

Getting and setting buffers, forward_pass()

-------------------------------------------------------------------------------

Trainers
========

.. _build_trainer:

Construction
------------

Network error_func, stepper


.. _monitor_stop:

Monitoring & Stopping
---------------------

add_monitor, StopIteration

.. _usage_trainer:

Usage
-----

train(), log


.. _randomization:

-------------------------------------------------------------------------------

Randomization
=============

set_global_seed() etc.