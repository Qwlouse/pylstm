#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from copy import deepcopy
import numpy as np
from pylstm import wrapper as pw
from pylstm.describable import Describable, get_description
from pylstm.randomness import reseeding_copy, Seedable
from pylstm.regularization.initializer import _evaluate_initializer
from pylstm.targets import Targets


class Network(Seedable, Describable):
    def __init__(self, layers, param_manager, fwd_state_manager, in_out_manager,
                 bwd_state_manager, error_func, architecture, seed=None):
        super(Network, self).__init__(seed=seed, category='network')
        self.layers = layers

        self.param_manager = param_manager
        self.grad_manager = deepcopy(param_manager)
        self.v_manager = deepcopy(param_manager)

        self.fwd_state_manager = fwd_state_manager
        self.r_fwd_state_manager = deepcopy(fwd_state_manager)
        self.bwd_state_manager = bwd_state_manager

        self.in_out_manager = in_out_manager
        self.r_in_out_manager = deepcopy(in_out_manager)
        self.delta_manager = deepcopy(in_out_manager)

        self.error_func = error_func

        self.description = {
            '@type': 'Network',
            'architecture': architecture,
            'initialization': {},
            'constraints': {},
            'regularization': {}
        }
        if seed is not None:
            self.description['$seed'] = seed

        self.targets = None
        self.error = None
        self.deltas = None

        self.regularizers = {}
        self.constraints = {}

        self.out_layer = self.layers.keys()[-1]

    def is_initialized(self):
        return self.param_manager.buffer is not None

    @property
    def in_buffer(self):
        """
        Access to the input buffer of the network, i.e. the data that gets
        passed to the lowest layers. (Note: This hides the context slices.)
        :return: input buffer
        :rtype: np.ndarray
        """
        return self.in_out_manager.get_source_view("InputLayer").as_array(
        )[1:, :, :]  # remove context slice

    @property
    def out_buffer(self):
        """
        Access to the output buffer of the network, i.e. the outputs of the
        topmost layer. (Note: This hides the context slices.)
        :return: output buffer
        :rtype: np.ndarray
        """
        return self.in_out_manager.get_source_view(self.out_layer).as_array(
        )[1:, :, :]  # remove context slice

    @property
    def out_delta_buffer(self):
        """
        Access to the delta buffer associated with the outputs of the topmost
        layer. (Note: This hides the context slices.)
        :return: output delta buffer
        :rtype: np.ndarray
        """
        return self.delta_manager.get_source_view(self.out_layer).as_array(
        )[1:, :, :]  # remove context slice

    @property
    def in_delta_buffer(self):
        """
        Access to the delta buffer associated with the inputs of the lowest
        layer. (Note: This hides the context slices.)
        :return: input delta buffer
        :rtype: np.ndarray
        """
        return self.delta_manager.get_source_view("InputLayer").as_array(
        )[1:, :, :]  # remove context slice

    @property
    def r_out_buffer(self):
        """
        Access to the output buffer of the R-forward pass.
        (Note: This hides the context slices.)
        :return: R-output buffer
        :rtype: np.ndarray
        """
        return self.r_in_out_manager.get_source_view(self.out_layer).as_array(
        )[1:, :, :]  # remove context slice

    @property
    def param_buffer(self):
        """
        Access to the parameter buffer of the network.
        This contains all the weight in one big flat array.
        :return: parameter buffer
        :rtype: np.ndarray
        """
        if self.is_initialized():
            return self.param_manager.buffer.as_array().flatten()

    @param_buffer.setter
    def param_buffer(self, buffer_view):
        """
        Set the parameter buffer that holds all the weights.
        """
        self.param_manager.ensure_initialization()
        if isinstance(buffer_view, pw.Matrix):
            self.param_manager.buffer.as_array().flat[:] = buffer_view.as_array().flat[:]
        else:
            self.param_manager.buffer.as_array().flat[:] = buffer_view.flat[:]
        self.enforce_constraints()

    @property
    def grad_buffer(self):
        """
        Access to the gradient buffer, that contains the gradients for all the
        weight as calculated by the calc_grad method.
        :return: gradient buffer
        :rtype: np.ndarray
        """
        return self.grad_manager.buffer.as_array()

    def enforce_constraints(self):
        for layer_name, layer_constraints in self.constraints.items():
            params = self.get_param_view_for(layer_name)
            for view, view_constraints in layer_constraints.items():
                for constraint in view_constraints:
                    params[view][:] = constraint(params[view])

    def get_param_size(self):
        """
        Returns the total size of all parameters.
        """
        return self.param_manager.calculate_size()

    def get_input_size(self):
        return self.layers["InputLayer"].out_size

    def get_output_size(self):
        return self.layers[self.out_layer].out_size

    def get_param_view_for(self, name):
        return self.param_manager.get_source_view(name)

    def get_fwd_state_for(self, name):
        return self.fwd_state_manager.get_source_view(name)

    def get_bwd_state_for(self, name):
        return self.bwd_state_manager.get_source_view(name)

    def get_input_view_for(self, name):
        return self.in_out_manager.get_sink_view(name).as_array(
        )[1:, :, :]  # remove context slice

    def get_output_view_for(self, name):
        return self.in_out_manager.get_source_view(name).as_array(
        )[1:, :, :]  # remove context slice

    def get_in_deltas_view_for(self, name):
        return self.delta_manager.get_sink_view(name).as_array(
        )[1:, :, :]  # remove context slice

    def get_out_deltas_view_for(self, name):
        return self.delta_manager.get_source_view(name).as_array(
        )[1:, :, :]  # remove context slice

    def clear_internal_state(self):
        # TODO: Optimization: we only need to clear the context slice
        if self.fwd_state_manager.buffer:
            self.fwd_state_manager.clear_buffer()
        if self.bwd_state_manager.buffer:
            self.bwd_state_manager.clear_buffer()

    def clear_context_slice(self):
        for n, l in self.layers.items():
            if self.in_out_manager.buffer:
                self.in_out_manager.get_source_view(n).as_array()[0:1, :, :] = 0

    def __getitem__(self, item):
        """
        Get the layer with the given name.
        """
        return self.layers[item]

    def set_buffer_manager_dimensions(self, t, b):
        # add one to the time dimension for context slice
        self.fwd_state_manager.set_dimensions(t + 1, b)
        self.r_fwd_state_manager.set_dimensions(t + 1, b)
        self.bwd_state_manager.set_dimensions(t + 1, b)
        self.in_out_manager.set_dimensions(t + 1, b)
        self.r_in_out_manager.set_dimensions(t + 1, b)
        self.delta_manager.set_dimensions(t + 1, b)

    def _copy_context(self):
        """
        Copies the content of all forward buffers and in_out buffer for the
        last timestep. The resulting dict can be copied into the
        context-frame (the 0th timestep), to make the network continue execution
        from where it left.
        See: _apply_context
        """
        end_t = self.fwd_state_manager.slice_count - 1
        context = dict()
        for n, l in self.layers.items()[1:]:
            fwd_state = self.fwd_state_manager.get_source_view(n)
            in_view = self.in_out_manager.get_sink_view(n)
            context[n] = {
                'fwd_state': fwd_state.copy_slice(end_t, end_t + 1),
                'in_view':  in_view.as_array()[end_t, :, :].copy()
            }

        context['__output_buffer__'] = self.in_out_manager.get_source_view(
            self.out_layer).as_array()[end_t, :, :].copy()
        return context

    def _apply_context(self, context):
        """
        Copies the context from _copy_context() into the context frame.
        """
        for n, l in self.layers.items()[1:]:
            fwd_state = self.fwd_state_manager.get_source_view(n)
            in_view = self.in_out_manager.get_sink_view(n).as_array()
            fwd_state.set_values(context[n]['fwd_state'], start=0)
            in_view[0, :, :] = context[n]['in_view']
        self.in_out_manager.get_source_view(self.out_layer).as_array()[0, :, :]\
            = context['__output_buffer__']

    def forward_pass(self, input_buffer, reset=True, context=None,
                     training_pass=False):
        self.targets = None
        self.error = None
        self.deltas = None

        if not reset and self.fwd_state_manager.buffer is not None:
            context = self._copy_context()

        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].out_size
        self.set_buffer_manager_dimensions(t, b)
        if reset:
            self.clear_internal_state()
            self.clear_context_slice()

        if context:
            self._apply_context(context)
        # inject the input buffer
        self.in_buffer[:] = input_buffer
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:]:
            param = self.param_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)

            out_view = self.in_out_manager.get_source_view(n)
            in_view = self.in_out_manager.get_sink_view(n)

            l.forward(param, fwd_state, in_view, out_view, training_pass)
        # read the output buffer
        return self.out_buffer

    def _calculate_deltas_and_error(self, targets):
        if self.error is None:
            assert isinstance(targets, Targets)
            self.targets = targets
            self.error, self.deltas = self.error_func(self.out_buffer,
                                                      self.targets)

    def calculate_error(self, targets):
        self._calculate_deltas_and_error(targets)
        return self.error

    def pure_backpass(self, deltas):
        t, b, f = deltas.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # clear all delta buffers
        self.delta_manager.clear_buffer()
        # inject delta_buffer
        self.out_delta_buffer[:] = deltas
        # execute all the intermediate layers backwards
        for n, l in self.layers.items()[-1:0:-1]:
            param = self.param_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            bwd_state = self.bwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            delta_in = self.delta_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.backward(param, fwd_state, bwd_state, out, delta_in, delta_out)
        # read the final delta buffer
        return self.in_delta_buffer

    def backward_pass(self, targets):
        self._calculate_deltas_and_error(targets)
        return self.pure_backpass(self.deltas)

    def calc_gradient(self):
        self.grad_manager.ensure_initialization()
        self.grad_manager.clear_buffer()
        for n, l in self.layers.items()[-1:0:-1]:
            if l.skip_training:
                continue
            param = self.param_manager.get_source_view(n)
            grad = self.grad_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            bwd_state = self.bwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.gradient(param, grad, fwd_state, bwd_state, out, input_view,
                       delta_out)

            if n in self.regularizers:
                regularizer = self.regularizers[n]
                for view, view_regularizers in regularizer.items():
                    view_param = param[view]
                    view_grad = grad[view]
                    for view_regularizer in view_regularizers:
                        view_grad[:] = view_regularizer(view_param, view_grad)

        return self.grad_buffer

    def r_forward_pass(self, input_buffer, v_buffer):
        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].out_size
        self.set_buffer_manager_dimensions(t, b)
        # inject the v value
        if isinstance(v_buffer, pw.Matrix):
            self.v_manager.initialize_buffer(v_buffer)
        else:
            self.v_manager.initialize_buffer(pw.Matrix(v_buffer))
        # inject the input buffer
        self.in_buffer[:] = input_buffer
        # set r input to 0
        self.r_in_out_manager.get_source_view("InputLayer").as_array()[:] = 0.0
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:]:
            param = self.param_manager.get_source_view(n)
            v = self.v_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            r_fwd_state = self.r_fwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            r_in = self.r_in_out_manager.get_sink_view(n)
            r_out = self.r_in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)

            l.Rpass(param, v, fwd_state, r_fwd_state, input_view, out, r_in, 
                    r_out)
            # read the output buffer
        return self.r_out_buffer

    def r_backward_pass(self, rvals, lambda_, mu):
        delta_buffer = rvals
        t, b, f = delta_buffer.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # clear all delta buffers
        self.delta_manager.clear_buffer()
        # inject delta_buffer
        self.out_delta_buffer[:] = delta_buffer
        # execute all the intermediate layers backwards
        for n, l in self.layers.items()[-1:0:-1]:
            param = self.param_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            bwd_state = self.bwd_state_manager.get_source_view(n)
            r_fwd_state = self.r_fwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            delta_in = self.delta_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            #l.backward(param, fwd_state, bwd_state, out, delta_in, delta_out)
            l.dampened_backward(param, fwd_state, bwd_state, out, delta_in,
                                delta_out, r_fwd_state, lambda_, mu)

        # read the final delta buffer
        return self.in_delta_buffer

    def hessian_pass(self, input_buffer, v_buffer, targets, lambda_=0., mu=0.,
                     matching_loss=True):
        y = self.forward_pass(input_buffer)
        self.r_forward_pass(input_buffer, v_buffer)
        if matching_loss:
            self.r_backward_pass(self.r_out_buffer, lambda_, mu)
        else:
            self._calculate_deltas_and_error(targets)

            rval = self.r_out_buffer
            for t in range(rval.shape[0]-1):
                for b in range(rval.shape[1]):
                    #rval[t+1, b, :] = self.deltas[t, b] *
                    #   np.inner(self.deltas[t, b], rval[t+1, b])
                    rval[t+1, b, :] = (np.diag(y[t, b]) -
                                       np.outer(y[t, b], y[t, b])).dot(
                                           rval[t+1, b])

            self.r_backward_pass(rval, lambda_, mu)
        return self.calc_gradient()

    def initialize(self, init_dict=None, seed=None, **kwargs):
        """
        This will initialize the network as specified.
        You can specify a seed to make the initialization reproducible.

        Example Usage:
            # you can set initializers in two equivalent ways:
            1) by passing a dictionary:
            >> initialize(net, {'RegularLayer': Uniform(),
                                'LstmLayer': Gaussian()})

            2) by using keyword arguments:
            >> initialize(net, RegularLayer=Uniform(), LstmLayer=Uniform())

            (you should not combine the two. If you do, however, then the
             keyword arguments take precedence)

            An initializer can either be a callable that takes
            (layer_name, view_name,  shape, seed) or something that converts to
            a numpy array. So for example:
            >> initialize(net,
                          LstmLayer=1,
                          RnnLayer=[1, 1, 1, 1, 1],
                          ForwardLayer=lambda ln, vn, s, seed: 1)

            You can specify a default initializer. If you don't a default=0 is
            assumed.
            >> initialize(net, default=Gaussian())

            To target only some of the weights of a layer can pass in a dict:
            >> initialize(net, LstmLayer={'HX': Gaussian(std=0.1),
                                          'IX': 0,
                                          'default': Uniform(-.1, .1)})
            The use of 'default' targets all unspecified views of the layer.
        """
        initializers = _update_references_with_dict(init_dict, kwargs)
        self._assert_view_reference_wellformed(initializers)
        self.description['initialization'] = initializers
        rnd = self.rnd['initialize'].get_new_random_state(seed)

        for layer_name, layer in self.layers.items()[1:]:
            views = self.get_param_view_for(layer_name)
            if views is None:
                continue
            for view_name, view in views.items():
                view_initializer = _get_default_aware(initializers, layer_name,
                                                      view_name, rnd)
                view[:] = _evaluate_initializer(view_initializer, layer_name,
                                                view_name, view.shape)

        self.enforce_constraints()

    def set_regularizers(self, reg_dict=None, seed=None, **kwargs):
        """
        Set weight regularizers for layers and even individual views of a layer.
        A regularizer has to be callable(function or object) with a single
        argument.

        Example Usage:
        # you can set regularizers for layers in two equivalent ways:
        1) by passing a dictionary:
        >> net.set_regularizers({'RegularLayer': L2(0.5), 'LstmLayer': L1(0.1)})

        2) by using keyword aguments:
        >> net.set_regularizers(RegularLayer=L2(0.5), LstmLayer=L1(0.1))

        (you should not combine the two. If you do, however, then the keyword
         arguments take precedence)

        You can specify a single Regularizer or a list of them:
        >> net.set_regularizers(RegularLayer=[L2(0.5), L1(0.1)])

        To target only some of the weights of a layer can pass in a dict:
        >> net.set_regularizers(LstmLayer={'HX': L2(0.5),
                                           'IX': None,
                                           'default': L1(0.5)})
        The use of 'defaut' acts as a default for all unset views of the layer.
        Passing None as a Regularizer is a way of specifying that this view
        should not be regularized. This is useful in combination with 'default'.

        """
        rnd = self.rnd['set_regularizers'].get_new_random_state(seed)
        regularizers = _update_references_with_dict(reg_dict, kwargs)
        self.description['regularization'] = regularizers
        self.regularizers = self._flatten_view_references(regularizers, rnd)
        _prune_view_references(self.regularizers)
        _ensure_all_references_are_lists(self.regularizers)

    def set_constraints(self, constraint_dict=None, seed=None, **kwargs):
        rnd = self.rnd['set_constraints'].get_new_random_state(seed)
        assert self.is_initialized()
        constraints = _update_references_with_dict(constraint_dict, kwargs)
        self.description['constraints'] = constraints
        self.constraints = self._flatten_view_references(constraints, rnd)
        _prune_view_references(self.constraints)
        _ensure_all_references_are_lists(self.constraints)
        self.enforce_constraints()

    def _assert_view_reference_wellformed(self, reference):
        if not isinstance(reference, dict):
            return

        all_layers = self.layers.keys()[1:]
        # assert only valid layers are specified
        for layer_name in reference:
            assert layer_name == 'default' or layer_name in all_layers, \
                "Unknown Layer '%s'.\nPossible layers are: %s" % \
                (layer_name, ", ".join(all_layers))

        # assert only valid views are specified
        for layer_name, layer_initializers in reference.items():
            if layer_name == 'default':
                continue
            param_view = self.get_param_view_for(layer_name)
            if isinstance(layer_initializers, dict):
                for view_name in layer_initializers:
                    assert view_name == 'default' or view_name in param_view, \
                        "Unknown view '%s' for '%s'.\nPossible views are: %s"\
                        % (view_name, layer_name, ", ".join(param_view.keys()))

    def _flatten_view_references(self, references, rnd, default=None):
        self._assert_view_reference_wellformed(references)
        flattened = dict()
        for layer_name, layer in self.layers.items()[1:]:
            flattened[layer_name] = dict()
            views = self.get_param_view_for(layer_name)
            for view_name, view in views.items():
                flattened[layer_name][view_name] = \
                    _get_default_aware(references, layer_name, view_name,
                                       rnd, default=default)
        return flattened

    def __describe__(self):
        description = get_description(self.description)
        description['error_function'] = get_description(self.error_func)
        return description

    @classmethod
    def __new_from_description__(cls, description):
        from .netbuilder import build_network_from_description
        net = build_network_from_description(description)
        return net


def _prune_view_references(references):
    """
    Delete all now empty layer references.
    """
    for lname, l in references.items():
        for vname in list(l.keys()):
            if not l[vname]:
                del l[vname]

    for lname in list(references.keys()):
        if not references[lname]:
            del references[lname]


def _ensure_all_references_are_lists(references):
    for lname, l in references.items():
        for vname, v in l.items():
            if l[vname] is None:
                l[vname] = []
            elif not isinstance(l[vname], list):
                l[vname] = [l[vname]]


def _get_default_aware(values, layer_name, view_name, rnd, default=0):
    """
    This function retrieves values from view reference dictionary that
    makes use of 'default'. These are used for initialize, set_regularizers, and
    set_constraints.
    """
    if not isinstance(values, dict):
        return values

    seed = rnd.generate_seed()

    if layer_name in values:
        layer_values = values[layer_name]
        if not isinstance(layer_values, dict):
            return reseeding_copy(layer_values, seed)

        if view_name in layer_values:
            return reseeding_copy(layer_values[view_name], seed)
        elif 'default' in layer_values:
            return reseeding_copy(layer_values['default'], seed)

    if 'default' in values:
        return reseeding_copy(values['default'], seed)

    return reseeding_copy(default, seed)


def _update_references_with_dict(refs, ref_dict):
    if refs is None:
        references = dict()
    elif isinstance(refs, dict):
        references = refs
    else:
        references = {'default': refs}

    if set(references.keys()) & set(ref_dict.keys()):
        raise TypeError('Conflicting values for %s!' %
                        sorted(set(references.keys()) & set(ref_dict.keys())))

    references.update(ref_dict)

    return references