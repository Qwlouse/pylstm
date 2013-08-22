#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from copy import deepcopy
from .. import wrapper as pw


class Network(object):
    def __init__(self, layers, param_manager, fwd_state_manager, in_out_manager,
                 bwd_state_manager, error_func, architecture):
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

        self.architecture = architecture

        self.error = None
        self.deltas = None

        self.regularizers = {}
        self.constraints = {}

        self.out_layer = self.layers.keys()[-1]

    @property
    def in_buffer(self):
        return self.in_out_manager.get_source_view("InputLayer").as_array()

    @property
    def out_buffer(self):
        return self.in_out_manager.get_source_view(self.out_layer).as_array()

    @property
    def param_buffer(self):
        return self.param_manager.buffer.as_array().flatten()

    @param_buffer.setter
    def param_buffer(self, buffer_view):
        """
        Set the parameter buffer that holds all the weights.
        """
        if isinstance(buffer_view, pw.Matrix):
            self.param_manager.initialize_buffer(buffer_view)
        else:
            self.param_manager.initialize_buffer(pw.Matrix(buffer_view))
        ## ensure constraints
        for layer_name, layer_constraints in self.constraints.items():
            params = self.get_param_view_for(layer_name)
            for view, view_constraints in layer_constraints.items():
                for constraint in view_constraints:
                    params[view][:] = constraint(params[view])

    @property
    def grad_buffer(self):
        return self.grad_manager.buffer.as_array()

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
        return self.in_out_manager.get_sink_view(name).as_array()

    def get_output_view_for(self, name):
        return self.in_out_manager.get_source_view(name).as_array()

    def get_in_deltas_view_for(self, name):
        return self.delta_manager.get_sink_view(name).as_array()

    def get_out_deltas_view_for(self, name):
        return self.delta_manager.get_source_view(name).as_array()

    def clear_internal_state(self):
        if self.fwd_state_manager.buffer:
            self.fwd_state_manager.buffer.as_array()[:] = 0.
        if self.bwd_state_manager.buffer:
            self.bwd_state_manager.buffer.as_array()[:] = 0.

    def __getitem__(self, item):
        """
        Get the layer with the given name.
        """
        return self.layers[item]

    def set_buffer_manager_dimensions(self, t, b):
        self.fwd_state_manager.set_dimensions(t, b)
        self.r_fwd_state_manager.set_dimensions(t, b)
        self.bwd_state_manager.set_dimensions(t, b)
        self.in_out_manager.set_dimensions(t, b)
        self.r_in_out_manager.set_dimensions(t, b)
        self.delta_manager.set_dimensions(t, b)

    def forward_pass(self, input_buffer):
        self.error = None
        self.deltas = None
        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].out_size
        self.set_buffer_manager_dimensions(t, b)
        # inject the input buffer
        self.in_buffer[:] = input_buffer
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:]:
            param = self.param_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)

            l.forward(param, fwd_state, input_view, out)
        # read the output buffer
        return self.out_buffer

    def calculate_error(self, T, M=None):
        if self.error is None:
            self.error, self.deltas = self.error_func(self.out_buffer, T, M)
        return self.error

    def pure_backpass(self, deltas):
        t, b, f = deltas.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # clear all delta buffers
        self.delta_manager.clear_buffer()
        # inject delta_buffer
        out_view = self.delta_manager.get_source_view(self.out_layer).as_array()
        out_view[:] = deltas
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
        return self.delta_manager.get_source_view("InputLayer").as_array()

    def backward_pass(self, T, M=None):
        if self.deltas is None:
            self.error, self.deltas = self.error_func(self.out_buffer, T, M)
        return self.pure_backpass(self.deltas)

    def calc_gradient(self):
        self.grad_manager.initialize_buffer(
            pw.Matrix(self.get_param_size()))
        for n, l in self.layers.items()[-1:0:-1]:
            param = self.param_manager.get_source_view(n)
            grad = self.grad_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            bwd_state = self.bwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.gradient(param, grad, fwd_state, bwd_state, out, input_view, delta_out)

            if n in self.regularizers:
                regularizer = self.regularizers[n]
                for view, view_regularizers in regularizer.items():
                    view_param = param[view]
                    view_grad = grad[view]
                    for view_regularizer in view_regularizers:
                        view_grad += view_regularizer(view_param)

        return self.grad_manager.buffer.as_array()

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

            l.Rpass(param, v, fwd_state, r_fwd_state, input_view, out, r_in, r_out)
            # read the output buffer
        return self.r_in_out_manager.get_source_view(self.out_layer).as_array()

    def r_backward_pass(self, lambda_, mu):
        delta_buffer = self.r_in_out_manager.get_source_view(self.out_layer).as_array()
        t, b, f = delta_buffer.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # clear all delta buffers
        self.delta_manager.clear_buffer()
        # inject delta_buffer
        out_view = self.delta_manager.get_source_view(self.out_layer).as_array()
        out_view[:] = delta_buffer
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
            l.dampened_backward(param, fwd_state, bwd_state, out, delta_in, delta_out, r_fwd_state, lambda_, mu)

        # read the final delta buffer
        return self.delta_manager.get_source_view("InputLayer").as_array()

    def hessian_pass(self, input_buffer, v_buffer, lambda_=0., mu=0.):
        t = input_buffer.shape[0]
        b = input_buffer.shape[1]
        self.forward_pass(input_buffer)
        self.r_forward_pass(input_buffer, v_buffer)
        self.r_backward_pass(lambda_, mu)
        return self.calc_gradient()

    def set_regularizers(self, reg_dict=(), **kwargs):
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
                                           'other': L1(0.5)})
        The use of 'other' sets all previously unset views of the layer.
        Passing None as a Regularizer is a way of specifying that this view
        should not be regularized. This is useful in combination with 'other'.

        """
        regularizers = dict(reg_dict)
        regularizers.update(kwargs)
        self._flatten_view_references(regularizers, self.regularizers)

    def set_constraints(self, constraint_dict=(), **kwargs):
        constraints = dict(constraint_dict)
        constraints.update(kwargs)
        self._flatten_view_references(constraints, self.constraints)

    def _flatten_view_references(self, references, flattened):
        allowed_layers = self.layers.keys()[1:]
        for layer_name, ref in references.items():
            assert layer_name in allowed_layers, "Unknown Layer '%s'.\n" \
                                                 "Possible layers are: %s" % \
                                                 (layer_name,
                                                 ", ".join(allowed_layers))
            if layer_name not in flattened:
                flattened[layer_name] = {}
            layer_references = flattened[layer_name]
            param_view = self.param_manager.get_source_view(layer_name)
            if isinstance(ref, dict):
                if 'other' in ref and ref['other'] is not None:
                    for view in param_view:
                        if view not in layer_references:
                            layer_references[view] = ensure_list(ref['other'])

                for view_name, r in ref.items():
                    if view_name == 'other':
                        continue
                    assert view_name in param_view, \
                        "Unknown view '%s' for '%s'.\nPossible views are: %s"\
                        % (view_name, layer_name, ", ".join(param_view.keys()))
                    layer_references[view_name] = ensure_list(r)
            else:
                for view in param_view:
                    layer_references[view] = ensure_list(ref)


def ensure_list(a):
    if isinstance(a, list):
        return a
    elif a is None:
        return []
    else:
        return [a]


