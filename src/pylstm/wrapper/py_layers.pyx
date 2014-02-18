#!/usr/bin/python
# coding=utf-8
cimport c_layers as cl
cimport c_matrix as cm
from cython.operator cimport dereference as deref
from cpython cimport bool
from libcpp.vector cimport vector

from py_matrix cimport Matrix
from py_matrix_container cimport create_MatrixContainer, MatrixContainer

cdef class BaseLayer:
    cdef cl.BaseLayer* layer
    cdef bool _skip_training

    @property
    def in_size(self):
        return self.layer.in_size

    @property
    def out_size(self):
        return self.layer.out_size

    def __cinit__(self):
        self.layer = NULL
        self._skip_training = False


    property skip_training:
        def __get__(self):
            return self._skip_training

        def __set__(self, value):
            self._skip_training = value
    
    def __dealloc(self):
        del self.layer

    def get_input_buffer_size(self, time_length=1, batch_size=1):
        return self.layer.in_size * time_length * batch_size

    def get_output_buffer_size(self, time_length=1, batch_size=1):
        return self.layer.out_size * time_length * batch_size

    def get_parameter_size(self, time_length=1, batch_size=1):
        return self.layer.get_weight_size()

    def get_fwd_state_size(self, time_length=1, batch_size=1):
        return self.layer.get_fwd_state_size(batch_size, time_length)

    def get_bwd_state_size(self, time_length=1, batch_size=1):
        return self.layer.get_bwd_state_size(batch_size, time_length)

    def create_input_view(self, input_buffer, time_length=1, batch_size=1):
        assert len(input_buffer) == self.get_input_buffer_size(time_length, batch_size)
        return input_buffer.reshape(time_length, batch_size, self.layer.in_size)

    def create_output_view(self, output_buffer, time_length=1, batch_size=1):
        assert len(output_buffer) == self.get_output_buffer_size(time_length, batch_size)
        return output_buffer.reshape(time_length, batch_size, self.layer.out_size)

    def create_param_view(self, Matrix param_buffer, time_length=1, batch_size=1):
        cdef cm.MatrixContainer* params = self.layer.create_parameter_view(param_buffer.c_obj)
        return create_MatrixContainer(params)
        
    def create_fwd_state(self, Matrix fwd_state_buffer, time_length=1, batch_size=1):
        cdef cm.MatrixContainer* fwd_state = self.layer.create_fwd_state_view(fwd_state_buffer.c_obj, batch_size, time_length)
        return create_MatrixContainer(fwd_state)

    def create_bwd_state(self, Matrix bwd_state_buffer, time_length=1, batch_size=1):
        cdef cm.MatrixContainer* bwd_state = self.layer.create_bwd_state_view(bwd_state_buffer.c_obj, batch_size, time_length)
        return create_MatrixContainer(bwd_state)

    def forward(self, MatrixContainer param, MatrixContainer fwd_state, Matrix in_view, Matrix out_view, bool training_pass):
        self.layer.forward_pass(deref(param.this_ptr), deref(fwd_state.this_ptr), in_view.c_obj, out_view.c_obj, training_pass)

    def backward(self, MatrixContainer param, MatrixContainer fwd_state, MatrixContainer err, Matrix out_view, Matrix in_deltas, Matrix out_deltas):
        self.layer.backward_pass(deref(param.this_ptr), deref(fwd_state.this_ptr), deref(err.this_ptr), out_view.c_obj, in_deltas.c_obj, out_deltas.c_obj)

    def gradient(self, MatrixContainer param, MatrixContainer grad, MatrixContainer fwd_state, MatrixContainer err, Matrix out_view, Matrix in_view, Matrix out_deltas):
        self.layer.gradient(deref(param.this_ptr), deref(grad.this_ptr), deref(fwd_state.this_ptr), deref(err.this_ptr), out_view.c_obj, in_view.c_obj, out_deltas.c_obj)

    def Rpass(self, MatrixContainer param, MatrixContainer v,  MatrixContainer fwd_state, MatrixContainer r_fwd_state, Matrix in_view, Matrix out_view, Matrix Rin_view, Matrix Rout_view):
        self.layer.Rpass(deref(param.this_ptr), deref(v.this_ptr),  deref(fwd_state.this_ptr), deref(r_fwd_state.this_ptr), in_view.c_obj, out_view.c_obj,Rin_view.c_obj, Rout_view.c_obj)

    def dampened_backward(self, MatrixContainer param, MatrixContainer fwd_state, MatrixContainer bwd_state, Matrix y, Matrix in_deltas, Matrix out_deltas, MatrixContainer r_fwd_state, double _lambda, double mu):
        self.layer.dampened_backward(deref(param.this_ptr), deref(fwd_state.this_ptr), deref(bwd_state.this_ptr), y.c_obj, in_deltas.c_obj, out_deltas.c_obj, deref(r_fwd_state.this_ptr), _lambda, mu)


    def __unicode__(self):
        return "<" + self.layer.get_typename() + ": in_size=%d out_size=%d>"%(int(self.layer.in_size), int(self.layer.out_size))

    def __repr__(self):
        return self.__unicode__()

    def __len__(self):
        return self.layer.out_size


def ctcpp_alpha(Y, T):
    t, b, f = Y.shape
    assert b == 1, "No multibatch support in ctcpp for now"
    Z =  2 * len(T) + 1
    alpha = Matrix(t, 1, Z)
    cl.ctc_alphas(Matrix(Y).c_obj, T, alpha.c_obj)
    return alpha.as_array()

def ctcpp_beta(Y, T):
    t, b, f = Y.shape
    assert b == 1, "No multibatch support in ctcpp for now"
    Z =  2 * len(T) + 1
    beta = Matrix(t, 1, Z)
    cl.ctc_betas(Matrix(Y).c_obj, T, beta.c_obj)
    return beta.as_array()

def ctcpp(Y, T):
    t, b, f = Y.shape
    assert b == 1, "No multibatch support in ctcpp for now"
    deltas = Matrix(t, b, f)
    deltas.c_obj.set_all_elements_to(float('-inf'))
    error = cl.ctc(Matrix(Y).c_obj, T, deltas.c_obj)
    return error, deltas

def create_layer(name, in_size, out_size, **kwargs):
    l = BaseLayer()
    name_lower = name.lower()

    cdef cm.ActivationFunction* act_fct = <cm.ActivationFunction*> &cm.Sigmoid

    expected_kwargs = {'act_func', 'skip_training'}
    if name_lower == "lstm97layer":
        expected_kwargs |= {'full_gradient', 'peephole_connections',
                           'forget_gate', 'output_gate', 'gate_recurrence',
                           'use_bias'}
    if name_lower == "lstmlayer":
        expected_kwargs |= {'delta_range'}
    if name_lower == "forwardlayer":
        expected_kwargs |= {'use_bias'}
    if name_lower == "dropoutlayer":
        expected_kwargs |= {'dropout_prob'}
    if name_lower == "lwtalayer":
        expected_kwargs |= {'block_size'}
    unexpected_kwargs = [k for k in kwargs if k not in expected_kwargs]
    if unexpected_kwargs:
        import warnings
        warnings.warn("Warning: got unexpected kwargs: %s"%unexpected_kwargs)

    if "act_func" in kwargs:
        af_name = kwargs["act_func"].lower()
        if af_name == "sigmoid":
            act_fct = <cm.ActivationFunction*> &cm.Sigmoid
        elif af_name == "tanh":
            act_fct = <cm.ActivationFunction*> &cm.Tanh
        elif af_name == "tanhx2":
            act_fct = <cm.ActivationFunction*> &cm.Tanhx2
        elif af_name in ["rectified_linear", "relu"]:
            act_fct = <cm.ActivationFunction*> &cm.RectifiedLinear
        elif af_name == "linear":
            act_fct = <cm.ActivationFunction*> &cm.Linear
        elif af_name == "softmax":
            act_fct = <cm.ActivationFunction*> &cm.Softmax
        # elif af_name == "winout":
        #     act_fct = <cm.ActivationFunction*> &cm.Winout
        elif af_name == "tanhscaled":
            act_fct = <cm.ActivationFunction*> &cm.TanhScaled
        else:
            raise AttributeError("No activation with name " + af_name)

    cdef cl.Lstm97Layer lstm97
    cdef cl.LstmLayer lstm_layer
    cdef cl.ForwardLayer forward_layer
    cdef cl.HfFinalLayer hf_final_layer
    cdef cl.DropoutLayer dropout_layer
    cdef cl.LWTALayer lwta_layer

    if name_lower == "forwardlayer":
        forward_layer = cl.ForwardLayer(act_fct)
        if 'use_bias' in kwargs:
            forward_layer.use_bias = kwargs['use_bias']
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.ForwardLayer](in_size, out_size, forward_layer))
    elif name_lower == "hffinallayer":
        hf_final_layer = cl.HfFinalLayer(act_fct)
        if 'use_bias' in kwargs:
            hf_final_layer.use_bias = kwargs['use_bias']
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.HfFinalLayer](in_size, out_size, hf_final_layer))
    elif name_lower == "rnnlayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.RnnLayer](in_size, out_size, cl.RnnLayer(act_fct)))
    elif name_lower == "arnnlayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.ArnnLayer](in_size, out_size, cl.ArnnLayer(act_fct)))
    elif name_lower == "mrnnlayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.MrnnLayer](in_size, out_size, cl.MrnnLayer(act_fct)))
    elif name_lower == "lstmlayer":
        lstm_layer = cl.LstmLayer(act_fct)
        if 'delta_range' in kwargs:
            lstm_layer.delta_range = kwargs['delta_range']
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.LstmLayer](in_size, out_size, lstm_layer))
    elif name_lower == "lstm97layer":
        lstm97 = cl.Lstm97Layer(act_fct)
        if 'full_gradient' in kwargs:
            lstm97.full_gradient = kwargs['full_gradient']
        if 'peephole_connections' in kwargs:
            lstm97.peephole_connections = kwargs['peephole_connections']
        if 'forget_gate' in kwargs:
            lstm97.forget_gate = kwargs['forget_gate']
        if 'output_gate' in kwargs:
            lstm97.output_gate = kwargs['output_gate']
        if 'gate_recurrence' in kwargs:
            lstm97.gate_recurrence = kwargs['gate_recurrence']
        if 'use_bias' in kwargs:
            lstm97.use_bias = kwargs['use_bias']

        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.Lstm97Layer](in_size, out_size, lstm97))
    elif name_lower == "reverselayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.ReverseLayer](in_size, out_size, cl.ReverseLayer()))
        l.skip_training = True
    elif name_lower == "dropoutlayer":
        if 'dropout_prob' in kwargs:
            dropout_layer.drop_prob = kwargs['dropout_prob']
        else:
            dropout_layer.drop_prob = 0.5
        if 'initial_state' in kwargs:
            dropout_layer.rnd_state = kwargs['initial_state']
        else:
            dropout_layer.rnd_state = 42
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.DropoutLayer](in_size, out_size, dropout_layer))
    elif name_lower == "lwtalayer":
        if 'block_size' in kwargs:
            lwta_layer.block_size = kwargs['block_size']
        else:
            lwta_layer.block_size = 2
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.LWTALayer](in_size, out_size, lwta_layer))
    else :
        raise AttributeError("No layer with name " + name)

    if "skip_training" in kwargs:
        l.skip_training = kwargs['skip_training']

    return l
