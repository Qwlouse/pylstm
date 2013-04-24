#!/usr/bin/python
# coding=utf-8
cimport c_layers as cl
cimport c_matrix as cm
from cython.operator cimport dereference as deref
from py_matrix cimport Matrix

cdef class ViewContainer:
    cdef cl.ViewContainer* this_ptr
    
    def __cinit__(self):
        self.this_ptr = NULL
        
    def __dealloc__(self):
        del self.this_ptr

    def __getattr__(self, item):
        if self.this_ptr.contains(item):
            b = Matrix()
            b.view = self.this_ptr[0][item]
            b.A = None
            return b.as_array()
        else:
            raise AttributeError("'%s' is not a valid view."%item)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __contains__(self, item):
        return self.this_ptr.contains(item)

    def keys(self):
        return self.this_ptr.get_view_names()

    def items(self):
        return [(n, self[n]) for n in self.keys()]

    def values(self):
        return [self[n] for n in self.keys()]

    def __unicode__(self):
        return "<" + self.this_ptr.get_typename() + ": " + ", ".join(self.keys()) + ">"

    def __repr__(self):
        return self.__unicode__()

    def __len__(self):
        return self.this_ptr.get_size()



cdef create_ViewContainer(cl.ViewContainer* c):
    bc = ViewContainer()
    bc.this_ptr = c
    return bc


cdef class BaseLayer:
    cdef cl.BaseLayer* layer

    @property
    def in_size(self):
        return self.layer.in_size

    @property
    def out_size(self):
        return self.layer.out_size

    def __cinit__(self):
        self.layer = NULL
    
    def __dealloc(self):
        del self.layer

    def get_input_buffer_size(self, time_length=1, batch_size=1):
        return self.layer.in_size * time_length * batch_size

    def get_output_buffer_size(self, time_length=1, batch_size=1):
        return self.layer.out_size * time_length * batch_size

    def get_param_size(self, time_length=1, batch_size=1):
        return self.layer.get_weight_size()

    def get_internal_state_size(self, time_length=1, batch_size=1):
        return self.layer.get_fwd_state_size(batch_size, time_length)

    def get_internal_error_state_size(self, time_length=1, batch_size=1):
        return self.layer.get_bwd_state_size(batch_size, time_length)

    def create_input_view(self, input_buffer, time_length=1, batch_size=1):
        assert len(input_buffer) == self.get_input_buffer_size(time_length, batch_size)
        return input_buffer.reshape(time_length, batch_size, self.layer.in_size)

    def create_output_view(self, output_buffer, time_length=1, batch_size=1):
        assert len(output_buffer) == self.get_output_buffer_size(time_length, batch_size)
        return output_buffer.reshape(time_length, batch_size, self.layer.out_size)

    def create_param_view(self, Matrix param_buffer, time_length=1, batch_size=1):
        cdef cl.ViewContainer* params = self.layer.create_parameter_view(param_buffer.view)
        return create_ViewContainer(params)
        
    def create_internal_view(self, Matrix internal_buffer, time_length=1, batch_size=1):
        cdef cl.ViewContainer* internal = self.layer.create_fwd_state_view(internal_buffer.view, batch_size, time_length)
        return create_ViewContainer(internal)

    def create_internal_error_view(self, Matrix internal_error_buffer, time_length=1, batch_size=1):
        cdef cl.ViewContainer* deltas = self.layer.create_bwd_state_view(internal_error_buffer.view, batch_size, time_length)
        return create_ViewContainer(deltas)

    def forward(self, ViewContainer param, ViewContainer internal, Matrix in_view, Matrix out_view):
        self.layer.forward_pass(deref(param.this_ptr), deref(internal.this_ptr), in_view.view, out_view.view)

    def backward(self, ViewContainer param, ViewContainer internal, ViewContainer err, Matrix out_view, Matrix in_deltas, Matrix out_deltas):
        self.layer.backward_pass(deref(param.this_ptr), deref(internal.this_ptr), deref(err.this_ptr), out_view.view, in_deltas.view, out_deltas.view)

    def gradient(self, ViewContainer param, ViewContainer grad, ViewContainer internal, ViewContainer err, Matrix out_view, Matrix in_view, Matrix out_deltas):
        self.layer.gradient(deref(param.this_ptr), deref(grad.this_ptr), deref(internal.this_ptr), deref(err.this_ptr), out_view.view, in_view.view, out_deltas.view)

    def Rpass(self, ViewContainer param, ViewContainer v,  ViewContainer internal, ViewContainer Rinternal, Matrix in_view, Matrix out_view, Matrix Rin_view, Matrix Rout_view):
        self.layer.Rpass(deref(param.this_ptr), deref(v.this_ptr),  deref(internal.this_ptr), deref(Rinternal.this_ptr), in_view.view, out_view.view,Rin_view.view, Rout_view.view)

    def Rbackward(self, ViewContainer param, ViewContainer internal, ViewContainer internal_deltas, Matrix in_deltas, Matrix out_deltas, ViewContainer Rinternal, double _lambda, double mu):
        self.layer.Rbackward(deref(param.this_ptr), deref(internal.this_ptr), deref(internal_deltas.this_ptr), in_deltas.view, out_deltas.view, deref(Rinternal.this_ptr), _lambda, mu)


    def __unicode__(self):
        return "<" + self.layer.get_typename() + ": in_size=%d out_size=%d>"%(int(self.layer.in_size), int(self.layer.out_size))

    def __repr__(self):
        return self.__unicode__()

    def __len__(self):
        return self.layer.out_size



def create_layer(name, in_size, out_size, **kwargs):
    l = BaseLayer()
    cdef cm.ActivationFunction* act_fct = <cm.ActivationFunction*> &cm.Sigmoid

    unexpected_kwargs = [k for k in kwargs if k not in {'act_func'}]
    if unexpected_kwargs:
        import warnings
        warnings.warn("Warning: got unexpected kwargs: %s"%unexpected_kwargs)

    if "act_func" in kwargs:
        af_name = kwargs["act_func"]
        if af_name.lower() == "sigmoid":
            act_fct = <cm.ActivationFunction*> &cm.Sigmoid
        elif af_name.lower() == "tanh":
            act_fct = <cm.ActivationFunction*> &cm.Tanh
        elif af_name.lower() == "tanhx2":
            act_fct = <cm.ActivationFunction*> &cm.Tanhx2
        elif af_name.lower() == "linear":
            act_fct = <cm.ActivationFunction*> &cm.Linear
        elif af_name.lower() == "softmax":
            act_fct = <cm.ActivationFunction*> &cm.Softmax

    if name.lower() == "regularlayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.RegularLayer](in_size, out_size, cl.RegularLayer(act_fct)))
    elif name.lower() == "rnnlayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.RnnLayer](in_size, out_size, cl.RnnLayer(act_fct)))
    elif name.lower() == "lstmlayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.LstmLayer](in_size, out_size, cl.LstmLayer(act_fct)))
    elif name.lower() == "reverselayer":
        l.layer = <cl.BaseLayer*> (new cl.Layer[cl.ReverseLayer](in_size, out_size, cl.ReverseLayer()))
    else :
        raise AttributeError("No layer with name " + name)
    return l
