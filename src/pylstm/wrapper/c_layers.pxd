from c_matrix cimport Matrix, ActivationFunction, MatrixContainer
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "fwd_layer.h":
    cppclass RegularLayer:
        RegularLayer()
        RegularLayer(ActivationFunction* f)
        bool use_bias


cdef extern from "rev_layer.h":
    cppclass ReverseLayer:
        ReverseLayer()


cdef extern from "rnn_layer.h":
    cppclass RnnLayer:
        RnnLayer()
        RnnLayer(ActivationFunction* f)


cdef extern from "lstm_layer.h":
    cppclass LstmLayer:
        LstmLayer()
        LstmLayer(ActivationFunction* f)


cdef extern from "lstm97_layer.h":
    cppclass Lstm97Layer:
        Lstm97Layer()
        Lstm97Layer(ActivationFunction* f)
        bool full_gradient
        bool peephole_connections
        bool forget_gate
        bool output_gate
        bool gate_recurrence
        bool use_bias


cdef extern from "layer.hpp":
    cppclass BaseLayer:
        size_t in_size
        size_t out_size

        BaseLayer(size_t in_size, size_t out_size)

        string get_typename()

        size_t get_weight_size() except +

        size_t get_fwd_state_size(size_t n_batches, size_t n_slices) except +

        size_t get_bwd_state_size(size_t n_batches, size_t n_slices) except +

        MatrixContainer* create_parameter_view(Matrix& w) except +

        MatrixContainer* create_fwd_state_view(Matrix&b, size_t n_batches, size_t n_slices) except +

        MatrixContainer* create_bwd_state_view(Matrix&b, size_t n_batches, size_t n_slices) except +

        void forward_pass(MatrixContainer& w, MatrixContainer& b, Matrix& x, Matrix& y) except +

        void backward_pass(MatrixContainer &w, MatrixContainer &b, MatrixContainer &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas) except +

        void gradient(MatrixContainer &w, MatrixContainer &grad, MatrixContainer &b, MatrixContainer &d, Matrix &y, Matrix& x, Matrix &out_deltas) except +

        void Rpass(MatrixContainer &w, MatrixContainer &v,  MatrixContainer &b, MatrixContainer &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry) except +

        void dampened_backward(MatrixContainer &w, MatrixContainer &b, MatrixContainer &d,Matrix &y, Matrix &in_deltas, Matrix &out_deltas, MatrixContainer &Rb, double _lambda, double mu) except +


    cppclass Layer[L]:
        Layer(size_t in_size, size_t out_size, L)
