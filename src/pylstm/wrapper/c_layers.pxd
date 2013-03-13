from c_matrix cimport Matrix, ActivationFunction
from libcpp.string cimport string
from libcpp.vector cimport vector
cdef extern from "fwd_layer.h":
    cppclass RegularLayer:
        RegularLayer()
        RegularLayer(ActivationFunction* f)

cdef extern from "lstm_layer.h":
    cppclass LstmLayer:
        LstmLayer()

cdef extern from "layer.hpp":
    cppclass ViewContainer:
        ViewContainer()
        int contains(string name)
        Matrix& operator[](string name)
        vector[string] get_view_names()
        size_t get_size()
        string get_typename()


    cppclass BaseLayer:
        size_t in_size
        size_t out_size

        BaseLayer(size_t in_size, size_t out_size)

        string get_typename()

        size_t get_weight_size() except +

        size_t get_fwd_state_size(size_t n_batches, size_t n_slices) except +

        size_t get_bwd_state_size(size_t n_batches, size_t n_slices) except +

        ViewContainer* create_weights_view(Matrix& w) except +

        ViewContainer* create_fwd_state_view(Matrix&b, size_t n_batches, size_t n_slices) except +

        ViewContainer* create_bwd_state_view(Matrix&b, size_t n_batches, size_t n_slices) except +

        void forward_pass(ViewContainer& w, ViewContainer& b, Matrix& x, Matrix& y) except +

        void backward_pass(ViewContainer &w, ViewContainer &b, ViewContainer &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas) except +

        void gradient(ViewContainer &w, ViewContainer &grad, ViewContainer &b, ViewContainer &d, Matrix &y, Matrix& x, Matrix &out_deltas) except +

        void Rpass(ViewContainer &w, ViewContainer &v,  ViewContainer &b, ViewContainer &Rb, Matrix &x, Matrix &y, Matrix &Ry) except +

        void Rbackward(ViewContainer &w, ViewContainer &b, ViewContainer &d, Matrix &in_deltas, Matrix &out_deltas, ViewContainer &Rb, double _lambda, double mu) except +


    cppclass Layer[L]:
        Layer(size_t in_size, size_t out_size, L)
