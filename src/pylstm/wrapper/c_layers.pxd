from c_matrix cimport Matrix
from libcpp.string cimport string

cdef extern from "fwd_layer.h":
    cppclass RegularLayer:
        pass

cdef extern from "layer.hpp":
    cppclass ViewContainer:
        ViewContainer()
        Matrix& operator[](string name)
        
    cppclass BaseLayer:
        size_t in_size
        size_t out_size
        
        BaseLayer(size_t in_size, size_t out_size)
    
        size_t get_weight_size()
    
        size_t get_fwd_state_size(size_t n_batches, size_t n_slices)
    
        size_t get_bwd_state_size(size_t n_batches, size_t n_slices)
    
        ViewContainer* create_weights_view(Matrix& w)
    
        ViewContainer* create_fwd_state_view(Matrix&b, size_t n_batches, size_t n_slices)
    
        ViewContainer* create_bwd_state_view(Matrix&b, size_t n_batches, size_t n_slices)
        
        void forward_pass(ViewContainer& w, ViewContainer& b, Matrix& x, Matrix& y)
    
        void backward_pass(ViewContainer &w, ViewContainer &b, ViewContainer &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas)
    
        void gradient(ViewContainer &w, ViewContainer &grad, ViewContainer &b, ViewContainer &d, Matrix &y, Matrix& x, Matrix &out_deltas)
   
   
cdef extern from "factory.hpp":
    BaseLayer* create_layer(string name, size_t in_size, size_t out_size)