from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map


cdef extern from "matrix.h":
    ctypedef double d_type

    cdef cppclass Matrix:
        int n_rows
        int n_columns
        int n_slices
        int size

        Matrix()
        Matrix(Matrix&)
        Matrix(int, int, int)
        Matrix(d_type* data_ptr, size_t n_rows, size_t n_columns, size_t n_slices)
        d_type& operator[](size_t) except +
        Matrix T() except +
        d_type& get(size_t row, size_t col, size_t slice) except +
        d_type* get_data() except +
        Matrix sub_matrix(size_t start, size_t n_rows, size_t n_columns, size_t n_slices) except +
        Matrix slice(size_t slice_index) except +
        Matrix slice(size_t start, size_t stop) except +
        Matrix row_slice(size_t row_index) except +
        Matrix row_slice(size_t start_row, size_t stop_row) except +
        void set_all_elements_to(d_type value) except +
        void print_me() except +


cdef extern from "matrix_operation.h":
#    void add(Matrix a, Matrix b, Matrix out)
    cppclass ActivationFunction:
        pass
    cppclass SoftmaxLayerActivation(ActivationFunction):
        pass
    cppclass WinoutActivation(ActivationFunction):
        pass

    ActivationFunction Sigmoid
    ActivationFunction Tanh
    ActivationFunction Tanhx2
    ActivationFunction Linear
    ActivationFunction RectifiedLinear
    SoftmaxLayerActivation Softmax
    WinoutActivation Winout
    ActivationFunction TanhScaled



    void add_into_b(Matrix a, Matrix b) except +

    void add_scalar(Matrix a, double b) except +

    void mult(Matrix a, Matrix b, Matrix out) except +

    void mult_add(Matrix a, Matrix b, Matrix out) except +

    void hard_compete_locally(Matrix mask, Matrix x, Matrix out, unsigned int block_size) except +

    void dot(Matrix a, Matrix b, Matrix out) except +

    void dot_add(Matrix a, Matrix b, Matrix out) except +

    void apply_sigmoid(Matrix a, Matrix out) except +

    void apply_tanh(Matrix a, Matrix out) except +

    void apply_tanhx2(Matrix a, Matrix out) except +

    bool equals(Matrix a, Matrix out) except +


cdef extern from "matrix_container.h":
    ctypedef map[string, Matrix] MatrixContainerSlice

    cppclass MatrixContainer:
        MatrixContainer()
        int contains(string name) except +
        Matrix& operator[](string name) except +
        vector[string] get_view_names() except +
        size_t get_size() except +
        string get_typename() except +
        MatrixContainerSlice* slice(size_t start, size_t stop) except +
        void set_values(MatrixContainerSlice* slice, size_t start) except +
