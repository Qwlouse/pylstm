from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


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
        d_type& operator[](size_t)
        Matrix T()
        d_type& get(size_t row, size_t col, size_t slice)
        d_type* get_data()
        Matrix sub_matrix(size_t start, size_t n_rows, size_t n_columns, size_t n_slices)
        Matrix slice(size_t slice_index)
        Matrix slice(size_t start, size_t stop)
        Matrix row_slice(size_t row_index)
        Matrix row_slice(size_t start_row, size_t stop_row)
        void set_all_elements_to(d_type value)
        void print_me()


cdef extern from "matrix_operation.h":
#    void add(Matrix a, Matrix b, Matrix out)
    cppclass ActivationFunction:
        pass
    cppclass SoftmaxLayerActivation(ActivationFunction):
        pass

    ActivationFunction Sigmoid
    ActivationFunction Tanh
    ActivationFunction Tanhx2
    ActivationFunction Linear
    ActivationFunction RectifiedLinear
    SoftmaxLayerActivation Softmax



    void add_into_b(Matrix a, Matrix b)

    void add_scalar(Matrix a, double b)

    void mult(Matrix a, Matrix b, Matrix out)

    void mult_add(Matrix a, Matrix b, Matrix out)

    void dot(Matrix a, Matrix b, Matrix out)

    void dot_add(Matrix a, Matrix b, Matrix out)

    void apply_sigmoid(Matrix a, Matrix out)

    void apply_tanh(Matrix a, Matrix out)

    void apply_tanhx2(Matrix a, Matrix out)

    bool equals(Matrix a, Matrix out)


cdef extern from "matrix_container.h":
    cppclass MatrixContainer:
        MatrixContainer()
        int contains(string name)
        Matrix& operator[](string name)
        vector[string] get_view_names()
        size_t get_size()
        string get_typename()