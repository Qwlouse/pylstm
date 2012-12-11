from libcpp cimport bool

cdef extern from "matrix_cpu.h":
    cdef cppclass MatrixView2DCPU:
        MatrixView2DCPU()

    cdef cppclass MatrixCPU:
        MatrixCPU(int, int, int)
        MatrixCPU(double*, int, int, int) #todo use d_type and size_t
        void print_me()
        MatrixView2DCPU standard_view_2d


cdef extern from "matrix_operation_cpu.h":
    void add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void add_into_b(MatrixView2DCPU a, MatrixView2DCPU b)

    void add_scalar(MatrixView2DCPU a, double b)

    void mult(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void mult_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void dot(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void dot_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void apply_sigmoid(MatrixView2DCPU a, MatrixView2DCPU out)

    void apply_tanh(MatrixView2DCPU a, MatrixView2DCPU out)

    void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out)

    bool equals(MatrixView2DCPU a, MatrixView2DCPU out)
