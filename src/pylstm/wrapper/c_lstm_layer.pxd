# coding=utf-8

from c_matrix cimport MatrixView2DCPU, MatrixView3DCPU

cdef extern from "lstm_layer.h":
    cdef cppclass LstmWeights:
        int n_inputs
        int n_cells
        MatrixView2DCPU IX, IH, IS
        MatrixView2DCPU FX, FH, FS
        MatrixView2DCPU ZX, ZH
        MatrixView2DCPU OX, OH, OS
        MatrixView2DCPU I_bias, F_bias, Z_bias, O_bias

        LstmWeights(int, int)
        int buffer_size()
        void allocate(MatrixView2DCPU)

    cdef cppclass LstmBuffers:
        LstmBuffers(int, int, int, int)
        int buffer_size()
        void allocate(MatrixView2DCPU)

    cdef cppclass LstmDeltas:
        LstmDeltas(int, int, int, int)
        int buffer_size()
        void allocate(MatrixView2DCPU)

    void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y)
    void lstm_backward(LstmWeights &w, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas)
    void lstm_grad(LstmWeights &w, LstmWeights &grad, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU input_batches)


cdef extern from "fwd_layer.h":
    cdef cppclass FwdWeights:
        FwdWeights(size_t, size_t)
        size_t buffer_size()

    cdef cppclass FwdBuffers:
        FwdBuffers(size_t, size_t, size_t, size_t)
        size_t buffer_size()

    cdef cppclass FwdDeltas :
        FwdDeltas(size_t , size_t , size_t , size_t )
        size_t buffer_size()

    void fwd_forward(FwdWeights &w, FwdBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y)
    void fwd_backward(FwdWeights &w, FwdBuffers &b, FwdDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas)
    void fwd_grad(FwdWeights &w, FwdWeights &grad, FwdBuffers &b, FwdDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU input_batches)
