#pragma once

#include <iostream>

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"
#include "matrix/matrix_container.h"

class RnnLayer {
public:
    const ActivationFunction* f;
    RnnLayer();
    explicit RnnLayer(const ActivationFunction* f);
    ~RnnLayer();

    class Parameters : public ::MatrixContainer {
    public:
        size_t n_inputs, n_cells;
        Matrix HX;
        Matrix HR;
        Matrix H_bias;

        Parameters(size_t n_inputs, size_t n_cells);
    };

    class FwdState : public ::MatrixContainer{
    public:
        ///Variables defining sizes
        size_t n_inputs, n_cells;
        size_t n_batches, time;

        Matrix Ha; //!< total input activations for all neurons in layer

        FwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
    };

    class BwdState: public ::MatrixContainer {
    public:
        ///Variables defining sizes
        size_t n_inputs, n_cells;
        size_t n_batches, time;

        Matrix Ha; //!< activations for all neurons in layer
        Matrix Hb;

        BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
    };

    void forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y);
    void backward(Parameters &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
    void gradient(Parameters &w, Parameters &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix &out_deltas);
    void Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry);
    void dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};
