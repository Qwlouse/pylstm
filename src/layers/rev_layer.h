#pragma once

#include "matrix/matrix.h"
#include "matrix/matrix_container.h"


class ReverseLayer {
public:
	ReverseLayer();
	~ReverseLayer();

	class Parameters : public ::MatrixContainer {
	public:
	    Parameters(size_t n_inputs, size_t n_cells);
	};

	class FwdState : public ::MatrixContainer {
	public:
	    FwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
	};

	class BwdState: public ::MatrixContainer {
	public:
	    BwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
	};

	void forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y);
	void backward(Parameters &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
	void gradient(Parameters &w, Parameters &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix &out_deltas);
    void Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry);
    void dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};

