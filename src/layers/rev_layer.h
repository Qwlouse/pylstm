#pragma once

#include "matrix/matrix.h"
#include "layer.hpp"


class ReverseLayer {
public:
	ReverseLayer();
	~ReverseLayer();

	class Weights : public ::ViewContainer {
	public:
	    Weights(size_t n_inputs, size_t n_cells);
	};

	class FwdState : public ::ViewContainer {
	public:
	    FwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
	};

	class BwdState: public ::ViewContainer {
	public:
	    BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
	};

	void forward(Weights &w, FwdState &b, Matrix &x, Matrix &y);
	void backward(Weights &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
	void gradient(Weights &w, Weights &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix &out_deltas);
    void Rpass(Weights &w, Weights &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry);
    void Rbackward(Weights &w, FwdState &b, BwdState &d, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};

