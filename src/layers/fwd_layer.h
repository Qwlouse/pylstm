#pragma once

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"
#include <iostream>
#include "layer.hpp"


class RegularLayer {
public:
	const ActivationFunction* f;
	RegularLayer();
	explicit RegularLayer(const ActivationFunction* f);
	~RegularLayer();
	///////////// Classes
	class Weights : public ::ViewContainer {
	public:
		size_t n_inputs, n_cells;
		///Variables defining sizes
		Matrix HX;  //!< inputs X, H, S to input gate I
		Matrix H_bias;   //!< bias to input gate, forget gate, state Z, output gate

		Weights(size_t n_inputs, size_t n_cells);
	};

	class FwdState : public ::ViewContainer{
	public:
		///Variables defining sizes
		size_t n_inputs, n_cells;
		size_t n_batches, time;

		//Views on all activations
		Matrix Ha; //!< Hidden unit activation and output

		FwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
	};

	struct BwdState: public ::ViewContainer {
		///Variables defining sizes
		size_t n_inputs, n_cells;
		size_t n_batches, time;

		//Views on all activations
		Matrix Ha, Hb; //Hidden unit activation and output

		BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
	};

	void forward(Weights &w, FwdState &b, Matrix &x, Matrix &y);
	void backward(Weights &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
	void gradient(Weights &w, Weights &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix &out_deltas);
};

typedef Layer<RegularLayer> RLayer;

