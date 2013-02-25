/**
 * \file fwd_layer.h
 * \brief Declares the whole exception hierarchy.
 *
 * \details
 */


#ifndef __FWD_LAYER_H__
#define __FWD_LAYER_H__

#include "matrix/matrix.h"
#include <iostream>

class FwdWeights {
public:
	static size_t estimate_size(size_t n_inputs, size_t n_cells);


	size_t n_inputs, n_cells;
	///Variables defining sizes
	Matrix HX;  //!< inputs X, H, S to input gate I
	Matrix H_bias;   //!< bias to input gate, forget gate, state Z, output gate

	FwdWeights(size_t n_inputs, size_t n_cells, Matrix& buffer);

	size_t buffer_size();
	void allocate(Matrix buffer_view);
};

struct FwdBuffers {
	///Variables defining sizes
	size_t n_inputs, n_outputs, n_cells;
	size_t n_batches, time;

	//Views on all activations
	Matrix Ha; //!< Hidden unit activation and output

	FwdBuffers(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);

	size_t buffer_size();
	void allocate(Matrix buffer_view);
};

struct FwdDeltas {
	///Variables defining sizes
	size_t n_inputs, n_outputs, n_cells;
	size_t n_batches, time;

	//Views on all activations
	Matrix Ha, Hb; //Hidden unit activation and output

	//MatrixView3DCPU temp_hidden, temp_hidden2; //temp values, neccessary?

	FwdDeltas(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
	size_t buffer_size();
	void allocate(Matrix buffer_view);
};

void fwd_forward(FwdWeights &w, FwdBuffers &b, Matrix &x, Matrix &y);
void fwd_backward(FwdWeights &w, FwdBuffers &b, FwdDeltas &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
void fwd_grad(FwdWeights &w, FwdWeights &grad, FwdBuffers &b, FwdDeltas &d, Matrix &y, Matrix input_batches);

#endif
