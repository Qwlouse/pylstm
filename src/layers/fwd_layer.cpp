/**
 * \file fwd_layer.cpp
 * \brief Implementation of the fwd_layer.
 */


#include "fwd_layer.h"
#include "matrix/matrix_operation.h"
#include <vector>
#include "Core.h"

using std::vector;

size_t FwdWeights::estimate_size(size_t n_inputs, size_t n_cells)
{
	return n_inputs * n_cells + n_cells;
}

void lay_out(Matrix& buffer, vector<Matrix*> views)
{
	size_t offset = 0;
	for (size_t i = 0; i < views.size(); ++i) {
		size_t rows = views[i]->n_rows;
		size_t cols = views[i]->n_columns;
		size_t slices = views[i]->n_slices;
		*views[i] = buffer.subslice(offset, rows, cols, slices);
		offset += views[i]->size;
		ASSERT(offset <= buffer.size);
	}
}

FwdWeights::FwdWeights(size_t n_inputs_, size_t n_cells_, Matrix& buffer) :
  n_inputs(n_inputs_), 
  n_cells(n_cells_), 
  HX(boost::shared_array<double>(NULL), 0, NORMAL, n_cells, n_inputs, 1),
  H_bias(boost::shared_array<double>(NULL), 0, NORMAL, n_cells, 1, 1)
{
	vector<Matrix*> views;
	views.push_back(&HX);
	views.push_back(&H_bias);
	lay_out(buffer, views);
}

size_t FwdWeights::buffer_size() {
  return HX.size + H_bias.size;  //!< inputs X, H, to cells H + bias to H

}
/*
FwdBuffers::FwdBuffers(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),
  Ha(n_cells, n_batches, time)
{}

size_t FwdBuffers::buffer_size() {
 //Views on all activations
  return Ha.size; //!< Hidden unit activation
    
}

void FwdBuffers::allocate(Matrix buffer_view) {
  vector<Matrix*> views;

  views.push_back(&Ha);

  lay_out(buffer_view, views);
}

FwdDeltas::FwdDeltas(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  ///Variables defining sizes
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),

  //Views on all activations
  Ha(n_cells, n_batches, time), Hb(n_cells, n_batches, time) //Input gate activation
{}

size_t FwdDeltas::buffer_size() {
  return Ha.size + Hb.size; //Hidden unit activation
}

void FwdDeltas::allocate(Matrix buffer_view) {
  vector<Matrix*> views;

  views.push_back(&Ha);
  views.push_back(&Hb);

  lay_out(buffer_view, views);
}

void fwd_forward(FwdWeights &w, FwdBuffers &b, Matrix &x, Matrix &y) {
  mult(w.HX, x.flatten(), b.Ha.flatten());

  add_vector_into(w.H_bias, b.Ha);
  apply_sigmoid(b.Ha, y);
}

void fwd_backward(FwdWeights &w, FwdBuffers &b, FwdDeltas &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas) {
  apply_sigmoid_deriv(y, d.Hb);

  dot(d.Hb, out_deltas, d.Ha);

  mult(w.HX.T(), d.Ha, in_deltas);
}

void fwd_grad(FwdWeights &w, FwdWeights &grad, FwdBuffers &b, FwdDeltas &d, Matrix &y, Matrix input_batches)  {

  mult(d.Ha, input_batches.T(), grad.HX);
  squash(d.Ha, grad.H_bias);
}
*/
