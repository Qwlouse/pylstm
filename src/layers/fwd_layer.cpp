#include "fwd_layer.h"
#include "matrix/matrix_operation.h"
#include <vector>
#include "Core.h"

using std::vector;

RegularLayer::RegularLayer():
	f(Sigmoid())
{ }

RegularLayer::RegularLayer(ActivationFunction f):
	f(f)
{ }


size_t RegularLayer::Weights::estimate_size(size_t n_inputs, size_t n_cells)
{
	return n_inputs * n_cells + n_cells;
}


RegularLayer::Weights::Weights(size_t n_inputs_, size_t n_cells_, Matrix& buffer) :
  n_inputs(n_inputs_), 
  n_cells(n_cells_), 
  HX(NULL, n_cells, n_inputs, 1),
  H_bias(NULL, n_cells, 1, 1)
{
	add_view("HX", &HX);
	add_view("H_bias", &H_bias);
	lay_out(buffer);
}

size_t RegularLayer::Weights::size() {
  return HX.size + H_bias.size;  //!< inputs X, H, to cells H + bias to H

}

////////////////////// Fwd Buffer /////////////////////////////////////////////

size_t RegularLayer::FwdState::estimate_size(size_t n_inputs, size_t n_cells, size_t n_batches_, size_t time_)
{
	return n_cells * n_batches_ * time_;
}

RegularLayer::FwdState::FwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_, Matrix& buffer) :
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),
  Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
	lay_out(buffer);
}

size_t RegularLayer::FwdState::size() {
 //Views on all activations
  return Ha.size; //!< Hidden unit activation
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

size_t RegularLayer::BwdState::estimate_size(size_t n_inputs, size_t n_cells, size_t n_batches_, size_t time_)
{
	return 2 * n_cells * n_batches_ * time_;
}

RegularLayer::BwdState::BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_, Matrix& buffer) :
    n_inputs(n_inputs_), n_cells(n_cells_),
    n_batches(n_batches_), time(time_),
    Ha(NULL, n_cells, n_batches, time),
	Hb(NULL, n_cells, n_batches, time)
{
	add_view("HX", &Ha);
	add_view("H_bias", &Hb);
	lay_out(buffer);
}

size_t RegularLayer::BwdState::size() {
 //Views on all activations
  return Ha.size; //!< Hidden unit activation
}


////////////////////// Methods /////////////////////////////////////////////
void RegularLayer::forward(RegularLayer::Weights &w, RegularLayer::FwdState &b, Matrix &x, Matrix &y) {
	size_t n_inputs = w.n_inputs;
	size_t n_cells = w.n_cells;
	size_t n_batches = b.n_batches;
	size_t n_slices = b.time;
	ASSERT(b.n_cells == n_cells);
	ASSERT(b.n_inputs == n_inputs);

	ASSERT(x.n_rows == n_inputs);
	ASSERT(x.n_columns == n_batches);
	ASSERT(x.n_slices == n_slices);

	ASSERT(y.n_rows == n_cells);
	ASSERT(y.n_columns == n_batches);
	ASSERT(y.n_slices == n_slices);

	for (int t = 0; t < n_slices; ++t) {
		mult(w.HX, x.slice(t), b.Ha.slice(t));
	}

	add_vector_into(w.H_bias, b.Ha);
	f.apply(b.Ha, y);
}



void RegularLayer::backward(RegularLayer::Weights &w, RegularLayer::FwdState &b, RegularLayer::BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas) {
	size_t n_inputs = w.n_inputs;
	size_t n_cells = w.n_cells;
	size_t n_batches = b.n_batches;
	size_t n_slices = b.time;
	ASSERT(b.n_cells == n_cells);
	ASSERT(b.n_inputs == n_inputs);

	ASSERT(in_deltas.n_rows == n_inputs);
	ASSERT(in_deltas.n_columns == n_batches);
	ASSERT(in_deltas.n_slices == n_slices);

	ASSERT(y.n_rows == n_cells);
	ASSERT(y.n_columns == n_batches);
	ASSERT(y.n_slices == n_slices);

	ASSERT(out_deltas.n_rows == n_cells);
	ASSERT(out_deltas.n_columns == n_batches);
	ASSERT(out_deltas.n_slices == n_slices);
    f.apply_deriv(y, d.Hb);
	dot(d.Hb, out_deltas, d.Ha);

    for (int t = 0; t < n_slices; ++t) {
        mult(w.HX.T(), d.Ha.slice(t), in_deltas.slice(t));
    }
}

void RegularLayer::gradient(RegularLayer::Weights &w, RegularLayer::Weights &grad, RegularLayer::FwdState &b, RegularLayer::BwdState &d, Matrix &y, Matrix& x, Matrix& out_deltas)
{
	size_t n_slices = x.n_slices;
	for (int t = 0; t < n_slices; ++t) {
		mult_add(d.Ha.slice(t), x.slice(t).T(), grad.HX);
	}

    squash(d.Ha, grad.H_bias);
}
