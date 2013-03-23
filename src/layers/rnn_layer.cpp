#include "matrix/matrix_operation.h"
#include <vector>
#include "Core.h"
#include "new_layer.h"

using std::vector;

RnnLayer::RnnLayer():
	f(&Sigmoid)
{ }

RnnLayer::RnnLayer(const ActivationFunction* f):
	f(f)
{ }

RnnLayer::~RnnLayer()
{
}

RnnLayer::Weights::Weights(size_t n_inputs_, size_t n_cells_) :
  n_inputs(n_inputs_), 
  n_cells(n_cells_), 
  HX(NULL, n_cells, n_inputs, 1),
  HR(NULL, n_cells, n_cells, 1),
  H_bias(NULL, n_cells, 1, 1)
{
    add_view("HX", &HX);
    add_view("HR", &HR);
    add_view("H_bias", &H_bias);
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

RnnLayer::FwdState::FwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),
  Ha(NULL, n_cells, n_batches, time),
  Hb(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
	add_view("Hb", &Hb);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

RnnLayer::BwdState::BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
    n_inputs(n_inputs_), n_cells(n_cells_),
    n_batches(n_batches_), time(time_),
    Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Methods /////////////////////////////////////////////
void RnnLayer::forward(RnnLayer::Weights &w, RnnLayer::FwdState &b, Matrix &x, Matrix &y) {
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

    mult(w.HX, x.flatten_time(), b.Ha.flatten_time());
    for (int t = 0; t < n_slices; ++t) {
      if (t) {
        mult_add(w.HR, x.slice(t-1), b.Ha.slice(t));
      }
      add_vector_into(w.H_bias, b.Ha.slice(t));
    }
    f->apply(b.Ha, y);
    copy(y, b.Hb);
}
void RnnLayer::backward(RnnLayer::Weights &w, RnnLayer::FwdState &b, RnnLayer::BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas) {
    
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

    f->apply_deriv(y, out_deltas, d.Ha);
    mult(w.HX.T(), d.Ha.slice(n_slices-1), in_deltas.slice(n_slices-1));

    for (size_t t = n_slices-2; t > 0; --t) {
        mult_add(w.HR.T(), d.Ha.slice(t+1), d.Ha.slice(t));
        mult(w.HX.T(), d.Ha.slice(t), in_deltas.slice(t));
    }
    
}

void RnnLayer::gradient(RnnLayer::Weights&, RnnLayer::Weights& grad, RnnLayer::FwdState& b, RnnLayer::BwdState& d, Matrix&, Matrix& x, Matrix&) {
    
    size_t n_slices = x.n_slices;
	for (int t = 0; t < n_slices; ++t) {
        mult_add(d.Ha.slice(t), x.slice(t).T(), grad.HX);
		mult_add(d.Ha.slice(t), b.Hb.slice(t).T(), grad.HR);
	}
    
    squash(d.Ha, grad.H_bias);
    
}

void RnnLayer::Rpass(Weights&, Weights&,  FwdState&, FwdState&, Matrix&, Matrix&, Matrix&)
{
    THROW(core::NotImplementedException("Rpass not implemented yet."));
}

void RnnLayer::Rbackward(Weights&, FwdState&, BwdState&, Matrix&, Matrix&, FwdState&, double, double)
{
    THROW(core::NotImplementedException("Rbackward pass not implemented yet."));
}

