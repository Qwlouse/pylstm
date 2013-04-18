#include "matrix/matrix_operation.h"
#include <vector>
#include "Core.h"
#include "rnn_layer.h"

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
  Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

RnnLayer::BwdState::BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
    n_inputs(n_inputs_), n_cells(n_cells_),
    n_batches(n_batches_), time(time_),
    Ha(NULL, n_cells, n_batches, time),
    Hb(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
	add_view("Hb", &Hb);
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
        mult_add(w.HR, y.slice(t-1), b.Ha.slice(t));
      }
      add_vector_into(w.H_bias, b.Ha.slice(t));
      f->apply(b.Ha.slice(t), y.slice(t));
    }
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

    f->apply_deriv(y.slice(n_slices-1), out_deltas.slice(n_slices-1), d.Ha.slice(n_slices-1));
    mult(w.HX.T(), d.Ha.slice(n_slices-1), in_deltas.slice(n_slices-1));

    for (int t = n_slices-2; t >= 0; --t) {
        copy(out_deltas.slice(t), d.Hb.slice(t));
        mult_add(w.HR.T(), d.Ha.slice(t+1), d.Hb.slice(t));
        f->apply_deriv(y.slice(t), d.Hb.slice(t), d.Ha.slice(t));
        mult(w.HX.T(), d.Ha.slice(t), in_deltas.slice(t));
    }
    
}

void RnnLayer::gradient(RnnLayer::Weights&, RnnLayer::Weights& grad, RnnLayer::FwdState& b, RnnLayer::BwdState& d, Matrix&y, Matrix& x, Matrix& out_deltas) {
    
    size_t n_slices = x.n_slices;
    mult_add(d.Ha.slice(0), x.slice(0).T(), grad.HX);
    for (int t = 1; t < n_slices; ++t) {
        mult_add(d.Ha.slice(t), x.slice(t).T(), grad.HX);
        mult_add(d.Ha.slice(t), y.slice(t-1).T(), grad.HR);
    }
    
    squash(d.Ha, grad.H_bias);
    
}

void RnnLayer::Rpass(Weights& w, Weights& v,  FwdState& b, FwdState& Rb, Matrix& x, Matrix& y, Matrix& Ry)
{
    size_t n_inputs = w.n_inputs;
    size_t n_cells = w.n_cells;
    ASSERT(v.n_inputs == n_inputs);
    ASSERT(v.n_cells == n_cells);

    size_t n_batches = b.n_batches;
    size_t n_slices = b.time;
    ASSERT(b.n_cells == n_cells);
    ASSERT(b.n_inputs == n_inputs);

    ASSERT(Rb.time == n_slices);
    ASSERT(Rb.n_batches == n_batches);
    ASSERT(Rb.n_inputs == n_inputs);
    ASSERT(Rb.n_cells == n_cells);

    ASSERT(x.n_rows == n_inputs);
    ASSERT(x.n_columns == n_batches);
    ASSERT(x.n_slices == n_slices);

    ASSERT(y.n_rows == n_cells);
    ASSERT(y.n_columns == n_batches);
    ASSERT(y.n_slices == n_slices);


    mult(v.HX, x.flatten_time(), Rb.Ha.flatten_time());

    for (int t = 0; t < n_slices; ++t) {
      if (t) {
        mult_add(v.HR, y.slice(t-1), Rb.Ha.slice(t));
        mult_add(w.HR, Ry.slice(t-1), Rb.Ha.slice(t));
      }
      add_vector_into(v.H_bias, Rb.Ha.slice(t));
      f->apply_deriv(y.slice(t), Rb.Ha.slice(t), Ry.slice(t));
    }

}

void RnnLayer::Rbackward(Weights&, FwdState&, BwdState&, Matrix&, Matrix&, FwdState&, double, double)
{
    THROW(core::NotImplementedException("Rbackward pass not implemented yet."));
}

