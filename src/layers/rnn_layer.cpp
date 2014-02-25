#include "rnn_layer.h"

#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"

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

RnnLayer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
  HX(NULL, n_cells, n_inputs, 1),
  HR(NULL, n_cells, n_cells, 1),
  H_bias(NULL, n_cells, 1, 1)
{
    add_view("HX", &HX);
    add_view("HR", &HR);
    add_view("H_bias", &H_bias);
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

RnnLayer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
  Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

RnnLayer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    Ha(NULL, n_cells, n_batches, time),
    Hb(NULL, n_cells, n_batches, time),
    tmp(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
	add_view("Hb", &Hb);
	add_view("tmp", &tmp);
}

////////////////////// Methods /////////////////////////////////////////////
void RnnLayer::forward(RnnLayer::Parameters& w, RnnLayer::FwdState& b, Matrix& x, Matrix& y, bool) {
    size_t n_slices = x.n_slices;
    mult(w.HX, x.slice(1,x.n_slices).flatten_time(), b.Ha.slice(1,b.Ha.n_slices).flatten_time());
    for (int t = 1; t < n_slices; ++t) {
        mult_add(w.HR, y.slice(t-1), b.Ha.slice(t));
        add_vector_into(w.H_bias, b.Ha.slice(t));
        f->apply(b.Ha.slice(t), y.slice(t));
    }
}
void RnnLayer::backward(RnnLayer::Parameters& w, RnnLayer::FwdState&b, RnnLayer::BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
    dampened_backward(w, b, d, y, in_deltas, out_deltas, b, 0., 0.);
}

void RnnLayer::gradient(RnnLayer::Parameters&, RnnLayer::Parameters& grad, RnnLayer::FwdState& , RnnLayer::BwdState& d, Matrix& y, Matrix& x, Matrix&) {
    size_t n_slices = x.n_slices;
    for (int t = 1; t < n_slices; ++t) {
        mult_add(d.Ha.slice(t), x.slice(t).T(), grad.HX);
        mult_add(d.Ha.slice(t), y.slice(t-1).T(), grad.HR);
    }
    
    squash(d.Ha.slice(1, n_slices), grad.H_bias);
}

void RnnLayer::Rpass(Parameters& w, Parameters& v,  FwdState&, FwdState& Rb, Matrix& x, Matrix& y, Matrix& Rx, Matrix& Ry)
{
    size_t n_slices = x.n_slices;
    mult(v.HX, x.slice(1,x.n_slices).flatten_time(), Rb.Ha.slice(1,Rb.Ha.n_slices).flatten_time());
    mult_add(w.HX, Rx.slice(1,x.n_slices).flatten_time(), Rb.Ha.slice(1,Rb.Ha.n_slices).flatten_time());
    for (int t = 1; t < n_slices; ++t) {
      mult_add(v.HR, y.slice(t-1), Rb.Ha.slice(t));
      mult_add(w.HR, Ry.slice(t-1), Rb.Ha.slice(t));

      add_vector_into(v.H_bias, Rb.Ha.slice(t));
      f->apply_deriv(y.slice(t), Rb.Ha.slice(t), Ry.slice(t));
    }

}

void RnnLayer::dampened_backward(Parameters& w, FwdState&, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas, FwdState& Rb, double lambda, double mu)
{
    size_t n_slices = y.n_slices;
    f->apply_deriv(y.slice(n_slices-1), out_deltas.slice(n_slices-1), d.Ha.slice(n_slices-1));
    // dampening for last timestep
    f->apply_deriv(y.slice(n_slices-1), Rb.Ha.slice(n_slices-1), d.tmp.slice(n_slices-1));
    scale_into(d.tmp.slice(n_slices-1), lambda * mu);
    add_into_b(d.tmp.slice(n_slices-1), d.Ha.slice(n_slices-1));
    for (int t = static_cast<int>(n_slices - 2); t >= 0; --t) {
        copy(out_deltas.slice(t), d.Hb.slice(t));
        mult_add(w.HR.T(), d.Ha.slice(t+1), d.Hb.slice(t));
        f->apply_deriv(y.slice(t), d.Hb.slice(t), d.Ha.slice(t));
        // dampening
        f->apply_deriv(y.slice(t), Rb.Ha.slice(t), d.tmp.slice(t));
        scale_into(d.tmp.slice(t), lambda * mu);
        add_into_b(d.tmp.slice(t), d.Ha.slice(t));
    }
    mult_add(w.HX.T(), d.Ha.slice(1,d.Ha.n_slices).flatten_time(), in_deltas.slice(1,in_deltas.n_slices).flatten_time());
}

