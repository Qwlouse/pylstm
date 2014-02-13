#include "forward_layer.h"

#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"

using std::vector;

ForwardLayer::ForwardLayer():
	f(&Sigmoid),
	use_bias(true)
{ }

ForwardLayer::ForwardLayer(const ActivationFunction* f):
	f(f),
    use_bias(true)
{ }

ForwardLayer::~ForwardLayer()
{
}

ForwardLayer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
    HX(NULL, n_cells, n_inputs, 1),
    H_bias(NULL, n_cells, 1, 1)
{
	add_view("HX", &HX);
	add_view("H_bias", &H_bias);
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

ForwardLayer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

ForwardLayer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Methods /////////////////////////////////////////////
void ForwardLayer::forward(ForwardLayer::Parameters &w, ForwardLayer::FwdState &b, Matrix &x, Matrix &y, bool) {
	mult(w.HX, x.flatten_time(), b.Ha.flatten_time());

    if (use_bias)
	    add_vector_into(w.H_bias, b.Ha);
	f->apply(b.Ha, y);
}

void ForwardLayer::backward(ForwardLayer::Parameters &w, ForwardLayer::FwdState &, ForwardLayer::BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas) {
    f->apply_deriv(y, out_deltas, d.Ha);
    mult_add(w.HX.T(), d.Ha.flatten_time(), in_deltas.flatten_time());
}

void ForwardLayer::gradient(ForwardLayer::Parameters&, ForwardLayer::Parameters& grad, ForwardLayer::FwdState&, ForwardLayer::BwdState& d, Matrix&, Matrix& x, Matrix&)
{
	mult_add(d.Ha.flatten_time(), x.flatten_time().T(), grad.HX);
	if (use_bias)
        squash(d.Ha, grad.H_bias);
}

void ForwardLayer::Rpass(Parameters &w, Parameters &v,  FwdState &, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry)
{
    // Rb.Ha = W Rx + V x
    mult(v.HX, x.flatten_time(), Rb.Ha.flatten_time());
    mult_add(w.HX, Rx.flatten_time(), Rb.Ha.flatten_time());

    if (use_bias)
        add_vector_into(v.H_bias, Rb.Ha);

	// Ry = f'(b.Ha)*Rb.Ha
    f->apply_deriv(y, Rb.Ha, Ry);
}

void ForwardLayer::dampened_backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas, FwdState&, double, double)
{
    backward(w, b, d, y, in_deltas, out_deltas);
}
