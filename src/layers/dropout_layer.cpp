#include "dropout_layer.h"

#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"

using std::vector;

DropoutLayer::DropoutLayer():
	drop_prob(0.5)
{ }

DropoutLayer::DropoutLayer(double drop_prob):
    drop_prob(drop_prob)
{ }

DropoutLayer::~DropoutLayer()
{
}

DropoutLayer::Parameters::Parameters(size_t n_inputs, size_t) :
    Mask(NULL, n_inputs, 1, 1)
{
	add_view("Mask", &Mask);
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

DropoutLayer::FwdState::FwdState(size_t n_inputs, size_t, size_t n_batches, size_t time) :
    Ha(NULL, n_inputs, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

DropoutLayer::BwdState::BwdState(size_t n_inputs, size_t, size_t n_batches, size_t time) :
    Ha(NULL, n_inputs, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Methods /////////////////////////////////////////////
void DropoutLayer::forward(DropoutLayer::Parameters &w, DropoutLayer::FwdState &b, Matrix &x, Matrix &) {
	dot(w.Mask, x.flatten_time(), b.Ha.flatten_time());

}

void DropoutLayer::backward(DropoutLayer::Parameters &w, DropoutLayer::FwdState &, DropoutLayer::BwdState &, Matrix &, Matrix &in_deltas, Matrix &out_deltas) {
    dot(w.Mask, out_deltas.flatten_time(), in_deltas.flatten_time());
}

void DropoutLayer::gradient(DropoutLayer::Parameters&, DropoutLayer::Parameters&, DropoutLayer::FwdState&, DropoutLayer::BwdState&, Matrix&, Matrix&, Matrix&)
{
}

void DropoutLayer::Rpass(Parameters &w, Parameters &v,  FwdState &, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry)
{
}

void DropoutLayer::dampened_backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas, FwdState&, double, double)
{
}
