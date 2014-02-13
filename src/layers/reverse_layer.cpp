#include "reverse_layer.h"

#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"

using std::vector;

ReverseLayer::ReverseLayer()
{ }

ReverseLayer::~ReverseLayer()
{ }

ReverseLayer::Parameters::Parameters(size_t, size_t)
{ }

////////////////////// Fwd Buffer /////////////////////////////////////////////

ReverseLayer::FwdState::FwdState(size_t, size_t, size_t, size_t)
{ }

////////////////////// Bwd Buffer /////////////////////////////////////////////

ReverseLayer::BwdState::BwdState(size_t, size_t, size_t, size_t)
{ }

////////////////////// Methods /////////////////////////////////////////////

void reverse(Matrix &x, Matrix &y)
{
    size_t n_slices = x.n_slices;
	for (int t = 0; t < n_slices; ++t) {
	    copy(x.slice(t), y.slice(n_slices-t-1));
	}
}

void reverse_add(Matrix &x, Matrix &y)
{
    size_t n_slices = x.n_slices;
	for (int t = 0; t < n_slices; ++t) {
	    add_into_b(x.slice(t), y.slice(n_slices-t-1));
	}
}


void ReverseLayer::forward(ReverseLayer::Parameters&, ReverseLayer::FwdState&, Matrix &x, Matrix &y, bool) {
	reverse(x, y);
}

void ReverseLayer::backward(ReverseLayer::Parameters&, ReverseLayer::FwdState&, ReverseLayer::BwdState&, Matrix&, Matrix& in_deltas, Matrix& out_deltas) {
	reverse_add(out_deltas, in_deltas);
}

void ReverseLayer::gradient(ReverseLayer::Parameters&, ReverseLayer::Parameters&, ReverseLayer::FwdState&, ReverseLayer::BwdState&, Matrix&, Matrix&, Matrix&)
{
    // no parameters ... no gradient
}

void ReverseLayer::Rpass(Parameters&, Parameters&,  FwdState&, FwdState&, Matrix&, Matrix&, Matrix& Rx, Matrix& Ry)
{
    reverse(Rx, Ry);
}

void ReverseLayer::dampened_backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas, FwdState&, double, double)
{
    backward(w, b, d, y, in_deltas, out_deltas);
}
