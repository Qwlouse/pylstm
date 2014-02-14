#include "lwta_layer.h"

#include <vector>

#include "Core.h"
#include <iostream>
#include "matrix/matrix_operation.h"

using std::vector;

LWTALayer::LWTALayer():
	block_size(2)
{ }

LWTALayer::LWTALayer(unsigned int block_size):
    block_size(block_size)
{ }

LWTALayer::~LWTALayer()
{
}

LWTALayer::Parameters::Parameters(size_t, size_t)
{
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

LWTALayer::FwdState::FwdState(size_t n_inputs, size_t, size_t n_batches, size_t time) :
    Mask(NULL, n_inputs, n_batches, time)
{
	add_view("Mask", &Mask);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

LWTALayer::BwdState::BwdState(size_t, size_t, size_t, size_t)
{
}

////////////////////// Methods /////////////////////////////////////////////
void LWTALayer::forward(LWTALayer::Parameters &, LWTALayer::FwdState &b, Matrix &x, Matrix &y, bool) {
    hard_compete_locally(b.Mask, x, y, block_size);
}

void LWTALayer::backward(LWTALayer::Parameters &, LWTALayer::FwdState &b, LWTALayer::BwdState &, Matrix &, Matrix &in_deltas, Matrix &out_deltas) {
    dot(b.Mask, out_deltas.flatten_time(), in_deltas.flatten_time());
}

void LWTALayer::gradient(LWTALayer::Parameters&, LWTALayer::Parameters&, LWTALayer::FwdState&, LWTALayer::BwdState&, Matrix&, Matrix&, Matrix&)
{
}

void LWTALayer::Rpass(Parameters &, Parameters &,  FwdState &, FwdState &, Matrix &, Matrix &, Matrix&, Matrix &)
{
}

void LWTALayer::dampened_backward(Parameters&, FwdState&, BwdState&, Matrix&, Matrix&, Matrix&, FwdState&, double, double)
{
}
