#include "dropout_layer.h"

#include <vector>

#include "Core.h"
#include <iostream>
#include "matrix/matrix_operation.h"

using std::vector;

DropoutLayer::DropoutLayer():
	drop_prob(0.5),
	rnd_state(42)
{ }

DropoutLayer::DropoutLayer(double drop_prob, unsigned int initial_state):
    drop_prob(drop_prob),
    rnd_state(initial_state)
{ }

DropoutLayer::~DropoutLayer()
{
}

DropoutLayer::Parameters::Parameters(size_t, size_t)
{
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

DropoutLayer::FwdState::FwdState(size_t n_inputs, size_t, size_t n_batches, size_t time) :
    Mask(NULL, n_inputs, n_batches, time)
{
	add_view("Mask", &Mask);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

DropoutLayer::BwdState::BwdState(size_t, size_t, size_t, size_t)
{
}

////////////////////// Methods /////////////////////////////////////////////
void DropoutLayer::forward(DropoutLayer::Parameters &, DropoutLayer::FwdState &b, Matrix &x, Matrix &y, bool training_pass) {
    if (training_pass) {
        for (size_t i = 0; i < b.Mask.n_rows; i++) {
            for (size_t j = 0; j < b.Mask.n_columns; j++) {
                for (size_t k = 0; k < b.Mask.n_slices; k++) {
                    double prob = (rand_r(&rnd_state) / (double(RAND_MAX) + 1.0));
                    b.Mask.get(i, j, k) = prob > drop_prob ? 1 : 0;
                    y.get(i, j, k) = b.Mask.get(i, j, k) * x.get(i, j, k);
                }
            }
        }
    }
    else {
        for (size_t i = 0; i < b.Mask.n_rows; i++) {
            for (size_t j = 0; j < b.Mask.n_columns; j++) {
                for (size_t k = 0; k < b.Mask.n_slices; k++) {
                    y.get(i, j, k) = (1 - drop_prob) * x.get(i, j, k);
                }
            }
        }
    }
}

void DropoutLayer::backward(DropoutLayer::Parameters &, DropoutLayer::FwdState &b, DropoutLayer::BwdState &, Matrix &, Matrix &in_deltas, Matrix &out_deltas) {
    dot(b.Mask, out_deltas.flatten_time(), in_deltas.flatten_time());
}

void DropoutLayer::gradient(DropoutLayer::Parameters&, DropoutLayer::Parameters&, DropoutLayer::FwdState&, DropoutLayer::BwdState&, Matrix&, Matrix&, Matrix&)
{
}

void DropoutLayer::Rpass(Parameters &, Parameters &,  FwdState &, FwdState &, Matrix &, Matrix &, Matrix&, Matrix &)
{
}

void DropoutLayer::dampened_backward(Parameters&, FwdState&, BwdState&, Matrix&, Matrix&, Matrix&, FwdState&, double, double)
{
}
