#include <gtest/gtest.h>

#include "Config.h"
#include "Core.h"

#include "layers/fwd_layer.h"
#include "matrix/matrix_operation.h"

TEST(FwdLayerTest, FwdWeightsConstruction)
{
	Matrix buffer = {{{1, 2, 3, 4, 5, 6, 7, 8}}};

	RegularLayer::Weights w(3, 2, buffer);
	ASSERT_EQ(8, w.size());
	ASSERT_EQ(w.HX.size, 6);
	ASSERT_EQ(w.H_bias.size, 2);
	ASSERT_EQ(w.HX[0], 1);
	ASSERT_EQ(w.HX[1], 2);
	ASSERT_EQ(w.HX[2], 3);
	ASSERT_EQ(w.HX[5], 6);
	ASSERT_EQ(w.H_bias[1], 8);
	ASSERT_EQ(w.H_bias[0], 7);
}

TEST(FwdLayerTest, FwdBuffersConstruction)
{
	Matrix buffer = {{{1, 2, 3, 4, 5, 6}}};

	RegularLayer::FwdState b(5, 1, 3, 2, buffer);
	ASSERT_EQ(6, b.size());
	ASSERT_EQ(b.Ha.size, 6);
	ASSERT_EQ(b.Ha[0], 1);
	ASSERT_EQ(b.Ha[1], 2);
	ASSERT_EQ(b.Ha[2], 3);
	ASSERT_EQ(b.Ha[5], 6);
}

template<typename Layer>
Matrix run_fwd(Matrix& X, size_t out_size, Matrix& W)
{
	size_t in_size = X.n_rows;
	size_t n_batches = X.n_columns;
	size_t n_slices = X.n_slices;
	Matrix Y(out_size, n_batches, n_slices);

	size_t fwd_state_size = Layer::FwdState::estimate_size(in_size, out_size, n_batches, n_slices);
	Matrix b(1, 1, fwd_state_size);
	typename Layer::FwdState fwd_state(in_size, out_size, n_batches, n_slices, b);

	size_t weight_size = Layer::Weights::estimate_size(in_size, out_size);
	if (W.size == 0) W = Matrix(1, 1, weight_size);
	ASSERT(W.size == weight_size);
	typename Layer::Weights weights(in_size, out_size, W);

	Layer l;
	l.forward(weights, fwd_state, X, Y);
	return Y;
}


TEST(FwdLayerTest, fwd_pass)
{
	size_t n_batches = 5;
	size_t n_slices = 6;
	Matrix X(3, n_batches, n_slices);
	Matrix W;
	Matrix Y = run_fwd<RegularLayer>(X, 4, W);
	Matrix expected(4, n_batches, n_slices);
	add_scalar(expected, 0.5);
	ASSERT_TRUE(equals(Y, expected));
}

TEST(FwdLayerTest, layer_wrapper)
{
	RLayer L(5, 4);
	Matrix W(1, 1, L.get_weight_size());
	Matrix X(L.in_size, 3, 2);
	Matrix Y = L.auto_forward_pass(W, X);
	Matrix expected(L.out_size, 3, 2);
	add_scalar(expected, 0.5);
	ASSERT_TRUE(equals(Y, expected));
}
