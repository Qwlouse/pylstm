#include <gtest/gtest.h>

#include "Config.h"

#include "layers/fwd_layer.h"
#include "matrix/matrix_operation.h"

TEST(FwdLayerTest, FwdWeightsConstruction)
{
	Matrix buffer = {{{1, 2, 3, 4, 5, 6, 7, 8}}};

	FwdWeights w(3, 2, buffer);
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

	FwdBuffers b(5, 1, 3, 2, buffer);
	ASSERT_EQ(6, b.buffer_size());
	ASSERT_EQ(b.Ha.size, 6);
	ASSERT_EQ(b.Ha[0], 1);
	ASSERT_EQ(b.Ha[1], 2);
	ASSERT_EQ(b.Ha[2], 3);
	ASSERT_EQ(b.Ha[5], 6);
}

TEST(FwdLayerTest, fwd_pass)
{
	size_t in_size = 3;
	size_t out_size = 4;
	size_t n_batches = 1;
	size_t n_slices = 1;
	Matrix X(in_size, n_batches, n_slices);
	Matrix Y(out_size, n_batches, n_slices);
	Matrix b(1, 1, FwdBuffers::estimate_size(in_size, out_size, n_batches, n_slices));
	FwdBuffers buffer(in_size, out_size, n_batches, n_slices, b);
	Matrix w(1, 1, FwdWeights::estimate_size(in_size, out_size));
	FwdWeights weights(in_size, out_size, w);

	fwd_forward(weights, buffer, X, Y);
	ASSERT_TRUE(equals(Y, Matrix({0.5, 0.5, 0.5, 0.5}).T()));
}
