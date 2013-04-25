#include <iostream>

#include <gtest/gtest.h>

#include "Config.h"
#include "Core.h"
#include "layers/fwd_layer.h"
#include "layers/layer.hpp"
#include "matrix/matrix_operation.h"

using namespace std;

TEST(FwdLayerTest, FwdParameterConstruction)
{
	Matrix buffer = {{{1, 2, 3, 4, 5, 6, 7, 8}}};

	RegularLayer::Parameters w(3, 2);
	w.lay_out(buffer);
	ASSERT_EQ(8, w.get_size());
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

	RegularLayer::FwdState b(5, 1, 3, 2);
	b.lay_out(buffer);
	ASSERT_EQ(6, b.get_size());
	ASSERT_EQ(b.Ha.size, 6);
	ASSERT_EQ(b.Ha[0], 1);
	ASSERT_EQ(b.Ha[1], 2);
	ASSERT_EQ(b.Ha[2], 3);
	ASSERT_EQ(b.Ha[5], 6);
}


TEST(FwdLayerTest, fwd_pass)
{
	size_t n_batches = 5;
	size_t n_slices = 6;
	Matrix X(3, n_batches, n_slices);


	//Matrix Y = run_fwd<RegularLayer>(X, 4, W);
	Matrix expected(4, n_batches, n_slices);
	add_scalar(expected, 0.5);
	//ASSERT_TRUE(equals(Y, expected));
}

TEST(FwdLayerTest, layer_wrapper)
{
	Layer<RegularLayer> L(5, 4);

	Matrix W(1, 1, L.get_weight_size());
	Matrix X(L.in_size, 3, 2);
	Matrix Y = L.auto_forward_pass(W, X);
	Matrix expected(L.out_size, 3, 2);
	add_scalar(expected, 0.5);
	ASSERT_TRUE(equals(Y, expected));
}


TEST(FwdLayerTest, layer_wrapper2)
{
	Layer<RegularLayer> layer(5, 4, RegularLayer(&Sigmoid));

	Matrix W(1, 1, layer.get_weight_size());
	Matrix X(layer.in_size, 3, 2);
	Matrix Y = layer.auto_forward_pass(W, X);
	//Matrix expected(layer.out_size, 3, 2);
	//add_scalar(expected, 0.5);
	//ASSERT_TRUE(equals(Y, expected));
	
}
