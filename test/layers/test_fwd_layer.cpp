#include <gtest/gtest.h>

#include "Config.h"

#include "layers/fwd_layer.h"

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
