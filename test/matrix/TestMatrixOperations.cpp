#include <string>
#include <iostream>
#include <gtest/gtest.h>

#include "Config.h"

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"

class MatrixOperationsTest : public ::testing::Test {
protected:
  MatrixOperationsTest() :
    m1({{2.,3.,5.},{3.,4.,55.}}),
    m2({{2.,3.,5.},{3.,4.,55.}}), // same as m1
    m3({{2.,3.,4.},{3.,4.,6.}}),
    m4({{2.,3.},{3.,4.}}),
    m5({{2.,3.},{3.,4.},{5.,55.}}), // m1 transposed
    m6({{2.,3., 4., 5.},{6., 7., 8., 9.}}), // m1 transposed
    m3d({{{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}},
        {{100, 101, 102, 103}, {110, 111, 112, 113}, {120, 121, 122, 123}}})
{  }

    Matrix m1, m2, m3, m4, m5, m6;
    Matrix m3d;
};

TEST_F(MatrixOperationsTest, check_1D_initializer_list)
{
	Matrix m = {0, 2, 4, 6};
	ASSERT_EQ(m.n_rows, 1);
	ASSERT_EQ(m.n_columns, 4);
	ASSERT_EQ(m.n_slices, 1);
	for (int i = 0; i < 4; ++i) {
		ASSERT_EQ(m.get(0, i, 0), 2*i);
	}
}

TEST_F(MatrixOperationsTest, check_2D_initializer_list)
{
	Matrix m = {{0, 2, 4, 6},
			{10, 12, 14, 16}};
	ASSERT_EQ(m.n_rows, 2);
	ASSERT_EQ(m.n_columns, 4);
	ASSERT_EQ(m.n_slices, 1);
	for (int r = 0; r < 2; ++r) {
		for (int c = 0; c < 4; ++c) {
			ASSERT_EQ(m.get(r, c, 0), 2*c + 10*r);
		}
	}
}

TEST_F(MatrixOperationsTest, check_3D_initializer_list)
{
	Matrix m = {{{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}},
			{{100, 101, 102, 103}, {110, 111, 112, 113}, {120, 121, 122, 123}}};
	ASSERT_EQ(m.n_rows, 3);
	ASSERT_EQ(m.n_columns, 4);
	ASSERT_EQ(m.n_slices, 2);
	for (int s = 0; s < 2; ++s) {
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 4; ++c) {
				ASSERT_EQ(m.get(r, c, s), c + 10*r + 100*s);
			}
		}
	}
}

TEST_F(MatrixOperationsTest, check_if_equals_works_as_expected)
{
	EXPECT_TRUE(equals(m1, m5.T()));
	EXPECT_TRUE(equals(m1.T(), m5));
}

TEST_F(MatrixOperationsTest, check_if_equals_works_on_transpose)
{
	EXPECT_TRUE(equals(m1, m1));
	EXPECT_TRUE(equals(m1, m2));

	EXPECT_FALSE(equals(m1, m3));
	EXPECT_FALSE(equals(m1, m4));
	EXPECT_FALSE(equals(m1, m5));
	EXPECT_FALSE(equals(m3, m4));
}

TEST_F(MatrixOperationsTest, check_if_add_into_b_works_as_expected)
{
	add_into_b(m1, m3);
	ASSERT_TRUE(equals(m3, Matrix({{4.,6.,9.},{6.,8.,61.}})));
}

TEST_F(MatrixOperationsTest, check_if_add_into_b_on_itself)
{
	add_into_b(m1, m1);
	ASSERT_TRUE(equals(m1, Matrix({{4.,6.,10.},{6.,8.,110.}})));
}

TEST_F(MatrixOperationsTest, check_if_add_into_b_work_on_3d)
{
	Matrix out(3, 4, 2);
	add_into_b(m3d, out);
	ASSERT_TRUE(equals(out, m3d));
	add_into_b(m3d, out);
	Matrix m = {{{0, 2, 4, 6}, {20, 22, 24, 26}, {40, 42, 44, 46}},
			{{200, 202, 204, 206}, {220, 222, 224, 226}, {240, 242, 244, 246}}};
	ASSERT_TRUE(equals(out, m));
}

TEST_F(MatrixOperationsTest, check_if_add_vector_into)
{
	Matrix v = {1, 2};
	add_vector_into(v, m1);
	Matrix expected = {{3.,4.,6.},{5.,6.,57.}};
	ASSERT_TRUE(equals(m1, expected));
}


TEST_F(MatrixOperationsTest, check_if_add_scalar_works_as_expected)
{
	add_scalar(m1, 10.);
	ASSERT_TRUE(equals(m1, Matrix({{12.,13.,15.},{13.,14.,65.}})));
}


TEST_F(MatrixOperationsTest, check_if_dot_works_as_expected)
{
	dot(m1, m3, m2);
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m3, Matrix({{2.,3.,4.},{3.,4.,6.}})));

	ASSERT_TRUE(equals(m2, Matrix({{4.,9.,20.},{9.,16.,330.}})));
}

TEST_F(MatrixOperationsTest, check_if_dot_works_with_smaller_b)
{
	dot(m1, Matrix({0,2}), m2);
	// check that m1 is unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m2, Matrix({{0.,0. ,0.},{6.,8.,110.}})));
}

/* TODO: these dont work!! be careful! Make them work...
TEST_F(MatrixOperationsTest, check_if_dot_works_with_a_b_a)
{
  dot(m1, m3, m1);
  // check that m1 and m3 are unchanged
  ASSERT_TRUE(equals(m3, Matrix({{2.,3.,4.},{3.,4.,6.}})));

  ASSERT_TRUE(equals(m1, Matrix({{4.,9.,20.},{9.,16.,330.}})));
}

TEST_F(MatrixOperationsTest, check_if_dot_works_with_a_b_b)
{
  dot(m1, m3, m3);
  // check that m1 and m3 are unchanged
  ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));

  ASSERT_TRUE(equals(m3, Matrix({{4.,9.,20.},{9.,16.,330.}})));
}
 */

TEST_F(MatrixOperationsTest, check_if_dot_add_works_as_expected)
{
	dot_add(m1, m3, m2);
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m3, Matrix({{2.,3.,4.},{3.,4.,6.}})));

	ASSERT_TRUE(equals(m2, Matrix({{6.,12.,25.},{12.,20.,385.}})));
}

TEST_F(MatrixOperationsTest, check_if_mult_works_as_expected)
{
	mult(m4, m1, m2);
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m4, Matrix({{2.,3.},{3.,4.}})));

	ASSERT_TRUE(equals(m2, Matrix({{13.,18.,175.},{18.,25.,235.}})));
}


TEST_F(MatrixOperationsTest, check_if_mult_works_transposed1)
{
	Matrix out(2, 3, 1);
	mult(m4.T(), m1, out);
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m4, Matrix({{2.,3.},{3.,4.}})));
	ASSERT_TRUE(equals(out, Matrix({{13.,18.,175.},{18.,25.,235.}})));
}

TEST_F(MatrixOperationsTest, check_if_mult_works_transposed2)
{
	Matrix out(3, 2, 1);
	mult(m1.T(), m4, out);
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m4, Matrix({{2.,3.},{3.,4.}})));
	ASSERT_TRUE(equals(out, Matrix({{13.,18.}, {18, 25}, {175., 235.}})));
}


TEST_F(MatrixOperationsTest, check_if_mult_works_transposed3)
{
	Matrix out(3, 2, 1);
	mult(m1.T(), m4.T(), out);
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m4, Matrix({{2.,3.},{3.,4.}})));
	ASSERT_TRUE(equals(out, Matrix({{13.,18.}, {18, 25}, {175., 235.}})));
}


TEST_F(MatrixOperationsTest, check_if_mult_add_works_as_expected)
{
  mult_add(m4, m1, m2);
  // check that m1 and m3 are unchanged
  ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
  ASSERT_TRUE(equals(m4, Matrix({{2.,3.},{3.,4.}})));

  ASSERT_TRUE(equals(m2, Matrix({{15.,21.,180.},{21.,29.,290.}})));
}

TEST_F(MatrixOperationsTest, check_if_apply_sigmoid_works_as_expected)
{
    Matrix range = {-2, -1, 0, 1, 2};
    Matrix expected = {0.11920292,  0.26894142,  0.5,  0.73105858,  0.88079708};
    Matrix actual(5, 1, 1);
    apply_sigmoid(range, actual);
    for (int i = 0; i < 5; ++i) {
        std::cout << expected[i] << ", " << actual[i] << std::endl;
        ASSERT_NEAR(expected[i], actual[i], 0.0000001);
    }
}

