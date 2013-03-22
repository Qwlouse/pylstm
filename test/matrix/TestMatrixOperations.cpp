#include <string>
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

TEST_F(MatrixOperationsTest, check_if_equals_works_as_expected)
{
	EXPECT_TRUE(equals(m1, m1));
	EXPECT_TRUE(equals(m1, m2));

	EXPECT_FALSE(equals(m1, m3));
	EXPECT_FALSE(equals(m1, m4));
	EXPECT_FALSE(equals(m1, m5));
	EXPECT_FALSE(equals(m3, m4));
}

TEST_F(MatrixOperationsTest, check_if_equals_works_on_transpose)
{
	EXPECT_TRUE(equals(m1, m5.T()));
	EXPECT_TRUE(equals(m1.T(), m5));
}

TEST_F(MatrixOperationsTest, check_if_equals_works_on_time_slices)
{
    Matrix a = {{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}};
    Matrix b = {{100, 101, 102, 103}, {110, 111, 112, 113}, {120, 121, 122, 123}};

	EXPECT_TRUE(equals(m3d.slice(0), a));
	EXPECT_TRUE(equals(m3d.slice(1), b));
}

TEST_F(MatrixOperationsTest, check_if_equals_works_on_multi_time_slices)
{
    Matrix a = {{{0, 7}}, {{1, 7}}, {{2, 7}}, {{3, 7}}, {{4, 7}}, {{5, 7}}};
    Matrix b = {{{3, 7}}, {{4, 7}}, {{5, 7}}};
    Matrix c = {{{1, 7}}, {{2, 7}}, {{3, 7}}, {{4, 7}}};

	EXPECT_TRUE(equals(a.slice(3, 5), b));
	EXPECT_TRUE(equals(a.slice(1, 4), c));
}

TEST_F(MatrixOperationsTest, check_if_equals_works_on_row_slices)
{
    Matrix a = {{{0, 1, 2, 3}}, {{100, 101, 102, 103}}};
    Matrix b = {{{10, 11, 12, 13}}, {{110, 111, 112, 113}}};
    Matrix c = {{{20, 21, 22, 23}}, {{120, 121, 122, 123}}};

	EXPECT_TRUE(equals(m3d.row_slice(0), a));
	EXPECT_TRUE(equals(m3d.row_slice(1), b));
	EXPECT_TRUE(equals(m3d.row_slice(2), c));
}

TEST_F(MatrixOperationsTest, check_if_equals_works_on_multi_row_slices)
{
    Matrix a = {{{0, 1, 2, 3}, {10, 11, 12, 13}}, {{100, 101, 102, 103}, {110, 111, 112, 113}}};
    Matrix b = {{{10, 11, 12, 13}, {20, 21, 22, 23}}, {{110, 111, 112, 113}, {120, 121, 122, 123}}};

	EXPECT_TRUE(equals(m3d.row_slice(0, 1), a));
	EXPECT_TRUE(equals(m3d.row_slice(1, 2), b));
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

TEST_F(MatrixOperationsTest, check_if_add_into_b_work_on_time_slices1)
{
	Matrix out(3, 4, 1);
	add_into_b(m3d.slice(0), out);
	Matrix e1 = {{{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}}};
	ASSERT_TRUE(equals(out, e1));
	add_into_b(m3d.slice(1), out);
	Matrix e2 = {{{100, 102, 104, 106}, {120, 122, 124, 126}, {140, 142, 144, 146}}};
	ASSERT_TRUE(equals(out, e2));
}

TEST_F(MatrixOperationsTest, check_if_add_into_b_work_on_time_slices2)
{
	Matrix out(3, 4, 2);
	add_into_b(m3d.slice(0), out.slice(0));
	add_into_b(m3d.slice(1), out.slice(1));
	ASSERT_TRUE(equals(out, m3d));
}

TEST_F(MatrixOperationsTest, check_if_add_into_b_work_on_row_sliced_a)
{
    Matrix out(1, 4, 2);
	add_into_b(m3d.row_slice(0), out);
	ASSERT_TRUE(equals(out, {{{0, 1, 2, 3}}, {{100, 101, 102, 103}}}));
	add_into_b(m3d.row_slice(1), out);
	ASSERT_TRUE(equals(out, {{{10, 12, 14, 16}}, {{210, 212, 214, 216}}}));
	add_into_b(m3d.row_slice(2), out);
	ASSERT_TRUE(equals(out, {{{30, 33, 36, 39}}, {{330, 333, 336, 339}}}));
}

TEST_F(MatrixOperationsTest, check_if_add_into_b_work_on_row_sliced_b)
{
    Matrix out(3, 4, 1);
    Matrix o = {1, 2, 3, 4};
	add_into_b(o, out.row_slice(0));
    ASSERT_TRUE(equals(out, {{{1, 2, 3, 4}, {0, 0, 0, 0}, {0, 0, 0, 0}}}));
	add_into_b(o, out.row_slice(1));
	ASSERT_TRUE(equals(out, {{{1, 2, 3, 4}, {1, 2, 3, 4}, {0, 0, 0, 0}}}));
	add_into_b(o, out.row_slice(2));
	ASSERT_TRUE(equals(out, {{{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}}}));
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

TEST_F(MatrixOperationsTest, check_if_add_scalar_works_as_expected_on_time_slice)
{
	add_scalar(m3d.slice(1), 1000.);
	Matrix expected({{{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}},
            {{1100, 1101, 1102, 1103}, {1110, 1111, 1112, 1113}, {1120, 1121, 1122, 1123}}});
	ASSERT_TRUE(equals(m3d, expected));
}

TEST_F(MatrixOperationsTest, check_if_add_scalar_works_as_expected_on_row_slice)
{
	add_scalar(m3d.row_slice(1), 1000.);
	Matrix expected({{{0, 1, 2, 3}, {1010, 1011, 1012, 1013}, {20, 21, 22, 23}},
            {{100, 101, 102, 103}, {1110, 1111, 1112, 1113}, {120, 121, 122, 123}}});
	ASSERT_TRUE(equals(m3d, expected));
}


TEST_F(MatrixOperationsTest, check_if_dot_works_as_expected)
{
	dot(m1, m3, m2);
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m3, Matrix({{2.,3.,4.},{3.,4.,6.}})));

	ASSERT_TRUE(equals(m2, Matrix({{4.,9.,20.},{9.,16.,330.}})));
}

TEST_F(MatrixOperationsTest, check_if_dot_works_with_row_slice)
{
	dot(m1.row_slice(0), m3.row_slice(1), m2.row_slice(0));
	// check that m1 and m3 are unchanged
	ASSERT_TRUE(equals(m1, Matrix({{2.,3.,5.},{3.,4.,55.}})));
	ASSERT_TRUE(equals(m3, Matrix({{2.,3.,4.},{3.,4.,6.}})));
	ASSERT_TRUE(equals(m2, Matrix({{6.,12.,30.},{3.,4.,55.}})));
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
        ASSERT_NEAR(expected[i], actual[i], 0.0000001);
    }
}

