#include <string>

#include <gtest/gtest.h>

#include "Config.h"

#include "matrix/matrix_cpu.h"
#include "matrix/matrix_operation_cpu.h"

class MatrixOperationsTest : public ::testing::Test {
protected:
  MatrixOperationsTest() :
    m1({{2.,3.,5.},{3.,4.,55.}}),
    m2({{2.,3.,5.},{3.,4.,55.}}), // same as m1
    m3({{2.,3.,4.},{3.,4.,6.}}),
    m4({{2.,3.},{3.,4.}}),
    m5({{2.,3.},{3.,4.},{5.,55.}}) // m1 transposed
  {  }

  MatrixCPU m1, m2, m3, m4, m5;
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

TEST_F(MatrixOperationsTest, check_if_add_into_b_works_as_expected)
{
  add_into_b(m1, m3);
  ASSERT_TRUE(equals(m3, MatrixCPU({{4.,6.,9.},{6.,8.,61.}})));
}

TEST_F(MatrixOperationsTest, check_if_add_scalar_works_as_expected)
{
  add_scalar(m1, 10.);
  ASSERT_TRUE(equals(m1, MatrixCPU({{12.,13.,15.},{13.,14.,65.}})));
}

TEST_F(MatrixOperationsTest, check_if_dot_works_as_expected)
{
  dot(m1, m3, m2);
  // check that m1 and m3 are unchanged
  ASSERT_TRUE(equals(m1, MatrixCPU({{2.,3.,5.},{3.,4.,55.}})));
  ASSERT_TRUE(equals(m3, MatrixCPU({{2.,3.,4.},{3.,4.,6.}})));

  ASSERT_TRUE(equals(m2, MatrixCPU({{4.,9.,20.},{9.,16.,330.}})));
}

TEST_F(MatrixOperationsTest, check_if_dot_add_works_as_expected)
{
  dot_add(m1, m3, m2);
  // check that m1 and m3 are unchanged
  ASSERT_TRUE(equals(m1, MatrixCPU({{2.,3.,5.},{3.,4.,55.}})));
  ASSERT_TRUE(equals(m3, MatrixCPU({{2.,3.,4.},{3.,4.,6.}})));

  ASSERT_TRUE(equals(m2, MatrixCPU({{6.,12.,25.},{12.,20.,385.}})));
}

TEST_F(MatrixOperationsTest, check_if_mult_works_as_expected)
{
  mult(m4, m1, m2);
  // check that m1 and m3 are unchanged
  ASSERT_TRUE(equals(m1, MatrixCPU({{2.,3.,5.},{3.,4.,55.}})));
  ASSERT_TRUE(equals(m4, MatrixCPU({{2.,3.},{3.,4.}})));

  ASSERT_TRUE(equals(m2, MatrixCPU({{13.,18.,175.},{18.,25.,235.}})));
}

TEST_F(MatrixOperationsTest, check_if_mult_add_works_as_expected)
{
  mult_add(m4, m1, m2);
  // check that m1 and m3 are unchanged
  ASSERT_TRUE(equals(m1, MatrixCPU({{2.,3.,5.},{3.,4.,55.}})));
  ASSERT_TRUE(equals(m4, MatrixCPU({{2.,3.},{3.,4.}})));

  ASSERT_TRUE(equals(m2, MatrixCPU({{15.,21.,180.},{21.,29.,290.}})));
}

