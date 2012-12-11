#include <string>

#include <gtest/gtest.h>

#include "Config.h"

#include "matrix/matrix_cpu.h"
#include "matrix/matrix_operation_cpu.h"

class MatrixTest : public ::testing::Test {
protected:
  MatrixTest() :
    m1({{2.,3.,5.},{3.,4.,55.}}),
    m2({{2.,3.,5.},{3.,4.,55.}}), // same as m1
    m3({{2.,3.,4.},{3.,4.,6.}}),
    m4({{2.,3.},{3.,4.}})
  {  }

  MatrixCPU m1, m2, m3, m4;
};

TEST_F(MatrixTest, check_equals)
{
  EXPECT_TRUE(equals(m1, m1));
  EXPECT_TRUE(equals(m1, m2));

  EXPECT_FALSE(equals(m1, m3));
  EXPECT_FALSE(equals(m1, m4));
  EXPECT_FALSE(equals(m3, m4));
}

