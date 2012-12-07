#include <string>

#include <gtest/gtest.h>

#include "Config.h"

#include "matrix/matrix_cpu.h"
#include "matrix/matrix_operation_cpu.h"

TEST(TestMatrix, check_if_one_is_true)
{
  ASSERT_TRUE(equals(MatrixCPU({{2.,3.,5.},{3.,4.,55.}}), MatrixCPU({{2.,3.,5.},{3.,4.,55.}})));

  ASSERT_TRUE(!equals(MatrixCPU({{2.,3.,4.},{3.,4.,55.}}), MatrixCPU({{2.,3.,5.},{3.,4.,55.}})));
  ASSERT_TRUE(!equals(MatrixCPU({{2.,3.,4.},{3.,4.,55.}}), MatrixCPU({{2.,3.},{3.,4.}})));

}

