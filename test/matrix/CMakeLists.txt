SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE(TestMatrixOperations TestMatrixOperations.cpp)
TARGET_LINK_LIBRARIES(TestMatrixOperations core matrix gtest gtest_main)
ADD_TEST(TestMatrixOperations ${CMAKE_CURRENT_BINARY_DIR}/TestMatrixOperations)

ADD_EXECUTABLE(TestMatrix TestMatrix.cpp)
TARGET_LINK_LIBRARIES(TestMatrix core matrix gtest gtest_main)
ADD_TEST(TestMatrix ${CMAKE_CURRENT_BINARY_DIR}/TestMatrix)