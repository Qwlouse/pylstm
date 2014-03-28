SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Test Exceptions
ADD_EXECUTABLE(TestExceptions Test_Exceptions.cpp)
TARGET_LINK_LIBRARIES(TestExceptions core gtest gtest_main)
ADD_TEST(TestExceptions ${CMAKE_CURRENT_BINARY_DIR}/TestExceptions)

# Test Assertions
IF(ASSERTS_ENABLED)
    ADD_EXECUTABLE(TestAssertions Test_Assertions.cpp)
    TARGET_LINK_LIBRARIES(TestAssertions core gtest gtest_main)
    ADD_TEST(TestAssertions ${CMAKE_CURRENT_BINARY_DIR}/TestAssertions)
ENDIF(ASSERTS_ENABLED)
