SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


ADD_EXECUTABLE(test_fwd_layer test_fwd_layer.cpp)
TARGET_LINK_LIBRARIES(test_fwd_layer core layers matrix gtest gtest_main)
ADD_TEST(TestLayer ${CMAKE_CURRENT_BINARY_DIR}/test_fwd_layer)