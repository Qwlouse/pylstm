# Set path for binaries
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test)

# GTest headers do not support -pedantic and -Weffc++
# so lets remove them
REMOVE_DEFINITIONS(-pedantic -Weffc++)
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(matrix)
ADD_SUBDIRECTORY(layers)

