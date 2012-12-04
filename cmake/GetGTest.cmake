# Module that checks out the current version of GTest and builds it as part of 
# of the build process.
###############################################################################
# This module defines:
# - GTest                 (the external project)
# - GTest_INCLUDE_PATH
# - GTest_LIBRARY
# - GTest_MAIN_LIBRARY
# - GTest_LIBRARIES
#
###############################################################################

# List of variables, which will set:
set(_SEARCHED_VARIABLES
    GTest_INCLUDE_PATH
    GTest_LIBRARY
    GTest_MAIN_LIBRARY
    GTest_LIBRARIES
    )
    
#------------------------------------------------------------------------------
# GTest External Project
#------------------------------------------------------------------------------
# Enable ExternalProject CMake module
INCLUDE(ExternalProject)

# Set default ExternalProject root directory
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_BASE ${CMAKE_BINARY_DIR}/ThirdParty)

# Add gtest
ExternalProject_Add(
    GTest
    SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk/
    TIMEOUT 10
    # Tell gtest to build and use dynamic libraries
    CMAKE_ARGS -Dgtest_force_shared_crt=ON
               -DBUILD_SHARED_LIBS=ON
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

#------------------------------------------------------------------------------
# GTEST_INCLUDE_PATH
#------------------------------------------------------------------------------
EXTERNALPROJECT_GET_PROPERTY(GTest SOURCE_DIR)
SET(GTEST_INCLUDE_PATH ${SOURCE_DIR}/include)
SET(SOURCE_DIR)

#------------------------------------------------------------------------------
# GTEST_LIBRARIES
#------------------------------------------------------------------------------
EXTERNALPROJECT_GET_PROPERTY(GTest BINARY_DIR)
# TODO on windows the ending should be .dll
# TODO but I couldn't get ${CMAKE_FIND_LIBRARY_SUFFIXES} to do the right thing
SET(GTEST_LIBRARY ${BINARY_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.so)
SET(GTEST_MAIN_LIBRARY ${BINARY_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.so)
SET(GTEST_LIBRARIES ${GTEST_MAIN_LIBRARY} ${GTEST_LIBRARY} pthread)
SET(BINARY_DIR)


