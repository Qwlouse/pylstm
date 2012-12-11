###############################################################################
###                          PY_LSTM BUILD SCRIPT                           ###
###############################################################################
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#==============================================================================
# Project Settings
#==============================================================================
PROJECT(PY_LSTM)
SET(VERSION_MAJOR 0)
SET(VERSION_MINOR 1)


#==============================================================================
# Set OS Variable
#==============================================================================
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   # Mac OS X specific code
   SET(OperatingSystem "Mac OS X")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # Linux specific code
    SET(OperatingSystem "Linux")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux") 

#==============================================================================
# Path Settings
#==============================================================================
# Read PY_LSTM_PATH from options, cache, environment variable, or . 
# (in that order)
IF(NOT PY_LSTM_PATH)
    SET(ENV_PY_LSTM_PATH $ENV{PY_LSTM_PATH})
    IF(ENV_PY_LSTM_PATH)
        FILE(TO_CMAKE_PATH $ENV{PY_LSTM_PATH} PY_LSTM_PATH)
    ELSE(ENV_PY_LSTM_PATH)
        SET(PY_LSTM_PATH ${CMAKE_HOME_DIRECTORY})
    ENDIF(ENV_PY_LSTM_PATH)
ENDIF(NOT PY_LSTM_PATH)

# Set Cmake module path
SET(CMAKE_MODULE_PATH "${PY_LSTM_PATH}/cmake")

#==============================================================================
# Options settings (including optional packages)
#==============================================================================
# Enable Logging (requires Log4cxx)
OPTION(LOGGING_ENABLED "Enable logging support (requires Log4cxx)" ON)
IF (LOGGING_ENABLED)
    FIND_PACKAGE(Log4cxx REQUIRED)
ENDIF(LOGGING_ENABLED)
SET(LOG_LEVEL 5 CACHE INTEGER "Compile-time log level (0 = NONE - 6 = TRACE)")


# Enable Asserts
OPTION(ASSERTS_ENABLED "Enable asserts" ON)

# Compile the tests
OPTION(UNIT_TESTS "Compile the unit tests (fetches GTest)" ON)
IF (UNIT_TESTS)
    INCLUDE(GetGTest)
ENDIF(UNIT_TESTS)

# Build the Documentation using Doxygen
OPTION(DOCUMENTATION "Build the Documentation using Doxygen" ON)
IF (DOCUMENTATION)
    FIND_PACKAGE(Doxygen)
ENDIF(DOCUMENTATION)

#==============================================================================
# Compiler Flags
#==============================================================================
SET( ENV{EXTERNAL_WARNINGS_NOT_ERRORS} TRUE)

IF(DEFINED MSVC)
    ADD_DEFINITIONS(/W3 /wd4996)
ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    ADD_DEFINITIONS(-Wall -Wextra -Wno-sign-compare -ansi -fPIC -std=c++0x #-Weffc++ -pedantic
                    -Wconversion )#-Werror)
ENDIF(DEFINED MSVC)

#==============================================================================
# External Libs
#==============================================================================

FIND_PACKAGE(Matlab2 REQUIRED)
INCLUDE_DIRECTORIES(${MATLAB_EXTERN_INCLUDE_PATH})

#==============================================================================
# Config header
#==============================================================================
# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/src/Config.h.in"
  "${PROJECT_BINARY_DIR}/src/Config.h"
  )
# add the to the search path for include files so that we will find Config.h
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}/src")

#==============================================================================
# Cython
#==============================================================================
# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
INCLUDE(UseCython)

# We reproduce the availability of the source files in the build tree.
# That way we can run the unchanged setup.py from the build dir and install
# both cython and pure python files.
ADD_CUSTOM_TARGET( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

#==============================================================================
# Process sources
#==============================================================================
ADD_SUBDIRECTORY(src)

#==============================================================================
# Testing
#==============================================================================
IF (UNIT_TESTS)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
ENDIF(UNIT_TESTS)

#==============================================================================
# Documentation
#==============================================================================
IF (DOCUMENTATION)
    # configure doxygen configuration file
    CONFIGURE_FILE(
        "${PROJECT_SOURCE_DIR}/config/doxygen.conf.in"
        "${PROJECT_BINARY_DIR}/config/doxygen.conf"
        @ONLY
    )

    ADD_CUSTOM_TARGET(doc
                      ${DOXYGEN_EXECUTABLE} 
                      ${PROJECT_BINARY_DIR}/config/doxygen.conf
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      COMMENT "Generating API documentation with Doxygen"
                      VERBATIM
                      )
ENDIF(DOCUMENTATION)

