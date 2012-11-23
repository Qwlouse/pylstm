/**
 * \file Assert.cpp
 * \brief Implementation of the assertFailed function.
 */

// ================================= Includes ==================================
#include "Assert.h"

#include "Exceptions.h"

// ============================ Method Definitions =============================
namespace core
{

void assertFailed(const char* expression, const char* filename, const char* functionName, int line)
{
    throw AssertFailedException("Assertion Failed!") << ::boost::throw_function(functionName)
                                                     << ::boost::throw_file(filename)
                                                     << ::boost::throw_line(line)
                                                     << ::core::Throw_Expression(expression);
}

}
