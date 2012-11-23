/** \file Assert.h
 *  \brief Header file that defines the ASSERT and VERIFY macros.
 */
#pragma once

// ================================= Includes =================================
#include <boost/current_function.hpp>

#include "Config.h"

// ============================ Method declaration ============================
namespace core
{
void assertFailed(const char* expression, const char* filename, const char* functionName, int line);
}

// ============================== Assert Macros ===============================
#ifdef ASSERTS_ENABLED
    #define ASSERT(expr) ((expr) ? \
                         static_cast<void>(0) : \
                         core::assertFailed(#expr, __FILE__, BOOST_CURRENT_FUNCTION, __LINE__))

    #define VERIFY(expr) ASSERT(expr)
#else /* Asserts disabled */
    #define ASSERT(expr) (static_cast<void>(0))
    #define VERIFY(expr) (static_cast<void>(expr))
#endif /* ASSERTS_ENABLED */


