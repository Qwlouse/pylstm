/**
 * \file Exceptions.h
 * \author Klaus Greff
 * \brief Declares the whole exception hierarchy.
 *
 * \details
 * The exceptions declared here are based on both, the standard exceptions and the boost exceptions.
 * Every exception can only be instantiated with a message as an argument. Supplementary information
 * can be added via the stream operator <<. The exceptions should be thrown using the \c THROW macro but
 * can be rethrown using the standard /c throw; .
 */
#pragma once

#include <ostream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/exception/all.hpp>

// ================================ THROW Macro ================================
/**
 * The standard way of throwing exceptions. You should always use this instead of
 * the ordinary \c throw because that way debugging informations like the current
 * line will be added to the exception.
 * \note This is just a convenience rename of \c BOOST_THROW_EXCEPTION.
 *
 * Usage example:
 * \code
 * // simple example
 * THROW(InvalidStateException("Can't open while locked"));
 * // adding some information
 * THROW(IllegalArgumentException("i is too big") << Throw_Argument<int>::Value(i));
 * \endcode
 *
 * However if you choose to re-throw an exception you should use the normal \c throw :
 * \code
 * ...
 * catch (Exception& e)
 * {
 *      // do some stuff
 *      throw;
 *      // NOT: THROW(e); because this will slice your exception.
 * }
 * \endcode
 *
 */
#define THROW BOOST_THROW_EXCEPTION

namespace core
{

// ================================ Abstract Exceptions ========================
/**
 * Abstract base class for the other abstract exception base classes.
 * \note DO NOT inherit from this class. If, however, you nevertheless decide to
 * you have to make sure the following:
 * - you need to also inherit from some std::exception derivative.
 * - you need to override the \c what() method
 * - your class needs to have a standard constructor (custom or auto generated)
 * - if you want your Exception to be useable as a normal stream you also have
 *   to provide two \c operator<< template functions as for example in
 *   RuntimeException.
 */
class Exception : public boost::exception
{
public:
    virtual ~Exception() throw() = 0;

    /**
     * Provides a thorough description of the exception including debugging information.
     */
    virtual const char* what() const throw();
};

/**
 * Abstract base class for all runtime errors, i.e. those that should be caught.
 *
 * \note It is recommended to only inherit virtually from this class.
 */
class RuntimeException : public Exception, public std::runtime_error
{
public:
    RuntimeException() throw();
    virtual const char* what() const throw();

    /**
     * Enable normal streaming for RuntimeException.
     * Every type that can be streamed to ostream can also be streamed to Exceptions.
     */
    template <typename T>
    core::RuntimeException& operator<< (const T& out);

    /**
     * Template specialization to preserve the \c error_info streaming behavior.
     */
    template <typename Tag, typename T>
    core::RuntimeException& operator<< (const boost::error_info<Tag, T>& out);
};

/**
 * Abstract base class for all logic errors, i.e. those that indicate a bug.
 *
 * \note It is recommended to only inherit virtually from this class.
 */
class LogicErrorException : public Exception, public std::logic_error
{
public:
    LogicErrorException() throw();
    virtual const char* what() const throw();

    /**
     * Enable normal streaming for LogicErrorException.
     * Every type that can be streamed to ostream can also be streamed to Exceptions.
     */
    template <typename T>
    core::LogicErrorException& operator<< (const T& out);

    /**
     * Template specialization to preserve the \c error_info streaming behavior.
     */
    template <typename Tag, typename T>
    core::LogicErrorException& operator<< (const boost::error_info<Tag, T>& out);
};

// ================================ Concrete Exceptions ========================

/**
 * Used to tell that an illegal argument has been passed to some function.
 *
 * The details of the argument should be passed via Throw_Argument.
 */
class IllegalArgumentException : public virtual RuntimeException
{
public:
    explicit IllegalArgumentException(const std::string& message) throw();
};

/**
 * Thrown if an error occurred while parsing the command line.
 */
class CommandLineException : public virtual RuntimeException
{
public:
    explicit CommandLineException(const std::string& message) throw();
};

/**
 * Thrown if an assertion fails.
 *
 * The assert statement is added as Throw_Expression.
 */
class AssertFailedException : public virtual LogicErrorException
{
public:
    explicit AssertFailedException(const std::string& message) throw();
};

/**
 * Indicates that some object is in an invalid state.
 */
class InvalidStateException : public virtual LogicErrorException
{
public:
    explicit InvalidStateException(const std::string& message) throw();
};

// ========================= Supplementary Information =========================

/**
 * Used internally to store the message of the exception.
 */
typedef boost::error_info<struct Throw_Message_, const std::string>
        Throw_Message;

/**
 * Used to store the expression of a failed assertion.
 */
typedef boost::error_info<struct Throw_Expression_, const char*>
        Throw_Expression;

/**
 * Used to store another exception.
 */
typedef boost::error_info<struct Throw_InnerException_, std::exception>
        Throw_InnerException;

/**
 * A template used to store additional information about the Argument in an IllegalArgumentException.
 *
 * Definition as a class is a workaround to get a template-typedef.
 * Usage example:
 * \code
 * THROW(IllegalArgumentException("i too big") << Throw_Argument<int>::Value(i));
 * \endcode
 */
template<class T>
class Throw_Argument
{
public:
    typedef boost::error_info<Throw_Argument, T> Value;
};


// ========================= Convienience Functions ============================
/**
 * Streaming operator to allow direct output of exceptions.
 *
 * The output is the \c what() message.
 *
 * Example:
 * \code
 * catch (Exception& e)
 * {
 *     cerr << e << endl;
 * }
 * \endcode
 */
std::ostream& operator<< (std::ostream& out, const Exception& cPoint);


}

#include "core/Exceptions.inl"

