#include "Exceptions.h"

#include <boost/exception/all.hpp>

namespace core
{

// =================================== Exception ==============================
Exception::~Exception() throw()
{ }

const char* Exception::what() const throw()
{
    const char* diagInfo = ::boost::diagnostic_information_what(*this);
    if (diagInfo != NULL)
        return diagInfo;
    else
        return "";
}

// =================================== RuntimeException =======================
RuntimeException::RuntimeException() throw()  :
    Exception(),
    std::runtime_error("DUMMY! Real message is stored as core::Throw_Message.")
{ }

const char* RuntimeException::what() const throw()
{
    return Exception::what();
}

// =================================== LogicErrorException ====================
LogicErrorException::LogicErrorException() throw()  :
    Exception(),
    std::logic_error("DUMMY! Real message is stored as core::Throw_Message.")
{ }

const char* LogicErrorException::what() const throw()
{
    return Exception::what();
}

// =================================== IllegalArgumentException ===============
IllegalArgumentException::IllegalArgumentException(const std::string& message) 
                                                                      throw() :
    RuntimeException()
{
    *this << Throw_Message(message);
}

// =================================== CommandLineException ===================
CommandLineException::CommandLineException(const std::string& message) 
                                                                      throw() :
    RuntimeException()
{
    *this << Throw_Message(message);
}

// =================================== AssertFailedException ==================
AssertFailedException::AssertFailedException(const std::string& message) 
                                                                      throw() :

    LogicErrorException()
{
    *this << Throw_Message(message);
}

// =================================== InvalidStateException ==================
InvalidStateException::InvalidStateException(const std::string& message) 
                                                                      throw() :
    LogicErrorException()
{
    *this << Throw_Message(message);
}

// =================================== Convienience Functions =================
std::ostream & operator<<(std::ostream& out, const Exception& e)
{
    out << e.what();
    return out;
}

}
