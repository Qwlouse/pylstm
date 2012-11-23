// Inline file to make gcc happy.

namespace core
{

template <typename T>
core::RuntimeException& core::RuntimeException::operator<< (const T& out)
{
    const std::string* message = ::boost::get_error_info<Throw_Message>( *this );
    if (message == NULL)
        message = new std::string("");
    std::stringstream ss;
    ss << *message << out;
    *this << Throw_Message(ss.str());
    return *this;
}


template <typename Tag, typename T>
core::RuntimeException& core::RuntimeException::operator<< (const boost::error_info<Tag, T>& out)
{
    boost::operator<< (static_cast<boost::exception&>(*this), out);
    return *this;
}


template <typename T>
core::LogicErrorException& core::LogicErrorException::operator<< (const T& out)
{
    const std::string* message = ::boost::get_error_info<Throw_Message>( *this );
    if (message == NULL)
        message = new std::string("");
    std::stringstream ss;
    ss << *message << out;
    *this << Throw_Message(ss.str());
    return *this;
}


template <typename Tag, typename T>
core::LogicErrorException& core::LogicErrorException::operator<< (const boost::error_info<Tag, T>& out)
{
    boost::operator<< (static_cast<boost::exception&>(*this), out);
    return *this;
}

}

