/**
 * \page exceptions Exception Handling
 *
 * \author Klaus Greff
 *
 *
 *
 * \section exceptions-intro Introduction
 * All exceptions are declared in "src/core/Exceptions.h".
 * This file also contains the THROW macro and the error_info structures,
 * so all you have to do in order to use them is to:
 * \code
 * #include "core/Exceptions.h"
 * \endcode
 *
 * All Exceptions belong to the core namespace.
 *
 *
 *
 * \section exceptions-throw How to Throw Exceptions
 * While you can still throw any core::Exception using the normal \c throw
 * keyword, it is strongly recommended to use the \c THROW macro like this:
 * \code
 * THROW(core::IllegalArgumentException("bad foo!"));
 * \endcode
 *
 * This adds valuable debugging information to the exception:
 *   - The name of the current file
 *   - The name of the current method
 *   - The current line number
 *
 * That way, you get an exception message like this:
 * \code
 * [...]/src/foo.cpp(124): Throw in function void bar()
 * Dynamic exception type: boost::exception_detail::clone_impl<core::IllegalArgumentException>
 * [core::ThrowMessage_*] = bad foo!
 * \endcode
 *
 * Note also that exceptions are thrown by value. DO NOT throw pointers to exceptions!
 * (This also means not to use the \c new operator.)
 *
 * If, however, you want to re-throw an exception you caught, then use this:
 * \code
 * catch (core::RuntimeException& e)
 * {
 *     // ...
 *     throw;
 * }
 * \endcode
 * If you use \c THROW(e); or \c throw \c e; instead, you will have the following problems:
 *   - THROW will overwrite the original filename, method name, and line number.
 *   - the newly thrown exception will be of the type given in the catch statement.
 *     So it was a derived exception you might loose type information.
 *
 *
 *
 * \section exceptions-catch How to Catch Exceptions
 * It is important you catch every exception by reference and not by value
 * or even by pointer. That way you preserve polymorphism and the actual type
 * of the caught exception, but do not have to deal with memory management:
 * \code
 *  // Good:
 *  catch (core::RuntimeException& e)
 *
 *  // Bad:
 *  catch (core::RuntimeException e)
 *
 *  // Also Bad:
 *  catch (core::RuntimeException* e)
 * \endcode
 *
 *
 *
 * \section exceptions-message The Exception Message
 * The core exceptions offer two different ways to pass a message. The first
 * one is using the message constructor taking a string as argument:
 * \code
 * THROW(core::IllegalArgumentException("your message here"));
 * \endcode
 *
 * The second one is with the stream operator \c <<. This option is best if you want to "construct" a message:
 * \code
 * THROW(core::IllegalArgumentException() << "your message here. i=" << i);
 * \endcode
 *
 * Both approaches can also be combined:
 * \code
 * THROW(core::IllegalArgumentException("your message here") << " i=" << i);
 * \endcode
 *
 *
 *
 * \section exceptions-errorInfo Structured Information
 * You can also add structured information to core::Exceptions. This information is not appended to the message string,
 * but stored as a kind of key/value pair. For example you could:
 * \code
 *   // Error information definition idiom (see below for details)
 *   typedef boost::error_info<struct ThrowPortNumber_, int>  ThrowPortNumber;
 *   typedef boost::error_info<struct ThrowModuleUrl_, std::string>  ThrowModuleUrl;
 *
 *   // ...
 *
 *   THROW(IllegalArgumentException("Invalid PortNumber!") << ThrowPortNumber(port) << ThrowModuleUrl(getUrl()));
 * \endcode
 * The added information is automatically added to the what() message like this:
 * \code
 * [...]/src/AM_Example.cpp(124): Throw in function void bar()
 * Dynamic exception type: boost::exception_detail::clone_impl<core::IllegalArgumentException>
 * [core::ThrowMessage_*] = Invalid PortNumber!
 * [core::ThrowPortNumber_*] = -1
 * [core::ThrowModuleUrl_*] = foo.bar.ExampleModule
 * \endcode
 *
 * If you need to extract some of the information this can be done like this:
 * \code
 * catch (core::Exception& e)
 * {
 *     const int* = e.getErrorInfo<ThrowPortNumber>();
 *     const std::string* = e.getErrorInfo<ThrowModuleUrl>();
 * }
 * \endcode
 *
 * To define an custom error info use this idiom:
 * \code
 *   typedef boost::error_info<struct #tag#, #type#>  #name#;
 * \endcode
 * Thereby \c #tag#, is a unique name for the dummy struct used internally to distinguish different error-infos.
 * The name of this tag will also appear in the what() message as the keyname.
 * Usually #name# plus an underscore is used.
 *
 * \c #type# simply denotes the type of the value(cannot be a reference), and \c #name# is the name of the error info
 * as used within the sourcecode.
 *
 * So you could for example define an ErrorInfo to store the name of a Module as:
 * \code
 *   typedef boost::error_info<struct ThrowModuleName_, std::string>  ThrowModuleName;
 * \endcode
 *
 *
 * \section exceptions-appending Appending Information to Exceptions
 * You can also add information to an Exception that was already thrown. To do so just catch the exception
 * (by reference!) and use the streaming operator as before to add whatever information you like. As before
 * error-infos are stored separately and anything else is appended to the message string.
 * Remember: It is important to re-throw the exception using plain \c throw;:
 * \code
 * catch (core::Exception& e)
 * {
 *      // add the module name to the exception and re-throw it
 *      e << ThrowModuleName("FooBar");
 *      throw;
 * }
 * \endcode
 *
 *
 *
 * \section exceptions-choosing-throw Throwing Guidelines
 *   - Always throw by value.
 *   - Make sure the exception is informative, either by its type or by its message.
 *   - Prefer to throw specific exceptions.
 *   - You may throw a LogicException if you don't want it to be caught and you give an informative message.
 *   - Don't throw an core::RuntimeExcepiton, but use a derived exception.
 *   - If no Exception matches your case, consider to define a new one (it's easy)
 *
 *
 *
 * \section exceptions-choosing-catch Catching Guidelines
 *   - Catch only specific exceptions.
 *   - Always catch by reference.
 *   - You can catch generic core::Exception if you re-throw it afterwards.
 *   - you should (almost) never catch everything: \c catch(...).
 *   - Don't catch exceptions inheriting from core::LogicErrorException, as they usually indicate a bug.
 *
 *
 *
 * \section exceptions-define-new Define Your Own Exception
 * Defining your own exceptions is easy. First you have to find a suitable base Exception. Consider:
 *   - any specific exception that can be considered a category for your exception.
 *   - \c core::LogicException if your exception indicates a programming error/bug and should not be caught
 *   - \c core::RuntimeExcepiton if your exception indicates a runtime error that might be caught and handled appropriately later.
 *
 * You can inherit from multiple exceptions, but make sure not to mix the two categories
 * \c LogicErrorExcepitons and \c RuntimeExcepitons. Also you should not inherit directly from \c core::Exception.
 *
 * After that declaring it is easy:
 * \code
 * class FooBarException : public virtual LogicException
 * {
 * public:
 *     FooBarException() throw();
 *     explicit FooBarException(const std::string& what) throw();
 * };
 * \endcode
 * And the implementation is easy too:
 * \code
 * FooBarException::FooBarException() throw() { }
 * FooBarException::FooBarException(const std::string& what) throw()
 *         : Exception(what)
 * {
 * }
 * \endcode
 *
 *
 */
