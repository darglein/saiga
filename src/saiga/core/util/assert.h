/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"

#include <string>

// similar to unix assert.h implementation

namespace Saiga
{
SAIGA_CORE_API extern void saiga_assert_fail(const std::string& __assertion, const char* __file, unsigned int __line,
                                             const char* __function, const std::string& __message);
}


#if defined _WIN32
#    define SAIGA_ASSERT_FUNCTION __FUNCSIG__
#    define SAIGA_SHORT_FUNCTION __FUNCSIG__
#elif defined __unix__
#    include <features.h>
#    if defined __cplusplus ? __GNUC_PREREQ(2, 6) : __GNUC_PREREQ(2, 4)
#        define SAIGA_ASSERT_FUNCTION __PRETTY_FUNCTION__
#        define SAIGA_SHORT_FUNCTION __FUNCTION__
#    else
#        if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#            define SAIGA_ASSERT_FUNCTION __func__
#            define SAIGA_SHORT_FUNCTION __func__
#        else
#            define SAIGA_ASSERT_FUNCTION ((const char*)0)
#            define SAIGA_SHORT_FUNCTION ((const char*)0)
#        endif
#    endif
#elif defined __APPLE__
#    define SAIGA_ASSERT_FUNCTION __PRETTY_FUNCTION__
#    define SAIGA_SHORT_FUNCTION __FUNCTION__
#else
#    error Unknown compiler.
#endif

#ifdef SAIGA_DEBUG
#    define SAIGA_DEBUG_ASSERT(expr) \
        ((expr) ? static_cast<void>(0) : Saiga::saiga_assert_fail(#expr, __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION, ""))

#else
#    define SAIGA_DEBUG_ASSERT(expr) \
        if (false)                   \
            static_cast<void>(expr); \
        else                         \
            static_cast<void>(0)
#endif


#if defined(SAIGA_ASSERTS)

#    define SAIGA_ASSERT_MSG(expr, msg) \
        ((expr) ? static_cast<void>(0)  \
                : Saiga::saiga_assert_fail(#expr, __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION, msg))

#else

//# define SAIGA_ASSERT_MSG(expr,msg)		(static_cast<void>(0))

// this is a trick so that no unused variable warnings are generated if a variable
// is only used in an assert
#    define SAIGA_ASSERT_MSG(expr, msg) \
        if (false)                      \
            static_cast<void>(expr);    \
        else                            \
            static_cast<void>(0)


#endif

#define SAIGA_ASSERT_NOMSG(expr) SAIGA_ASSERT_MSG(expr, "")


// With this trick SAIGA_ASSERT is overloaded for 1 and 2 arguments. (With and without message)
#define GET_SAIGA_ASSERT_MACRO(_1, _2, NAME, ...) NAME
#define SAIGA_ASSERT(...) GET_SAIGA_ASSERT_MACRO(__VA_ARGS__, SAIGA_ASSERT_MSG, SAIGA_ASSERT_NOMSG, 0)(__VA_ARGS__)


// Creates an assert with message and terminates the program.
// This is always enabled and works even if asserts are disabled.
#define SAIGA_EXIT_ERROR(msg) (Saiga::saiga_assert_fail((msg), __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION, ""))
