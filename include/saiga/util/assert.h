/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"


//similar to unix assert.h implementation

namespace Saiga {
SAIGA_GLOBAL extern void saiga_assert_fail (const char *__assertion, const char *__file,
               unsigned int __line, const char *__function, const char *__message);
//      throw __attribute__ ((__noreturn__));
}

# if defined WIN32
#   define SAIGA_ASSERT_FUNCTION	__FUNCSIG__
# else
#include <features.h>
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define SAIGA_ASSERT_FUNCTION	__PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define SAIGA_ASSERT_FUNCTION	__func__
#  else
#   define SAIGA_ASSERT_FUNCTION	((const char *) 0)
#  endif
# endif
# endif



#if defined(SAIGA_ASSERTS)

# define SAIGA_ASSERT_MSG(expr,msg)							\
  ((expr)								\
   ? static_cast<void>(0)						\
   : Saiga::saiga_assert_fail (#expr, __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION,msg))

#else

//# define SAIGA_ASSERT_MSG(expr,msg)		(static_cast<void>(0))

//this is a trick so that no unused variable warnings are generated if a variable
//is only used in an assert
# define SAIGA_ASSERT_MSG(expr,msg)         \
   if(false) static_cast<void>(expr)


#endif

#define SAIGA_ASSERT_NOMSG(expr) SAIGA_ASSERT_MSG(expr,"")


//With this trick SAIGA_ASSERT is overloaded for 1 and 2 arguments. (With and without message)
#define GET_SAIGA_ASSERT_MACRO(_1,_2,NAME,...) NAME
#define SAIGA_ASSERT(...) GET_SAIGA_ASSERT_MACRO(__VA_ARGS__, SAIGA_ASSERT_MSG, SAIGA_ASSERT_NOMSG, 0)(__VA_ARGS__)


//#undef assert

