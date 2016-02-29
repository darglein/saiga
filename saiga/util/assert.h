#include "saiga/config.h"

//enables asserts when build with saiga_debug or saiga_testing
#if defined(SAIGA_DEBUG) || defined(SAIGA_TESTING)
    #undef NDEBUG
#else
    #ifndef NDEBUG
        #define NDEBUG
    #endif
#endif

#include <assert.h>
