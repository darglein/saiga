/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/assert.h"

#include <csignal>

namespace Saiga {

void saiga_assert_fail (const char *__assertion, const char *__file,
               unsigned int __line, const char *__function, const char *__message){

    std::cout << "Assertion '" << __assertion << "' failed!" << std::endl;
    std::cout << "  File: " << __file << ":" << __line << std::endl;
    std::cout << "  Function: " << __function << std::endl;
    std::cout << "  Message: " << __message << std::endl;

#ifdef _MSC_VER
    //breakpoint for visual studio
    __debugbreak();
#endif

    //raise sigint signal for gdb debugger
    std::raise(SIGINT);
    exit(1);
}

}
