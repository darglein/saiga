/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>

namespace Saiga
{
void saiga_assert_fail(const std::string& __assertion, const char* __file, unsigned int __line, const char* __function,
                       const std::string& __message)
{
    std::cout << "Assertion '" << __assertion << "' failed!" << std::endl;
    std::cout << "  File: " << __file << ":" << __line << std::endl;
    std::cout << "  Function: " << __function << std::endl;
    if (!__message.empty()) std::cout << "  Message: " << __message << std::endl;

    // stops and raise SIGABRT
    std::abort();
}

}  // namespace Saiga
