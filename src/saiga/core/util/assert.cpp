/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/assert.h"

#include "saiga/core/util/crash.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>

namespace Saiga
{
void saiga_assert_fail(const std::string& __assertion, const char* __file, unsigned int __line, const char* __function,
                       const std::string& __message)
{
    // stderr: diagnostics (and gtest death tests match stderr)
    std::cerr << "Saiga Assertion '" << __assertion << "' failed!" << std::endl;
    std::cerr << "  File: " << __file << ":" << __line << std::endl;
    std::cerr << "  Function: " << __function << std::endl;
    if (!__message.empty()) std::cerr << "  Message: " << __message << std::endl;

    printCurrentStack();
    // stops and raise SIGABRT
    std::abort();
}

}  // namespace Saiga
