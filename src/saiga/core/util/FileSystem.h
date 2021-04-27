/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/saiga_modules.h"

#if __has_include(<filesystem>)
#    include <filesystem>
#else
#    include <experimental/filesystem>

// Make the experimental::filesystem namespace in std
// After this we can just write std::filesystem
namespace std
{
namespace filesystem = experimental::filesystem;
}

#endif
