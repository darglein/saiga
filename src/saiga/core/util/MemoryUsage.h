
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <iostream>
namespace Saiga
{
struct MemoryInfo
{
    bool valid = false;

    // in bytes
    size_t current_memory_used = 0;
    size_t max_memory_used     = 0;
};

SAIGA_CORE_API extern MemoryInfo GetMemoryInfo();

SAIGA_CORE_API extern std::ostream& operator<<(std::ostream& strm, const MemoryInfo& mem_info);
}  // namespace Saiga
