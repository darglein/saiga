/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>
#include <vector>

namespace Saiga
{

struct Unzipfile
{
    std::string name;
    std::vector<char> data;
};


// Extracts all files inside a .zip into memory
SAIGA_CORE_API std::vector<Unzipfile> UnzipToMemory(const std::string& path);


}  // namespace Saiga
