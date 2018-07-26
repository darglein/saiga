/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include <saiga/util/array_view.h>

#include <vector>

namespace Saiga{
namespace File {


SAIGA_GLOBAL std::vector<unsigned char> loadFileBinary(const std::string& file);
SAIGA_GLOBAL std::string                loadFileString(const std::string& file);


SAIGA_GLOBAL void saveFileBinary(const std::string& file, array_view<const char> data);
}
}
