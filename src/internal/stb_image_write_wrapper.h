/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/managedImage.h"


namespace Saiga
{
SAIGA_LOCAL bool saveImageSTB(const std::string& path, const Image& img);

SAIGA_LOCAL std::vector<uint8_t> compressImageSTB(const Image& img);


}  // namespace Saiga
