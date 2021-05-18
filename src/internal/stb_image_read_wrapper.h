/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/managedImage.h"


namespace Saiga
{
SAIGA_LOCAL bool loadImageSTB(const std::string& path, Image& img);

SAIGA_LOCAL bool decompressImageSTB(Image& img, std::vector<uint8_t>& data);
}  // namespace Saiga
