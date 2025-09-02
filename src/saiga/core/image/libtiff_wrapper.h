#pragma once
#include "saiga/saiga_modules.h"

#ifdef SAIGA_USE_LIBTIFF

#    include "saiga/core/image/managedImage.h"

namespace Saiga
{
SAIGA_LOCAL bool loadImageLibTiff(const std::filesystem::path& path, Image& img);
SAIGA_LOCAL bool saveImageLibTiff(const std::filesystem::path& path, const Image& img);

SAIGA_LOCAL bool loadImageFromMemoryLibTiff(const void* data, size_t size, Image& img);
}  // namespace Saiga

#endif
