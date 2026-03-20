#pragma once
#include "saiga/saiga_modules.h"

#ifdef SAIGA_USE_LIBAVIF

#    include "saiga/core/image/managedImage.h"

namespace Saiga
{
SAIGA_LOCAL bool loadImageLibAVIF(const std::filesystem::path& path, Image& img);
SAIGA_LOCAL bool saveImageLibAVIF(const std::filesystem::path& path, const Image& img);

SAIGA_LOCAL bool loadImageFromMemoryLibAVIF(const void* data, size_t size, Image& img);
}  // namespace Saiga

#endif
