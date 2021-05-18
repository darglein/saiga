/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

/*
 * credits for write PNG:
 * http://www.libpng.org/pub/png/book/chapter15.html
 * credits for read PNG:
 * http://www.libpng.org/pub/png/book/chapter13.html
 * http://blog.nobel-joergensen.com/2010/11/07/loading-a-png-as-texture-in-opengl-using-libpng/
 */

#include "managedImage.h"

#ifdef SAIGA_USE_PNG

namespace Saiga
{
namespace LibPNG
{
enum class Compression
{
    fast,
    medium,
    best
};

SAIGA_CORE_API bool save(const std::string& path, const Image& img, bool invertY = false,
                         Compression compression = Compression::medium);
SAIGA_CORE_API bool load(const std::string& path, Image& img, bool invertY = false);

}  // namespace LibPNG
}  // namespace Saiga

#endif
