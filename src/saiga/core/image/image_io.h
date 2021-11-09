/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include <optional>
#include "managedImage.h"
namespace Saiga
{
enum class ImageCompression
{
    fast,
    medium,
    best
};

struct ImageSaveFlags
{
    ImageCompression compression = ImageCompression::medium;
};

struct ImageLoadFlags
{
};

// Base class for image IO.
// There are different specializations for this class for libPNG and freeimage
class SAIGA_CORE_API ImageIO
{
   public:
    ImageIO() {}

    virtual bool Save2File(const std::string& path, const Image& img, ImageSaveFlags flags = ImageSaveFlags()) = 0;

    virtual std::vector<unsigned char> Save2Memory(const Image& img,
                                                   ImageSaveFlags flags = ImageSaveFlags()) = 0;

    virtual std::optional<Image> LoadFromFile(const std::string& path, ImageLoadFlags flags = ImageLoadFlags())   = 0;
    virtual std::optional<Image> LoadFromMemory(void* data, size_t size, ImageLoadFlags flags = ImageLoadFlags()) = 0;
};
}  // namespace Saiga
