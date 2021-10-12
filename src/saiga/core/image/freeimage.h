/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

#include "image.h"
#include "imageMetaData.h"

#ifdef SAIGA_USE_FREEIMAGE

class fipImage;

namespace Saiga
{
namespace FIP
{
// loads an image with freeimage converts it to Image and reads metadata if != 0
SAIGA_CORE_API bool load(const std::string& path, Image& img, ImageMetadata* metaData = nullptr);
SAIGA_CORE_API bool loadFromMemory(ArrayView<const char> data, Image& img);

// writes an image to file with freeimage
SAIGA_CORE_API bool save(const std::string& path, const Image& img);
SAIGA_CORE_API std::vector<unsigned char> saveToMemory(const Image& img, std::string file_extension = ".jpg");

// helper functions if you want to actually have fipimages
// these are used by load and save from above
SAIGA_CORE_API bool loadFIP(const std::string& path, fipImage& img);
SAIGA_CORE_API bool saveFIP(
    const std::string& path,
    fipImage& img);  // fipImage is not const to ensure compatibility with version 3.18.0 of freeimage

// conersion between saiga's image and freeimage's fipimage
// these are used by load and save from above
SAIGA_CORE_API extern void convert(const fipImage& src, Image& dest);
SAIGA_CORE_API extern void convert(const Image& src,
                                   fipImage& dest);  // copy the src image because we need to flip red and blue :(

// reads the meta data of a fipimage
// is used by load from above
SAIGA_CORE_API extern void getMetaData(fipImage& img, ImageMetadata& metaData);
SAIGA_CORE_API extern void printAllMetaData(fipImage& img);

}  // namespace FIP

}  // namespace Saiga

#endif
