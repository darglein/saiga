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

#include "image_io.h"

#ifdef SAIGA_USE_FREEIMAGE

//class fipImage;

namespace Saiga
{
class SAIGA_CORE_API ImageIOLibFreeimage : public ImageIO
{
   public:
    virtual bool Save2File(const std::string& path, const Image& img, ImageSaveFlags flags = ImageSaveFlags()) override;

    virtual std::vector<unsigned char> Save2Memory(const Image& img, ImageSaveFlags flags = ImageSaveFlags()) override
    {
        return {};
    }

    virtual std::optional<Image> LoadFromFile(const std::string& path,
                                              ImageLoadFlags flags = ImageLoadFlags()) override;
    virtual std::optional<Image> LoadFromMemory(void* data, size_t size,
                                                ImageLoadFlags flags = ImageLoadFlags()) override
    {
        return {};
    }


    // loads an image with freeimage converts it to Image and reads metadata if != 0
    // bool load(const std::string& path, Image& img, ImageMetadata* metaData = nullptr);
    // bool loadFromMemory(ArrayView<const char> data, Image& img);
    //
    // // writes an image to file with freeimage
    // bool save(const std::string& path, const Image& img);
    // std::vector<unsigned char> saveToMemory(const Image& img, std::string file_extension = ".jpg");
    //
    // // helper functions if you want to actually have fipimages
    // // these are used by load and save from above
    // bool loadFIP(const std::string& path, fipImage& img);
    // bool saveFIP(const std::string& path,
    //              fipImage& img);  // fipImage is not const to ensure compatibility with version 3.18.0 of freeimage
};
//
//// conersion between saiga's image and freeimage's fipimage
//// these are used by load and save from above
//SAIGA_CORE_API extern void convert(const fipImage& src, Image& dest);
//SAIGA_CORE_API extern void convert(const Image& src,
//                                   fipImage& dest);  // copy the src image because we need to flip red and blue :(
//
//// reads the meta data of a fipimage
//// is used by load from above
//SAIGA_CORE_API extern void getMetaData(fipImage& img, ImageMetadata& metaData);
//SAIGA_CORE_API extern void printAllMetaData(fipImage& img);


}  // namespace Saiga

#endif
