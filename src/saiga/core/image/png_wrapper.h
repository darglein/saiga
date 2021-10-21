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

#include "image_io.h"
#ifdef SAIGA_USE_PNG


#    include <png.h>
#    include <zlib.h>


namespace Saiga
{
class SAIGA_CORE_API ImageIOLibPNG : public ImageIO
{
   public:
    virtual bool Save2File(const std::string& path, const Image& img, ImageSaveFlags flags = ImageSaveFlags()) override;

    virtual std::vector<unsigned char> Save2Memory(const Image& img,
                                                   ImageSaveFlags flags = ImageSaveFlags()) override;

    virtual std::optional<Image> LoadFromFile(const std::string& path,
                                              ImageLoadFlags flags = ImageLoadFlags()) override;
    virtual std::optional<Image> LoadFromMemory(void* data, size_t size,
                                                ImageLoadFlags flags = ImageLoadFlags()) override;

   private:
};


}  // namespace Saiga

#endif
