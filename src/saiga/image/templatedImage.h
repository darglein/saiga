/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/managedImage.h"

namespace Saiga
{
template <typename T>
class TemplatedImage : public Image
{
   public:
    using TType    = ImageTypeTemplate<T>;
    using ViewType = ImageView<T>;
    using Image::create;

    TemplatedImage() : Image(TType::type) {}
    TemplatedImage(int h, int w) : Image(h, w, TType::type) {}
    TemplatedImage(const std::string& file) { load(file); }

    // Note: This creates a copy of img
    TemplatedImage(ImageView<T> img)
    {
        setFormatFromImageView(img);
        Image::create();
        img.copyTo(getImageView());
    }

    void create(int h, int w) { Image::create(h, w, TType::type); }
    void create(int h, int w, int p) { Image::create(h, w, p, TType::type); }

    inline T& operator()(int y, int x) { return rowPtr(y)[x]; }


    inline T* rowPtr(int y)
    {
        auto ptr = data8() + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }

    ImageView<T> getImageView() { return Image::getImageView<T>(); }

    ImageView<const T> getConstImageView() const { return Image::getConstImageView<T>(); }



    T* data() { return reinterpret_cast<T*>(data8()); }
    const T* data() const { return reinterpret_cast<const T*>(data8()); }

    operator ImageView<T>() { return getImageView(); }
    operator ImageView<const T>() const { return getConstImageView(); }

    // Load + type check
    bool load(const std::string& path)
    {
        auto r = Image::load(path);
        SAIGA_ASSERT(type == TType::type);
        return r;
    }
};



}  // namespace Saiga
