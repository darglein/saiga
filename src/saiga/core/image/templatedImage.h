/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "managedImage.h"

#include <iostream>

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
    TemplatedImage(ImageDimensions dimensions) : Image(dimensions, TType::type) {}
    TemplatedImage(const std::string& file)
    {
        auto res = load(file);
        if (!res)
        {
            SAIGA_EXIT_ERROR("Could not load file " + file);
        }
    }

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
        if (!r)
        {
            return false;
        }

        if (type != TType::type)
        {
            std::cerr << "Image type does not match template argument!" << std::endl;
            std::cerr << "Loaded:   " << channels(type) << "/" << (int)elementType(type) << std::endl;
            std::cerr << "Template: " << channels(TType::type) << "/" << (int)elementType(TType::type) << std::endl;
            std::cerr << "Path:     " << path << std::endl;
            SAIGA_EXIT_ERROR("Image Load failed!");
        }
        SAIGA_ASSERT(type == TType::type);
        return r;
    }
};



}  // namespace Saiga
