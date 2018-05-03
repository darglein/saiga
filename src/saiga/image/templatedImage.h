/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/managedImage.h"

namespace Saiga {

template<typename T>
class TemplatedImage : public Image
{
public:

    using TType = ImageTypeTemplate<T>;

    TemplatedImage() : Image(TType::type) {}
    TemplatedImage(int h, int w) : Image(h,w,TType::type){}
    TemplatedImage(std::string file) {
        load(file);
        SAIGA_ASSERT(type == TType::type);
    }

    // Note: This creates a copy of img
    TemplatedImage(ImageView<T> img)
    {
        setFormatFromImageView(img);
        create();
        img.copyTo(getImageView());
    }

    inline
    T& operator()(int y, int x){
        return rowPtr(y)[x];
    }


    inline
    T* rowPtr(int y){
        auto ptr = data8() + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }

    ImageView<T> getImageView()
    {
        return Image::getImageView<T>();
    }

    operator ImageView<T>() { return getImageView(); }
};



}
