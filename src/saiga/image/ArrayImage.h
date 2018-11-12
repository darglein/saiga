/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/managedImage.h"

namespace Saiga {

/**
 * Simple class to treat arrays of arbitrary types as images.
 */

template<typename T>
class ArrayImage : public ImageBase
{
public:
    std::vector<T> _data;

    ArrayImage(int h, int w) : ImageBase(h,w,w*sizeof(T)), _data(w*h) {}


    ImageView<T> getImageView()
    {
        ImageView<T> res(*this);
        res.data = data();
        return res;
    }

    ImageView<const T> getConstImageView() const
    {
        ImageView<const T> res(*this);
        res.data = data();
        return res;
    }



    T* data() { return reinterpret_cast<T*>(_data.data()); }
    const T* data() const { return reinterpret_cast<const T*>(_data.data()); }

    operator ImageView<T>() { return getImageView(); }
    operator ImageView<const T>() const { return getConstImageView(); }

};



}
