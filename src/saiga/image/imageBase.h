/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/common.h"

namespace Saiga {

/**
 * @brief The ImageBase struct
 *
 * The base of all images. It does not manage memory.
 * It only captures the key concept of an image, which is a two dimensional array
 * with a 'pitch' offset between rows.
 */
struct SAIGA_GLOBAL ImageBase
{
    union{
        int width;
        int cols;
    };
    union{
        int height;
        int rows;
    };

    size_t pitchBytes;



    HD inline
    ImageBase()
     : ImageBase(0,0,0)
    {
        static_assert(sizeof(ImageBase) == 16, "ImageView size wrong!");
    }

    HD inline
    ImageBase(int h, int w , int p)
        : width(w),height(h),pitchBytes(p) {}


    //size in bytes
    HD inline
    size_t size() const
    {
        return height * pitchBytes;
    }





    HD inline
    bool inImage(int y, int x){
        return x >= 0 && x < width && y >=0 && y < height;
    }

    template<typename AT>
    HD inline
    bool inImage(AT y, AT x){
        return x >= 0 && x <= AT(width-1) && y >=0 && y <= AT(height-1);
    }


    HD inline
    void clampToEdge(int& y, int& x){
#ifdef SAIGA_ON_DEVICE
        x = min(max(0,x),width-1);
        y = min(max(0,y),height-1);
#else
        x = std::min(std::max(0,x),width-1);
        y = std::min(std::max(0,y),height-1);
#endif
    }

    //minimum distance of the pixel to all edges
    HD inline
    int distanceFromEdge(int y, int x){
        int x0 = x;
        int x1 = width - 1 - x;
        int y0 = y;
        int y1 = height - 1 - y;
#ifdef SAIGA_ON_DEVICE
        return min(x0,min(x1,min(y0,y1)));
#else
        return std::min(x0,std::min(x1,std::min(y0,y1)));
#endif
    }

};

}
