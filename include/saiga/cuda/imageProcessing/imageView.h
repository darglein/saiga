/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/common.h"
#include "saiga/util/assert.h"
#include "saiga/util/glm.h"
#include <algorithm>

namespace Saiga {


template<typename T>
struct ImageView{
    union{
        int width;
        int cols;
    };
    union{
        int height;
        int rows;
    };
    //    int width, height;
    //    int pitch; //important: the pitch is not in bytes!!!
    int pitchBytes;

    union{
        void* data;
        uint8_t* data8;
    };



    HD inline
    ImageView(){
        static_assert(sizeof(ImageView<T>) == 24, "ImageView size wrong!");
    }

    HD inline
    ImageView(int w, int h , int p, void* data)
        : width(w),height(h),pitchBytes(p),data(data) {}

    HD inline
    ImageView(int w, int h, void* data)
        : width(w),height(h),pitchBytes(w*sizeof(T)),data(data) {}

    //size in bytes
    HD inline
    int size(){
        return height * pitchBytes;
    }

    //a view to a sub image
    HD inline
    ImageView<T> subImageView(int startX, int startY, int w, int h){
#ifdef ON_DEVICE
#else
        SAIGA_ASSERT(startX >= 0 && startX < width);
        SAIGA_ASSERT(startY >= 0 && startY < height);
        SAIGA_ASSERT(startX + w <= width);
        SAIGA_ASSERT(startY + h <= height);
#endif
        ImageView<T> iv(w,h,pitchBytes,&(*this)(startX,startY));
        return iv;
    }

    HD inline
    T& operator()(int x, int y){
#ifdef ON_DEVICE
#else
        SAIGA_ASSERT(inImage(x,y));
#endif
        //        return data[y * pitch + x];
//        uint8_t* data8 = reinterpret_cast<uint8_t*>(data);
//        data8 += y * pitchBytes + x * sizeof(T);
        auto ptr = data8 + y * pitchBytes + x * sizeof(T);
        return reinterpret_cast<T*>(ptr)[0];
    }

    HD inline
    T* rowPtr(int y){
//        uint8_t* data8 = reinterpret_cast<uint8_t*>(data);
//        data8 += y * pitchBytes;
        auto ptr = data8 + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }

    //bilinear interpolated pixel with clamp to edge boundary
    HD inline
    T inter(float sx, float sy){

        int x0 = glm::floor(sx);
        int y0 = glm::floor(sy);

        //interpolation weights
        float ax = sx - x0;
        float ay = sy - y0;

        if(x0 < 0){ x0=0;ax=0;};
        if ( x0>=width ) {x0=width-1;ax=0;}
        if ( y0<0 ) {y0=0;ay=0;}
        if ( y0>=height ) {y0=height-1;ay=0;}


#ifdef ON_DEVICE
        int x1 = min(x0 + 1, width - 1);
        int y1 = min(y0 + 1, height - 1);
#else
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);
#endif



        T res = ((*this)(x0,y0) * (1.0f - ax) + (*this)(x1,y0) * (ax)) * (1.0f - ay) +
                ((*this)(x0,y1) * (1.0f - ax) + (*this)(x1,y1) * (ax)) * (ay);
        return res;
    }

    HD inline
    bool inImage(int x, int y){
        return x >= 0 && x < width && y >=0 && y < height;
    }

    //minimum distance of the pixel to all edges
    HD inline
    int distanceFromEdge(int x, int y){
        int x0 = x;
        int x1 = width - 1 - x;
        int y0 = y;
        int y1 = height - 1 - y;
#ifdef ON_DEVICE
        return min(x0,min(x1,min(y0,y1)));
#else
        return std::min(x0,std::min(x1,std::min(y0,y1)));
#endif
    }

    template<typename AT>
    HD inline
    bool inImage(AT x, AT y){
        return x >= 0 && x <= AT(width-1) && y >=0 && y <= AT(height-1);
    }

    template<typename AT>
    HD inline
    void multWithScalar(AT a){
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                (*this)(x,y) *= a;
            }
        }
    }

    HD inline
    void clampToEdge(int& x, int& y){
#ifdef ON_DEVICE
        x = min(max(0,x),width-1);
        y = min(max(0,y),height-1);
#else
        x = std::min(std::max(0,x),width-1);
        y = std::min(std::max(0,y),height-1);
#endif
    }
};

//multiple images that are stored in memory consecutively
template<typename T>
struct ImageArrayView{
    ImageView<T> imgStart;
    int n;

    ImageArrayView(){}
    ImageArrayView(ImageView<T> _imgStart, int _n) : imgStart(_imgStart), n(_n) {}

    HD inline
    ImageView<T> at(int i){
        ImageView<T> res = imgStart;
        res.data =  imgStart.data8 + imgStart.size() * i;
        return res;
    }

    HD inline
    ImageView<T> operator[](int i){ return at(i); }
};

}
