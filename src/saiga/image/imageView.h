/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/imageBase.h"
#include "saiga/util/imath.h"
#include "saiga/util/assert.h"
#include <algorithm>

#if 0
#if defined(SAIGA_USE_CUDA)
#include <vector_types.h>
#else
#if !defined (__VECTOR_TYPES_H__)
struct uchar3
{
    unsigned char x, y, z;
};

struct SAIGA_ALIGN(4) uchar4
{
    unsigned char x, y, z, w;
};
#endif
#endif
#endif

namespace Saiga {


template<typename T>
struct SAIGA_TEMPLATE ImageView : public ImageBase
{
    union{
        void* data;
        uint8_t* data8;
    };

    HD inline
    ImageView(){
        static_assert(sizeof(ImageView<T>) == 24, "ImageView size wrong!");
    }

    HD inline
    ImageView(int h, int w , int p, void* data)
        : ImageBase(h,w,p) , data(data) {}

    HD inline
    ImageView(int h, int w, void* data)
        : ImageBase(h,w,w*sizeof(T)), data(data) {}


    HD inline
    ImageView(const ImageBase& base)
        : ImageBase(base) {}

    //size in bytes
    HD inline
    int size(){
        return height * pitchBytes;
    }

    //a view to a sub image
    HD inline
    ImageView<T> subImageView(int startY, int startX, int h, int w){
#ifdef SAIGA_ON_HOST
        SAIGA_ASSERT(startX >= 0 && startX < width);
        SAIGA_ASSERT(startY >= 0 && startY < height);
        SAIGA_ASSERT(startX + w <= width);
        SAIGA_ASSERT(startY + h <= height);
#endif
        ImageView<T> iv(h,w,pitchBytes,&(*this)(startY,startX));
        return iv;
    }

    /**
     * @brief setSubImage
     * Copies the image "img" to the region starting at [startY,startX] of this image.
     */
    inline
    void setSubImage(int startY, int startX, ImageView<T> img)
    {
        SAIGA_ASSERT(img.width + startX <= width && img.height + startY <= height);

        for(int i = 0; i < img.height; ++i){
            for(int j =0; j < img.width; ++j)
            {
                (*this)(startY + i, startX + j) = img(i,j);
            }
        }
    }

    //does not change the data
    HD inline
    ImageView<T> yFlippedImageView()
    {
        ImageView<T> fy = *this;
        fy.data = rowPtr(height-1);
        fy.pitchBytes = -pitchBytes;
        return fy;
    }

    HD inline
    T& operator()(int y, int x){
        return rowPtr(y)[x];
    }

    HD inline
    const T& operator()(int y, int x) const{
        return rowPtr(y)[x];
    }


    HD inline
    T* rowPtr(int y){
        auto ptr = data8 + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }

    HD inline
    const T* rowPtr(int y) const{
        auto ptr = data8 + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }


    //bilinear interpolated pixel with clamp to edge boundary
    HD inline
    T inter(float sy, float sx){

        int x0 = iFloor(sx);
        int y0 = iFloor(sy);

        //interpolation weights
        float ax = sx - x0;
        float ay = sy - y0;

        if(x0 < 0){ x0=0;ax=0;};
        if ( x0>=width ) {x0=width-1;ax=0;}
        if ( y0<0 ) {y0=0;ay=0;}
        if ( y0>=height ) {y0=height-1;ay=0;}


#ifdef SAIGA_ON_DEVICE
        int x1 = min(x0 + 1, width - 1);
        int y1 = min(y0 + 1, height - 1);
#else
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);
#endif

        T res = ((*this)(y0,x0) * (1.0f - ax) + (*this)(y1,x0) * (ax)) * (1.0f - ay) +
                ((*this)(y0,x1) * (1.0f - ax) + (*this)(y1,x1) * (ax)) * (ay);
        return res;
    }

    template<typename AT>
    HD inline
    void multWithScalar(AT a){
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                (*this)(y,x) *= a;
            }
        }
    }

    template<typename AT>
    HD inline
    void add(AT a){
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                (*this)(y,x) += a;
            }
        }
    }

    template<typename AT>
    HD inline
    void set(AT a){
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                (*this)(y,x) = a;
            }
        }
    }

    template<typename T2>
    inline
    void copyTo(ImageView<T2> a) const
    {
        SAIGA_ASSERT(height == a.height && width == a.width);
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                a(y,x) = (*this)(y,x);
            }
        }
    }

    template<typename T2, typename MT>
    inline
    void copyTo(ImageView<T2> a, MT alpha) const
    {
        SAIGA_ASSERT(height == a.height && width == a.width);
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                a(y,x) = (*this)(y,x) * alpha;
            }
        }
    }

    template<typename T2>
    inline
    void copyToScaleDownEverySecond(ImageView<T2> a) const
    {
        SAIGA_ASSERT(height/2 == a.height && width/2 == a.width);
        for(int y = 0; y < a.height; ++y){
            for(int x = 0; x < a.width; ++x){
                a(y,x) = (*this)(y*2,x*2);
            }
        }
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

    HD inline
    T clampedRead(int y, int x){
        clampToEdge(y,x);
        return (*this)(y,x);
    }


    HD inline
    T borderRead(int y, int x, const T& borderValue){
        return inImage(y,x) ? (*this)(y,x) : borderValue;
    }

    inline
    void findMinMax(T& minV, T& maxV)
    {
        minV = std::numeric_limits<T>::max();
        maxV = std::numeric_limits<T>::min();
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                auto v = (*this)(y,x);
                minV = std::min(minV,v);
                maxV = std::max(maxV,v);
            }
        }
    }

    inline
    void findMinMaxOutlier(T& minV, T& maxV, T outlier)
    {
        minV = std::numeric_limits<T>::max();
        maxV = std::numeric_limits<T>::min();
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                auto v = (*this)(y,x);
                if(v != outlier)
                {
                    minV = std::min(minV,v);
                    maxV = std::max(maxV,v);
                }
            }
        }
    }

    template<typename V>
    inline void setChannel(int c, V v)
    {
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                (*this)(y,x)[c] = v;
            }
        }
    }

    inline void swapChannels(int c1, int c2)
    {
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                std::swap((*this)(y,x)[c1],(*this)(y,x)[c2]);
            }
        }
    }

    /**
     * Normalizes the image so that all values are in the range 0/1
     */
    inline
    void normalize()
    {
        T minV, maxV;
        findMinMax(minV,maxV);
        add(-minV);
        multWithScalar(T(1) / maxV);
    }


    inline
    void flipY()
    {
        for(int y = 0; y < height / 2; ++y){
            for(int x = 0; x < width; ++x){
                std::swap((*this)(y,x),(*this)(height-y-1,x));
            }
        }
    }

	inline
		void flipX()
	{
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width / 2; ++x) {
				std::swap((*this)(y, x), (*this)(y, width - x - 1));
			}
		}
	}

    //write only if the point is in the image
    HD inline
    void clampedWrite(int y, int x, const T& v){
        if(inImage(y,x))
            (*this)(y,x) = v;
    }


    template<typename T2>
    friend std::ostream& operator<<(std::ostream& os, const ImageView<T2>& iv);
};

template<typename T>
inline
std::ostream& operator<<(std::ostream& os, const ImageView<T>& iv)
{
    os << "ImageView " << iv.width << "x" << iv.height << " " << iv.pitchBytes;
    return os;
}

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

    //    HD inline
    //    T& operator()(int x, int y, int z){
    //        auto ptr = imgStart.data8 + z * imgStart.size() + y * imgStart.pitchBytes + x * sizeof(T);
    //        return reinterpret_cast<T*>(ptr)[0];
    //    }

    HD inline
    T& atIARVxxx(int z, int y, int x){
        auto ptr = imgStart.data8 + z * imgStart.size() + y * imgStart.pitchBytes + x * sizeof(T);
        return reinterpret_cast<T*>(ptr)[0];
    }
};

}
