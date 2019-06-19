/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/imageViewIterators.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/math/imath.h"

#include "floatTexels.h"
#include "imageBase.h"

#include <algorithm>
#include <array>

#if 0
#    if defined(SAIGA_USE_CUDA)
#        include <vector_types.h>
#    else
#        if !defined(__VECTOR_TYPES_H__)
struct uchar3
{
    unsigned char x, y, z;
};

struct SAIGA_ALIGN(4) uchar4
{
    unsigned char x, y, z, w;
};
#        endif
#    endif
#endif

namespace Saiga
{
template <typename T>
struct SAIGA_TEMPLATE ImageView : public ImageBase
{
    // Get the void* and uint8_t* types with the same constness as T
    using RawDataType  = typename std::conditional<std::is_const<T>::value, const void, void>::type;
    using RawDataType8 = typename std::conditional<std::is_const<T>::value, const uint8_t, uint8_t>::type;
    using NoConstType  = typename std::remove_const<T>::type;
    using Type         = T;
    union {
        RawDataType* data;
        RawDataType8* data8;
        T* dataT;
    };

    HD inline ImageView() { static_assert(sizeof(ImageView<T>) == 24, "ImageView size wrong!"); }

    HD inline ImageView(int h, int w, int p, void* data) : ImageBase(h, w, p), data(data) {}

    HD inline ImageView(int h, int w, void* data) : ImageBase(h, w, w * sizeof(T)), data(data) {}

    HD inline ImageView(int h, int w, int p, const void* data) : ImageBase(h, w, p), data(data) {}

    HD inline ImageView(int h, int w, const void* data) : ImageBase(h, w, w * sizeof(T)), data(data) {}


    HD inline explicit ImageView(const ImageBase& base) : ImageBase(base) {}

    // convert T to const T in constructor
    HD inline ImageView(const ImageView<NoConstType>& other) : ImageBase(other), data(other.data) {}

    // size in bytes
    HD inline int size() { return height * pitchBytes; }

    // a view to a sub image
    HD inline ImageView<T> subImageView(int startY, int startX, int h, int w)
    {
#ifdef SAIGA_ON_HOST
        SAIGA_ASSERT(startX >= 0 && startX < width);
        SAIGA_ASSERT(startY >= 0 && startY < height);
        SAIGA_ASSERT(startX + w <= width);
        SAIGA_ASSERT(startY + h <= height);
#endif
        ImageView<T> iv(h, w, pitchBytes, &(*this)(startY, startX));
        return iv;
    }

    /**
     * @brief setSubImage
     * Copies the image "img" to the region starting at [startY,startX] of this image.
     */
    inline void setSubImage(int startY, int startX, ImageView<T> img)
    {
        SAIGA_ASSERT(img.width + startX <= width && img.height + startY <= height);

        for (int i = 0; i < img.height; ++i)
        {
            for (int j = 0; j < img.width; ++j)
            {
                (*this)(startY + i, startX + j) = img(i, j);
            }
        }
    }

    // does not change the data
    HD inline ImageView<T> yFlippedImageView()
    {
        ImageView<T> fy = *this;
        fy.data         = rowPtr(height - 1);
        //        fy.pitchBytes = -pitchBytes;
        //        fy.pitchBytes = 0xFFFFFFFFFFFFFFFF - pitchBytes + 1;
        fy.pitchBytes = ~pitchBytes + 1;
        return fy;
    }

    HD inline T& operator()(int y, int x) { return rowPtr(y)[x]; }

    HD inline const T& operator()(int y, int x) const { return rowPtr(y)[x]; }


    HD inline T* rowPtr(int y)
    {
        auto ptr = data8 + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }

    HD inline const T* rowPtr(int y) const
    {
        auto ptr = data8 + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }

    // bilinear interpolated pixel with UV values [0,1]
    HD inline T interUV(float u, float v) const { return inter(v * (height - 1), u * (width - 1)); }

    // bilinear interpolated pixel with clamp to edge boundary
    HD inline T inter(float sy, float sx) const
    {
        int x0 = iFloor(sx);
        int y0 = iFloor(sy);

        // interpolation weights
        float ax = sx - x0;
        float ay = sy - y0;

        if (x0 < 0)
        {
            x0 = 0;
            ax = 0;
        };
        if (x0 >= width)
        {
            x0 = width - 1;
            ax = 0;
        }
        if (y0 < 0)
        {
            y0 = 0;
            ay = 0;
        }
        if (y0 >= height)
        {
            y0 = height - 1;
            ay = 0;
        }


#ifdef SAIGA_ON_DEVICE
        int x1 = min(x0 + 1, width - 1);
        int y1 = min(y0 + 1, height - 1);
#else
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);
#endif

        // We need to convert the texel to a floating point type for interpolation.
        // This enables us to interpolate classic 8-bit rgb images.
        TexelFloatConverter<T, false> ttf;

        auto b00 = ttf.toFloat((*this)(y0, x0));
        auto b01 = ttf.toFloat((*this)(y0, x1));
        auto b10 = ttf.toFloat((*this)(y1, x0));
        auto b11 = ttf.toFloat((*this)(y1, x1));

        auto res = (b00 * (1.0f - ax) + b01 * (ax)) * (1.0f - ay) + (b10 * (1.0f - ax) + b11 * (ax)) * (ay);

        return ttf.fromFloat(res);
    }

    template <typename AT>
    HD inline void multWithScalar(AT a)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                (*this)(y, x) *= a;
            }
        }
    }

    template <typename AT>
    HD inline void add(AT a)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                (*this)(y, x) += a;
            }
        }
    }

    template <typename AT>
    HD inline void set(AT a)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                (*this)(y, x) = a;
            }
        }
    }

    template <typename T2>
    inline void copyTo(ImageView<T2> a) const
    {
        SAIGA_ASSERT(height == a.height && width == a.width);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                a(y, x) = (*this)(y, x);
            }
        }
    }

    template <typename T2, typename Op>
    inline void copyToTransform(ImageView<T2> a, Op op) const
    {
        SAIGA_ASSERT(height == a.height && width == a.width);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                a(y, x) = op((*this)(y, x));
            }
        }
    }

    template <typename T2, typename MT>
    inline void copyTo(ImageView<T2> a, MT alpha) const
    {
        SAIGA_ASSERT(height == a.height && width == a.width);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                a(y, x) = (*this)(y, x) * alpha;
            }
        }
    }

    template <typename T2>
    inline void copyToScaleDownEverySecond(ImageView<T2> a) const
    {
        SAIGA_ASSERT(height / 2 == a.height && width / 2 == a.width);
        for (int y = 0; y < a.height; ++y)
        {
            for (int x = 0; x < a.width; ++x)
            {
                a(y, x) = (*this)(y * 2, x * 2);
            }
        }
    }


    template <typename T2, bool LOW = true>
    inline void copyToScaleDownMedian(ImageView<T2> dst) const
    {
        SAIGA_ASSERT(height / 2 == dst.height && width / 2 == dst.width);
        for (int i = 0; i < dst.height; ++i)
        {
            for (int j = 0; j < dst.width; ++j)
            {
                std::array<float, 4> vs;
                for (int di = 0; di < 2; ++di)
                {
                    for (int dj = 0; dj < 2; ++dj)
                    {
                        vs[di * 2 + dj] = (*this)(i * 2 + di, j * 2 + dj);
                    }
                }
                std::sort(vs.begin(), vs.end());
                dst(i, j) = LOW ? vs[1] : vs[2];
            }
        }
    }


    /**
     * Copies this image to the target image.
     * The target image can have a different size.
     * The image will be scaled with bilinear interpolation.
     */
    template <typename T2>
    inline void copyScaleLinear(ImageView<T2> a) const
    {
        for (int y = 0; y < a.height; ++y)
        {
            for (int x = 0; x < a.width; ++x)
            {
                float u_long = (x + 0.5f) / a.width;
                float v_long = (y + 0.5f) / a.height;

                float x2 = u_long * width - 0.5f;
                float y2 = v_long * height - 0.5f;

                a(y, x) = this->inter(y2, x2);
            }
        }
    }

    /**
     * Copies this image to the target image.
     * The target image must be a power of 2 smaller.
     * The resulting pixels will be averaged.
     *
     * Factor must be a power of 2!!!
     */
    template <typename T2>
    inline void copyScaleDownPow2(ImageView<T2> a, int factor) const
    {
        SAIGA_ASSERT(height / factor == a.height && width / factor == a.width);

        using TFC = TexelFloatConverter<T, false>;
        TFC ttf;


        float div = 1.0f / (factor * factor);

        for (int y = 0; y < a.height; ++y)
        {
            int gy = y * factor;
            for (int x = 0; x < a.width; ++x)
            {
                int gx = x * factor;

                typename TFC::FloatType sum(0);

                // Average inner patch
                for (int i = 0; i < factor; ++i)
                {
                    for (int j = 0; j < factor; ++j)
                    {
                        T2 value = (*this)(gy + i, gx + j);
                        sum += ttf.toFloat(value);
                    }
                }

                sum *= div;
                a(y, x) = ttf.fromFloat(sum);
            }
        }
    }


    /**
     * Computes the gradient in x direction.
     * Zentral difference is used for all pixels except the border.
     * At the border forward/backward difference is used.
     */
    inline void gx(ImageView<T> gradient) const
    {
        SAIGA_ASSERT(height == gradient.height && width == gradient.width);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                auto zentralDifference = (*this)(y, x + 1) - (*this)(y, x - 1);
                gradient(y, x)         = zentralDifference / T(2);
            }
            // left border (forward difference)
            gradient(y, 0) = (*this)(y, 1) - (*this)(y, 0);
            // right border (backward difference)
            gradient(y, w - 1) = (*this)(y, w - 1) - (*this)(y, w - 2);
        }
    }

    /**
     * Computes the gradient in y direction.
     * See 'gx' for more information.
     */
    inline void gy(ImageView<T> gradient) const
    {
        SAIGA_ASSERT(height == gradient.height && width == gradient.width);

        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                auto zentralDifference = (*this)(y + 1, x) - (*this)(y - 1, x);
                gradient(y, x)         = zentralDifference / T(2);
            }
        }


        for (int x = 0; x < width; ++x)
        {
            // upper border (forward difference)
            gradient(0, x) = (*this)(1, x) - (*this)(0, x);
            // lower border (backward difference)
            gradient(h - 1, x) = (*this)(h - 1, x) - (*this)(h - 2, x);
        }
    }



    HD inline void clampToEdge(int& y, int& x)
    {
#ifdef SAIGA_ON_DEVICE
        x = min(max(0, x), width - 1);
        y = min(max(0, y), height - 1);
#else
        x      = std::min(std::max(0, x), width - 1);
        y      = std::min(std::max(0, y), height - 1);
#endif
    }

    HD inline T clampedRead(int y, int x)
    {
        clampToEdge(y, x);
        return (*this)(y, x);
    }


    HD inline T borderRead(int y, int x, const T& borderValue) { return inImage(y, x) ? (*this)(y, x) : borderValue; }

    inline void findMinMax(T& minV, T& maxV)
    {
        minV = std::numeric_limits<T>::max();
        maxV = std::numeric_limits<T>::min();
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                auto v = (*this)(y, x);
                minV   = std::min(minV, v);
                maxV   = std::max(maxV, v);
            }
        }
    }

    inline void findMinMaxOutlier(T& minV, T& maxV, T outlier)
    {
        minV = std::numeric_limits<T>::max();
        maxV = std::numeric_limits<T>::min();
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                auto v = (*this)(y, x);
                if (v != outlier)
                {
                    minV = std::min(minV, v);
                    maxV = std::max(maxV, v);
                }
            }
        }
    }

    template <typename V>
    inline void setChannel(int c, V v)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                (*this)(y, x)[c] = v;
            }
        }
    }

    inline void swapChannels(int c1, int c2)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                std::swap((*this)(y, x)[c1], (*this)(y, x)[c2]);
            }
        }
    }

    /**
     * Normalizes the image so that all values are in the range 0/1
     */
    inline void normalize()
    {
        T minV, maxV;
        findMinMax(minV, maxV);
        add(-minV);
        multWithScalar(T(1) / maxV);
    }


    inline void flipY()
    {
        for (int y = 0; y < height / 2; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                std::swap((*this)(y, x), (*this)(height - y - 1, x));
            }
        }
    }

    inline void flipX()
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width / 2; ++x)
            {
                std::swap((*this)(y, x), (*this)(y, width - x - 1));
            }
        }
    }

    // write only if the point is in the image
    HD inline void clampedWrite(int y, int x, const T& v)
    {
        if (inImage(y, x)) (*this)(y, x) = v;
    }


    using ImageIterator = ImageIteratorRowmajor<ImageView<T>>;
    /**
     * This is the recommended way to iterate over all pixels of an image:
     *
     *   for (auto row : image)
     *   {
     *       for (auto p : row)
     *       {
     *           std::cout << p.x() << " " << p.y() << " " << p.value() << std::endl;
     *       }
     *   }
     */
    ImageIterator begin() { return ImageIteratorRowmajor<ImageView<T>>(*this, 0); }
    ImageIterator end() { return ImageIteratorRowmajor<ImageView<T>>(*this, h); }


    template <typename T2>
    friend std::ostream& operator<<(std::ostream& os, const ImageView<T2>& iv);
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const ImageView<T>& iv)
{
    os << "ImageView " << iv.width << "x" << iv.height << " " << iv.pitchBytes << " ptr: " << iv.data;
    return os;
}

// multiple images that are stored in memory consecutively
template <typename T>
struct ImageArrayView
{
    ImageView<T> imgStart;
    int n;

    ImageArrayView() {}
    ImageArrayView(ImageView<T> _imgStart, int _n) : imgStart(_imgStart), n(_n) {}

    HD inline ImageView<T> at(int i)
    {
        ImageView<T> res = imgStart;
        res.data         = imgStart.data8 + imgStart.size() * i;
        return res;
    }

    HD inline ImageView<T> operator[](int i) { return at(i); }

    //    HD inline
    //    T& operator()(int x, int y, int z){
    //        auto ptr = imgStart.data8 + z * imgStart.size() + y * imgStart.pitchBytes + x * sizeof(T);
    //        return reinterpret_cast<T*>(ptr)[0];
    //    }

    HD inline T& atIARVxxx(int z, int y, int x)
    {
        auto ptr = imgStart.data8 + z * imgStart.size() + y * imgStart.pitchBytes + x * sizeof(T);
        return reinterpret_cast<T*>(ptr)[0];
    }
};

}  // namespace Saiga
