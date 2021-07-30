/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/Range.h"
namespace Saiga
{
struct ImageDimensions
{
    union
    {
        int h;
        int height;
        int r;
        int rows;
    };
    union
    {
        int w;
        int width;
        int c;
        int cols;
    };
    HD ImageDimensions() : h(0), w(0) {}
    HD ImageDimensions(int h, int w) : h(h), w(w) {}


    HD inline ivec2 Vector() const { return ivec2(w, h); }
    HD inline std::pair<int, int> pair() const { return {h, w}; }
    HD inline bool valid() const { return h > 0 && w > 0; }
    HD inline explicit operator bool() { return valid(); }
    HD inline bool operator==(const ImageDimensions& other) const { return pair() == other.pair(); }
    HD inline bool operator!=(const ImageDimensions& other) const { return !((*this) == other); }


    HD inline bool inImage(int y, int x) const { return x >= 0 && x < width && y >= 0 && y < height; }
    HD inline bool inImage(ivec2 ip) const { return inImage(ip.y(), ip.x()); }

    // This is the fast version of the check, which uses the trick that a negative number converted
    // to an unsigned is very large.
    HD inline bool inImage2(int y, int x) const { return (((unsigned)x) < width) & (((unsigned)y) < height); }

    template <typename AT>
    HD inline bool inImage(AT y, AT x) const
    {
        return x >= 0 && x <= AT(width - 1) && y >= 0 && y <= AT(height - 1);
    }


    HD inline void clampToEdge(int& y, int& x) const
    {
#ifdef SAIGA_ON_DEVICE
        x = std::min(max(0, x), width - 1);
        y = std::min(max(0, y), height - 1);
#else
        x = std::min(std::max(0, x), width - 1);
        y = std::min(std::max(0, y), height - 1);
#endif
    }

    // minimum distance of the pixel to all edges
    HD inline int distanceFromEdge(int y, int x) const
    {
        int x0 = x;
        int x1 = width - 1 - x;
        int y0 = y;
        int y1 = height - 1 - y;
        return std::min(x0, std::min(x1, std::min(y0, y1)));
    }

    /**
     * Usefull for range-based iteration over the image.
     * Example:
     *
     * for(auto i : img.rowRange())
     *   for(auto j : img.colRange())
     *      img(i,j) = 0;
     */
    HD inline Range<int> rowRange(int border = 0) const { return {border, rows - border}; }
    HD inline Range<int> colRange(int border = 0) const { return {border, cols - border}; }
};

/**
 * @brief The ImageBase struct
 *
 * The base of all images. It does not manage memory.
 * It only captures the key concept of an image, which is a two dimensional array
 * with a 'pitch' offset between rows.
 */
struct SAIGA_CORE_API ImageBase : public ImageDimensions
{
    size_t pitchBytes;

    HD inline ImageBase() : ImageBase(0, 0, 0)
    {
        // static_assert(sizeof(ImageBase) == 16, "ImageBase size wrong!");
    }

    HD inline ImageBase(int h, int w, int p) : ImageDimensions(h, w), pitchBytes(p) {}


    HD inline ImageDimensions dimensions() const { return *this; }

    // size in bytes
    HD inline size_t size() const { return height * pitchBytes; }
};

}  // namespace Saiga
