/**

 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include <vector>

namespace Saiga
{
// AABB-like iRectangle with integer coordinates
// the index marked with begin is inclusive and the index 'end' is exclusive
template <int N>
class SAIGA_TEMPLATE iRect
{
   public:
    using Vec = Vector<int, N>;

    Vec begin, end;


    iRect() : begin(Vec::Zero()), end(Vec::Zero()) {}
    iRect(const Vec& begin, const Vec& end) : begin(begin), end(end) {}
    iRect(const Vec& index) : begin(index), end(index + Vec::Ones()) {}

    // Union
    iRect(const iRect& a, const iRect& b)
        : begin(a.begin.array().min(b.begin.array())), end(a.end.array().max(b.end.array()))
    {
    }

    Vec Size() const { return end - begin; }
    int Volume() const
    {
        auto s = Size();
        return s(0) * s(1) * s(2);
    }

    bool Contains(const Vec& index) const
    {
        return (((index.array() >= begin.array()).any()) && (index.array() < end.array()).any());
    }

    bool Contains(const iRect& other) const { return Contains(other.begin) && Contains(other.end - Vec::Ones()); }


    int Distance(const iRect& other) const
    {
        int dx = std::max({begin.x() - other.end.x(), 0, other.begin.x() - end.x()});
        int dy = std::max({begin.y() - other.end.y(), 0, other.begin.y() - end.y()});
        int dz = std::max({begin.z() - other.end.z(), 0, other.begin.z() - end.z()});
        return std::max({dx, dy, dz});
    }

    std::vector<Vec> ToPoints() const
    {
        std::vector<Vec> result;
        for (int z = begin(2); z < end(2); ++z)
        {
            for (int y = begin(1); y < end(1); ++y)
            {
                for (int x = begin(0); x < end(0); ++x)
                {
                    result.push_back({x, y, z});
                }
            }
        }
        return result;
    }

    iRect Expand(int radius) const
    {
        return iRect(begin - Vec(radius, radius, radius), end + Vec(radius, radius, radius));
    }
};

template <int N>
std::ostream& operator<<(std::ostream& strm, const iRect<N>& rect)
{
    strm << rect.begin.transpose() << " " << rect.end.transpose();
    return strm;
}

}  // namespace Saiga
