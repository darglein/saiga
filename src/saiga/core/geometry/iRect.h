/**

 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include <iostream>
#include <vector>

#include <unordered_map>

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

    void setZero()
    {
        begin.setZero();
        end.setZero();
    }

    bool Empty() const { return end == begin; }

    Vec Size() const { return end - begin; }

    Vec Center() const { return (begin + end) / 2; }
    int Volume() const
    {
        auto s = Size();
        return s(0) * s(1) * s(2);
    }

    bool operator==(const iRect& other) const { return begin == other.begin && end == other.end; }

    iRect Add(const ivec3& value) const { return iRect(begin + value, end + value); }

    bool Contains(const Vec& index) const
    {
        return (((index.array() >= begin.array()).all()) && (index.array() < end.array()).all());
    }

    bool Contains(const iRect& other) const { return Contains(other.begin) && Contains(other.end - Vec::Ones()); }

    bool Intersect(const iRect other) const
    {
        if (Empty() || other.Empty()) return false;
        int dx = std::max({begin.x() - other.end.x(), other.begin.x() - end.x()});
        int dy = std::max({begin.y() - other.end.y(), other.begin.y() - end.y()});
        int dz = std::max({begin.z() - other.end.z(), other.begin.z() - end.z()});

        return dx < 0 && dy < 0 && dz < 0;
    }

    // Distance > 0  the rectangles have at least 1 cell space between them
    // Distance == 0 the rectangles are touching
    // Distance < 0  the rectangles are overlapping
    int Distance(const iRect& other) const
    {
        //        SAIGA_ASSERT(!Empty());
        //        SAIGA_ASSERT(!other.Empty());

        int dx = std::max({begin.x() - other.end.x(), other.begin.x() - end.x()});
        int dy = std::max({begin.y() - other.end.y(), other.begin.y() - end.y()});
        int dz = std::max({begin.z() - other.end.z(), other.begin.z() - end.z()});
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


    int maxDimension() const
    {
        auto d = Size();

        int m  = -1;
        int mi = -1;

        for (int i = 0; i < 3; ++i)
        {
            if (d[i] > m)
            {
                mi = i;
                m  = d[i];
            }
        }
        return mi;
    }


    bool ShrinkOtherToThis(const iRect other, iRect& output_inner, iRect& output_outer_l, iRect& output_outer_r)
    {
        if (Contains(other))
        {
            // This fully contains the other
            // -> the other can be eliminated
            output_inner = other;
            output_outer_l.setZero();
            output_outer_r.setZero();
            return true;
        }

        Vector<bool, 3> begin_test = begin.array() <= other.begin.array();
        Vector<bool, 3> end_test   = end.array() >= other.end.array();
        Vector<bool, 3> contain    = begin_test.array() && end_test.array();

        int sum_contain = contain.cast<int>().array().sum();


        {
            bool contain_x = begin.x() <= other.begin.x() && end.x() >= other.end.x();
            bool contain_y = begin.y() <= other.begin.y() && end.y() >= other.end.y();
            bool contain_z = begin.z() <= other.begin.z() && end.z() >= other.end.z();

            Vector<bool, 3> test;
            test(0) = contain_x;
            test(1) = contain_y;
            test(2) = contain_z;
            SAIGA_ASSERT(contain == test);
            int sum_contain_test = contain_x + contain_y + contain_z;
            SAIGA_ASSERT(sum_contain == sum_contain_test);
        }

        SAIGA_ASSERT(sum_contain <= 2);
        if (sum_contain <= 1) return false;

        int axis = (!contain(0) * 0) + (!contain(1) * 1) + (!contain(2) * 2);
        //        int move_begin = begin_test(axis);

        if (end(axis) <= other.begin(axis) || begin(axis) >= other.end(axis))
        {
            //            std::cout << "fake overlap" << std::endl;
            return false;
        }

        //        if (other.begin(axis) < begin(axis) && other.end(axis) > end(axis))
        //        {
        //            //                    std::cout << "double shrink" << std::endl;
        //            return false;
        //        }


        output_inner = other;

        output_outer_l.setZero();
        output_outer_r.setZero();
        //        if (move_begin)
        if (other.begin(axis) < begin(axis))
        {
            output_outer_l           = other;
            output_outer_l.end(axis) = begin(axis);
            output_inner.begin(axis) = begin(axis);
        }

        if (other.end(axis) > end(axis))
        {
            output_outer_r             = other;
            output_outer_r.begin(axis) = end(axis);
            output_inner.end(axis)     = end(axis);
        }
        //        else



        SAIGA_ASSERT(output_outer_l.Volume() + output_outer_r.Volume() + output_inner.Volume() == other.Volume());

        //        std::cout << *this << " | " << other << " = " << sum_contain << " | " << contain.transpose() << " = "
        //        << axis
        //                  << " | " << move_begin << std::endl;
        //        std::cout << (*this).Add(-begin) << std::endl;
        //        std::cout << other.Add(-begin) << std::endl;
        //        std::cout << output_inner.Add(-begin) << std::endl;
        //        std::cout << output_outer.Add(-begin) << std::endl;
        SAIGA_ASSERT(!output_inner.Empty());
        SAIGA_ASSERT(Contains(output_inner));
        SAIGA_ASSERT(output_outer_l.Empty() || !Intersect(output_outer_l));
        SAIGA_ASSERT(output_outer_r.Empty() || !Intersect(output_outer_r));

        return true;
    }
};

template <int N>
std::ostream& operator<<(std::ostream& strm, const iRect<N>& rect)
{
    strm << rect.begin.transpose() << " " << rect.end.transpose();
    return strm;
}
}  // namespace Saiga

namespace std
{
template <>
struct hash<Saiga::ivec3>
{
    std::size_t operator()(const Saiga::ivec3& k) const
    {
        using std::hash;
        using std::size_t;
        using std::string;

        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:

        return ((hash<int>()(k(0)) ^ (hash<int>()(k(1)) << 1)) >> 1) ^ (hash<int>()(k(2)) << 1);
    }
};

}  // namespace std

namespace Saiga
{
template <int N>
struct PointHashMap
{
    using Rect = iRect<N>;

    void Add(const Rect& r, int n)
    {
        for (int z = r.begin(2); z < r.end(2); ++z)
        {
            for (int y = r.begin(1); y < r.end(1); ++y)
            {
                for (int x = r.begin(0); x < r.end(0); ++x)
                {
                    map[ivec3(x, y, z)] += n;
                }
            }
        }
    }

    bool AllGreater(const Rect& r, int n)
    {
        for (int z = r.begin(2); z < r.end(2); ++z)
        {
            for (int y = r.begin(1); y < r.end(1); ++y)
            {
                for (int x = r.begin(0); x < r.end(0); ++x)
                {
                    if (map[ivec3(x, y, z)] <= n) return false;
                }
            }
        }
        return true;
    }

    std::unordered_map<ivec3, int> map;
};

}  // namespace Saiga
