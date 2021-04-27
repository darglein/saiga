/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/config.h"
#include "saiga/core/geometry/iRect.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

#include <iomanip>
#include <sstream>

namespace Saiga
{
namespace RectangularDecomposition
{
using PointView     = ArrayView<const ivec3>;
using Rect          = iRect<3>;
using RectangleList = std::vector<Rect>;


struct Cost
{
    virtual float operator()(const Rect& r) const                   = 0;
    virtual float operator()(const RectangleList& rectangles) const = 0;
};

// We define the convolution cost as a weighted sum of the expaned rectangles.
// The weights are given as
// [c, r0, r1, r2, ....]
//
// c is a costant cost
// ri is the rectangle volume expanded by i.
struct VolumeCost : public Cost
{
    VolumeCost(const std::vector<float>& weights = {0, 1}) : weights(weights) {}


    float operator()(const Rect& r) const override
    {
        if (r.Empty()) return 0;

        if (r.Volume() > max_volume) return std::numeric_limits<float>::infinity();

        float result = weights[0];
        for (int i = 1; i < (int)weights.size(); ++i)
        {
            result += r.Expand(i - 1).Volume() * weights[i];
        }
        return result;
    }
    virtual float operator()(const RectangleList& rectangles) const override
    {
        SAIGA_ASSERT(!weights.empty());
        float sum = 0;
        for (auto& r : rectangles)
        {
            sum += (*this)(r);
        }
        return sum;
    }

    std::vector<float> weights;

    // if the volume is larger than this value we set the cost to inf
    int max_volume = 1000 * 1000;
};



template <int D>
struct DiscreteBVH
{
    using Rect = iRect<D>;

    struct Node
    {
        Rect box;
        struct
        {
            uint32_t _inner : 1;
            uint32_t _left : 31;
        };
        uint32_t _right;
    };

    struct SortRectByAxis
    {
        SortRectByAxis(int a) : axis(a) {}
        bool operator()(const Rect& A, const Rect& B)
        {
            auto a = A.Center();
            auto b = B.Center();
            return a[axis] < b[axis];
        }
        int axis;
    };

    DiscreteBVH(std::vector<Rect>& elements) : leaves(elements) { construct(); }

    Rect computeBox(int start, int end)
    {
        SAIGA_ASSERT(start != end);

        Rect result = leaves[start];

        for (int i = start + 1; i < end; ++i)
        {
            result = Rect(result, leaves[i]);
        }
        return result;
    }

    void DistanceIntersect(Rect r, int distance, std::vector<int>& result, int node = 0)
    {
        SAIGA_ASSERT(node >= 0 && node < (int)nodes.size());
        Node& n = nodes[node];
        if (n.box.Distance(r) > distance) return;

        if (n._inner)
        {
            DistanceIntersect(r, distance, result, n._left);
            DistanceIntersect(r, distance, result, n._right);
        }
        else
        {
            // Leaf node -> intersect with triangles
            for (uint32_t i = n._left; i < n._right; ++i)
            {
                if (leaves[i].Distance(r) <= distance)
                {
                    result.push_back(i);
                }
            }
        }
    }



    void construct()
    {
        nodes.reserve(leaves.size());
        construct(0, leaves.size());
    }


    int construct(int start, int end)
    {
        SAIGA_ASSERT(end - start > 0);
        int nodeid = nodes.size();
        nodes.push_back({});
        auto& node = nodes.back();

        node.box = computeBox(start, end);

        if (end - start <= leaf_elements)
        {
            // leaf node
            node._inner = 0;
            node._left  = start;
            node._right = end;
        }
        else
        {
            node._inner = 1;
            int axis    = node.box.maxDimension();
            sortByAxis(start, end, axis);

            int mid = (start + end) / 2;

            int l = construct(start, mid);
            int r = construct(mid, end);

            // reload node, because the reference from above might be broken
            auto& node2  = nodes[nodeid];
            node2._left  = l;
            node2._right = r;
        }

        return nodeid;
    }

    void sortByAxis(int start, int end, int axis)
    {
        std::sort(leaves.begin() + start, leaves.begin() + end, SortRectByAxis(axis));
    }
    int leaf_elements = 5;
    std::vector<Rect>& leaves;
    std::vector<Node> nodes;
};

inline int Volume(const RectangleList& rectangles, int radius = 0)
{
    int v = 0;
    for (auto r : rectangles)
    {
        v += r.Expand(radius).Volume();
    }
    return v;
}

inline std::string to_string(const RectangleList& rectangles)
{
    std::stringstream strm;
    strm << "[RectangleList] N = " << std::setw(6) << rectangles.size();
    strm << " V0 = " << std::setw(6) << Volume(rectangles);
    strm << " V1 = " << std::setw(6) << Volume(rectangles, 1);
    strm << " V2 = " << std::setw(6) << Volume(rectangles, 2);

    //    strm << "  V0 = " << std::setw(6) << decomp.Volume() << "  V1 = " << std::setw(6) << decomp.ExpandedVolume(1)
    //         << "  V2 = " << std::setw(6) << decomp.ExpandedVolume(2);
    return strm.str();
}

}  // namespace RectangularDecomposition
}  // namespace Saiga
