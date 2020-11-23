/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/iRect.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

#include "aabb.h"
#include "intersection.h"
#include "ray.h"
#include "triangle.h"

namespace Saiga
{
namespace RectangularDecomposition
{
using Rect = iRect<3>;


SAIGA_CORE_API std::vector<ivec3> RemoveDuplicates(ArrayView<const ivec3> points);

struct SAIGA_CORE_API Decomposition
{
    std::vector<Rect> rectangles;
    bool ContainsAll(ArrayView<const ivec3> points) const;

    // Remove all unnecessary rectangles.
    // This are rectangles which are duplicates or fully contained in other rectangles.
    Decomposition RemoveFullyContained() const;

    Decomposition ShrinkIfPossible() const;


    Decomposition MergeNeighborsSave() const;

    std::vector<std::pair<int, int>> NeighborList(int distance) const;

    std::vector<int> AllIntersectingRects(const Rect& r)
    {
        std::vector<int> result;
        //        for (auto r2 : rectangles)
        for (int i = 0; i < rectangles.size(); ++i)
        {
            if (r.Intersect(rectangles[i]))
            {
                result.push_back(i);
            }
        }
        return result;
    }

    void RemoveEmpty()
    {
        rectangles.erase(std::remove_if(rectangles.begin(), rectangles.end(), [](auto& a) { return a.Empty(); }),
                         rectangles.end());
    }

    int Volume() const
    {
        int sum = 0;
        for (auto& r : rectangles) sum += r.Volume();
        return sum;
    }

    int ExpandedVolume(int radius) const
    {
        int sum = 0;
        for (auto& r : rectangles) sum += r.Expand(radius).Volume();
        return sum;
    }
};  // namespace RectangularDecomposition

SAIGA_CORE_API std::ostream& operator<<(std::ostream& strm, const Decomposition& decomp);


class SAIGA_CORE_API RectangularDecompositionAlgorithm
{
   public:
    virtual ~RectangularDecompositionAlgorithm() {}
    virtual Decomposition Compute(ArrayView<const ivec3> points) = 0;

    virtual Decomposition Optimize(const Decomposition& decomp) { return decomp; }


    // We define the convolution cost as a weighted sum of the expaned rectangles.
    // The weights are given as
    // [c, r0, r1, r2, ....]
    //
    // c is a costant cost
    // ri is the rectangle volume expanded by i.
    float ConvolutionCost(const Rect& rect)
    {
        SAIGA_ASSERT(!conv_cost_weights.empty());
        if (rect.Empty()) return 0;


        float result = conv_cost_weights[0];
        for (int i = 1; i < conv_cost_weights.size(); ++i)
        {
            result += rect.Expand(i - 1).Volume() * conv_cost_weights[i];
        }

        return result;
    }

    float ConvolutionCost(const Decomposition& decomp)
    {
        float sum = 0;
        for (auto& r : decomp.rectangles) sum += ConvolutionCost(r);
        return sum;
    }
    std::vector<float> conv_cost_weights = {0, 1};
};

// =========================================
// Actual decompositions which generate a list of rectangles from a point cloud
// =========================================

class SAIGA_CORE_API TrivialRectangularDecomposition : public RectangularDecompositionAlgorithm
{
   public:
    // The trivial decomposition converts each element into a 1x1 rectangle.
    virtual Decomposition Compute(ArrayView<const ivec3> points) override;
};

class SAIGA_CORE_API RowMergeDecomposition : public RectangularDecompositionAlgorithm
{
   public:
    // Combines all neighboring elements in x-direction with the same (y,z) coordinate.
    virtual Decomposition Compute(ArrayView<const ivec3> points) override;
};

class SAIGA_CORE_API OctTreeDecomposition : public RectangularDecompositionAlgorithm
{
   public:
    // Combines all neighboring elements in x-direction with the same (y,z) coordinate.
    virtual Decomposition Compute(ArrayView<const ivec3> points) override;

    float merge_factor = 1.0;
};

// =========================================
// These 'decompositions' are actually only optimizing an existing decomposition
// therefore they also implement the 'optimize' virtual function
// =========================================

class SAIGA_CORE_API SaveMergeDecomposition : public RectangularDecompositionAlgorithm
{
   public:
    // Combines all neighboring elements in x-direction with the same (y,z) coordinate.
    virtual Decomposition Compute(ArrayView<const ivec3> points) override;
    virtual Decomposition Optimize(const Decomposition& decomp) override;
};



class SAIGA_CORE_API GrowAndShrinkDecomposition : public RectangularDecompositionAlgorithm
{
   public:
    // Combines all neighboring elements in x-direction with the same (y,z) coordinate.
    virtual Decomposition Compute(ArrayView<const ivec3> points) override;


    int its          = 1000;
    int converge_its = 100;

   private:
    double current_best       = 1e50;
    int not_improved_in_a_row = 0;

    // <decomp, cost>
    std::vector<std::pair<Decomposition, double>> decomps;

    void RandomStepGrow(Decomposition& decomp);
    void RandomStepMerge(Decomposition& decomp);

    std::pair<Decomposition, double>& Best()
    {
        auto min_ele =
            std::min_element(decomps.begin(), decomps.end(), [](auto& a, auto& b) { return a.second < b.second; });
        return *min_ele;
    }
};

}  // namespace RectangularDecomposition
}  // namespace Saiga
