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
};

SAIGA_CORE_API std::ostream& operator<<(std::ostream& strm, const Decomposition& decomp);


class SAIGA_CORE_API RectangularDecompositionAlgorithm
{
   public:
    virtual ~RectangularDecompositionAlgorithm() {}
    virtual Decomposition Compute(ArrayView<const ivec3> points) = 0;
};

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

class SAIGA_CORE_API GrowAndShrinkDecomposition : public RectangularDecompositionAlgorithm
{
   public:
    // Combines all neighboring elements in x-direction with the same (y,z) coordinate.
    virtual Decomposition Compute(ArrayView<const ivec3> points) override;


    int its = 5000;

   private:
    int cost_radius = 0;
    int N           = 10;

    int current_best          = 94755745;
    int not_improved_in_a_row = 0;

    // <decomp, cost>
    std::vector<std::pair<Decomposition, int>> decomps;

    void RandomStepGrow(Decomposition& decomp);
    void RandomStepMerge(Decomposition& decomp);
    int Cost(const Decomposition& decomp)
    {
        return decomp.ExpandedVolume(cost_radius) * 100 + decomp.rectangles.size();
    }

    std::pair<Decomposition, int>& Best()
    {
        auto min_ele =
            std::min_element(decomps.begin(), decomps.end(), [](auto& a, auto& b) { return a.second < b.second; });
        return *min_ele;
    }
};

}  // namespace RectangularDecomposition
}  // namespace Saiga
