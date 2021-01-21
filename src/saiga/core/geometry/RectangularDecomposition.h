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

#include "RectilinearCover.h"
#include "aabb.h"
#include "intersection.h"
#include "ray.h"
#include "triangle.h"

namespace Saiga
{
namespace RectangularDecomposition
{
SAIGA_CORE_API std::vector<ivec3> RemoveDuplicates(ArrayView<const ivec3> points);

struct SAIGA_CORE_API Decomposition
{
    std::vector<Rect> rectangles;
    bool ContainsAll(ArrayView<const ivec3> points) const;

    // Remove all unnecessary rectangles.
    // This are rectangles which are duplicates or fully contained in other rectangles.
    Decomposition RemoveFullyContained() const;


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


    VolumeCost cost;
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
    int not_improved_in_a_row = 0;

    // <decomp, cost>
    Decomposition current_decomp;
    float current_cost;

    void RandomStepGrow(RectangleList& decomp);
    void RandomStepMerge(RectangleList& decomp);
};

}  // namespace RectangularDecomposition
}  // namespace Saiga
