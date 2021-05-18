/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "RectangularDecomposition.h"
#include "RectilinearCover.h"

namespace Saiga
{
namespace RectangularDecomposition
{
using RectangleList = std::vector<Rect>;

SAIGA_CORE_API void MergeNeighborsSave(RectangleList& rectangles);

// returns the number of iterations
SAIGA_CORE_API int MergeNeighbors(RectangleList& rectangles, const Cost& cost, int max_iterations);

SAIGA_CORE_API void MergeShrink(PointView points, RectangleList& rectangles, int its, int converge_its,
                                const Cost& cost);


// Removes all rectangles with volume == 0
SAIGA_CORE_API void RemoveEmpty(RectangleList& rectangles);

SAIGA_CORE_API void ShrinkIfPossible(RectangleList& rectangles);
SAIGA_CORE_API void ShrinkIfPossible2(RectangleList& rectangles, PointView points);

std::vector<std::pair<int, int>> NeighborList(RectangleList& rectangles, int distance);

std::vector<int> AllIntersectingRects(const RectangleList& rectangles, const Rect& r);

}  // namespace RectangularDecomposition
}  // namespace Saiga
