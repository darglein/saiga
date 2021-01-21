/**
 * Copyright (c) 2017 Darius Rückert
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

// Removes all rectangles with volume == 0
void RemoveEmpty(RectangleList& rectangles);


void MergeNeighborsSave(RectangleList& rectangles);

void MergeNeighbors(RectangleList& rectangles, const Cost& cost);

std::vector<std::pair<int, int>> NeighborList(RectangleList& rectangles, int distance);

std::vector<int> AllIntersectingRects(const RectangleList& rectangles, const Rect& r);

}  // namespace RectangularDecomposition
}  // namespace Saiga
