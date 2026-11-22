/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainEditBlock (docs/engine_design.md §12.1; S22): 16 x 16 terrain deltas (edited height minus the generated base,
// in height units) of one space, keyed in TerrainEditStore by (k >> 4) << 32 | (i >> 4) of the space samples (i, k).
// Chunk borders fall on block borders (S = 256 / spacing is a multiple of 16), and every sample has exactly one copy.

#pragma once

#include <cstdint>

namespace Saiga
{
namespace Engine
{
struct TerrainEditBlock
{
    // Row-major: index (k & 15) * 16 + (i & 15); an unallocated block reads 0
    int32_t delta[256] = {};
};

}  // namespace Engine
}  // namespace Saiga
