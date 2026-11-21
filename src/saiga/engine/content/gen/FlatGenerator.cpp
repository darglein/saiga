/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/gen/FlatGenerator.h"

namespace Saiga
{
namespace Engine
{
void zeroGrassMask(const WorldPlan& plan, GrassMask& out)
{
    const int32_t spacing = plan.space.baseSpacing == 1 ? 1 : 2;
    out.size              = 256 / spacing;
    out.density.assign(uint32_t(out.size) * uint32_t(out.size), 0);
}

void FlatGenerator::grassMask(uint16_t, const WorldPlan& plan, ChunkId, GrassMask& out) const
{
    zeroGrassMask(plan, out);
}

}  // namespace Engine
}  // namespace Saiga
