/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/gen/AuthoredSpaceGenerator.h"

#include "saiga/engine/content/gen/FlatGenerator.h"

#include <utility>

namespace Saiga
{
namespace Engine
{
AuthoredSpaceGenerator::AuthoredSpaceGenerator(std::string name, std::string prefab, uint16_t prefabVersion)
    : generatorName(std::move(name)), prefab(std::move(prefab)), prefabVersion(prefabVersion)
{
}

void AuthoredSpaceGenerator::plan(uint16_t, WorldPlan& plan) const
{
    // S21's site origin (E4): the centre of the middle chunk
    const int64_t size   = int64_t(plan.space.sizeChunks) * 256;
    const int64_t origin = int64_t(plan.space.sizeChunks / 2) * 256 + 128;
    Site s;
    s.key           = 0;
    s.prefab        = prefab;
    s.prefabVersion = prefabVersion;
    s.x             = origin;
    s.z             = origin;
    s.baseHeight    = 0;
    // The whole space
    s.footprint[0] = int32_t(-origin);
    s.footprint[1] = int32_t(-origin);
    s.footprint[2] = int32_t(size - origin);
    s.footprint[3] = int32_t(size - origin);
    plan.sites.push_back(s);
}

void AuthoredSpaceGenerator::grassMask(uint16_t, const WorldPlan& plan, ChunkId, GrassMask& out) const
{
    zeroGrassMask(plan, out);
}

}  // namespace Engine
}  // namespace Saiga
