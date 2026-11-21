/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ContentHash.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/core/Hash.h"

namespace Saiga
{
namespace Engine
{
namespace
{
void addView(Hasher& h, const GenerationTemplate& t)
{
    h.addString(t.name).addU8(uint8_t(t.kind)).addU32(t.templateIndex).addI32(t.baseSpacing).addI32(t.sizeChunks);
    h.addU8(t.seaLevel ? 1 : 0).addI32(t.seaLevel ? *t.seaLevel : 0);
}

void addGrid(Hasher& h, const PlanGrid& g)
{
    h.addI32(g.cellSize).addI32(g.originX).addI32(g.originZ).addI32(g.width).addI32(g.height);
    h.addU32(uint32_t(g.values.size()));
    for (int32_t v : g.values) h.addI32(v);
}
}  // namespace

uint64_t planHash(const WorldPlan& plan, uint16_t pinnedVersion, const std::vector<uint64_t>& prefabHashes)
{
    SAIGA_ASSERT(prefabHashes.size() == plan.sites.size());
    Hasher h(fnv1a64("saiga.content.plan.v1"));
    h.addU64(plan.seed).addU32(pinnedVersion);
    addView(h, plan.space);
    h.addU32(uint32_t(plan.sites.size()));
    for (size_t i = 0; i < plan.sites.size(); ++i)
    {
        const Site& s = plan.sites[i];
        h.addU32(s.key).addString(s.prefab).addU32(s.prefabVersion).addU64(prefabHashes[i]);
        h.addI64(s.x).addI64(s.z).addU8(s.quarterTurns).addI32(s.baseHeight);
        for (int k = 0; k < 4; ++k) h.addI32(s.footprint[k]);
    }
    h.addU32(uint32_t(plan.spaces.size()));
    for (const SpaceInstance& s : plan.spaces)
        h.addU32(s.templateIndex).addU64(s.seed).addU32(s.parent).addU32(s.siteKey).addU8(s.link).addU8(s.origin);
    addGrid(h, plan.biomes);
    addGrid(h, plan.rivers);
    return h.value();
}

uint64_t templateViewHash(const GenerationTemplate& view)
{
    Hasher h(fnv1a64("saiga.content.template.v1"));
    addView(h, view);
    return h.value();
}

uint64_t combinedPlanHash(const std::vector<CombinedPlanEntry>& spaces)
{
    Hasher h(fnv1a64("saiga.content.plans.v1"));
    h.addU32(uint32_t(spaces.size()));
    for (const CombinedPlanEntry& s : spaces) h.addU32(s.templateIndex).addString(s.generator).addU64(s.planHash);
    return h.value();
}

}  // namespace Engine
}  // namespace Saiga
