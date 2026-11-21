/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/gen/SectionHash.h"

#include "saiga/engine/content/ChunkContent.h"
#include "saiga/engine/core/Hash.h"

namespace Saiga
{
namespace Engine
{
uint64_t hashHeights(const std::vector<int32_t>* samples, int32_t samplesPerSide, int32_t baseSpacing)
{
    Hasher h(fnv1a64("saiga.content.heights.v1"));
    if (!samples) return h.addU32(0).addU32(0).value();
    h.addU32(uint32_t(samplesPerSide)).addU32(uint32_t(baseSpacing));
    for (int32_t v : *samples) h.addI32(v);
    return h.value();
}

uint64_t hashTierOne(const std::vector<TierOneInstance>& instances)
{
    Hasher h(fnv1a64("saiga.content.tier1.v1"));
    h.addU32(uint32_t(instances.size()));
    for (const TierOneInstance& t : instances)
    {
        h.addU32(t.key.generatorSlot).addU32(t.key.counter).addU32(t.meshSlot);
        h.addI32(t.x).addI32(t.z).addI32(t.y);
        h.addU32(t.yaw).addU32(t.scale).addU32(t.flags);
    }
    return h.value();
}

uint64_t hashGrass(const GrassMask& mask)
{
    Hasher h(fnv1a64("saiga.content.grass.v1"));
    h.addU32(uint32_t(mask.size));
    for (uint8_t v : mask.density) h.addU8(v);
    return h.value();
}

uint64_t hashStamps(const std::vector<StampRecord>& records)
{
    Hasher h(fnv1a64("saiga.content.stamps.v1"));
    h.addU32(uint32_t(records.size()));
    for (const StampRecord& r : records)
    {
        h.addU32(r.siteKey).addU32(r.entity).addString(r.prefab).addU32(r.prefabVersion).addString(r.className);
        h.addI64(r.x).addI64(r.z).addI32(r.y).addU32(r.yaw);
    }
    return h.value();
}

uint64_t hashStatics(const std::vector<StaticPart>& parts)
{
    Hasher h(fnv1a64("saiga.content.statics.v1"));
    h.addU32(uint32_t(parts.size()));
    for (const StaticPart& p : parts)
    {
        h.addU32(p.siteKey).addU32(p.part).addString(p.mesh);
        h.addI64(p.x).addI64(p.z).addI32(p.y).addU32(p.yaw);
    }
    return h.value();
}

uint64_t hashTotal(const ChunkId& chunk, uint16_t pinnedVersion, uint16_t placementVersion,
                   const ContentHashes& sections)
{
    Hasher h(fnv1a64("saiga.content.total.v1"));
    h.addU32(chunk.space).addI32(chunk.x).addI32(chunk.z).addU32(pinnedVersion).addU32(placementVersion);
    h.addU64(sections.heights).addU64(sections.tier1).addU64(sections.grass).addU64(sections.stamps);
    h.addU64(sections.statics);
    return h.value();
}

}  // namespace Engine
}  // namespace Saiga
