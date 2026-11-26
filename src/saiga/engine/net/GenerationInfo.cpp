/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/GenerationInfo.h"

#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr size_t kMaxNameBytes  = 256;
constexpr size_t kMaxGenerators = 1024;
constexpr size_t kMaxSlots      = 255;
}  // namespace

GenerationInfo generationInfoOf(const World& world)
{
    const GenerationPlans& plans = world.generation().plans();
    GenerationInfo m;
    m.seed = plans.seed;
    for (const GeneratorList::Entry* e : plans.generators->byName())
    {
        GenerationInfo::Generator g;
        g.name    = e->generator->name();
        g.pinned  = plans.pinned.at(g.name);
        g.current = plans.current.at(g.name);
        for (const GeneratorSlot& s : e->slots) g.slots.push_back({s.name, s.flags});
        m.generators.push_back(std::move(g));
    }
    m.spaceCount   = uint32_t(plans.spaces.size());
    m.combinedHash = plans.combinedHash;
    return m;
}

void encodeGenerationInfo(const GenerationInfo& m, std::vector<uint8_t>& out)
{
    putU64(out, m.seed);
    putU16(out, uint16_t(m.generators.size()));
    for (const auto& g : m.generators)
    {
        putString(out, g.name);
        putU16(out, g.pinned);
        putU16(out, g.current);
        putU16(out, uint16_t(g.slots.size()));
        for (const auto& s : g.slots)
        {
            putString(out, s.name);
            putU32(out, s.flags);
        }
    }
    putU32(out, m.spaceCount);
    putU64(out, m.combinedHash);
}

bool decodeGenerationInfo(WireReader& r, GenerationInfo& out, std::string& error)
{
    out                       = GenerationInfo();
    out.seed                  = r.u64();
    const uint16_t generators = r.u16();
    if (!r.ok) return (error = "generation info: truncated", false);
    // Each generator takes at least 7 bytes, each slot 5
    if (generators > kMaxGenerators || uint64_t(generators) * 7 > r.remaining())
        return (error = "generation info: the generator count does not fit", false);
    out.generators.resize(generators);
    for (auto& g : out.generators)
    {
        g.name               = r.string(kMaxNameBytes);
        g.pinned             = r.u16();
        g.current            = r.u16();
        const uint16_t slots = r.u16();
        if (!r.ok) return (error = "generation info: truncated or a name above 256 bytes", false);
        if (slots > kMaxSlots || uint64_t(slots) * 5 > r.remaining())
            return (error = "generation info: the slot count does not fit", false);
        g.slots.resize(slots);
        for (auto& s : g.slots)
        {
            s.name  = r.string(kMaxNameBytes);
            s.flags = r.u32();
        }
    }
    out.spaceCount   = r.u32();
    out.combinedHash = r.u64();
    if (!r.ok) return (error = "generation info: truncated or a name above 256 bytes", false);
    return true;
}

}  // namespace Engine
}  // namespace Saiga
