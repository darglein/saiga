/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/GenerationStateMessage.h"

#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
GenerationStateMessage generationStateOf(const World& world, uint32_t epoch)
{
    const GenerationPlans& plans = world.generation().plans();
    GenerationStateMessage m;
    m.epoch = epoch;
    m.seed  = plans.seed;
    for (const GeneratorList::Entry* e : plans.generators->byName())
    {
        GenerationStateMessage::Generator g;
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

void encodeGenerationState(const GenerationStateMessage& m, std::vector<uint8_t>& out)
{
    out.clear();
    BinaryWriter w(out);
    w.write(m.epoch);
    w.write(m.seed);
    w.write(uint16_t(m.generators.size()));
    for (const auto& g : m.generators)
    {
        w.write(g.name);
        w.write(g.pinned);
        w.write(g.current);
        w.write(uint16_t(g.slots.size()));
        for (const auto& s : g.slots)
        {
            w.write(s.name);
            w.write(s.flags);
        }
    }
    w.write(m.spaceCount);
    w.write(m.combinedHash);
}

bool decodeGenerationState(const uint8_t* data, size_t size, GenerationStateMessage& out, std::string& error)
{
    out = GenerationStateMessage();
    BinaryReader r(data, size);
    r.read(out.epoch);
    r.read(out.seed);
    uint16_t generators = 0;
    r.read(generators);
    // Each generator takes at least 14 bytes, each slot 12
    if (r.ok() && uint64_t(generators) * 14 > r.remaining())
    {
        error = "GenerationState: the generator count does not fit";
        return false;
    }
    out.generators.resize(r.ok() ? generators : 0);
    for (auto& g : out.generators)
    {
        r.read(g.name);
        r.read(g.pinned);
        r.read(g.current);
        uint16_t slots = 0;
        r.read(slots);
        if (!r.ok()) break;
        if (uint64_t(slots) * 12 > r.remaining())
        {
            error = "GenerationState: the slot count does not fit";
            return false;
        }
        g.slots.resize(slots);
        for (auto& s : g.slots)
        {
            r.read(s.name);
            r.read(s.flags);
        }
    }
    r.read(out.spaceCount);
    r.read(out.combinedHash);
    if (!r.ok())
    {
        error = "GenerationState: truncated (" + r.error() + ")";
        return false;
    }
    if (r.remaining() != 0)
    {
        error = "GenerationState: bytes left over";
        return false;
    }
    return true;
}

std::string generationStateDifference(const World& client, const GenerationStateMessage& host)
{
    const GenerationStateMessage self = generationStateOf(client, host.epoch);
    if (host.seed != self.seed)
        return "host seed " + std::to_string(host.seed) + ", this client's setup seed " + std::to_string(self.seed);
    auto names = [](const GenerationStateMessage& m)
    {
        std::string s;
        for (const auto& g : m.generators) s += (s.empty() ? "" : ", ") + g.name;
        return s;
    };
    if (host.generators.size() != self.generators.size())
        return "host generators [" + names(host) + "], this client's [" + names(self) + "]";
    for (size_t i = 0; i < host.generators.size(); ++i)
    {
        const auto& h = host.generators[i];
        const auto& c = self.generators[i];
        if (h.name != c.name) return "host generators [" + names(host) + "], this client's [" + names(self) + "]";
        if (h.pinned != c.pinned)
            return "host uses generator " + h.name + " pinned v" + std::to_string(h.pinned) + ", this client pins v" +
                   std::to_string(c.pinned);
        if (h.current != c.current)
            return "host places generator " + h.name + " at v" + std::to_string(h.current) +
                   ", this client's newest version is v" + std::to_string(c.current);
        bool sameSlots = h.slots.size() == c.slots.size();
        for (size_t k = 0; sameSlots && k < h.slots.size(); ++k)
            sameSlots = h.slots[k].name == c.slots[k].name && h.slots[k].flags == c.slots[k].flags;
        if (!sameSlots) return "generator " + h.name + ": the host's slot table differs from this client's";
    }
    if (host.spaceCount != self.spaceCount)
        return "host has " + std::to_string(host.spaceCount) + " spaces, this client " +
               std::to_string(self.spaceCount);
    if (host.combinedHash != self.combinedHash)
        return "the host's plans differ from this client's (combined plan hash)";
    return {};
}

}  // namespace Engine
}  // namespace Saiga
