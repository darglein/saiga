/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ContentMessages.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/VisitBinary.h"

namespace Saiga
{
namespace Engine
{
void encodeTombstoneModification(uint32_t epoch, SpaceId space, const std::vector<TombstoneEntry>& entries,
                                 std::vector<uint8_t>& out)
{
    SAIGA_ASSERT(!entries.empty() && entries.size() <= kMaxTombstoneEntries, "TombstoneModification: 1..4096 entries");
    out.clear();
    BinaryWriter w(out);
    w.write(epoch);
    w.write(space);
    w.write(uint16_t(entries.size()));
    for (const TombstoneEntry& e : entries)
    {
        SAIGA_ASSERT(e.key.generatorSlot < 256, "TombstoneModification: a slot above 255");
        w.write(e.x);
        w.write(e.z);
        w.write(e.placementVersion);
        w.write(e.sequence);
        w.write(uint8_t(e.key.generatorSlot));
        w.write(e.key.counter);
    }
}

bool decodeTombstoneModification(const uint8_t* data, size_t size, TombstoneModificationMessage& out,
                                 std::string& error)
{
    out = TombstoneModificationMessage();
    BinaryReader r(data, size);
    uint16_t count = 0;
    r.read(out.epoch);
    r.read(out.space);
    r.read(count);
    if (!r.ok() || count == 0 || count > kMaxTombstoneEntries || size_t(count) * 19 != r.remaining())
    {
        error = "TombstoneModification: a malformed header or an entry count not in 1..4096 (" + std::to_string(count) +
                ")";
        return false;
    }
    out.entries.resize(count);
    for (TombstoneEntry& e : out.entries)
    {
        uint8_t slot = 0;
        r.read(e.x);
        r.read(e.z);
        r.read(e.placementVersion);
        r.read(e.sequence);
        r.read(slot);
        r.read(e.key.counter);
        e.key.generatorSlot = slot;
    }
    if (!r.ok() || r.remaining() != 0)
    {
        error = "TombstoneModification: truncated or bytes left over";
        return false;
    }
    return true;
}

}  // namespace Engine
}  // namespace Saiga
