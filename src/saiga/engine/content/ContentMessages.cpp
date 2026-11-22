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

void encodeChunkRecordState(uint32_t epoch, SpaceId space, const std::vector<ChunkRecordPart>& parts,
                            std::vector<uint8_t>& out)
{
    size_t keys = 0;
    for (const ChunkRecordPart& p : parts) keys += p.keys.size();
    SAIGA_ASSERT(!parts.empty() && parts.size() <= kMaxRecordParts && keys <= kMaxRecordKeys,
                 "ChunkRecordState: 1..1024 parts and at most 8192 keys");
    out.clear();
    BinaryWriter w(out);
    w.write(epoch);
    w.write(space);
    w.write(uint16_t(parts.size()));
    w.write(uint32_t(keys));
    for (const ChunkRecordPart& p : parts)
    {
        w.write(p.x);
        w.write(p.z);
        w.write(p.placementVersion);
        w.write(uint32_t(p.keys.size()));
        for (const InstanceKey& k : p.keys)
        {
            SAIGA_ASSERT(k.generatorSlot < 256, "ChunkRecordState: a slot above 255");
            w.write(uint8_t(k.generatorSlot));
            w.write(k.counter);
        }
    }
}

bool decodeChunkRecordState(const uint8_t* data, size_t size, ChunkRecordStateMessage& out, std::string& error)
{
    out = ChunkRecordStateMessage();
    BinaryReader r(data, size);
    uint16_t parts = 0;
    uint32_t total = 0;
    r.read(out.epoch);
    r.read(out.space);
    r.read(parts);
    r.read(total);
    if (!r.ok() || parts == 0 || parts > kMaxRecordParts || total > kMaxRecordKeys ||
        uint64_t(parts) * 14 + uint64_t(total) * 5 != r.remaining())
    {
        error = "ChunkRecordState: a malformed header, a part count not in 1..1024 or more than 8192 keys";
        return false;
    }
    out.parts.resize(parts);
    uint64_t seen = 0;
    for (ChunkRecordPart& p : out.parts)
    {
        uint32_t n = 0;
        r.read(p.x);
        r.read(p.z);
        r.read(p.placementVersion);
        r.read(n);
        if (!r.ok() || seen + n > total)
        {
            error = "ChunkRecordState: the parts' key counts exceed the key total";
            return false;
        }
        seen += n;
        p.keys.resize(n);
        for (InstanceKey& k : p.keys)
        {
            uint8_t slot = 0;
            r.read(slot);
            r.read(k.counter);
            k.generatorSlot = slot;
        }
    }
    if (!r.ok() || seen != total || r.remaining() != 0)
    {
        error = "ChunkRecordState: truncated, the key total differs or bytes left over";
        return false;
    }
    return true;
}

}  // namespace Engine
}  // namespace Saiga
