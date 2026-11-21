/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// parseSave: the sections of a save blob (SaveFormat.h), checked for structure.

#include "saiga/engine/save/SaveReader.h"

#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"

#include <cmath>
#include <cstring>
#include <tuple>

namespace Saiga
{
namespace Engine
{
namespace
{
bool readBlocks(BinaryReader& r, size_t base, const SchemaTable& schema, SchemaKind kind,
                std::vector<ParsedSave::Block>& out, std::string& error)
{
    uint32_t n = 0;
    r.read(n);
    if (r.ok() && n > schema.types.size())
    {
        error = "more value blocks than schema types";
        return false;
    }
    std::vector<bool> seen(schema.types.size(), false);
    for (uint32_t i = 0; i < n && r.ok(); ++i)
    {
        ParsedSave::Block b;
        uint64_t length = 0;
        r.read(b.schemaIndex);
        r.read(b.count);
        r.read(length);
        if (!r.ok()) break;
        if (b.schemaIndex >= schema.types.size() || schema.types[b.schemaIndex].kind != kind)
        {
            error = "a value block names schema entry " + std::to_string(b.schemaIndex) + ", which is not a " +
                    (kind == SchemaKind::Component ? "component" : "event");
            return false;
        }
        if (seen[b.schemaIndex])
        {
            error = "two value blocks of " + schema.types[b.schemaIndex].name;
            return false;
        }
        seen[b.schemaIndex] = true;
        b.begin             = base + r.position();
        if (!r.skip(length)) break;
        b.end = base + r.position();
        // Every value takes at least one byte (components at least their 4-byte index)
        if (b.count > uint64_t(b.end - b.begin) && (kind == SchemaKind::Component || b.count > kMaxEmptyElements))
        {
            error = "a value block of " + schema.types[b.schemaIndex].name + " has more values than bytes";
            return false;
        }
        out.push_back(b);
    }
    if (!r.ok())
    {
        if (error.empty()) error = r.error();
        return false;
    }
    if (!r.atEnd())
    {
        error = "bytes left over after the value blocks";
        return false;
    }
    return true;
}
}  // namespace

bool parseTerrainSection(const uint8_t* data, size_t size, uint32_t version, TerrainSection& out, std::string& error)
{
    out = TerrainSection();
    if (version != kTerrainSectionVersion)
    {
        error = "unknown version " + std::to_string(version);
        return false;
    }
    BinaryReader r(data, size);
    auto fail = [&](const std::string& m)
    {
        error = m;
        return false;
    };
    uint32_t spaceCount = 0;
    r.read(spaceCount);
    // Each space takes at least 9 bytes
    if (!r.ok() || spaceCount == 0 || spaceCount > 65536 || uint64_t(spaceCount) * 9 > r.remaining())
        return fail("the space count " + std::to_string(spaceCount) + " does not fit into the section");
    out.spaces.resize(spaceCount);
    std::vector<int32_t> scratch;
    for (uint32_t si = 0; si < spaceCount; ++si)
    {
        TerrainSection::Space& sp = out.spaces[si];
        r.read(sp.space);
        r.read(sp.spacing);
        r.read(sp.N);
        uint32_t chunks = 0;
        r.read(chunks);
        if (!r.ok()) return fail(r.error());
        const std::string where = "space " + std::to_string(sp.space);
        if (si > 0 && sp.space <= out.spaces[si - 1].space) return fail(where + ": the spaces are not ascending");
        if (sp.spacing != 1 && sp.spacing != 2)
            return fail(where + ": the base spacing " + std::to_string(sp.spacing) + " is not 1 or 2");
        if (sp.N != 256 / sp.spacing + 1)
            return fail(where + ": N " + std::to_string(sp.N) + " does not match the base spacing");
        // Each chunk takes at least 13 bytes
        if (chunks == 0 || uint64_t(chunks) * 13 > r.remaining())
            return fail(where + ": the chunk count " + std::to_string(chunks) + " does not fit into the section");
        sp.chunks.resize(chunks);
        const size_t count = size_t(sp.N) * sp.N;
        for (uint32_t ci = 0; ci < chunks; ++ci)
        {
            TerrainSection::Chunk& c = sp.chunks[ci];
            uint8_t flags            = 0;
            r.read(c.x);
            r.read(c.z);
            r.read(c.sequence);
            r.read(flags);
            if (!r.ok()) return fail(r.error());
            const std::string at = where + " chunk (" + std::to_string(c.x) + ", " + std::to_string(c.z) + ")";
            if (ci > 0 && std::tie(c.x, c.z) <= std::tie(sp.chunks[ci - 1].x, sp.chunks[ci - 1].z))
                return fail(at + ": the chunks are not in ascending ChunkId order");
            if (c.sequence == 0) return fail(at + ": sequence 0");
            if (flags & ~uint8_t(1)) return fail(at + ": unknown flags " + std::to_string(flags));
            c.hasDeltas = (flags & 1) != 0;
            if (!c.hasDeltas) continue;
            uint64_t length = 0;
            r.read(length);
            if (!r.ok()) return fail(r.error());
            if (length > r.remaining()) return fail(at + ": the encoded deltas are longer than the section");
            c.deltas.resize(count);
            std::string e;
            if (!TerrainDeltaCodec::decode(data + r.position(), size_t(length), true, count, c.deltas.data(), &e))
                return fail(at + ": " + e);
            r.skip(length);
            bool any = false;
            for (int32_t v : c.deltas) any = any || v != 0;
            if (!any) return fail(at + ": deltas present but all zero");
        }
    }
    if (!r.atEnd()) return fail("bytes left over");
    return true;
}

bool parseSpaceTable(BinaryReader& r, uint32_t version, SpaceTable& out, std::string& error)
{
    out = SpaceTable();
    if (version != kSpaceTableVersion)
    {
        error = "unknown version " + std::to_string(version);
        return false;
    }
    uint32_t n = 0;
    r.read(n);
    if (!r.ok()) return false;
    // Each row takes at least 5 bytes (an empty name's length and the kind)
    if (n == 0 || n > 65535 || uint64_t(n) * 5 > r.remaining())
    {
        error = "the space count " + std::to_string(n) + " is not in 1..65535 or does not fit into the section";
        return false;
    }
    out.rows.resize(n);
    for (uint32_t i = 0; i < n && r.ok(); ++i)
    {
        uint32_t length = 0;
        r.read(length);
        if (!r.ok()) break;
        if (length > 256 || length > r.remaining())
        {
            error = "space " + std::to_string(i) + ": a name of " + std::to_string(length) + " bytes";
            return false;
        }
        std::string& name = out.rows[i].name;
        name.resize(length);
        for (uint32_t k = 0; k < length; ++k)
        {
            uint8_t c = 0;
            r.read(c);
            name[k] = char(c);
        }
        uint8_t kind = 0;
        r.read(kind);
        if (r.ok() && kind > uint8_t(SpaceKind::Enclosed))
        {
            error = "space " + std::to_string(i) + ": the unknown kind " + std::to_string(kind);
            return false;
        }
        out.rows[i].kind = SpaceKind(kind);
    }
    if (!r.ok()) return false;
    if (!r.atEnd())
    {
        error = "bytes left over";
        return false;
    }
    return true;
}

bool parseSave(const std::vector<uint8_t>& data, ParsedSave& out, std::string& error)
{
    out = ParsedSave();
    if (data.size() < 8 || std::memcmp(data.data(), kSaveMagic, 4) != 0)
    {
        error = "not a world save (no SGWS header)";
        return false;
    }
    BinaryReader top(data.data() + 4, data.size() - 4);
    uint32_t format = 0;
    top.read(format);
    if (format != kSaveFormatVersion)
    {
        error = "save format version " + std::to_string(format) + " (this build reads version " +
                std::to_string(kSaveFormatVersion) + ")";
        return false;
    }
    // The order table (D7): an optional entry whose tag is not the next section is absent
    constexpr size_t kEntries = sizeof(kSaveSections) / sizeof(kSaveSections[0]);
    size_t next               = 0;
    while (next < kEntries)
    {
        // The blob may end only when every remaining entry is optional
        bool requiredLeft = false;
        for (size_t i = next; i < kEntries; ++i) requiredLeft = requiredLeft || !kSaveSections[i].optional;
        if (top.atEnd() && !requiredLeft) break;
        uint32_t tag    = 0;
        uint64_t length = 0;
        top.read(tag);
        top.read(length);
        if (!top.ok())
        {
            error = "truncated save: " + top.error();
            return false;
        }
        // The candidates: the next entry and the optional ones after it, up to the first required one
        size_t found = kEntries;
        std::string expected;
        for (size_t i = next; i < kEntries; ++i)
        {
            expected += (expected.empty() ? "" : " or ") + tagName(kSaveSections[i].tag);
            if (kSaveSections[i].tag == tag)
            {
                found = i;
                break;
            }
            if (!kSaveSections[i].optional) break;
        }
        if (found == kEntries)
        {
            error = "unexpected section " + tagName(tag) + " (expected " + expected + ")";
            return false;
        }
        next                         = found + 1;
        const SaveSectionEntry entry = kSaveSections[found];
        if (length > top.remaining())
        {
            error = "section " + tagName(tag) + " is longer than the save";
            return false;
        }
        const size_t base = 4 + top.position();
        BinaryReader r(data.data() + base, size_t(length));
        top.skip(length);
        std::string sectionError;
        bool ok          = true;
        uint32_t version = 0;
        if (entry.optional)
        {
            r.read(version);
            if (r.ok() && (version == 0 || version > entry.maxVersion))
            {
                error = tagName(tag) + ": section format version " + std::to_string(version) + " (this build reads " +
                        std::to_string(entry.maxVersion) + ")";
                return false;
            }
        }
        if (entry.optional)
        {
            // One case per optional entry: its parse function or its raw range (the recipe of the S21 plan)
            switch (tag)
            {
                case kTagSpaces:
                    out.spaces.emplace();
                    ok = r.ok() && parseSpaceTable(r, version, *out.spaces, sectionError);
                    break;
                case kTagTerrain:
                    // A raw range, decoded by the loader's own pass (parseTerrainSection) before the commit
                    {
                        // (Assigned, not emplaced: Clang treats the nested aggregate with member initializers as not
                        // default-constructible inside std::optional)
                        ParsedSave::Range range;
                        range.version = version;
                        range.begin   = base + r.position();
                        range.end     = base + size_t(length);
                        out.terrain   = range;
                        ok            = r.ok() && r.skip(r.remaining());
                        break;
                    }
                default:
                    SAIGA_ASSERT(false, "parseSave: an optional section without a case");
                    break;
            }
        }
        else if (tag == kTagHeader)
        {
            r.read(out.engineVersion);
            r.read(out.stepsPerSecond);
            r.read(out.steps);
            r.read(out.time);
            if (r.ok() && (!std::isfinite(out.time) || !std::isfinite(out.stepsPerSecond) || out.stepsPerSecond <= 0))
            {
                error = "HEAD: the time and the step rate must be finite (and the step rate positive)";
                return false;
            }
        }
        else if (tag == kTagSchema)
            ok = out.schema.read(r, sectionError);
        else if (tag == kTagEntities)
        {
            uint32_t slots = 0;
            r.read(slots);
            // At most 2^24 slots (a few bytes per free slot in the save become a Transform each in memory)
            if (r.ok() && (slots == 0 || slots > (1u << 24) || uint64_t(slots - 1) * 6 > r.remaining()))
            {
                error = "ENTS: the slot table does not fit into the section";
                return false;
            }
            out.slots.resize(r.ok() ? slots : 0);
            for (uint32_t i = 1; i < slots && r.ok(); ++i)
            {
                ParsedSave::Slot& s = out.slots[i];
                r.read(s.generation);
                r.read(s.state);
                r.read(s.flags);
            }
            r.read(out.freeList);
        }
        else if (tag == kTagComponents)
            ok = readBlocks(r, base, out.schema, SchemaKind::Component, out.components, sectionError);
        else if (tag == kTagEvents)
            ok = readBlocks(r, base, out.schema, SchemaKind::Event, out.events, sectionError);
        else if (tag == kTagActions)
        {
            uint64_t n = 0;
            r.read(n);
            // Each action takes kActionBytes: peer (1), sequence (8), player (8), action (4), aim (24)
            constexpr uint64_t kActionBytes = 45;
            if (r.ok() && n > r.remaining() / kActionBytes)
            {
                error = "ACTN: more actions than bytes";
                return false;
            }
            for (uint64_t i = 0; i < n && r.ok(); ++i)
            {
                ParsedSave::Action a;
                r.read(a.peer);
                r.read(a.sequence);
                r.read(a.player);
                r.read(a.action);
                r.read(a.aim);
                out.actions.push_back(a);
            }
            r.read(out.actionSequence);
        }
        else if (tag == kTagState)
        {
            r.read(out.rngState);
            r.read(out.rngIncrement);
            r.read(out.persistentCounter);
            if (r.ok() && (out.rngIncrement & 1) == 0)
            {
                error = "STAT: the RNG increment must be odd";
                return false;
            }
        }
        if (!ok || !r.ok())
        {
            error = tagName(tag) + ": " + (sectionError.empty() ? r.error() : sectionError);
            return false;
        }
        if (!r.atEnd() && tag != kTagComponents && tag != kTagEvents && !entry.optional)
        {
            error = tagName(tag) + ": bytes left over";
            return false;
        }
    }
    if (!top.atEnd())
    {
        uint32_t tag = 0;
        top.read(tag);
        error = "unknown section " + tagName(tag) + " after the last one";
        return false;
    }
    return true;
}

}  // namespace Engine
}  // namespace Saiga
