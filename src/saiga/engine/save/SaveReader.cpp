/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// parseSave: the sections of a save blob (SaveFormat.h), checked for structure.

#include "saiga/engine/save/SaveReader.h"

#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/save/SaveFormat.h"

#include <cstring>

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
    const uint32_t order[] = {kTagHeader, kTagSchema, kTagEntities, kTagComponents, kTagEvents, kTagActions, kTagState};
    for (uint32_t expected : order)
    {
        uint32_t tag    = 0;
        uint64_t length = 0;
        top.read(tag);
        top.read(length);
        if (!top.ok())
        {
            error = "truncated save: " + top.error();
            return false;
        }
        if (tag != expected)
        {
            error = "unexpected section " + tagName(tag) + " (expected " + tagName(expected) + ")";
            return false;
        }
        if (length > top.remaining())
        {
            error = "section " + tagName(tag) + " is longer than the save";
            return false;
        }
        const size_t base = 4 + top.position();
        BinaryReader r(data.data() + base, size_t(length));
        top.skip(length);
        std::string sectionError;
        bool ok = true;
        if (tag == kTagHeader)
        {
            r.read(out.engineVersion);
            r.read(out.stepsPerSecond);
            r.read(out.steps);
            r.read(out.time);
        }
        else if (tag == kTagSchema)
            ok = out.schema.read(r, sectionError);
        else if (tag == kTagEntities)
        {
            uint32_t slots = 0;
            r.read(slots);
            if (r.ok() && (slots == 0 || uint64_t(slots - 1) * 6 > r.remaining()))
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
            // Each action takes 49 bytes
            if (r.ok() && n > r.remaining() / 49)
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
        if (!r.atEnd() && tag != kTagComponents && tag != kTagEvents)
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
