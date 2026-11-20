/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/VisitBinary.h"

#include <cstdint>
#include <string>
#include <vector>

// The save blob of World::save (docs/engine_design.md §15.1, S11): the magic "SGWS", the format version, then sections
// (a 4-character tag and a byte length). Little-endian, the binary visit form for values. An unknown section is a
// load error.
//
//   HEAD  engine version (string), steps per second, step index, time
//   SCHM  the schema table (SchemaTable::write)
//   ENTS  slot count; per slot from 1: generation (u32), state (u8: 0 free, 2 alive), flags (u8); the free list
//   COMP  per saved component type: schema index (u32), count (u64), byte length (u64), then per component the entity
//         index (u32) and the binary form, in the pool's dense order (Transform: index order)
//   EVNT  per event type with pending events: schema index, count, byte length, the events in emit order
//   ACTN  the queued actions (peer u8, sequence u64, player Entity, action u32, aim Vec3), the action sequence
//   STAT  the RNG state (state, increment), the Persistent counter

namespace Saiga
{
namespace Engine
{
constexpr uint32_t kSaveFormatVersion = 1;
constexpr uint8_t kSaveMagic[4]       = {'S', 'G', 'W', 'S'};

constexpr uint32_t saveTag(const char (&t)[5])
{
    return uint32_t(uint8_t(t[0])) | uint32_t(uint8_t(t[1])) << 8 | uint32_t(uint8_t(t[2])) << 16 |
           uint32_t(uint8_t(t[3])) << 24;
}
constexpr uint32_t kTagHeader     = saveTag("HEAD");
constexpr uint32_t kTagSchema     = saveTag("SCHM");
constexpr uint32_t kTagEntities   = saveTag("ENTS");
constexpr uint32_t kTagComponents = saveTag("COMP");
constexpr uint32_t kTagEvents     = saveTag("EVNT");
constexpr uint32_t kTagActions    = saveTag("ACTN");
constexpr uint32_t kTagState      = saveTag("STAT");

inline std::string tagName(uint32_t tag)
{
    std::string s;
    for (int i = 0; i < 4; ++i)
    {
        const char c = char((tag >> (8 * i)) & 0xFF);
        s += (c >= 32 && c < 127) ? c : '?';
    }
    return s;
}

// Overwrites a u64 written earlier at position at (a length known only afterwards)
inline void patchU64(std::vector<uint8_t>& out, size_t at, uint64_t value)
{
    for (int i = 0; i < 8; ++i) out[at + size_t(i)] = uint8_t(value >> (8 * i));
}

// Writes one section: begin() remembers the length field, end() fills it in
class SectionWriter
{
   public:
    explicit SectionWriter(std::vector<uint8_t>& out) : out(out), w(out) {}
    BinaryWriter& writer() { return w; }
    void begin(uint32_t tag)
    {
        w.write(tag);
        lengthAt = out.size();
        w.write(uint64_t(0));
    }
    void end() { patchU64(out, lengthAt, uint64_t(out.size() - lengthAt - 8)); }

   private:
    std::vector<uint8_t>& out;
    BinaryWriter w;
    size_t lengthAt = 0;
};

}  // namespace Engine
}  // namespace Saiga
