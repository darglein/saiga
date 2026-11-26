/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The generation info (docs/engine_design.md §13.1, §13.2, §15.3; S23a's GenerationState, S26): what a client needs to
// generate exactly the host's content. A part of S26's JoinAccept and Resync (JoinMessages.h), present when the host's
// world writes generation state (World::writesGenerationState); the client adopts the host's pins (E11 of S23a,
// adoptablePins in JoinCheck.h).
//
//   seed (u64), generator count (u16); per generator in name order: name (string: varint length + bytes), pinned
//   version (u16), current placement version (u16), slot count (u16), per slot the name (string) and the flags (u32);
//   then the space count (u32) and the combined plan hash (u64)

#pragma once

#include "saiga/engine/net/WireFormat.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

struct GenerationInfo
{
    uint64_t seed = 0;
    struct Slot
    {
        std::string name;
        uint32_t flags = 0;
        bool operator==(const Slot& o) const { return name == o.name && flags == o.flags; }
    };
    struct Generator
    {
        std::string name;
        uint16_t pinned = 0, current = 0;
        std::vector<Slot> slots;
        bool operator==(const Generator& o) const
        {
            return name == o.name && pinned == o.pinned && current == o.current && slots == o.slots;
        }
    };
    std::vector<Generator> generators;
    uint32_t spaceCount   = 0;
    uint64_t combinedHash = 0;
    bool operator==(const GenerationInfo& o) const
    {
        return seed == o.seed && generators == o.generators && spaceCount == o.spaceCount &&
               combinedHash == o.combinedHash;
    }
};

// The info of a world's current generation
SAIGA_ENGINE_API GenerationInfo generationInfoOf(const World& world);
// Appends the encoding to out
SAIGA_ENGINE_API void encodeGenerationInfo(const GenerationInfo& info, std::vector<uint8_t>& out);
// Untrusted input: false with the error (truncated, a name above 256 bytes, more than 1024 generators or 255 slots of a
// generator); the reader stands after the info
SAIGA_ENGINE_API bool decodeGenerationInfo(WireReader& r, GenerationInfo& out, std::string& error);

}  // namespace Engine
}  // namespace Saiga
