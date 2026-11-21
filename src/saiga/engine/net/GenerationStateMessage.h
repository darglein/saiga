/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// GenerationState (docs/engine_design.md §13.1, §13.2, §15.3; S23a): the pre-S26 subset of the handshake. The host
// sends what a client checks to generate exactly the host's content: a raw engine message (EngineMessage::
// GenerationState), host -> client, Channel::State, reliable, with the S16 epoch; only by a host whose world writes
// generation state (World::writesGenerationState), after Resync and before the spawn set on a join, right after Resync
// on a host load. The client validates and refuses (World::generationMismatch); adopting the host's pins is S26's.
//
//   epoch (u32), seed (u64), generator count (u16); per generator in name order: name (string: u64 length + bytes),
//   pinned version (u16), current placement version (u16), slot count (u16), per slot the name (string) and the flags
//   (u32); then the space count (u32) and the combined plan hash (u64)

#pragma once

#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

struct GenerationStateMessage
{
    uint32_t epoch = 0;
    uint64_t seed  = 0;
    struct Slot
    {
        std::string name;
        uint32_t flags = 0;
    };
    struct Generator
    {
        std::string name;
        uint16_t pinned = 0, current = 0;
        std::vector<Slot> slots;
    };
    std::vector<Generator> generators;
    uint32_t spaceCount   = 0;
    uint64_t combinedHash = 0;
};

// The message of a world's generation state
SAIGA_ENGINE_API GenerationStateMessage generationStateOf(const World& world, uint32_t epoch);
SAIGA_ENGINE_API void encodeGenerationState(const GenerationStateMessage& m, std::vector<uint8_t>& out);
// Untrusted input: false with the error (truncated, more than 65535 generators or slots, bytes left over)
SAIGA_ENGINE_API bool decodeGenerationState(const uint8_t* data, size_t size, GenerationStateMessage& out,
                                            std::string& error);
// The first difference between the host's state and this client's ("host uses generator test_terrain pinned v1, this
// client pins v2"); empty if the client generates what the host generates
SAIGA_ENGINE_API std::string generationStateDifference(const World& client, const GenerationStateMessage& host);

}  // namespace Engine
}  // namespace Saiga
