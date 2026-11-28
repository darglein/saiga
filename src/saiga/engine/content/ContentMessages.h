/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The tombstone message (docs/engine_design.md §13.1, §13.2; S23b): a raw engine message host -> clients on
// Channel::State, reliable, with the replication epoch (ordered with Resync, the spawn set and S22's terrain messages);
// S26 sends each peer only the entries of the chunks it subscribed (a subscription's record arrives in S26's
// ChunkSnapshot, net/ChunkMessages.h).
//
//   TombstoneModification  epoch (u32), space (u16), entry count (u16, 1..4096); per entry: chunk x, z (i32), placement
//                          version (u16), the chunk's sequence after the change (u32), slot (u8), counter (u32)
//
// The decoder checks only what needs no world (the structure and the bounds); the client's handler checks the rest
// (ClientChunkSync::onTombstoneModification).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
constexpr size_t kMaxTombstoneEntries = 4096;

struct TombstoneEntry
{
    int32_t x = 0, z = 0;
    uint16_t placementVersion = 0;
    uint32_t sequence         = 0;
    InstanceKey key;
};

struct TombstoneModificationMessage
{
    uint32_t epoch = 0;
    SpaceId space  = 0;
    std::vector<TombstoneEntry> entries;
};

SAIGA_ENGINE_API void encodeTombstoneModification(uint32_t epoch, SpaceId space,
                                                  const std::vector<TombstoneEntry>& entries,
                                                  std::vector<uint8_t>& out);
// Untrusted input: false with the error (truncated, a count out of bounds, a slot above 255, bytes left over)
SAIGA_ENGINE_API bool decodeTombstoneModification(const uint8_t* data, size_t size, TombstoneModificationMessage& out,
                                                  std::string& error);

}  // namespace Engine
}  // namespace Saiga
