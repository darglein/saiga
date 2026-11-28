/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The terrain message (docs/engine_design.md §13.2; S22): a raw engine message host -> clients on Channel::State,
// reliable, with the replication epoch; S26 sends each peer only the parts of the chunks it subscribed (the chunk state
// of a subscription is S26's ChunkSnapshot, net/ChunkMessages.h). Values are absolute deltas (new height minus base),
// never increments or the brush; every rectangle is encoded on its own by TerrainDeltaCodec (predictor reset per part).
//
//   TerrainModification  flags (u8: bit 0 zstd), epoch (u32), space (u16), part count (u16, >= 1); per part: chunk x,
//                        z (i32), the chunk's sequence after the change (u32), the chunk-local box x0, z0, w, h (u16,
//                        w, h >= 1), the encoded length (varint) and the w * h deltas
//
// The decoder checks only what needs no world (the structure, the flags, the codec's bounds); the client's handler
// checks the rest (ClientChunkSync::onTerrainModification).

#pragma once

#include "saiga/engine/terrain/TerrainEdit.h"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct TerrainModificationMessage
{
    uint32_t epoch = 0;
    SpaceId space  = 0;
    std::vector<TerrainPart> parts;
};

// zstd when it is smaller than the raw varints (large boxes)
SAIGA_ENGINE_API void encodeTerrainModification(uint32_t epoch, SpaceId space, const std::vector<TerrainPart>& parts,
                                                std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeTerrainModification(const uint8_t* data, size_t size, TerrainModificationMessage& out,
                                                std::string& error);

}  // namespace Engine
}  // namespace Saiga
