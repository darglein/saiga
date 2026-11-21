/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The terrain messages (docs/engine_design.md §13.2; S22): raw engine messages host -> clients on Channel::State,
// reliable, with the replication epoch. Values are absolute deltas (new height minus base), never increments or the
// brush; every rectangle is encoded on its own by TerrainDeltaCodec (predictor reset per part and per chunk).
//
//   TerrainModification  flags (u8: bit 0 zstd), epoch (u32), space (u16), part count (u16, >= 1); per part: chunk x,
//                        z (i32), the chunk's sequence after the change (u32), the chunk-local box x0, z0, w, h (u16,
//                        w, h >= 1), the encoded length (varint) and the w * h deltas
//   TerrainChunkState    flags (u8: bit 0 zstd, bit 1 deltas present), epoch (u32), space (u16), chunk x, z (i32),
//                        sequence (u32), N (u16); with deltas: the encoded length (varint) and the dense N x N deltas
//
// The decoders check only what needs no world (the structure, the flags, the codec's bounds); the client's handlers
// check the rest (Replication::onTerrainModification / onTerrainChunkState).

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

struct TerrainChunkStateMessage
{
    uint32_t epoch = 0;
    ChunkId chunk;
    uint32_t sequence = 0;
    uint16_t N        = 0;
    bool hasDeltas    = false;
    std::vector<int32_t> deltas;  // N * N with deltas
};

// zstd when it is smaller than the raw varints (large boxes)
SAIGA_ENGINE_API void encodeTerrainModification(uint32_t epoch, SpaceId space, const std::vector<TerrainPart>& parts,
                                                std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeTerrainModification(const uint8_t* data, size_t size, TerrainModificationMessage& out,
                                                std::string& error);
SAIGA_ENGINE_API void encodeTerrainChunkState(uint32_t epoch, const ChunkId& chunk, uint32_t sequence, uint16_t N,
                                              const std::vector<int32_t>* deltas, std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeTerrainChunkState(const uint8_t* data, size_t size, TerrainChunkStateMessage& out,
                                              std::string& error);

}  // namespace Engine
}  // namespace Saiga
