/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainDeltaCodec (docs/engine_design.md §13.2, §15.1; S22): the encoding of a rectangle of terrain deltas, shared by
// the TERR save section, the TerrainModification message and S26's ChunkSnapshot. The values in row-major order (row =
// z) as zigzag varints of the difference to the previous value; the predictor starts at 0 for every rectangle (each
// TERR chunk, each message part, each snapshot), so every rectangle decodes on its own; then optionally zstd at a
// fixed level. The decoder treats its input as untrusted: it bounds the decompressed size (count * 4 bytes: a zigzag
// difference of two deltas within +-kMaxTerrainDelta is below 2^26, at most 4 varint bytes), the value count and
// |delta| <= kMaxTerrainDelta, and leaves no bytes over.

#pragma once

#include "saiga/export.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// Every delta lies within +-2^24 (two heights within +-kTerrainHeightLimit)
constexpr int32_t kMaxTerrainDelta = 1 << 24;
// The fixed zstd level of the codec
constexpr int kTerrainDeltaZstdLevel = 3;

namespace TerrainDeltaCodec
{
// Appends the count values (each within +-kMaxTerrainDelta, asserted) to out: raw varints, or with zstd one zstd frame
// of them
SAIGA_ENGINE_API void encode(const int32_t* values, size_t count, bool zstd, std::vector<uint8_t>& out);
// Decodes exactly count values from all size bytes; false with an error (out partially written) otherwise
SAIGA_ENGINE_API bool decode(const uint8_t* data, size_t size, bool zstd, size_t count, int32_t* out,
                             std::string* error);
}  // namespace TerrainDeltaCodec

}  // namespace Engine
}  // namespace Saiga
