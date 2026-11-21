/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ChunkRecords.h"

#include <utility>

namespace Saiga
{
namespace Engine
{
const ChunkRecord* ChunkRecords::find(const ChunkId& chunk) const
{
    auto it = byChunk.find(chunk);
    return it == byChunk.end() ? nullptr : &it->second;
}

void ChunkRecords::markStamped(const ChunkId& chunk, uint16_t currentVersion)
{
    stampedChunks.insert(chunk);
    if (!byChunk.count(chunk)) byChunk[chunk].placementVersion = currentVersion;
    statistics.chunks++;
}

void ChunkRecords::replace(std::map<ChunkId, ChunkRecord> records, std::set<ChunkId> stamped)
{
    byChunk       = std::move(records);
    stampedChunks = std::move(stamped);
}

void ChunkRecords::clear()
{
    byChunk.clear();
    stampedChunks.clear();
}

}  // namespace Engine
}  // namespace Saiga
