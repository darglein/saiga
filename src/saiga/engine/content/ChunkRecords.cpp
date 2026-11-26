/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ChunkRecords.h"

#include "saiga/core/util/assert.h"

#include <algorithm>
#include <utility>

namespace Saiga
{
namespace Engine
{
bool sortedUniqueKeys(const std::vector<InstanceKey>& keys)
{
    for (size_t i = 1; i < keys.size(); ++i)
        if (!(keys[i - 1] < keys[i])) return false;
    return true;
}

const ChunkRecord* ChunkRecords::find(const ChunkId& chunk) const
{
    auto it = byChunk.find(chunk);
    return it == byChunk.end() ? nullptr : &it->second;
}

size_t ChunkRecords::tombstonedChunks() const
{
    size_t n = 0;
    for (const auto& kv : byChunk) n += kv.second.tombstones.empty() ? 0 : 1;
    return n;
}

void ChunkRecords::markStamped(const ChunkId& chunk, uint16_t currentVersion)
{
    stampedChunks.insert(chunk);
    if (!byChunk.count(chunk)) byChunk[chunk].placementVersion = currentVersion;
    statistics.chunks++;
}

ChunkRecord& ChunkRecords::reserve(const ChunkId& chunk, uint16_t version, size_t more)
{
    auto it = byChunk.find(chunk);
    if (it == byChunk.end())
    {
        it                          = byChunk.emplace(chunk, ChunkRecord()).first;
        it->second.placementVersion = version;
    }
    it->second.tombstones.reserve(it->second.tombstones.size() + more);
    return it->second;
}

ChunkRecords::Map::node_type ChunkRecords::makeRecord(const ChunkId& chunk, uint16_t version, size_t capacity) const
{
    Map scratch;
    ChunkRecord r;
    r.placementVersion = version;
    r.tombstones.reserve(capacity);
    scratch.emplace(chunk, std::move(r));
    return scratch.extract(scratch.begin());
}

void ChunkRecords::reserveKeys(const ChunkId& chunk, size_t more)
{
    auto it = byChunk.find(chunk);
    SAIGA_ASSERT(it != byChunk.end(), "ChunkRecords::reserveKeys: no record");
    it->second.tombstones.reserve(it->second.tombstones.size() + more);
}

void ChunkRecords::insertRecord(Map::node_type&& node)
{
    const auto r = byChunk.insert(std::move(node));
    SAIGA_ASSERT(r.inserted, "ChunkRecords::insertRecord: the chunk has a record");
}

bool ChunkRecords::insert(const ChunkId& chunk, const InstanceKey& key)
{
    auto it = byChunk.find(chunk);
    SAIGA_ASSERT(it != byChunk.end(), "ChunkRecords::insert: the record was not reserved");
    std::vector<InstanceKey>& t = it->second.tombstones;
    auto pos                    = std::lower_bound(t.begin(), t.end(), key);
    if (pos != t.end() && *pos == key) return false;
    SAIGA_ASSERT(t.size() < t.capacity(), "ChunkRecords::insert: no reserved capacity");
    t.insert(pos, key);
    statistics.tombstones++;
    return true;
}

void ChunkRecords::replace(Map records, std::set<ChunkId> stamped)
{
    byChunk       = std::move(records);
    stampedChunks = std::move(stamped);
}

void ChunkRecords::erase(const ChunkId& chunk)
{
    byChunk.erase(chunk);
}

ChunkRecord& ChunkRecords::set(const ChunkId& chunk, uint16_t placementVersion, std::vector<InstanceKey> keys)
{
    ChunkRecord& r     = byChunk[chunk];
    r.placementVersion = placementVersion;
    r.tombstones       = std::move(keys);
    return r;
}

void ChunkRecords::clear()
{
    byChunk.clear();
    stampedChunks.clear();
}

void ChunkRecords::injectForTests(const ChunkId& chunk, uint16_t placementVersion, const std::vector<InstanceKey>& keys)
{
    SAIGA_ASSERT(sortedUniqueKeys(keys), "ChunkRecords::injectForTests: the keys are not sorted and unique");
    ChunkRecord& r = reserve(chunk, placementVersion, keys.size());
    SAIGA_ASSERT(r.placementVersion == placementVersion, "ChunkRecords::injectForTests: another placement version");
    for (const InstanceKey& k : keys) insert(chunk, k);
}

}  // namespace Engine
}  // namespace Saiga
