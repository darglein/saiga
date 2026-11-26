/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/ChunkStore.h"

#include "saiga/engine/save/SaveDirectoryFormat.h"

namespace Saiga
{
namespace Engine
{
const ChunkEntry* ChunkStore::entry(const ChunkId& c) const
{
    auto it = entries.find(c);
    return it == entries.end() ? nullptr : &it->second;
}

bool ChunkStore::hasGroups(const ChunkId& c) const
{
    const ChunkEntry* e = entry(c);
    return e && !e->groups.empty() && !isSpaceRecordKey(c);
}

bool ChunkStore::needsRead(const ChunkId& c) const
{
    const ChunkEntry* e = entry(c);
    if (!e || isSpaceRecordKey(c)) return false;
    for (const StoredGroup& g : e->groups)
        if (!g.bytes) return true;
    return false;
}

size_t ChunkStore::groupCount(const ChunkId& c) const
{
    const ChunkEntry* e = entry(c);
    return e && !isSpaceRecordKey(c) ? e->groups.size() : 0;
}

size_t ChunkStore::chunksWithGroups() const
{
    size_t n = 0;
    for (const auto& [c, e] : entries) n += (!isSpaceRecordKey(c) && !e.groups.empty()) ? 1 : 0;
    return n;
}

size_t ChunkStore::totalGroups() const
{
    size_t n = 0;
    for (const auto& [c, e] : entries)
        if (!isSpaceRecordKey(c)) n += e.groups.size();
    return n;
}

uint32_t ChunkStore::internTable(std::shared_ptr<const SchemaTable> table)
{
    std::vector<uint8_t> bytes;
    BinaryWriter w(bytes);
    table->write(w);
    for (size_t i = 0; i < tableBytes.size(); ++i)
        if (tableBytes[i] == bytes) return uint32_t(i);
    tableList.push_back(std::move(table));
    tableBytes.push_back(std::move(bytes));
    return uint32_t(tableList.size() - 1);
}

uint32_t ChunkStore::currentTable(const World& world)
{
    if (currentNumber < 0)
        currentNumber = int64_t(internTable(std::make_shared<const SchemaTable>(SchemaTable::fromWorld(world))));
    return uint32_t(currentNumber);
}

void ChunkStore::addGroup(const ChunkId& c, StoredGroup g)
{
    g.id          = nextGroupId++;
    g.recordIndex = -1;
    ChunkEntry& e = entries[c];
    e.groups.push_back(std::move(g));
    e.dirty = true;
}

bool ChunkStore::takeGroups(const ChunkId& c, std::vector<StoredGroup>& out)
{
    out.clear();
    auto it = entries.find(c);
    if (it == entries.end()) return true;
    for (const StoredGroup& g : it->second.groups)
        if (!g.bytes) return false;
    out.swap(it->second.groups);
    it->second.dirty = true;
    return true;
}

void ChunkStore::putBack(const ChunkId& c, std::vector<StoredGroup> groups)
{
    if (groups.empty()) return;
    ChunkEntry& e = entries[c];
    groups.insert(groups.end(), std::make_move_iterator(e.groups.begin()), std::make_move_iterator(e.groups.end()));
    e.groups.swap(groups);
}

void ChunkStore::markDirty(const ChunkId& c)
{
    if (!tracking) return;
    entries[c].dirty = true;
}

void ChunkStore::setSpaceUnloaded(SpaceId s, bool isUnloaded)
{
    if (isUnloaded)
        unloaded.insert(s);
    else
        unloaded.erase(s);
    // The space record changes with it (written by the next commit, dropped from the index when loaded again)
    entries[spaceRecordKey(s)].dirty = true;
    if (!tracking) prune(spaceRecordKey(s));
}

void ChunkStore::fillBytes(const ChunkId& c, const std::vector<RecordGroup>& groups)
{
    auto it = entries.find(c);
    if (it == entries.end()) return;
    for (StoredGroup& g : it->second.groups)
    {
        if (g.bytes || g.recordIndex < 0 || size_t(g.recordIndex) >= groups.size()) continue;
        const RecordGroup& r = groups[size_t(g.recordIndex)];
        if (r.schema != g.schema || !r.bytes || fnv1a64Bytes(r.bytes->data(), r.bytes->size()) != g.hash) continue;
        g.bytes = r.bytes;
    }
}

void ChunkStore::dropBytes(const ChunkId& c)
{
    auto it = entries.find(c);
    if (it == entries.end() || isSpaceRecordKey(c)) return;
    for (StoredGroup& g : it->second.groups)
        if (g.recordIndex >= 0) g.bytes.reset();
}

void ChunkStore::prune(const ChunkId& c)
{
    auto it = entries.find(c);
    if (it == entries.end()) return;
    const ChunkEntry& e  = it->second;
    const bool keepSpace = isSpaceRecordKey(c) && unloaded.count(c.space);
    if (e.groups.empty() && !e.indexed && !(e.dirty && tracking) && !keepSpace) entries.erase(it);
}

void ChunkStore::replace(StoreStage&& stage)
{
    gen++;
    unloaded = std::move(stage.unloadedSpaces);
    if (stage.fromDirectory)
    {
        // The directory's tables and index: clean (the index matches the loaded commit)
        tableList.clear();
        tableBytes.clear();
        for (auto& t : stage.tables) internTable(t);
        currentNumber = -1;
        entries       = std::move(stage.indexEntries);
        for (auto& [c, groups] : stage.groups) entries[c].groups = std::move(groups);
        // Every group gets its store-wide id (a commit's completion maps the frozen groups by it)
        for (auto& [c, e] : entries)
            for (StoredGroup& g : e.groups) g.id = nextGroupId++;
        return;
    }
    // A blob load: the blob's tables interned into this store's numbering (a directory's tables stay: its header is
    // append-only)
    if (!tracking)
    {
        tableList.clear();
        tableBytes.clear();
    }
    currentNumber = -1;
    std::vector<uint32_t> number;
    for (auto& t : stage.tables) number.push_back(internTable(t));
    // The old index entries stay (the next commit drops the chunks without state); everything else is replaced
    std::map<ChunkId, ChunkEntry> next;
    for (auto& [c, e] : entries)
    {
        if (!e.indexed) continue;
        ChunkEntry k;
        k.indexed = true;
        k.record  = e.record;
        k.summary = e.summary;
        k.dirty   = tracking;
        next.emplace(c, std::move(k));
    }
    for (auto& [c, groups] : stage.groups)
    {
        ChunkEntry& e = next[c];
        for (StoredGroup& g : groups)
        {
            g.schema      = number[g.schema];
            g.id          = nextGroupId++;
            g.recordIndex = -1;
        }
        e.groups = std::move(groups);
        e.dirty  = tracking;
    }
    for (SpaceId s : unloaded) next[spaceRecordKey(s)].dirty = tracking;
    entries.swap(next);
}

}  // namespace Engine
}  // namespace Saiga
