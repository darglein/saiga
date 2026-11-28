/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ChunkStore (docs/engine_design.md §15.1; S25), owned by the World (World::chunkStore(), host only): the stored entity
// groups of unloaded chunks and the space records of unloaded spaces, with the schema tables they use. Without a save
// directory a memory store (the groups' bytes in memory); with one it also holds the index of the records of the newest
// commit (each entry's location and summary) and which chunks differ from it (dirty). One list per chunk (G4): each
// group in memory, on disk (in the current record), or both. A group's bytes leave memory only when it is in the
// current record and the chunk is not requested or resident (S24's state None): at a commit's completion and when the
// residency unloads the chunk. The modification state (terrain deltas, chunk records) stays in its own stores (G2); the
// store only tracks which chunks differ from the newest commit.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/save/ChunkRecordCodec.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/export.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

struct StoredGroup
{
    uint64_t id = 0;           // a store-wide serial (not saved): identifies it across commits
    GroupBytes bytes;          // the group form; null: only on disk
    uint32_t schema     = 0;   // the schema table number (the store's numbering = the directory header's)
    uint64_t hash       = 0;   // FNV-1a 64 of the bytes (kept when the bytes are dropped)
    int32_t recordIndex = -1;  // its position in the chunk's current record (-1: not in it)
};

struct ChunkEntry
{
    // The groups in stored order (oldest first); the single list (G4)
    std::vector<StoredGroup> groups;
    // The newest successful commit's index entry (indexed: the index names the chunk; location: its record, none for a
    // summary-only entry)
    bool indexed = false;
    std::optional<RecordLocation> record;
    IndexSummary summary;
    bool dirty = false;  // the state differs from the newest commit
};

// The store part of a load (staged before the commit; WorldLoader)
struct StoreStage
{
    // The tables in the blob's EGRP numbering (0: the blob's SCHM)
    std::vector<std::shared_ptr<const SchemaTable>> tables;
    std::map<ChunkId, std::vector<StoredGroup>> groups;  // chunk groups by chunk; space groups by spaceRecordKey
    std::set<SpaceId> unloadedSpaces;
    // A directory open: the index of the loaded commit (the entries keep their locations; the store starts clean)
    bool fromDirectory = false;
    std::map<ChunkId, ChunkEntry> indexEntries;
};

// A directory open's input of the loader: the terrain and the chunk records assembled from the records of the loaded
// commit (instead of TERR and CREC), the store part (the index, the unloaded spaces)
struct DirectoryLoad
{
    std::optional<TerrainSection> terrain;
    std::optional<ChunkRecordSection> records;
    StoreStage store;
};

class SAIGA_ENGINE_API ChunkStore
{
   public:
    struct Stats
    {
        uint64_t groupsUnloaded = 0, entitiesUnloaded = 0, farUnloads = 0, unloadCandidates = 0, groupsReloaded = 0,
                 entitiesReloaded = 0, droppedReferences = 0, failedGroups = 0, reloadHookErrors = 0,
                 spacesUnloaded = 0, spacesReloaded = 0, recordsRead = 0, syncReads = 0, recordsWritten = 0,
                 bytesWritten = 0, commits = 0, failedCommits = 0, compactions = 0, autoRecords = 0;
        double lastFreezeMs = 0, maxFreezeMs = 0, lastCommitWriteMs = 0;
    };

    ChunkStore()                             = default;
    ChunkStore(const ChunkStore&)            = delete;
    ChunkStore& operator=(const ChunkStore&) = delete;

    const std::map<ChunkId, ChunkEntry>& chunks() const { return entries; }
    const ChunkEntry* entry(const ChunkId& c) const;
    // Chunk groups (in memory or on disk); space records are not counted
    bool hasGroups(const ChunkId& c) const;
    // A group whose bytes are only on disk (PartStored wanted)
    bool needsRead(const ChunkId& c) const;
    size_t groupCount(const ChunkId& c) const;
    // Chunks with groups and their group count (space records excluded)
    size_t chunksWithGroups() const;
    size_t totalGroups() const;
    bool spaceUnloaded(SpaceId s) const { return unloaded.count(s) != 0; }
    const std::set<SpaceId>& unloadedSpaces() const { return unloaded; }
    // Groups or unloaded spaces exist (EGRP, the hash part)
    bool hasContent() const { return totalGroups() > 0 || !unloaded.empty(); }
    const Stats& stats() const { return statistics; }
    Stats& statsForUpdate() { return statistics; }

    // ---- schema tables (the directory header's numbering when a directory is open)
    const std::vector<std::shared_ptr<const SchemaTable>>& tables() const { return tableList; }
    // The number of a table with equal write bytes, else appended
    uint32_t internTable(std::shared_ptr<const SchemaTable> table);
    // The number of the world's current table (computed once per store generation)
    uint32_t currentTable(const World& world);

    // ---- engine-internal writers (main thread)
    // Unload: appends the group (a new id) and marks the chunk dirty
    void addGroup(const ChunkId& c, StoredGroup g);
    // The groups of a chunk for its reload: false while a group's bytes are only on disk; else all groups (out), the
    // entry keeps none and is dirty
    bool takeGroups(const ChunkId& c, std::vector<StoredGroup>& out);
    // A failed group goes back (front: its original order before the next groups)
    void putBack(const ChunkId& c, std::vector<StoredGroup> groups);
    // The chunk's state changed (modification state, auto records); a no-op without a directory
    void markDirty(const ChunkId& c);
    void setTracking(bool on) { tracking = on; }
    bool trackingDirty() const { return tracking; }
    // Space records
    void setSpaceUnloaded(SpaceId s, bool isUnloaded);
    // The bytes of the groups by recordIndex (a read's completion); false if the entry changed
    void fillBytes(const ChunkId& c, const std::vector<RecordGroup>& groups);
    // The rule of when bytes leave memory: groups in the current record lose their bytes
    void dropBytes(const ChunkId& c);
    // A load's commit: the staged state replaces the store (blob loads: every group in memory, every chunk with state
    // or groups dirty when a directory is open; a directory open: the index entries, clean)
    void replace(StoreStage&& stage);
    // Entries without groups, index entry or dirt are removed
    void prune(const ChunkId& c);
    std::map<ChunkId, ChunkEntry>& entriesForUpdate() { return entries; }
    // Bumped by every load and open: completions of IO tasks of another generation apply nothing
    uint64_t generation() const { return gen; }
    void bumpGeneration() { gen++; }

   private:
    std::map<ChunkId, ChunkEntry> entries;
    std::set<SpaceId> unloaded;
    std::vector<std::shared_ptr<const SchemaTable>> tableList;
    std::vector<std::vector<uint8_t>> tableBytes;
    int64_t currentNumber = -1;
    uint64_t nextGroupId  = 1;
    uint64_t gen          = 1;
    bool tracking         = false;
    Stats statistics;
};

}  // namespace Engine
}  // namespace Saiga
