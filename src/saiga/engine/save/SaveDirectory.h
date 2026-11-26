/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SaveDirectory (docs/engine_design.md §15.1; S25), owned by the World while a directory is open: the open, the commits
// and the reads of stored groups.
//
// A commit (G10): between steps the main thread freezes the world-state blob (World::save's world-state form), a
// snapshot of the index (every entry's location and summary, the region file table with each file's committed end) and
// every dirty chunk (copy-on-write references of its terrain blocks, its chunk record, its group list with shared byte
// references); the IO lane writes, on the streaming pool: 0. truncates the region files it touches to their committed
// end, 1. the header if a schema table was appended (tmp, sync, rename, sync the directory), 2. decides compaction (a
// region file below half live bytes and at least 1 MiB gets generation g + 1 with its live records copied verbatim),
// 3. appends one record per frozen chunk (a disk-only group's bytes read from the old record; an unreadable old record:
// its groups are omitted and counted, G14), 4. syncs every file and every directory with a new entry, 5. writes the
// world-state file (tmp, sync, rename, sync the directory: the commit point), 6. removes world.<c - 2>.sgw and the
// region files no kept commit names. The completion (main thread) takes the new locations, maps the frozen groups by
// id, drops the bytes of groups in the record whose chunk is not requested or resident, and counts; a failure marks the
// frozen chunks dirty again and reuses the commit number.
//
// The open: the lock, then either an adoption (no world-state file of any kind: the current world becomes commit 1) or
// the newest valid commit (else its backup): its world-state blob and index, the records of chunks with deltas or
// tombstones and every space record, through the loader (validate-then-commit; only corruption tries the next
// candidate); then the cleanup (corrupt candidates renamed to .corrupt, .tmp files and unreferenced files removed, kept
// region files truncated to the largest end a kept commit records).

#pragma once

#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/SaveFileSystem.h"
#include "saiga/engine/save/SaveIoQueue.h"
#include "saiga/engine/save/SaveTypes.h"

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace Saiga
{
namespace Engine
{
class World;

class SAIGA_ENGINE_API SaveDirectory
{
   public:
    SaveDirectory(World& world, const std::string& path, const SaveDirectoryOptions& options,
                  std::shared_ptr<SaveFileSystem> fileSystem);
    ~SaveDirectory();
    SaveDirectory(const SaveDirectory&)            = delete;
    SaveDirectory& operator=(const SaveDirectory&) = delete;

    // The open (World::openSaveDirectory installs the object first: the load's residency rebuild reads through it)
    SaveOpenResult open();
    // Freezes and submits commit c (0: refused or deferred)
    uint64_t commit();
    bool waitForCommits();
    // Reads every disk-only group into memory and unlocks (the store becomes a memory store)
    bool close();
    SaveStatus status() const;
    // After a completed step (between steps, before the step listeners): a pending commit request, the automatic
    // commit interval, the lane's completions
    void afterStep();
    // A committed load replaced the store (blob or open): the commit clock restarts at the loaded time
    void afterLoad();
    // Before a load replaces the store: the commits are waited for and the queued reads cancelled
    void beforeLoad();
    // The stored-groups part of the residency (PartStored): a synchronous read / an asynchronous one on the lane
    void readStoredSync(const ChunkId& chunk);
    void requestStoredRead(const ChunkId& chunk, int priority, std::function<void()> done);
    // save(): the bytes of a disk-only group, read now (mutates nothing; counted in syncReads)
    bool readGroupForSave(const ChunkId& chunk, const StoredGroup& group, GroupBytes& out) const;
    // The streaming pool is about to be reset (TerrainSystem::replaceSources)
    void beforePoolReset() { lane.poolReset(); }
    bool synchronousIo() const;
    SaveIoQueue& ioQueue() { return lane; }
    SaveFileSystem& fileSystem() { return *fs; }
    uint64_t committed() const { return committedCommit; }
    uint64_t inFlight() const { return inFlightCommit; }

   private:
    struct FrozenGroup
    {
        uint64_t id     = 0;
        uint32_t schema = 0;
        uint64_t hash   = 0;
        GroupBytes bytes;       // null: read from the old record
        int32_t oldIndex = -1;  // its position in the old record
    };
    struct FrozenChunk;
    struct Snapshot;
    struct Result;

    World& w;
    std::string path;
    SaveDirectoryOptions options;
    std::shared_ptr<SaveFileSystem> fs;
    SaveIoQueue lane;
    bool locked = false;
    // The newest successful commit and the commit being written
    uint64_t committedCommit = 0, inFlightCommit = 0;
    bool commitRequested    = false;
    uint64_t lastCommitStep = 0;
    double lastCommitTime   = 0;
    std::string lastError;
    // The header's tables (store numbering) and whether the header is on disk
    size_t headerTables = 0;
    bool headerWritten  = false;
    // The region file table of the newest commit (with each file's committed end) and of its backup
    std::map<RegionKey, uint64_t> files, backupFiles;
    uint64_t regionBytes = 0;

    std::shared_ptr<Snapshot> freeze(uint64_t commit, bool firstCommit);
    void complete(const Snapshot& snap, const Result& result);
    static void writeCommit(const Snapshot& snap, SaveFileSystem& fs, Result& result);
    void cleanupAfterOpen(uint64_t loaded, const std::map<RegionKey, uint64_t>& keptEnds,
                          const std::vector<std::string>& corrupt);
};

}  // namespace Engine
}  // namespace Saiga
