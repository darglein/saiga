/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The client's chunk sync (docs/engine_design.md §13.2, §12.3; S26, D9-D11, D16-D19), owned by Replication: per
// subscribed chunk Pending (a ticket and a queue of live parts) or Synced. A streaming client subscribes the chunks its
// ChunkResidency has Requested or Resident plus the holds (forEachChunk, diffed in phase 8); a client of a world
// without streaming subscribes every space as a whole (ChunkInterest all) and learns from the host's ChunkIndex which
// chunks have state (Pending with ticket 0: their snapshots follow; every other chunk Synced at sequence 0).
//
// ChunkSnapshot replaces a Pending chunk's state when its ticket is current (an older one is stale), then the queue
// applies: parts with sequence <= the snapshot's are dropped, the rest applied each at current + 1. Live parts of a
// Synced chunk apply at current + 1; a gap (or a queue above kMaxQueuedPartsPerChunk) re-requests the chunk: Pending
// with a new ticket, its present state kept until the new snapshot. An unsubscribed chunk's state is dropped (D10:
// records, entries, the samples of its closed range that no other subscribed chunk contains). The interest is staged
// in phase 8 and applied by commit() (abort() discards it; re-requests and `all` subscriptions stay unsent until a
// committed phase 8 sent them).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/content/ContentMessages.h"
#include "saiga/engine/terrain/TerrainEdit.h"
#include "saiga/export.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

class SAIGA_ENGINE_API ClientChunkSync
{
   public:
    static constexpr size_t kMaxQueuedPartsPerChunk = 1024;
    enum class State : uint8_t
    {
        None,
        Pending,
        Synced
    };
    // A live part or tombstone entry queued behind a pending snapshot
    struct Queued
    {
        bool tombstone    = false;
        uint32_t sequence = 0;
        TerrainPart part;
        TombstoneEntry entry;
    };
    struct Entry
    {
        State state     = State::None;
        uint32_t ticket = 0;
        // A chunk of a space whose ChunkIndex has not arrived (its live parts are queued)
        bool awaitingIndex = false;
        std::vector<Queued> queue;
    };
    struct Stats
    {
        uint64_t subscribed = 0, unsubscribed = 0, snapshotsApplied = 0, snapshotsStale = 0, indexesApplied = 0,
                 partsQueued = 0, partsDroppedNotSubscribed = 0, partsDroppedOld = 0, partsApplied = 0, rerequests = 0,
                 queueOverflows = 0, interestMessages = 0, interestCapped = 0;
    };

    explicit ClientChunkSync(World& world);

    State state(const ChunkId& chunk) const;
    // 0 for None (and a chunk of an indexed space without its own subscription)
    uint32_t ticket(const ChunkId& chunk) const;
    size_t queuedParts(const ChunkId& chunk) const;
    size_t queuedParts() const;
    size_t pendingCount() const;
    size_t syncedCount() const;
    // The subscribed chunks with an entry, in ChunkId order (an indexed space's chunks without state have none)
    const std::map<ChunkId, Entry>& chunks() const { return table; }
    // A world without streaming: the space's ChunkIndex of the current epoch was applied; every space
    bool spaceIndexed(SpaceId space) const;
    bool allIndexed() const;
    const Stats& stats() const { return statistics; }

    // Explicit interest on top of the residency, reference-counted (a no-op in a world without streaming)
    void hold(const ChunkId& chunk);
    void release(const ChunkId& chunk);
    const std::map<ChunkId, int>& holds() const { return held; }

    // ---- phase 1 handlers (client)
    void onSnapshot(uint8_t from, const uint8_t* data, size_t size);
    void onIndex(uint8_t from, const uint8_t* data, size_t size);
    void onTerrainModification(uint8_t from, const uint8_t* data, size_t size);
    void onTombstoneModification(uint8_t from, const uint8_t* data, size_t size);

    // ---- phase 8 (Replication::extract, also stepNetworkOnly) and the step end
    void extract();
    void commit();
    void abort();
    // The client was admitted: the `all` subscriptions of a world without streaming
    void onJoined();
    // A Resync (the stores were cleared by Replication::clearReplicatedChunkState): the table, the spaces' index
    // state, the unsent and staged interest; every wanted chunk is subscribed again in the new epoch
    void reset();

   private:
    enum class SpaceSync : uint8_t
    {
        None,
        AllPending,
        Indexed
    };
    World& w;
    std::map<ChunkId, Entry> table;
    std::vector<SpaceSync> spaces;
    std::map<ChunkId, int> held;
    uint32_t nextTicket = 1;
    // Re-requests (chunk -> ticket) and `all` subscriptions not yet sent by a committed phase 8
    std::map<ChunkId, uint32_t> unsentAdds;
    std::set<SpaceId> unsentAll;
    // Staged by extract
    struct StagedAdd
    {
        ChunkId chunk;
        uint32_t ticket = 0;
        bool fresh      = false;
    };
    std::vector<StagedAdd> stagedAdds;
    std::vector<ChunkId> stagedRemoves;
    std::vector<SpaceId> stagedAll;
    Stats statistics;
    std::vector<uint8_t> bytes;

    uint32_t newTicket();
    void reject(uint8_t from, const std::string& why, bool content = false);
    // The chunk becomes Pending with a new ticket and an empty queue; its state is kept (D9)
    void rerequest(const ChunkId& chunk, const char* why);
    // Queues a part or entry of a Pending chunk (overflow: re-request)
    void enqueue(Entry& e, const ChunkId& chunk, Queued q);
    Entry* pendingEntry(const ChunkId& chunk);
    // Applies Synced parts of one space (each at current + 1, checked by the caller); one onTerrainChanged
    void applyParts(SpaceId space, const std::vector<const TerrainPart*>& parts);
    // Applies one tombstone entry; false: its placement version is not the chunk's effective one
    bool applyTombstone(SpaceId space, const TombstoneEntry& e);
    // Applies the queue of a chunk whose state is current (after a snapshot or an index); false after a re-request
    bool drainQueue(const ChunkId& chunk, std::vector<Queued> queue);
    // D10: the chunk's state goes (unsubscribed)
    void drop(const ChunkId& chunk);
};

}  // namespace Engine
}  // namespace Saiga
