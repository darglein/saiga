/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The host's chunk subscriptions (docs/engine_design.md §13.2; S26, D1, D7, D12, D19), owned by Replication: per
// admitted peer the chunks it subscribed (with the ticket of each subscription) and the spaces it subscribed as a whole
// (a world without streaming), the ChunkIndex messages of new `all` subscriptions and the FIFO snapshot queue over all
// peers. ChunkInterest applies to the table in its phase-1 handler; phase 8 sends the indexes and serves the queue
// under WorldConfig::network.chunkSnapshotBytesPerStep (at least one snapshot per step), encoding each snapshot from
// the current TerrainEditStore and ChunkRecords. What phase 8 served is staged: commit() pops it, abort() keeps it.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/net/ChunkMessages.h"
#include "saiga/export.h"

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

class SAIGA_ENGINE_API HostChunkSubscriptions
{
   public:
    struct Stats
    {
        uint64_t interestMessages = 0, adds = 0, removes = 0, droppedEpoch = 0, snapshotsSent = 0, snapshotBytes = 0,
                 indexesSent = 0, partsFiltered = 0, entriesFiltered = 0;
    };

    explicit HostChunkSubscriptions(World& world);

    // The chunk is subscribed by the peer (also through an `all` subscription of its space)
    bool subscribed(uint8_t peer, const ChunkId& chunk) const;
    // The space is subscribed as a whole (D19)
    bool subscribedAll(uint8_t peer, SpaceId space) const;
    // The per-chunk subscriptions of a peer (an `all` space counts its chunks with state)
    size_t subscribedCount(uint8_t peer) const;
    size_t queuedSnapshots(uint8_t peer) const;
    size_t queuedSnapshots() const;
    // The peers with a table, ascending
    std::vector<uint8_t> peers() const;
    const Stats& stats() const { return statistics; }
    Stats& statsForUpdate() { return statistics; }

    // Phase 1: a ChunkInterest of an admitted peer (the World's admission gate; rejected input otherwise)
    void onInterest(uint8_t from, const uint8_t* data, size_t size);
    // Phase 8 (both extract paths), after the host state list: the indexes of new `all` subscriptions, then the
    // snapshot queue under the budget; returns the snapshot bytes sent
    size_t serve(uint32_t epoch);
    // The snapshots the last serve sent
    size_t servedCount() const { return served.size(); }
    void commit();
    void abort();
    // A committed host load (D11): every table and queue goes
    void clear();
    // A peer that left loses its table
    void removePeer(uint8_t peer);

    // The snapshot of a chunk from the World's current modification state (also the benchmark): sequence, the record
    // when it has tombstones or another placement version than the current one, the dense deltas when it has any
    static void snapshotOf(const World& world, uint32_t epoch, uint32_t ticket, const ChunkId& chunk,
                           ChunkSnapshotMessage& out);
    // The chunks of a space whose snapshot carries something (sequence > 0 or a record as above), in ChunkId order
    static std::vector<ChunkId> chunksWithState(const World& world, SpaceId space);

   private:
    struct Peer
    {
        std::map<ChunkId, uint32_t> subs;    // per-chunk subscriptions -> ticket
        std::set<SpaceId> all;               // spaces subscribed as a whole
        std::map<ChunkId, uint32_t> queued;  // queued snapshots -> ticket (the newest)
    };
    World& w;
    std::map<uint8_t, Peer> table;
    // FIFO over all peers; an entry whose (peer, chunk) is no longer in Peer::queued is skipped
    std::deque<std::pair<uint8_t, ChunkId>> queue;
    // `all` subscriptions whose ChunkIndex is sent in the next phase 8
    std::vector<std::pair<uint8_t, SpaceId>> pendingIndex;
    // Staged by serve: the queue entries walked, the served (peer, chunk, ticket), the indexes sent with their chunks
    size_t walked      = 0;
    size_t servedBytes = 0;
    std::vector<std::pair<std::pair<uint8_t, ChunkId>, uint32_t>> served;
    std::vector<std::pair<uint8_t, std::vector<ChunkId>>> indexed;
    size_t indexesStaged = 0;
    Stats statistics;
    void enqueue(uint8_t peer, const ChunkId& chunk, uint32_t ticket);
    std::vector<uint8_t> bytes;
};

}  // namespace Engine
}  // namespace Saiga
