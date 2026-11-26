/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/HostChunkSubscriptions.h"

#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <limits>
#include <set>

namespace Saiga
{
namespace Engine
{
namespace
{
// A record that differs from the effective record of a chunk without one (S23b's rule)
bool recordPresent(const World& w, const ChunkId& c, const ChunkRecord& r)
{
    return !r.tombstones.empty() || r.placementVersion != w.generation().space(c.space).current;
}

ChunkId chunkIdOf(SpaceId space, int32_t x, int32_t z)
{
    ChunkId c;
    c.space = space;
    c.x     = x;
    c.z     = z;
    return c;
}
}  // namespace

HostChunkSubscriptions::HostChunkSubscriptions(World& world) : w(world) {}

bool HostChunkSubscriptions::subscribed(uint8_t peer, const ChunkId& chunk) const
{
    auto it = table.find(peer);
    if (it == table.end()) return false;
    return it->second.all.count(chunk.space) != 0 || it->second.subs.count(chunk) != 0;
}

bool HostChunkSubscriptions::subscribedAll(uint8_t peer, SpaceId space) const
{
    auto it = table.find(peer);
    return it != table.end() && it->second.all.count(space) != 0;
}

size_t HostChunkSubscriptions::subscribedCount(uint8_t peer) const
{
    auto it = table.find(peer);
    if (it == table.end()) return 0;
    size_t n = it->second.subs.size();
    for (SpaceId s : it->second.all) n += chunksWithState(w, s).size();
    return n;
}

size_t HostChunkSubscriptions::queuedSnapshots(uint8_t peer) const
{
    auto it = table.find(peer);
    return it == table.end() ? 0 : it->second.queued.size();
}

size_t HostChunkSubscriptions::queuedSnapshots() const
{
    size_t n = 0;
    for (const auto& kv : table) n += kv.second.queued.size();
    // An index not yet sent queues its snapshots
    return n + pendingIndex.size();
}

std::vector<uint8_t> HostChunkSubscriptions::peers() const
{
    std::vector<uint8_t> out;
    for (const auto& kv : table) out.push_back(kv.first);
    return out;
}

void HostChunkSubscriptions::enqueue(uint8_t peer, const ChunkId& chunk, uint32_t ticket)
{
    Peer& p = table[peer];
    auto it = p.queued.find(chunk);
    if (it != p.queued.end())
    {
        // D12: one snapshot with the newest ticket
        it->second = ticket;
        return;
    }
    p.queued.emplace(chunk, ticket);
    queue.push_back({peer, chunk});
}

void HostChunkSubscriptions::onInterest(uint8_t from, const uint8_t* data, size_t size)
{
    if (!w.isHost() || from == 0)
    {
        w.rejectInput(from, "ChunkInterest: only clients send it to the host");
        return;
    }
    ChunkInterestMessage m;
    std::string error;
    if (!decodeChunkInterest(data, size, m, error))
    {
        w.rejectInput(from, error);
        return;
    }
    statistics.interestMessages++;
    // D11: an interest of another epoch was in flight across a host load
    if (m.epoch != w.replication().epoch())
    {
        statistics.droppedEpoch++;
        return;
    }
    if (m.space >= w.spaceCount())
    {
        w.rejectInput(from, "ChunkInterest: unknown space " + std::to_string(m.space));
        return;
    }
    Peer& p = table[from];
    if (m.all)
    {
        // D19: once per space and epoch, only in a world without streaming
        if (w.streaming())
        {
            w.rejectInput(from, "ChunkInterest: an all subscription in a streaming world");
            return;
        }
        if (p.all.insert(m.space).second) pendingIndex.push_back({from, m.space});
        return;
    }
    const bool allSpace = p.all.count(m.space) != 0;
    if (!w.streaming() && (!allSpace || !m.removes.empty()))
    {
        w.rejectInput(from,
                      "ChunkInterest: chunk entries in a world without streaming (only a re-request inside a "
                      "subscribed space)");
        return;
    }
    // Validation of the whole message before anything applies
    const int32_t n = w.spaceTemplate(m.space).sizeChunks;
    std::set<std::pair<int32_t, int32_t>> seen;
    auto valid = [&](int32_t x, int32_t z) { return x >= 0 && z >= 0 && x < n && z < n && seen.insert({x, z}).second; };
    size_t newCount = p.subs.size();
    for (const auto& r : m.removes)
    {
        if (!valid(r.first, r.second))
        {
            w.rejectInput(from, "ChunkInterest: a chunk outside the space or twice in one message");
            return;
        }
        if (p.subs.count(chunkIdOf(m.space, r.first, r.second))) newCount--;
    }
    for (const ChunkInterestMessage::Add& a : m.adds)
    {
        if (!valid(a.x, a.z))
        {
            w.rejectInput(from, "ChunkInterest: a chunk outside the space or twice in one message");
            return;
        }
        if (!allSpace && !p.subs.count(chunkIdOf(m.space, a.x, a.z))) newCount++;
    }
    if (newCount > w.worldConfig().network.maxChunksPerPeer)
    {
        w.rejectInput(from, "ChunkInterest: more than " + std::to_string(w.worldConfig().network.maxChunksPerPeer) +
                                " subscribed chunks");
        return;
    }
    for (const auto& r : m.removes)
    {
        const ChunkId c = chunkIdOf(m.space, r.first, r.second);
        p.subs.erase(c);
        p.queued.erase(c);
        statistics.removes++;
    }
    for (const ChunkInterestMessage::Add& a : m.adds)
    {
        const ChunkId c = chunkIdOf(m.space, a.x, a.z);
        if (!allSpace) p.subs[c] = a.ticket;
        enqueue(from, c, a.ticket);
        statistics.adds++;
    }
}

std::vector<ChunkId> HostChunkSubscriptions::chunksWithState(const World& world, SpaceId space)
{
    std::vector<ChunkId> out;
    for (const auto& kv : world.terrainEdits().chunks(space))
        if (kv.second.sequence > 0) out.push_back(kv.first);
    const ChunkRecords& records = world.contentRecords();
    ChunkId first;
    first.space = space;
    first.x = first.z = std::numeric_limits<int32_t>::min();
    for (auto it = records.records().lower_bound(first); it != records.records().end() && it->first.space == space;
         ++it)
        if (recordPresent(world, it->first, it->second)) out.push_back(it->first);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void HostChunkSubscriptions::snapshotOf(const World& world, uint32_t epoch, uint32_t ticket, const ChunkId& chunk,
                                        ChunkSnapshotMessage& out)
{
    out                        = ChunkSnapshotMessage();
    out.epoch                  = epoch;
    out.ticket                 = ticket;
    out.chunk                  = chunk;
    const TerrainEditStore& st = world.terrainEdits();
    out.sequence               = st.sequence(chunk);
    const bool openWorld       = world.spaceTemplate(chunk.space).kind == SpaceKind::OpenWorld;
    out.N                      = openWorld ? uint16_t(st.chunkSamples(chunk.space) + 1) : 0;
    if (openWorld && st.hasDeltas(chunk))
    {
        out.hasDeltas = true;
        st.denseDelta(chunk, out.deltas);
    }
    if (const ChunkRecord* r = world.contentRecords().find(chunk))
        if (recordPresent(world, chunk, *r))
        {
            out.hasRecord        = true;
            out.placementVersion = r->placementVersion;
            out.keys             = r->tombstones;
        }
}

size_t HostChunkSubscriptions::serve(uint32_t epoch)
{
    // The indexes of new `all` subscriptions (not budgeted); their snapshots are queued when the step commits
    indexed.clear();
    indexesStaged = pendingIndex.size();
    for (const auto& [peer, space] : pendingIndex)
    {
        if (!subscribedAll(peer, space)) continue;
        ChunkIndexMessage index;
        index.epoch                     = epoch;
        index.space                     = space;
        const std::vector<ChunkId> list = chunksWithState(w, space);
        for (const ChunkId& c : list) index.chunks.push_back({c.x, c.z});
        bytes.clear();
        encodeChunkIndex(index, bytes);
        w.sendRaw(peer, MessageId(EngineMessage::ChunkIndex), bytes.data(), bytes.size());
        statistics.indexesSent++;
        indexed.push_back({peer, list});
    }
    // The snapshot queue: FIFO under the budget, at least one per step
    const size_t budget = w.worldConfig().network.chunkSnapshotBytesPerStep;
    size_t total        = 0;
    walked              = 0;
    served.clear();
    // The (peer, chunk) pairs served in this call (a set: the queue can hold thousands of entries per step)
    std::set<std::pair<uint8_t, ChunkId>> servedPairs;
    ChunkSnapshotMessage snapshot;
    for (size_t i = 0; i < queue.size(); ++i)
    {
        if (!served.empty() && total >= budget) break;
        const auto& [peer, chunk] = queue[i];
        walked++;
        auto pt = table.find(peer);
        if (pt == table.end()) continue;
        auto qt = pt->second.queued.find(chunk);
        if (qt == pt->second.queued.end()) continue;
        if (servedPairs.count({peer, chunk})) continue;
        snapshotOf(w, epoch, qt->second, chunk, snapshot);
        bytes.clear();
        encodeChunkSnapshot(snapshot, bytes);
        // A snapshot that would exceed the budget waits for the next step (it stays outside the walked prefix); the
        // first one of a step is always sent, so an oversized snapshot cannot block the queue
        if (!served.empty() && total + bytes.size() > budget)
        {
            walked--;
            break;
        }
        servedPairs.insert({peer, chunk});
        w.sendRaw(peer, MessageId(EngineMessage::ChunkSnapshot), bytes.data(), bytes.size());
        total += bytes.size();
        served.push_back({{peer, chunk}, qt->second});
    }
    servedBytes = total;
    return total;
}

void HostChunkSubscriptions::commit()
{
    for (size_t i = 0; i < walked && !queue.empty(); ++i) queue.pop_front();
    for (const auto& s : served)
    {
        auto pt = table.find(s.first.first);
        if (pt == table.end()) continue;
        auto qt = pt->second.queued.find(s.first.second);
        if (qt != pt->second.queued.end() && qt->second == s.second) pt->second.queued.erase(qt);
        statistics.snapshotsSent++;
    }
    statistics.snapshotBytes += servedBytes;
    servedBytes = 0;
    pendingIndex.erase(pendingIndex.begin(), pendingIndex.begin() + std::ptrdiff_t(indexesStaged));
    for (const auto& [peer, list] : indexed)
        for (const ChunkId& c : list)
            if (subscribedAll(peer, c.space)) enqueue(peer, c, 0);
    walked        = 0;
    indexesStaged = 0;
    served.clear();
    indexed.clear();
}

void HostChunkSubscriptions::abort()
{
    servedBytes   = 0;
    walked        = 0;
    indexesStaged = 0;
    served.clear();
    indexed.clear();
}

void HostChunkSubscriptions::clear()
{
    table.clear();
    queue.clear();
    pendingIndex.clear();
    abort();
}

void HostChunkSubscriptions::removePeer(uint8_t peer)
{
    table.erase(peer);
    pendingIndex.erase(std::remove_if(pendingIndex.begin(), pendingIndex.end(),
                                      [&](const std::pair<uint8_t, SpaceId>& x) { return x.first == peer; }),
                       pendingIndex.end());
}

}  // namespace Engine
}  // namespace Saiga
