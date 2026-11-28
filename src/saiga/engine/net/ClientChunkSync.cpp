/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/ClientChunkSync.h"

#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/ChunkMessages.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainMessages.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Saiga
{
namespace Engine
{
namespace
{
ChunkId chunkIdOf(SpaceId space, int32_t x, int32_t z)
{
    ChunkId c;
    c.space = space;
    c.x     = x;
    c.z     = z;
    return c;
}

SampleRect chunkBox(const TerrainEditStore& st, const ChunkId& c)
{
    const int64_t S = st.chunkSamples(c.space);
    SampleRect r;
    r.i0 = int64_t(c.x) * S;
    r.k0 = int64_t(c.z) * S;
    r.i1 = r.i0 + S;
    r.k1 = r.k0 + S;
    return r;
}

// The keys of one chunk against the client's generation (slots and counters in range)
bool checkKeys(const World& w, const SpaceGeneration& g, const std::vector<InstanceKey>& keys, std::string& error)
{
    if (keys.empty()) return true;
    const size_t slots = w.generators().find(g.generatorId)->slots.size();
    if (slots == 0)
    {
        error = "keys in space " + std::to_string(g.id) + " whose generator has no slots";
        return false;
    }
    for (const InstanceKey& k : keys)
    {
        if (k.generatorSlot >= slots)
        {
            error =
                "slot " + std::to_string(k.generatorSlot) + " beyond the slot table (" + std::to_string(slots) + ")";
            return false;
        }
        if (k.counter >= g.generator->candidatesPerChunk(k.generatorSlot))
        {
            error = "counter " + std::to_string(k.counter) + " of slot " + std::to_string(k.generatorSlot) +
                    " out of range";
            return false;
        }
    }
    return true;
}
}  // namespace

ClientChunkSync::ClientChunkSync(World& world) : w(world), spaces(world.spaceCount(), SpaceSync::None) {}

// ---- queries --------------------------------------------------------------------------------------------------------

ClientChunkSync::State ClientChunkSync::state(const ChunkId& chunk) const
{
    auto it = table.find(chunk);
    if (it != table.end()) return it->second.state;
    if (chunk.space >= spaces.size()) return State::None;
    if (spaces[chunk.space] == SpaceSync::Indexed) return State::Synced;
    if (spaces[chunk.space] == SpaceSync::AllPending) return State::Pending;
    return State::None;
}

uint32_t ClientChunkSync::ticket(const ChunkId& chunk) const
{
    auto it = table.find(chunk);
    return it == table.end() ? 0 : it->second.ticket;
}

size_t ClientChunkSync::queuedParts(const ChunkId& chunk) const
{
    auto it = table.find(chunk);
    return it == table.end() ? 0 : it->second.queue.size();
}

size_t ClientChunkSync::queuedParts() const
{
    size_t n = 0;
    for (const auto& kv : table) n += kv.second.queue.size();
    return n;
}

size_t ClientChunkSync::pendingCount() const
{
    size_t n = 0;
    for (const auto& kv : table) n += kv.second.state == State::Pending ? 1 : 0;
    return n;
}

size_t ClientChunkSync::syncedCount() const
{
    size_t n = 0;
    for (const auto& kv : table) n += kv.second.state == State::Synced ? 1 : 0;
    return n;
}

bool ClientChunkSync::spaceIndexed(SpaceId space) const
{
    return space < spaces.size() && spaces[space] == SpaceSync::Indexed;
}

bool ClientChunkSync::allIndexed() const
{
    for (SpaceSync s : spaces)
        if (s != SpaceSync::Indexed) return false;
    return true;
}

void ClientChunkSync::hold(const ChunkId& chunk)
{
    if (!w.streaming()) return;
    // The host rejects a whole ChunkInterest for one chunk outside its space: a bad hold must not poison the others
    SAIGA_ASSERT(chunk.space < w.spaceCount() && chunk.x >= 0 && chunk.z >= 0 &&
                     chunk.x < w.spaceTemplate(chunk.space).sizeChunks &&
                     chunk.z < w.spaceTemplate(chunk.space).sizeChunks,
                 "ClientChunkSync::hold: the chunk is outside its space");
    held[chunk]++;
}

void ClientChunkSync::release(const ChunkId& chunk)
{
    auto it = held.find(chunk);
    if (it == held.end()) return;
    if (--it->second <= 0) held.erase(it);
}

// ---- helpers --------------------------------------------------------------------------------------------------------

uint32_t ClientChunkSync::newTicket()
{
    if (nextTicket == 0) nextTicket = 1;
    return nextTicket++;
}

void ClientChunkSync::reject(uint8_t from, const std::string& why, bool content)
{
    w.rejectInput(from, why);
    if (content) SAIGA_LOG("content", "rejected a chunk message from peer %d: %s", int(from), why.c_str());
}

void ClientChunkSync::rerequest(const ChunkId& chunk, const char* why)
{
    Entry& e        = table[chunk];
    e.state         = State::Pending;
    e.ticket        = newTicket();
    e.awaitingIndex = false;
    takeQueue(e);
    unsentAdds[chunk] = e.ticket;
    statistics.rerequests++;
    w.netCounts.chunkRerequests++;
    SAIGA_LOG("net", "chunk (%d, %d) of space %d re-requested (%s), ticket %u", int(chunk.x), int(chunk.z),
              int(chunk.space), why, e.ticket);
}

void ClientChunkSync::enqueue(Entry& e, const ChunkId& chunk, Queued q)
{
    const size_t bytes = sizeof(Queued) + q.part.deltas.size() * sizeof(int32_t);
    if (e.queue.size() >= kMaxQueuedPartsPerChunk || e.queuedBytes + bytes > kMaxQueuedBytesPerChunk ||
        queuedBytesTotal + bytes > kMaxQueuedBytesTotal)
    {
        // Dropped with the queue: the new snapshot covers every part sent before the new subscription
        statistics.queueOverflows++;
        rerequest(chunk, "queue overflow");
        return;
    }
    e.queue.push_back(std::move(q));
    e.queuedBytes += bytes;
    queuedBytesTotal += bytes;
    statistics.partsQueued++;
}

std::vector<ClientChunkSync::Queued> ClientChunkSync::takeQueue(Entry& e)
{
    queuedBytesTotal -= std::min(queuedBytesTotal, e.queuedBytes);
    e.queuedBytes = 0;
    std::vector<Queued> q;
    q.swap(e.queue);
    return q;
}

ClientChunkSync::Entry* ClientChunkSync::pendingEntry(const ChunkId& chunk)
{
    auto it = table.find(chunk);
    if (it != table.end()) return it->second.state == State::Pending ? &it->second : nullptr;
    if (chunk.space < spaces.size() && spaces[chunk.space] == SpaceSync::AllPending)
    {
        Entry& e        = table[chunk];
        e.state         = State::Pending;
        e.awaitingIndex = true;
        return &e;
    }
    return nullptr;
}

void ClientChunkSync::applyParts(SpaceId space, const std::vector<const TerrainPart*>& apply)
{
    if (apply.empty()) return;
    TerrainEditStore& st = w.terrainEditsForUpdate();
    const int64_t S      = st.chunkSamples(space);
    // Reserve: the blocks of non-zero changed deltas, the entries, the change-list slot
    std::vector<TerrainEditStore::EntryNode> entries;
    for (const TerrainPart* p : apply)
    {
        for (int z = 0; z < p->h; ++z)
            for (int x = 0; x < p->w; ++x)
                st.reserveFor(space, int64_t(p->chunk.x) * S + p->x0 + x, int64_t(p->chunk.z) * S + p->z0 + z,
                              p->deltas[size_t(z * p->w + x)]);
        if (!st.entry(p->chunk)) entries.push_back(st.makeEntry(p->chunk));
    }
    w.reserveTerrainChange();
    for (TerrainEditStore::EntryNode& e : entries) st.insertEntry(std::move(e));
    SampleRect box;
    box.i0 = box.k0 = std::numeric_limits<int64_t>::max();
    box.i1 = box.k1 = std::numeric_limits<int64_t>::min();
    for (const TerrainPart* p : apply)
    {
        const int64_t gi = int64_t(p->chunk.x) * S + p->x0, gk = int64_t(p->chunk.z) * S + p->z0;
        for (int z = 0; z < p->h; ++z)
            for (int x = 0; x < p->w; ++x) st.writeDelta(space, gi + x, gk + z, p->deltas[size_t(z * p->w + x)]);
        st.setSequence(p->chunk, p->sequence);
        box.i0 = std::min(box.i0, gi);
        box.k0 = std::min(box.k0, gk);
        box.i1 = std::max(box.i1, gi + p->w - 1);
        box.k1 = std::max(box.k1, gk + p->h - 1);
        statistics.partsApplied++;
    }
    w.onTerrainChanged(space, box);
    w.replication().counts.terrainMessagesApplied++;
}

bool ClientChunkSync::applyTombstone(SpaceId space, const TombstoneEntry& e)
{
    const ChunkId c = chunkIdOf(space, e.x, e.z);
    if (e.placementVersion != w.effectivePlacementVersion(c)) return false;
    TerrainEditStore& st = w.terrainEditsForUpdate();
    ChunkRecords& rec    = w.contentRecordsForUpdate();
    if (!st.entry(c)) st.insertEntry(st.makeEntry(c));
    rec.reserve(c, e.placementVersion, 1);
    rec.insert(c, e.key);
    st.setSequence(c, e.sequence);
    w.replication().counts.tombstonesApplied++;
    w.onRecordChanged(c);
    return true;
}

bool ClientChunkSync::drainQueue(const ChunkId& chunk, std::vector<Queued> queue)
{
    const TerrainEditStore& st = w.terrainEdits();
    for (size_t i = 0; i < queue.size(); ++i)
    {
        Queued& q              = queue[i];
        const uint32_t current = st.sequence(chunk);
        if (q.sequence <= current)
        {
            statistics.partsDroppedOld++;
            continue;
        }
        if (q.sequence > current + 1)
        {
            SAIGA_LOG("net", "a gap in the queue of chunk (%d, %d) of space %d: sequence %u after %u", int(chunk.x),
                      int(chunk.z), int(chunk.space), q.sequence, current);
            w.netCounts.modificationGaps++;
            // The rest of the queue goes: the missing sequence never arrives (one ordered channel), the new snapshot
            // covers every part sent before the host received the new subscription
            rerequest(chunk, "a gap");
            return false;
        }
        if (q.tombstone)
        {
            if (!applyTombstone(chunk.space, q.entry))
            {
                reject(0, "TombstoneModification: a queued entry of another placement version", true);
                rerequest(chunk, "a queued entry of another placement version");
                return false;
            }
        }
        else
            applyParts(chunk.space, {&q.part});
    }
    return true;
}

// ---- snapshots and indexes ------------------------------------------------------------------------------------------

void ClientChunkSync::onSnapshot(uint8_t from, const uint8_t* data, size_t size)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "ChunkSnapshot: only the host sends it to clients");
        return;
    }
    ChunkSnapshotMessage m;
    std::string error;
    if (!decodeChunkSnapshot(data, size, m, error))
    {
        reject(from, error, true);
        return;
    }
    // Another epoch: dropped (not counted, D11)
    if (m.epoch != w.replication().epoch()) return;
    const ChunkId c = m.chunk;
    // Validation against the world, in scratch (nothing applies on a failure)
    if (c.space >= w.spaceCount())
    {
        reject(from, "ChunkSnapshot: unknown space " + std::to_string(c.space), true);
        return;
    }
    const SpaceGeneration& g    = w.generation().space(c.space);
    const bool openWorld        = w.spaceTemplate(c.space).kind == SpaceKind::OpenWorld;
    const TerrainEditStore& st0 = w.terrainEdits();
    if (c.x < 0 || c.z < 0 || c.x >= g.settings.sizeChunks || c.z >= g.settings.sizeChunks)
    {
        reject(from, "ChunkSnapshot: a chunk outside its space", true);
        return;
    }
    if (m.N != (openWorld ? uint16_t(st0.chunkSamples(c.space) + 1) : uint16_t(0)))
    {
        reject(from, "ChunkSnapshot: N " + std::to_string(m.N) + " does not match the space", true);
        return;
    }
    // A terrain sequence only in an open-world space (an enclosed one has no terrain store; TERR would refuse the
    // entry on save), and deltas only with a sequence (no samples without an entry)
    if ((!openWorld && m.sequence > 0) || (m.hasDeltas && m.sequence == 0))
    {
        reject(from, "ChunkSnapshot: a terrain sequence or deltas the chunk's space cannot have", true);
        return;
    }
    if (m.hasRecord)
    {
        if (m.placementVersion < g.pinned || m.placementVersion > g.current)
        {
            reject(from,
                   "ChunkSnapshot: placement version " + std::to_string(m.placementVersion) + " not in v" +
                       std::to_string(g.pinned) + "..v" + std::to_string(g.current),
                   true);
            return;
        }
        if (!checkKeys(w, g, m.keys, error))
        {
            reject(from, "ChunkSnapshot: " + error, true);
            return;
        }
    }
    // Only the snapshot of the current subscription (D7)
    auto it = table.find(c);
    if (it == table.end() || it->second.state != State::Pending || it->second.awaitingIndex ||
        it->second.ticket != m.ticket)
    {
        statistics.snapshotsStale++;
        return;
    }
    // Apply: the closed range, the sequence, the record
    TerrainEditStore& st     = w.terrainEditsForUpdate();
    ChunkRecords& records    = w.contentRecordsForUpdate();
    const ChunkRecord before = w.effectiveRecord(c);
    const bool hadDeltas     = openWorld && st.hasDeltas(c);
    if (openWorld)
    {
        const int64_t N = m.N, i0 = int64_t(c.x) * (N - 1), k0 = int64_t(c.z) * (N - 1);
        auto value = [&](int64_t i, int64_t k) { return m.hasDeltas ? m.deltas[size_t(k * N + i)] : 0; };
        for (int64_t k = 0; k < N; ++k)
            for (int64_t i = 0; i < N; ++i) st.reserveFor(c.space, i0 + i, k0 + k, value(i, k));
        if (!st.entry(c) && m.sequence > 0) st.insertEntry(st.makeEntry(c));
        w.reserveTerrainChange();
        for (int64_t k = 0; k < N; ++k)
            for (int64_t i = 0; i < N; ++i) st.writeDelta(c.space, i0 + i, k0 + k, value(i, k));
    }
    if (m.sequence > 0)
        st.setSequence(c, m.sequence);
    else
        st.removeEntry(c);
    if (m.hasRecord)
        records.set(c, m.placementVersion, std::move(m.keys));
    else
        records.erase(c);
    if (hadDeltas || (openWorld && st.hasDeltas(c))) w.onTerrainChanged(c.space, chunkBox(st, c));
    if (w.effectiveRecord(c) != before) w.onRecordChanged(c);
    statistics.snapshotsApplied++;
    // The queue (D9), then Synced
    std::vector<Queued> queue;
    queue = takeQueue(it->second);
    if (!drainQueue(c, std::move(queue))) return;
    auto now = table.find(c);
    if (now == table.end()) return;
    now->second.state = State::Synced;
    // A chunk of an indexed space needs no entry once Synced
    if (spaces[c.space] == SpaceSync::Indexed && !w.streaming()) table.erase(now);
}

void ClientChunkSync::onIndex(uint8_t from, const uint8_t* data, size_t size)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "ChunkIndex: only the host sends it to clients");
        return;
    }
    ChunkIndexMessage m;
    std::string error;
    if (!decodeChunkIndex(data, size, m, error))
    {
        reject(from, error, true);
        return;
    }
    if (m.epoch != w.replication().epoch()) return;
    if (m.space >= w.spaceCount())
    {
        reject(from, "ChunkIndex: unknown space " + std::to_string(m.space), true);
        return;
    }
    const int32_t n = w.spaceTemplate(m.space).sizeChunks;
    if (m.chunks.size() > size_t(n) * size_t(n))
    {
        reject(from, "ChunkIndex: more chunks than the space has", true);
        return;
    }
    for (const auto& xz : m.chunks)
        if (xz.first < 0 || xz.second < 0 || xz.first >= n || xz.second >= n)
        {
            reject(from, "ChunkIndex: a chunk outside its space", true);
            return;
        }
    if (spaces[m.space] != SpaceSync::AllPending)
    {
        statistics.snapshotsStale++;
        return;
    }
    // The listed chunks wait for their snapshots (ticket 0); every other chunk is Synced at sequence 0 and its queued
    // parts apply from sequence 1
    spaces[m.space] = SpaceSync::Indexed;
    statistics.indexesApplied++;
    std::vector<ChunkId> listed;
    for (const auto& xz : m.chunks) listed.push_back(chunkIdOf(m.space, xz.first, xz.second));
    for (const ChunkId& c : listed)
    {
        Entry& e = table[c];
        // A chunk re-requested while the space waited for its index (a gap or a queue overflow) keeps its own ticket:
        // the host's only snapshot of it answers that request
        if (e.state == State::Pending && e.ticket != 0)
        {
            e.awaitingIndex = false;
            continue;
        }
        e.state         = State::Pending;
        e.ticket        = 0;
        e.awaitingIndex = false;
    }
    std::vector<ChunkId> unlisted;
    ChunkId first = chunkIdOf(m.space, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min());
    for (auto it = table.lower_bound(first); it != table.end() && it->first.space == m.space; ++it)
        if (it->second.awaitingIndex) unlisted.push_back(it->first);
    for (const ChunkId& c : unlisted)
    {
        std::vector<Queued> queue;
        queue                  = takeQueue(table[c]);
        table[c].awaitingIndex = false;
        if (drainQueue(c, std::move(queue))) table.erase(c);
    }
}

// ---- live modifications ---------------------------------------------------------------------------------------------

void ClientChunkSync::onTerrainModification(uint8_t from, const uint8_t* data, size_t size)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "TerrainModification: only the host sends it to clients");
        return;
    }
    TerrainModificationMessage m;
    std::string error;
    if (!decodeTerrainModification(data, size, m, error))
    {
        w.rejectInput(from, error);
        return;
    }
    if (m.epoch != w.replication().epoch())
    {
        w.replication().counts.terrainDroppedEpoch++;
        return;
    }
    // Validation against the world, in scratch: nothing changes before every part passed
    if (m.space >= w.spaceCount() || w.spaceTemplate(m.space).kind != SpaceKind::OpenWorld)
    {
        w.rejectInput(from, "TerrainModification: unknown or enclosed space " + std::to_string(m.space));
        return;
    }
    const TerrainEditStore& st = w.terrainEdits();
    const int64_t S = st.chunkSamples(m.space), N = S + 1;
    std::map<std::pair<int64_t, int64_t>, int32_t> shared;
    std::set<ChunkId> partChunks;
    for (const TerrainPart& p : m.parts)
    {
        if (p.chunk.x < 0 || p.chunk.z < 0 || p.chunk.x >= st.sizeChunks(m.space) ||
            p.chunk.z >= st.sizeChunks(m.space) || p.x0 + p.w > N || p.z0 + p.h > N || p.sequence == 0)
        {
            w.rejectInput(from, "TerrainModification: a part outside the space or its tile, or sequence 0");
            return;
        }
        // One part per chunk (PR review of S22: two parts of a new chunk could move its sequence backward)
        if (!partChunks.insert(p.chunk).second)
        {
            w.rejectInput(from, "TerrainModification: two parts of one chunk");
            return;
        }
        // Parts of one message that share samples (chunk borders) carry equal values there
        for (int z = 0; z < p.h; ++z)
            for (int x = 0; x < p.w; ++x)
            {
                const int64_t li = p.x0 + x, lk = p.z0 + z;
                if (li != 0 && li != S && lk != 0 && lk != S) continue;
                const auto key      = std::make_pair(int64_t(p.chunk.x) * S + li, int64_t(p.chunk.z) * S + lk);
                const int32_t v     = p.deltas[size_t(z * p.w + x)];
                auto [it, inserted] = shared.emplace(key, v);
                if (!inserted && it->second != v)
                {
                    w.rejectInput(from, "TerrainModification: parts disagree on a shared sample");
                    return;
                }
            }
    }
    // Per part by the chunk's state (D9)
    std::vector<const TerrainPart*> apply;
    for (TerrainPart& p : m.parts)
    {
        const State s = state(p.chunk);
        if (s == State::None)
        {
            statistics.partsDroppedNotSubscribed++;
            continue;
        }
        if (s == State::Pending)
        {
            Entry* e = pendingEntry(p.chunk);
            Queued q;
            q.sequence = p.sequence;
            q.part     = std::move(p);
            enqueue(*e, q.part.chunk, std::move(q));
            continue;
        }
        const uint32_t current = st.sequence(p.chunk);
        if (p.sequence <= current)
        {
            w.replication().counts.terrainPartsDuplicate++;
            continue;
        }
        if (p.sequence > current + 1)
        {
            SAIGA_LOG("net", "TerrainModification: a gap in chunk (%d, %d) of space %d: sequence %u after %u",
                      int(p.chunk.x), int(p.chunk.z), int(m.space), p.sequence, current);
            w.netCounts.modificationGaps++;
            // Not applied, not queued: the new snapshot includes it (it was sent before the new subscription)
            rerequest(p.chunk, "a gap");
            continue;
        }
        apply.push_back(&p);
    }
    applyParts(m.space, apply);
}

void ClientChunkSync::onTombstoneModification(uint8_t from, const uint8_t* data, size_t size)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "TombstoneModification: only the host sends it to clients");
        return;
    }
    TombstoneModificationMessage m;
    std::string error;
    if (!decodeTombstoneModification(data, size, m, error))
    {
        reject(from, error, true);
        return;
    }
    if (m.epoch != w.replication().epoch())
    {
        w.replication().counts.recordsDroppedEpoch++;
        return;
    }
    // An open-world space only (as TerrainModification): a tombstone bumps the chunk's terrain sequence, which an
    // enclosed space cannot hold
    if (m.space >= w.spaceCount() || w.spaceTemplate(m.space).kind != SpaceKind::OpenWorld)
    {
        reject(from, "TombstoneModification: not an open-world space " + std::to_string(m.space), true);
        return;
    }
    // Validation of every entry before anything applies (the placement version of Synced chunks; a queued entry is
    // checked when it applies)
    const SpaceGeneration& g = w.generation().space(m.space);
    for (const TombstoneEntry& e : m.entries)
    {
        const std::string where = "TombstoneModification: chunk (" + std::to_string(e.x) + ", " + std::to_string(e.z) +
                                  ") of space " + std::to_string(m.space);
        if (e.x < 0 || e.z < 0 || e.x >= g.settings.sizeChunks || e.z >= g.settings.sizeChunks || e.sequence == 0)
        {
            reject(from, where + ": outside the space or sequence 0", true);
            return;
        }
        const ChunkId c = chunkIdOf(m.space, e.x, e.z);
        if (state(c) == State::Synced && e.placementVersion != w.effectivePlacementVersion(c))
        {
            reject(from,
                   where + ": placement version " + std::to_string(e.placementVersion) +
                       ", the chunk's effective version " + std::to_string(w.effectivePlacementVersion(c)),
                   true);
            return;
        }
        if (!checkKeys(w, g, {e.key}, error))
        {
            reject(from, where + ": " + error, true);
            return;
        }
    }
    // Per entry in order by the chunk's state
    const TerrainEditStore& st = w.terrainEdits();
    for (const TombstoneEntry& e : m.entries)
    {
        const ChunkId c = chunkIdOf(m.space, e.x, e.z);
        const State s   = state(c);
        if (s == State::None)
        {
            statistics.partsDroppedNotSubscribed++;
            continue;
        }
        if (s == State::Pending)
        {
            Entry* pe = pendingEntry(c);
            Queued q;
            q.tombstone = true;
            q.sequence  = e.sequence;
            q.entry     = e;
            enqueue(*pe, c, std::move(q));
            continue;
        }
        const uint32_t current = st.sequence(c);
        if (e.sequence <= current)
        {
            w.replication().counts.tombstonesDuplicate++;
            continue;
        }
        if (e.sequence > current + 1)
        {
            SAIGA_LOG("net", "TombstoneModification: a gap in chunk (%d, %d) of space %d: sequence %u after %u",
                      int(e.x), int(e.z), int(m.space), e.sequence, current);
            w.netCounts.modificationGaps++;
            rerequest(c, "a gap");
            continue;
        }
        applyTombstone(m.space, e);
    }
    w.replication().counts.recordMessagesApplied++;
}

// ---- interest (phase 8) ---------------------------------------------------------------------------------------------

void ClientChunkSync::extract()
{
    stagedAdds.clear();
    stagedRemoves.clear();
    stagedAll.clear();
    if (w.isHost() || w.joinState() != JoinState::Joined) return;
    const uint32_t epoch = w.replication().epoch();
    auto send            = [&](const ChunkInterestMessage& m)
    {
        bytes.clear();
        encodeChunkInterest(m, bytes);
        w.sendRaw(0, MessageId(EngineMessage::ChunkInterest), bytes.data(), bytes.size());
        statistics.interestMessages++;
    };
    // The subscriptions after this diff (all spaces; the host's maxChunksPerPeer counts them per peer)
    size_t total = table.size();
    for (size_t si = 0; si < w.spaceCount(); ++si)
    {
        const SpaceId space = SpaceId(si);
        std::vector<ChunkId> removes;
        std::vector<StagedAdd> adds;
        if (!w.streaming())
        {
            if (unsentAll.count(space))
            {
                ChunkInterestMessage m;
                m.all   = true;
                m.epoch = epoch;
                m.space = space;
                send(m);
                stagedAll.push_back(space);
            }
        }
        else
        {
            // The wanted set: the residency's Requested and Resident chunks and the holds (D1, S24 interface 3)
            std::set<ChunkId> wanted;
            w.residency().forEachChunk(space, [&](const ChunkId& c, ChunkState) { wanted.insert(c); });
            ChunkId first = chunkIdOf(space, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min());
            for (auto it = held.lower_bound(first); it != held.end() && it->first.space == space; ++it)
                wanted.insert(it->first);
            for (auto it = table.lower_bound(first); it != table.end() && it->first.space == space; ++it)
                if (!wanted.count(it->first)) removes.push_back(it->first);
            for (const ChunkId& c : wanted)
                if (!table.count(c)) adds.push_back(StagedAdd{c, newTicket(), true});
        }
        // Re-requests (also inside an `all` space)
        ChunkId first = chunkIdOf(space, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min());
        for (auto it = unsentAdds.lower_bound(first); it != unsentAdds.end() && it->first.space == space; ++it)
            if (std::find(removes.begin(), removes.end(), it->first) == removes.end())
                adds.push_back(StagedAdd{it->first, it->second, false});
        if (removes.empty() && adds.empty()) continue;
        // Nearest first (the host's FIFO serves the chunk under a held player first), then ChunkId order
        std::vector<Vec3> centres;
        for (const StreamingCentre& c : w.residency().centres())
            if (c.space == space) centres.push_back(c.position);
        auto distance = [&](const ChunkId& c)
        {
            double best = std::numeric_limits<double>::infinity();
            for (const Vec3& p : centres)
                best = std::min(best, std::hypot(double(c.x) * 256 + 128 - p.x(), double(c.z) * 256 + 128 - p.z()));
            return best;
        };
        std::stable_sort(adds.begin(), adds.end(),
                         [&](const StagedAdd& a, const StagedAdd& b) { return distance(a.chunk) < distance(b.chunk); });
        // The host rejects a whole ChunkInterest above maxChunksPerPeer (in the network schema, so both sides agree):
        // the client never asks for more; the farthest new chunks wait (they stay Requested) until others are removed
        const size_t limit = w.worldConfig().network.maxChunksPerPeer;
        total -= std::min(total, removes.size());
        std::vector<StagedAdd> kept;
        for (const StagedAdd& add : adds)
            if (!add.fresh)
                kept.push_back(add);  // a re-request: already subscribed
            else if (total < limit)
            {
                kept.push_back(add);
                total++;
            }
        if (kept.size() < adds.size())
        {
            // The limit must cover the chunks of the kept discs (a setup error otherwise): logged once
            if (statistics.interestCapped++ == 0)
                SAIGA_LOG("net",
                          "the chunk subscriptions reached maxChunksPerPeer (%zu): chunks wait unsubscribed; raise the "
                          "limit above the chunks within the kept distance of every centre",
                          limit);
        }
        adds.swap(kept);
        if (removes.empty() && adds.empty()) continue;
        // Messages of at most kMaxInterestEntries entries: removes first
        size_t r = 0, a = 0;
        while (r < removes.size() || a < adds.size())
        {
            ChunkInterestMessage m;
            m.epoch = epoch;
            m.space = space;
            while (r < removes.size() && m.removes.size() < kMaxInterestEntries)
            {
                m.removes.push_back({removes[r].x, removes[r].z});
                r++;
            }
            while (a < adds.size() && m.removes.size() + m.adds.size() < kMaxInterestEntries)
            {
                m.adds.push_back(ChunkInterestMessage::Add{adds[a].chunk.x, adds[a].chunk.z, adds[a].ticket});
                a++;
            }
            send(m);
        }
        stagedRemoves.insert(stagedRemoves.end(), removes.begin(), removes.end());
        stagedAdds.insert(stagedAdds.end(), adds.begin(), adds.end());
    }
}

void ClientChunkSync::commit()
{
    for (const StagedAdd& a : stagedAdds)
    {
        if (a.fresh)
        {
            Entry& e = table[a.chunk];
            e.state  = State::Pending;
            e.ticket = a.ticket;
            statistics.subscribed++;
        }
        auto it = unsentAdds.find(a.chunk);
        if (it != unsentAdds.end() && it->second == a.ticket) unsentAdds.erase(it);
    }
    for (SpaceId s : stagedAll)
    {
        unsentAll.erase(s);
        if (spaces[s] == SpaceSync::None) spaces[s] = SpaceSync::AllPending;
    }
    for (const ChunkId& c : stagedRemoves)
    {
        unsentAdds.erase(c);
        drop(c);
    }
    stagedAdds.clear();
    stagedRemoves.clear();
    stagedAll.clear();
}

void ClientChunkSync::abort()
{
    stagedAdds.clear();
    stagedRemoves.clear();
    stagedAll.clear();
}

void ClientChunkSync::drop(const ChunkId& c)
{
    if (auto it = table.find(c); it != table.end()) takeQueue(it->second);
    table.erase(c);
    statistics.unsubscribed++;
    TerrainEditStore& st  = w.terrainEditsForUpdate();
    ChunkRecords& records = w.contentRecordsForUpdate();
    const bool openWorld  = w.spaceTemplate(c.space).kind == SpaceKind::OpenWorld;
    const bool hadDeltas  = openWorld && st.hasDeltas(c);
    const bool hadRecord  = records.find(c) != nullptr;
    // The shared border with another subscribed chunk (Pending or Synced) stays
    st.dropChunk(c, [&](const ChunkId& n) { return table.count(n) != 0; });
    records.erase(c);
    if (hadDeltas)
    {
        w.reserveTerrainChange();
        w.onTerrainChanged(c.space, chunkBox(st, c));
    }
    if (hadRecord) w.onRecordChanged(c);
}

void ClientChunkSync::onJoined()
{
    if (w.streaming()) return;
    for (size_t s = 0; s < w.spaceCount(); ++s) unsentAll.insert(SpaceId(s));
}

void ClientChunkSync::reset()
{
    table.clear();
    queuedBytesTotal = 0;
    std::fill(spaces.begin(), spaces.end(), SpaceSync::None);
    unsentAdds.clear();
    unsentAll.clear();
    abort();
    onJoined();
}

}  // namespace Engine
}  // namespace Saiga
