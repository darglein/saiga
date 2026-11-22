/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Terraforming over the network (docs/engine_design.md §13.2; S22): the TerrainModification and TerrainChunkState
// messages, the client's validation and application, sequences and gaps, the host state list over steps and aborts,
// late joiners and host loads, over LoopbackTransport.

#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/EngineMessages.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/net/Varint.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"
#include "saiga/engine/terrain/TerrainEdit.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainMessages.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace TerrainTestHooks;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

std::function<void(World&)> g_phase3, g_phase4;
void phase3System(World& w, double)
{
    if (g_phase3) g_phase3(w);
}
void phase4System(World& w, double)
{
    if (g_phase4) g_phase4(w);
}

WorldSetup terrainNetSetup()
{
    WorldSetup s;
    s.seed      = 3;
    s.profiling = false;
    s.config.spaceTemplates.push_back(terraformTestTemplate(2));
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Behaviour, "test_terrain_net3", phase3System, RunOn::Both);
        l.add(Phase::Gameplay, "test_terrain_net4", phase4System, RunOn::Both);
    };
    return s;
}

TerrainBrush brush(TerrainBrush::Op op, double x, double z, double radius, double strength, double y = 0)
{
    TerrainBrush b;
    b.op       = op;
    b.center   = Vec3(x, y, z);
    b.radius   = radius;
    b.strength = strength;
    return b;
}

ChunkId chunk(int32_t x, int32_t z)
{
    ChunkId c;
    c.x = x;
    c.z = z;
    return c;
}

// A transport that holds back every event of the inner one (the Connected events of the peers too) until release()
class HeldTransport : public Transport
{
   public:
    explicit HeldTransport(Transport& inner) : inner(inner) {}
    uint8_t localPeer() const override { return inner.localPeer(); }
    void send(uint8_t to, Channel channel, const std::vector<uint8_t>& packet, Reliability reliability) override
    {
        inner.send(to, channel, packet, reliability);
    }
    bool poll(TransportEvent& event) override
    {
        TransportEvent e;
        while (inner.poll(e)) held.push_back(std::move(e));
        if (holding || held.empty()) return false;
        event = std::move(held.front());
        held.pop_front();
        return true;
    }
    TransportStats stats() const override { return inner.stats(); }
    void release() { holding = false; }

   private:
    Transport& inner;
    bool holding = true;
    std::deque<TransportEvent> held;
};

// A host and clients over loopback; step() steps the host, then every client. holdHost: the host's events wait in a
// HeldTransport until hostTransport->release()
struct Net
{
    std::vector<std::unique_ptr<LoopbackTransport>> t;
    std::unique_ptr<HeldTransport> hostTransport;
    std::unique_ptr<World> host;
    std::vector<std::unique_ptr<World>> clients;
    explicit Net(int n = 1, int delay = 0, bool createClients = true, bool holdHost = false)
    {
        LoopbackOptions o;
        o.delayRounds = delay;
        t             = LoopbackTransport::createNetwork(1 + n, o);
        WorldSetup s  = terrainNetSetup();
        s.transport   = t[0].get();
        if (holdHost)
        {
            hostTransport = std::make_unique<HeldTransport>(*t[0]);
            s.transport   = hostTransport.get();
        }
        host = std::make_unique<World>(s);
        if (createClients)
            for (int i = 1; i <= n; ++i) addClient(i);
    }
    void addClient(int i)
    {
        WorldSetup s = terrainNetSetup();
        s.transport  = t[size_t(i)].get();
        s.role       = NetRole::Client;
        s.localPeer  = uint8_t(i);
        clients.push_back(std::make_unique<World>(s));
    }
    World& client(int i = 1) { return *clients[size_t(i - 1)]; }
    void step(int n = 1)
    {
        for (int k = 0; k < n; ++k)
        {
            host->step();
            for (auto& c : clients) c->step();
        }
    }
};

class TerrainNet : public ::testing::Test
{
   protected:
    void SetUp() override { g_phase3 = g_phase4 = nullptr; }
    void TearDown() override { g_phase3 = g_phase4 = nullptr; }
};

// The client's state equals the host's: the chunks with state, their sequences, dense deltas and counts
void expectSameState(const World& host, const World& client)
{
    const TerrainEditStore& a = host.terrainEdits();
    const TerrainEditStore& b = client.terrainEdits();
    ASSERT_EQ(a.chunkCount(), b.chunkCount());
    std::vector<int32_t> da, db;
    for (const auto& [c, e] : a.chunks(0))
    {
        const TerrainChunkEdits* f = b.entry(c);
        ASSERT_NE(f, nullptr) << c.x << ", " << c.z;
        EXPECT_EQ(e.sequence, f->sequence) << c.x << ", " << c.z;
        EXPECT_EQ(e.nonZeroSamples, f->nonZeroSamples) << c.x << ", " << c.z;
        a.denseDelta(c, da);
        b.denseDelta(c, db);
        EXPECT_EQ(da, db) << c.x << ", " << c.z;
    }
}

size_t pendingEvents(const World& w)
{
    return w.events.find(TerrainEditedEvent::id)->pendingCount();
}

std::vector<uint8_t> modification(uint32_t epoch, const std::vector<TerrainPart>& parts)
{
    std::vector<uint8_t> out;
    encodeTerrainModification(epoch, 0, parts, out);
    return out;
}

TerrainPart part(int32_t cx, int32_t cz, uint32_t sequence, int x0, int z0, int w, int h, int32_t value)
{
    TerrainPart p;
    p.chunk    = chunk(cx, cz);
    p.sequence = sequence;
    p.x0       = x0;
    p.z0       = z0;
    p.w        = w;
    p.h        = h;
    p.deltas.assign(size_t(w) * size_t(h), value);
    return p;
}
}  // namespace

// ---- encoding

TEST(TerrainMessages, RoundTrips)
{
    // Small (raw) and large (zstd) modifications
    for (int size : {3, 40})
    {
        std::vector<TerrainPart> parts   = {part(1, 2, 7, 120, 3, size, size, -5), part(2, 2, 3, 0, 3, 2, size, 9)};
        parts[0].deltas[1]               = kMaxTerrainDelta;
        const std::vector<uint8_t> bytes = modification(4, parts);
        EXPECT_EQ(bytes[0], size == 40 ? 1 : 0);
        TerrainModificationMessage m;
        std::string e;
        ASSERT_TRUE(decodeTerrainModification(bytes.data(), bytes.size(), m, e)) << e;
        EXPECT_EQ(m.epoch, 4u);
        ASSERT_EQ(m.parts.size(), 2u);
        for (size_t i = 0; i < 2; ++i)
        {
            EXPECT_EQ(m.parts[i].chunk, parts[i].chunk);
            EXPECT_EQ(m.parts[i].sequence, parts[i].sequence);
            EXPECT_EQ(m.parts[i].x0, parts[i].x0);
            EXPECT_EQ(m.parts[i].z0, parts[i].z0);
            EXPECT_EQ(m.parts[i].w, parts[i].w);
            EXPECT_EQ(m.parts[i].h, parts[i].h);
            EXPECT_EQ(m.parts[i].deltas, parts[i].deltas);
        }
    }
    // Chunk states with and without deltas
    std::vector<int32_t> dense(129 * 129, 0);
    dense[500] = 77;
    for (bool deltas : {false, true})
    {
        std::vector<uint8_t> bytes;
        encodeTerrainChunkState(2, chunk(3, 4), 9, 129, deltas ? &dense : nullptr, bytes);
        TerrainChunkStateMessage m;
        std::string e;
        ASSERT_TRUE(decodeTerrainChunkState(bytes.data(), bytes.size(), m, e)) << e;
        EXPECT_EQ(m.chunk, chunk(3, 4));
        EXPECT_EQ(m.sequence, 9u);
        EXPECT_EQ(m.hasDeltas, deltas);
        EXPECT_EQ(m.deltas, deltas ? dense : std::vector<int32_t>());
    }
}

// ---- the client's validation

TEST_F(TerrainNet, InvalidMessagesAreRejectedAndApplyNothing)
{
    Net n;
    n.step();
    World& c          = n.client();
    const uint64_t h0 = c.terrain().baseHookSamples();
    std::vector<std::vector<uint8_t>> bad;
    // Parts of one message disagree on a shared sample (the corner (128, 128) of chunks (0,0) and (1,0))
    bad.push_back(modification(0, {part(0, 0, 1, 128, 128, 1, 1, 5), part(1, 0, 1, 0, 128, 1, 1, 6)}));
    // Sequence 0, a chunk outside the space, a box outside the tile
    bad.push_back(modification(0, {part(0, 0, 0, 10, 10, 2, 2, 5)}));
    bad.push_back(modification(0, {part(16, 0, 1, 10, 10, 2, 2, 5)}));
    bad.push_back(modification(0, {part(0, 0, 1, 128, 10, 2, 2, 5)}));
    // Regression (PR review): two interior parts of one new chunk were both applied (two entries reserved for it, the
    // sequence moved from 2 back to 1)
    bad.push_back(modification(0, {part(3, 3, 2, 10, 10, 2, 2, 5), part(3, 3, 1, 20, 20, 2, 2, 6)}));
    // A delta beyond 2^24 (by hand: the encoder asserts)
    {
        std::vector<uint8_t> m = {0, 0, 0, 0, 0, 0, 0, 1, 0};
        for (int i = 0; i < 12; ++i) m.push_back(i == 8 ? 1 : 0);  // x, z, sequence 1
        for (int i = 0; i < 4; ++i)
        {
            m.push_back(i < 2 ? 5 : 1);
            m.push_back(0);
        }
        std::vector<uint8_t> v;
        writeVarint(v, zigzag(int64_t(kMaxTerrainDelta) + 1));
        writeVarint(m, v.size());
        m.insert(m.end(), v.begin(), v.end());
        bad.push_back(m);
    }
    // A truncated zstd stream after valid parts
    {
        std::vector<uint8_t> m = modification(0, {part(0, 0, 1, 0, 0, 20, 20, 3), part(1, 0, 1, 0, 0, 20, 20, 3)});
        m.resize(m.size() - 4);
        bad.push_back(m);
    }
    // Unknown flags, an unknown space, bytes left over
    {
        std::vector<uint8_t> m = modification(0, {part(0, 0, 1, 10, 10, 2, 2, 5)});
        m[0]                   = 4;
        bad.push_back(m);
        std::vector<uint8_t> s;
        encodeTerrainModification(0, 5, {part(0, 0, 1, 10, 10, 2, 2, 5)}, s);
        bad.push_back(s);
        std::vector<uint8_t> o = modification(0, {part(0, 0, 1, 10, 10, 2, 2, 5)});
        o.push_back(0);
        bad.push_back(o);
    }
    // A chunk state with another N, sequence 0, unknown flags
    {
        std::vector<int32_t> d(257 * 257, 1);
        std::vector<uint8_t> a, b;
        encodeTerrainChunkState(0, chunk(1, 1), 1, 257, &d, a);
        encodeTerrainChunkState(0, chunk(1, 1), 0, 129, nullptr, b);
        for (auto* m : {&a, &b})
        {
            n.host->sendRaw(1, MessageId(EngineMessage::TerrainChunkState), m->data(), m->size());
        }
        std::vector<uint8_t> f = b;
        f[0]                   = 8;
        n.host->sendRaw(1, MessageId(EngineMessage::TerrainChunkState), f.data(), f.size());
    }
    for (const auto& m : bad) n.host->sendRaw(1, MessageId(EngineMessage::TerrainModification), m.data(), m.size());
    const uint64_t rejected = c.netStats().rejected;
    n.step();
    EXPECT_EQ(c.netStats().rejected - rejected, bad.size() + 3);
    EXPECT_EQ(c.terrainEdits().chunkCount(), 0u);
    EXPECT_EQ(c.terrainEdits().blockCount(0), 0u);
    EXPECT_EQ(c.terrain().baseHookSamples(), h0);
    EXPECT_EQ(pendingEvents(c), 0u);
    // The host never accepts them
    const uint64_t hostRejected   = n.host->netStats().rejected;
    const std::vector<uint8_t> ok = modification(0, {part(0, 0, 1, 10, 10, 2, 2, 5)});
    n.client().sendRaw(0, MessageId(EngineMessage::TerrainModification), ok.data(), ok.size());
    n.step();
    EXPECT_EQ(n.host->netStats().rejected - hostRejected, 1u);
    EXPECT_EQ(n.host->terrainEdits().chunkCount(), 0u);
}

// Regression: a neighbour check rejected the first of two adjacent edited chunks of a snapshot
TEST_F(TerrainNet, HostAuthorityOverNeighbours)
{
    Net n;
    n.host->editTerrain(0, brush(TerrainBrush::Add, 256, 256, 20, 2));
    n.step();
    // A later host state, sent as chunk states only (the client's neighbours are stale or absent)
    World& c = n.client();
    expectSameState(*n.host, c);
    n.host->editTerrain(0, brush(TerrainBrush::Subtract, 256, 256, 20, 3));
    n.host->editTerrain(0, brush(TerrainBrush::Smooth, 258, 250, 16, 1));
    // The client misses the modifications: a stale state of chunk (0, 0) arrives first (its border differs from the
    // host's new one, the neighbours (1, 0), (0, 1), (1, 1) keep the old state)
    n.host->replication().clearHostStates();
    n.host->replication().sendChunkSnapshot(1);
    const uint64_t rejected = c.netStats().rejected;
    n.step();
    EXPECT_EQ(c.netStats().rejected, rejected);
    expectSameState(*n.host, c);
    for (int64_t k = 100; k <= 160; ++k)
        for (int64_t i = 100; i <= 160; ++i) ASSERT_EQ(c.terrainSample(0, i, k), n.host->terrainSample(0, i, k));
    // A state whose neighbour is absent applies too; the neighbour's own state later agrees on the border
    std::vector<int32_t> a(129 * 129, 0), b(129 * 129, 0);
    for (int k = 10; k <= 20; ++k)
    {
        a[size_t(k * 129 + 128)] = 9;
        b[size_t(k * 129 + 0)]   = 9;
    }
    a[size_t(50 * 129 + 50)] = 4;
    std::vector<uint8_t> ma, mb;
    encodeTerrainChunkState(0, chunk(5, 5), 1, 129, &a, ma);
    encodeTerrainChunkState(0, chunk(6, 5), 1, 129, &b, mb);
    n.host->sendRaw(1, MessageId(EngineMessage::TerrainChunkState), ma.data(), ma.size());
    n.step();
    EXPECT_EQ(c.terrainEdits().sequence(chunk(5, 5)), 1u);
    const TerrainChunkEdits* e55 = c.terrainEdits().entry(chunk(5, 5));
    ASSERT_NE(e55, nullptr);
    EXPECT_EQ(e55->nonZeroSamples, 12u);
    EXPECT_EQ(c.terrainEdits().delta(0, 6 * 128, 5 * 128 + 15), 9);  // the absent neighbour's border reads it
    n.host->sendRaw(1, MessageId(EngineMessage::TerrainChunkState), mb.data(), mb.size());
    n.step();
    const TerrainChunkEdits* e65 = c.terrainEdits().entry(chunk(6, 5));
    ASSERT_NE(e65, nullptr);
    EXPECT_EQ(e65->nonZeroSamples, 11u);
    e55 = c.terrainEdits().entry(chunk(5, 5));
    ASSERT_NE(e55, nullptr);
    EXPECT_EQ(e55->nonZeroSamples, 12u);
    EXPECT_EQ(c.netStats().rejected, rejected);
}

// ---- edits reach clients

TEST_F(TerrainNet, EditsReachClientsAtDelay0And2)
{
    for (int delay : {0, 2})
    {
        Net n(1, delay);
        n.step(3);
        size_t listedInStep = 0;
        g_phase4            = [&](World& w)
        {
            if (!w.isHost() && !w.pendingTerrainChanges().empty()) listedInStep = w.pendingTerrainChanges().size();
        };
        const TerrainEditResult r = applyTerrainEdit(*n.host, 0, brush(TerrainBrush::Add, 256, 256, 9, 1));
        ASSERT_EQ(r.chunks.size(), 4u);
        n.step();
        // The host's one event (emitted between steps) is readable in that step
        EXPECT_EQ(n.host->events.read<TerrainEditedEvent>().size(), 1u);
        if (delay == 2)
        {
            EXPECT_EQ(n.client().terrainEdits().chunkCount(), 0u);
            n.step(2);
        }
        EXPECT_EQ(listedInStep, 1u);
        expectSameState(*n.host, n.client());
        EXPECT_EQ(n.client().replication().stats().terrainMessagesApplied, 1u);
        // One event on the client, with the host's box
        ASSERT_EQ(pendingEvents(n.client()), 1u);
        n.step();
        const auto& ev = n.client().events.read<TerrainEditedEvent>();
        ASSERT_EQ(ev.size(), 1u);
        EXPECT_EQ(ev[0].x0, double(r.i0) * 2);
        EXPECT_EQ(ev[0].x1, double(r.i1) * 2);
        EXPECT_EQ(ev[0].z0, double(r.k0) * 2);
        EXPECT_EQ(ev[0].z1, double(r.k1) * 2);
        EXPECT_EQ(n.client().netStats().modificationGaps, 0u);
        g_phase4 = nullptr;
    }
}

TEST_F(TerrainNet, ZeroDeltasAllocateNoBlock)
{
    Net n;
    n.step();
    // An edit back to the base in a block the client has not allocated
    const std::vector<uint8_t> m = modification(0, {part(2, 2, 1, 10, 10, 4, 4, 0)});
    n.host->sendRaw(1, MessageId(EngineMessage::TerrainModification), m.data(), m.size());
    n.step();
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(2, 2)), 1u);
    EXPECT_EQ(n.client().terrainEdits().blockCount(0), 0u);
    EXPECT_FALSE(n.client().terrainEdits().hasDeltas(chunk(2, 2)));
}

TEST_F(TerrainNet, DuplicatesGapsAndEpochs)
{
    Net n;
    n.step();
    World& c  = n.client();
    auto send = [&](const std::vector<uint8_t>& m)
    {
        n.host->sendRaw(1, MessageId(EngineMessage::TerrainModification), m.data(), m.size());
        n.step();
    };
    send(modification(0, {part(3, 3, 1, 5, 5, 2, 2, 100)}));
    EXPECT_EQ(c.terrainEdits().delta(0, 3 * 128 + 5, 3 * 128 + 5), 100);
    // A duplicate (sequence <= current) is dropped
    send(modification(0, {part(3, 3, 1, 5, 5, 2, 2, 200)}));
    EXPECT_EQ(c.terrainEdits().delta(0, 3 * 128 + 5, 3 * 128 + 5), 100);
    EXPECT_EQ(c.replication().stats().terrainPartsDuplicate, 1u);
    // A gap is applied, logged and counted
    send(modification(0, {part(3, 3, 4, 5, 5, 2, 2, 300)}));
    EXPECT_EQ(c.terrainEdits().delta(0, 3 * 128 + 5, 3 * 128 + 5), 300);
    EXPECT_EQ(c.terrainEdits().sequence(chunk(3, 3)), 4u);
    EXPECT_EQ(c.netStats().modificationGaps, 1u);
    // A message of another epoch is dropped
    send(modification(7, {part(3, 3, 5, 5, 5, 2, 2, 400)}));
    EXPECT_EQ(c.terrainEdits().delta(0, 3 * 128 + 5, 3 * 128 + 5), 300);
    EXPECT_EQ(c.replication().stats().terrainDroppedEpoch, 1u);
}

// ---- the host state list over steps

TEST_F(TerrainNet, HostStateListOverStepsAndAborts)
{
    Net n;
    n.step();
    // Between steps: sent by the host's stepNetworkOnly when that runs first
    n.host->editTerrain(0, brush(TerrainBrush::Add, 600, 600, 6, 1));
    n.host->stepNetworkOnly();
    EXPECT_TRUE(n.host->replication().hostStateList().empty());
    n.client().step();
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(2, 2)), 1u);
    // An aborted phase 4 after an edit applied at the phase-3 flush keeps the entry; the next step sends it
    bool edited = false, thrown = false;
    g_phase3 = [&](World& w)
    {
        if (!w.isHost() || edited) return;
        edited = true;
        w.editTerrain(0, brush(TerrainBrush::Add, 600, 600, 6, 1));
    };
    g_phase4 = [&](World& w)
    {
        if (!w.isHost() || thrown) return;
        thrown = true;
        throw std::runtime_error("abort phase 4");
    };
    EXPECT_THROW(n.host->step(), std::runtime_error);
    EXPECT_EQ(n.host->terrainEdits().sequence(chunk(2, 2)), 2u);
    EXPECT_EQ(n.host->replication().hostStateList().size(), 1u);
    n.step();
    EXPECT_TRUE(n.host->replication().hostStateList().empty());
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(2, 2)), 2u);
    g_phase3 = g_phase4 = nullptr;
    // An aborted phase 8 resends
    n.host->editTerrain(0, brush(TerrainBrush::Add, 600, 600, 6, 1));
    bool once                     = false;
    n.host->debugAfterReplication = [&]()
    {
        if (once) return;
        once = true;
        throw std::runtime_error("abort phase 8");
    };
    EXPECT_THROW(n.host->step(), std::runtime_error);
    n.step();
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(2, 2)), 3u);
    expectSameState(*n.host, n.client());
    EXPECT_EQ(n.client().netStats().modificationGaps, 0u);
    n.host->debugAfterReplication = nullptr;
    // A host without peers keeps no entries
    World alone(terrainNetSetup());
    alone.editTerrain(0, brush(TerrainBrush::Add, 600, 600, 6, 1));
    EXPECT_EQ(alone.replication().hostStateList().size(), 1u);
    alone.step();
    EXPECT_TRUE(alone.replication().hostStateList().empty());
}

// Regression (PR review of S22): an edit recorded during stepNetworkOnly's extract (after the host state list was sent)
// was applied by the flush after it and its entry was cleared by the commit unsent: the client never received it
TEST_F(TerrainNet, EditAppliedAfterTheSendWaitsForTheNextExtract)
{
    Net n;
    n.step();
    bool once                     = false;
    n.host->debugAfterReplication = [&]()
    {
        if (once) return;
        once = true;
        n.host->editTerrain(0, brush(TerrainBrush::Add, 600, 600, 6, 1));
    };
    n.host->stepNetworkOnly();
    n.host->debugAfterReplication = nullptr;
    EXPECT_EQ(n.host->terrainEdits().sequence(chunk(2, 2)), 1u);
    ASSERT_EQ(n.host->replication().hostStateList().size(), 1u);
    n.step();
    EXPECT_TRUE(n.host->replication().hostStateList().empty());
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(2, 2)), 1u);
    expectSameState(*n.host, n.client());
}

// ---- late joiners and host loads

TEST_F(TerrainNet, LateJoinerGetsEveryChunkState)
{
    for (bool editInJoinStep : {false, true})
    {
        // The host sees the peer join only after the edits (PR review: the loopback peer was connected from the first
        // step, so the join-step edit reached it as a regular modification)
        Net n(1, 0, false, true);
        n.host->editTerrain(0, brush(TerrainBrush::Add, 256, 256, 12, 2));  // four chunks
        n.host->editTerrain(0, brush(TerrainBrush::Subtract, 1000, 700, 9, 1));
        n.host->terrainEditsForUpdate().bumpSequence(chunk(10, 10));  // a state without deltas
        n.step(3);
        EXPECT_TRUE(n.host->connectedPeers().empty());
        n.addClient(1);
        n.hostTransport->release();
        if (editInJoinStep) n.host->editTerrain(0, brush(TerrainBrush::Smooth, 256, 256, 12, 1));
        n.host->step();
        EXPECT_EQ(n.host->peersJoined(), std::vector<uint8_t>{1});
        n.client().step();
        n.step();
        expectSameState(*n.host, n.client());
        EXPECT_EQ(n.client().terrainEdits().sequence(chunk(10, 10)), 1u);
        EXPECT_FALSE(n.client().terrainEdits().hasDeltas(chunk(10, 10)));
        EXPECT_EQ(n.client().netStats().modificationGaps, 0u);
        EXPECT_EQ(n.client().netStats().rejected, 0u);
    }
}

TEST_F(TerrainNet, HostLoadSendsResyncThenTheStates)
{
    Net n;
    n.host->editTerrain(0, brush(TerrainBrush::Add, 256, 256, 12, 2));  // a corner edit in the save
    const std::vector<uint8_t> saved = n.host->save();
    const std::vector<uint8_t> empty = World(terrainNetSetup()).save();
    n.host->editTerrain(0, brush(TerrainBrush::Add, 256, 256, 12, 2));
    n.host->editTerrain(0, brush(TerrainBrush::Add, 1500, 900, 12, 2));
    n.step(2);
    expectSameState(*n.host, n.client());
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(0, 0)), 2u);
    // The save has lower sequences: the client ends with the loaded state
    n.host->load(saved);
    n.step(2);
    expectSameState(*n.host, n.client());
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(0, 0)), 1u);
    EXPECT_EQ(n.client().terrainEdits().sequence(chunk(5, 3)), 0u);
    EXPECT_EQ(n.client().replication().epoch(), 1u);
    // A save without edits: the client's entries are cleared, one event per cleared chunk with deltas
    n.host->load(empty);
    n.host->step();
    n.client().step();
    EXPECT_EQ(n.client().terrainEdits().chunkCount(), 0u);
    EXPECT_EQ(n.client().terrainEdits().nonZeroSamples(0), 0u);
    EXPECT_EQ(pendingEvents(n.client()), 4u);
    EXPECT_EQ(n.client().netStats().modificationGaps, 0u);
}
