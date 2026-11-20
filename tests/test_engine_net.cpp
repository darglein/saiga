/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The transport and network messages (docs/engine_design.md §13.1, §13.6; stage S15): the loopback network (rounds,
// delay, loss, bandwidth, fragmentation, star topology, disconnects), the reassembler with hostile input, message
// registration, coalescing, delivery order, aborted steps, loads and the world API's net fields.

#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/net/Fragmenter.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Varint.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

std::vector<uint8_t> bytes(size_t n, uint8_t seed = 0)
{
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 31 + seed);
    return v;
}

// All events of one round
std::vector<TransportEvent> round(Transport& t)
{
    std::vector<TransportEvent> out;
    TransportEvent e;
    while (t.poll(e)) out.push_back(e);
    return out;
}
std::vector<TransportEvent> packets(const std::vector<TransportEvent>& events)
{
    std::vector<TransportEvent> out;
    for (const auto& e : events)
        if (e.type == TransportEvent::Packet) out.push_back(e);
    return out;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Varints

TEST(Varint, RoundTripAndRejects)
{
    for (uint64_t v : {uint64_t(0), uint64_t(1), uint64_t(127), uint64_t(128), uint64_t(1) << 56, ~uint64_t(0)})
    {
        std::vector<uint8_t> b;
        writeVarint(b, v);
        EXPECT_EQ(b.size(), varintSize(v));
        size_t pos = 0;
        uint64_t r = 0;
        ASSERT_TRUE(readVarint(b.data(), b.size(), pos, r));
        EXPECT_EQ(r, v);
    }
    EXPECT_EQ(unzigzag(zigzag(-5)), -5);
    size_t pos = 0;
    uint64_t r = 0;
    const std::vector<uint8_t> overlong(11, 0x80);
    EXPECT_FALSE(readVarint(overlong.data(), overlong.size(), pos, r));
    std::vector<uint8_t> overflow(9, 0xFF);
    overflow.push_back(0x02);  // more than 64 bits
    pos = 0;
    EXPECT_FALSE(readVarint(overflow.data(), overflow.size(), pos, r));
    const std::vector<uint8_t> nonMinimal{0x80, 0x00};
    pos = 0;
    EXPECT_FALSE(readVarint(nonMinimal.data(), nonMinimal.size(), pos, r));
    const std::vector<uint8_t> truncated{0x80};
    pos = 0;
    EXPECT_FALSE(readVarint(truncated.data(), truncated.size(), pos, r));
}

// ---------------------------------------------------------------------------------------------------------------------
// Loopback transport

TEST(Loopback, ConnectedFirstAndReliableOrderPerChannel)
{
    auto [host, client] = LoopbackTransport::createPair();
    host->send(1, Channel::Control, {1}, Reliability::Reliable);
    host->send(1, Channel::State, {10}, Reliability::Reliable);
    host->send(1, Channel::Control, {2}, Reliability::Reliable);
    host->send(1, Channel::State, {11}, Reliability::Reliable);
    const auto events = round(*client);
    ASSERT_GE(events.size(), 5u);
    EXPECT_EQ(events[0].type, TransportEvent::Connected);
    EXPECT_EQ(events[0].peer, 0);
    std::vector<uint8_t> control, state;
    for (const auto& e : packets(events)) (e.channel == Channel::Control ? control : state).push_back(e.packet[0]);
    EXPECT_EQ(control, (std::vector<uint8_t>{1, 2}));
    EXPECT_EQ(state, (std::vector<uint8_t>{10, 11}));
    EXPECT_EQ(client->localPeer(), 1);
}

TEST(Loopback, DelayInRounds)
{
    LoopbackOptions o;
    o.delayRounds = 2;
    auto net      = LoopbackTransport::createNetwork(2, o);
    EXPECT_TRUE(packets(round(*net[1])).empty());  // round 1: Connected only
    net[0]->send(1, Channel::Control, {7}, Reliability::Reliable);
    EXPECT_TRUE(packets(round(*net[1])).empty());   // round 2
    EXPECT_TRUE(packets(round(*net[1])).empty());   // round 3
    EXPECT_EQ(packets(round(*net[1])).size(), 1u);  // round 4: 1 + 1 + delay 2
}

namespace
{
// The unreliable packets the client receives from 200 sent with loss (the values sent)
std::set<int> deliveredWithLoss(uint64_t seed, bool otherTraffic)
{
    LoopbackOptions o;
    o.dropRate = 0.5;
    o.seed     = seed;
    auto net   = LoopbackTransport::createNetwork(3, o);
    std::set<int> got;
    for (int i = 0; i < 200; ++i)
    {
        net[0]->send(1, Channel::State, {uint8_t(i)}, Reliability::Unreliable);
        // Traffic on another link must not change this link's loss pattern
        if (otherTraffic) net[0]->send(2, Channel::State, {uint8_t(i)}, Reliability::Unreliable);
        for (const auto& e : packets(round(*net[1]))) got.insert(e.packet[0]);
    }
    return got;
}
}  // namespace

TEST(Loopback, SeededLossOnlyForUnreliablePackets)
{
    const auto a = deliveredWithLoss(1, false);
    EXPECT_EQ(a, deliveredWithLoss(1, false));  // reproducible
    EXPECT_EQ(a, deliveredWithLoss(1, true));   // independent of other links
    EXPECT_NE(a, deliveredWithLoss(2, false));  // another seed, another pattern
    EXPECT_GT(a.size(), 50u);
    EXPECT_LT(a.size(), 150u);
    // Reliable packets are never dropped
    LoopbackOptions o;
    o.dropRate = 0.9;
    auto net   = LoopbackTransport::createNetwork(2, o);
    for (int i = 0; i < 100; ++i) net[0]->send(1, Channel::Control, {uint8_t(i)}, Reliability::Reliable);
    EXPECT_EQ(packets(round(*net[1])).size(), 100u);
    EXPECT_GT(net[0]->stats().packetsOut[0], 0u);
}

TEST(Loopback, Fragmentation)
{
    for (size_t size : {size_t(1200), size_t(1201), size_t(4) << 20})
    {
        auto [host, client] = LoopbackTransport::createPair();
        const auto data     = bytes(size, uint8_t(size));
        host->send(1, Channel::ChunkData, data, Reliability::Reliable);
        const auto got = packets(round(*client));
        ASSERT_EQ(got.size(), 1u) << size;
        EXPECT_EQ(got[0].packet, data);
        const uint64_t fragments = host->stats().fragmentsOut;
        EXPECT_EQ(fragments, size <= kMaxPacket ? 0u : (size + kFragmentPayload - 1) / kFragmentPayload) << size;
        EXPECT_EQ(client->stats().fragmentsIn, fragments);
    }
}

TEST(LoopbackDeathTest, OversizedPackets)
{
    auto unreliable = []()
    {
        auto [host, client] = LoopbackTransport::createPair();
        host->send(1, Channel::State, bytes(kMaxPacket + 1), Reliability::Unreliable);
    };
    EXPECT_DEATH(unreliable(), "unreliable packet");
}

// A large ChunkData packet does not delay Control under a budget, and the budget never stalls the link
TEST(Loopback, ChannelsIndependentUnderABudget)
{
    LoopbackOptions o;
    o.bytesPerRound = 20000;
    auto net        = LoopbackTransport::createNetwork(2, o);
    round(*net[1]);
    net[0]->send(1, Channel::ChunkData, bytes(1 << 20), Reliability::Reliable);
    net[0]->send(1, Channel::Control, {42}, Reliability::Reliable);
    int controlRound = -1, chunkRound = -1;
    for (int r = 0; r < 200 && chunkRound < 0; ++r)
        for (const auto& e : packets(round(*net[1])))
        {
            if (e.channel == Channel::Control) controlRound = r;
            if (e.channel == Channel::ChunkData) chunkRound = r;
        }
    ASSERT_GE(chunkRound, 0);
    ASSERT_GE(controlRound, 0);
    EXPECT_LT(controlRound, chunkRound);
    EXPECT_LE(controlRound, 1);
}

TEST(Loopback, BudgetBelowAPacketSlowsButNeverStalls)
{
    LoopbackOptions o;
    o.bytesPerRound = 100;
    auto net        = LoopbackTransport::createNetwork(2, o);
    net[0]->send(1, Channel::Control, bytes(1200), Reliability::Reliable);
    int arrived = -1;
    for (int r = 0; r < 20 && arrived < 0; ++r)
        if (!packets(round(*net[1])).empty()) arrived = r;
    EXPECT_GE(arrived, 10);  // the bucket fills 100 bytes per round up to one packet
    EXPECT_LE(arrived, 13);
}

// Snapshots (unreliable, every round) and a chunk transfer under a budget: the snapshots are not starved, the chunk
// completes
TEST(Loopback, SnapshotsDuringAChunkTransfer)
{
    LoopbackOptions o;
    o.bytesPerRound = 400;
    auto net        = LoopbackTransport::createNetwork(2, o);
    round(*net[1]);
    net[0]->send(1, Channel::ChunkData, bytes(20000), Reliability::Reliable);
    int snapshots = 0, rounds = 0;
    bool chunk = false;
    for (; rounds < 300 && !chunk; ++rounds)
    {
        net[0]->send(1, Channel::State, bytes(1100), Reliability::Unreliable);
        for (const auto& e : packets(round(*net[1])))
        {
            if (e.channel == Channel::State) snapshots++;
            if (e.channel == Channel::ChunkData) chunk = true;
        }
    }
    EXPECT_TRUE(chunk) << rounds;
    // The snapshots keep arriving (their half of the budget, 200 bytes per round, passes one 1100-byte packet about
    // every sixth round)
    EXPECT_GE(snapshots, rounds / 8) << rounds;
    EXPECT_GT(net[1]->stats().droppedBandwidth, 0u);
}

// Regression (S15 review): with other budget and packet sizes the reliable stream starved forever; unreliable and
// reliable data now have a bucket each (half the budget each while both have data)
TEST(Loopback, ReliableProgressForAnyBudget)
{
    for (auto [budget, snapshot] : {std::pair<size_t, size_t>{800, 800}, {400, 600}, {400, 1100}, {100, 1200}})
    {
        LoopbackOptions o;
        o.bytesPerRound = budget;
        auto net        = LoopbackTransport::createNetwork(2, o);
        net[0]->send(1, Channel::ChunkData, bytes(20000), Reliability::Reliable);
        bool chunk = false;
        int rounds = 0, snapshots = 0;
        for (; rounds < 2000 && !chunk; ++rounds)
        {
            net[0]->send(1, Channel::State, bytes(snapshot), Reliability::Unreliable);
            net[0]->send(1, Channel::Control, bytes(20), Reliability::Reliable);  // small reliable traffic too
            for (const auto& e : packets(round(*net[1])))
            {
                if (e.channel == Channel::ChunkData) chunk = true;
                if (e.channel == Channel::State) snapshots++;
            }
        }
        EXPECT_TRUE(chunk) << budget << " " << snapshot;
        EXPECT_GT(snapshots, 0) << budget << " " << snapshot;
    }
}

// Regression (CodeRabbit, S15): a stream of reliable Control packets that used the whole reliable bucket every round
// starved a fragmented ChunkData packet forever (the 1200-byte fragment never fit); the channel whose turn it is now
// keeps it until its unit fits
TEST(Loopback, ControlTrafficDoesNotStarveAFragmentedChunk)
{
    LoopbackOptions o;
    o.bytesPerRound = 400;
    auto net        = LoopbackTransport::createNetwork(2, o);
    net[0]->send(1, Channel::ChunkData, bytes(20000), Reliability::Reliable);
    bool chunk = false;
    int rounds = 0, controls = 0;
    for (; rounds < 2000 && !chunk; ++rounds)
    {
        net[0]->send(1, Channel::Control, bytes(400), Reliability::Reliable);
        for (const auto& e : packets(round(*net[1])))
        {
            if (e.channel == Channel::ChunkData) chunk = true;
            if (e.channel == Channel::Control) controls++;
        }
    }
    EXPECT_TRUE(chunk) << rounds;
    EXPECT_GT(controls, rounds / 8) << rounds;  // Control keeps its turns
}

// A peer that goes away before the other end's first poll is still announced first (no Disconnected without Connected)
TEST(Loopback, ConnectedBeforeDisconnected)
{
    auto net = LoopbackTransport::createNetwork(2);
    net[1]->send(0, Channel::Control, {1}, Reliability::Reliable);
    net[1].reset();
    const auto events = round(*net[0]);
    ASSERT_EQ(events.size(), 3u);
    EXPECT_EQ(events[0].type, TransportEvent::Connected);
    EXPECT_EQ(events[1].type, TransportEvent::Packet);
    EXPECT_EQ(events[2].type, TransportEvent::Disconnected);
}

TEST(Loopback, StarTopologyAndDisconnect)
{
    auto net = LoopbackTransport::createNetwork(3);
    // A client sees only the host
    auto events = round(*net[1]);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].type, TransportEvent::Connected);
    EXPECT_EQ(events[0].peer, 0);
    EXPECT_EQ(round(*net[0]).size(), 2u);  // the host sees both clients
    net[1]->send(2, Channel::Control, {1}, Reliability::Reliable);
    EXPECT_EQ(net[1]->stats().droppedNoPeer, 1u);
    // A client that goes away: its earlier packets first, then Disconnected
    net[2]->send(0, Channel::Control, {5}, Reliability::Reliable);
    net[2].reset();
    events = round(*net[0]);
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].type, TransportEvent::Packet);
    EXPECT_EQ(events[1].type, TransportEvent::Disconnected);
    EXPECT_EQ(events[1].peer, 2);
    net[0]->send(2, Channel::Control, {1}, Reliability::Reliable);
    EXPECT_EQ(net[0]->stats().droppedNoPeer, 1u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Fragments as untrusted input

TEST(Reassembler, HostileHeaders)
{
    auto fragment = [](uint32_t seq, uint32_t index, uint32_t count, size_t payload)
    {
        std::vector<uint8_t> f;
        for (uint32_t v : {seq, index, count})
            for (int i = 0; i < 4; ++i) f.push_back(uint8_t(v >> (8 * i)));
        f.resize(f.size() + payload, 7);
        return f;
    };
    std::vector<uint8_t> out;
    std::string error;
    {
        Reassembler r;
        EXPECT_EQ(r.add(fragment(0, 2, 2, kFragmentPayload), out, error),
                  Reassembler::Result::Error);  // index >= count
    }
    {
        Reassembler r;
        EXPECT_EQ(r.add(fragment(0, 0, 0, 10), out, error), Reassembler::Result::Error);  // count 0
    }
    {
        Reassembler r;
        EXPECT_EQ(r.add(fragment(0, 0, kMaxFragments + 1, kFragmentPayload), out, error), Reassembler::Result::Error);
    }
    {
        Reassembler r;
        EXPECT_EQ(r.add(fragment(0, 0, 3, kFragmentPayload), out, error), Reassembler::Result::Incomplete);
        EXPECT_EQ(r.add(fragment(0, 1, 4, kFragmentPayload), out, error), Reassembler::Result::Error);  // count changed
    }
    {
        Reassembler r;
        EXPECT_EQ(r.add(fragment(0, 0, 3, kFragmentPayload), out, error), Reassembler::Result::Incomplete);
        EXPECT_EQ(r.add(fragment(0, 0, 3, kFragmentPayload), out, error), Reassembler::Result::Error);  // duplicate
    }
    {
        // Out of order is fine
        Reassembler r;
        EXPECT_EQ(r.add(fragment(5, 1, 2, 3), out, error), Reassembler::Result::Incomplete);
        EXPECT_EQ(r.add(fragment(5, 0, 2, kFragmentPayload), out, error), Reassembler::Result::Complete);
        EXPECT_EQ(out.size(), kFragmentPayload + 3);
    }
    {
        // A short fragment that is not the last
        Reassembler r;
        EXPECT_EQ(r.add(fragment(0, 0, 2, 10), out, error), Reassembler::Result::Error);
    }
    {
        // Too many incomplete packets
        Reassembler r;
        Reassembler::Result last = Reassembler::Result::Incomplete;
        for (uint32_t s = 0; s < 100 && last != Reassembler::Result::Error; ++s)
            last = r.add(fragment(s, 0, 2, kFragmentPayload), out, error);
        EXPECT_EQ(last, Reassembler::Result::Error);
    }
}

// A bad fragment on the wire breaks the stream: counted, and both ends are disconnected
TEST(Loopback, InjectedBadFragmentDisconnects)
{
    auto net = LoopbackTransport::createNetwork(2);
    round(*net[0]);
    round(*net[1]);
    std::vector<uint8_t> bad(kFragmentHeader + 4, 0);
    bad[4] = 9;  // index 9 of count 0
    net[0]->injectFragment(0, 1, Channel::Control, bad);
    const auto events = round(*net[1]);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].type, TransportEvent::Disconnected);
    EXPECT_EQ(net[1]->stats().rejectedInput, 1u);
    const auto hostEvents = round(*net[0]);
    ASSERT_EQ(hostEvents.size(), 1u);
    EXPECT_EQ(hostEvents[0].type, TransportEvent::Disconnected);
}

// ---------------------------------------------------------------------------------------------------------------------
// Messages

namespace
{
struct Ping
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "Ping";
    static constexpr uint32_t version = 1;
    uint32_t value                    = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};
struct Note
{
    static constexpr MessageId id     = 65;
    static constexpr const char* name = "Note";
    static constexpr uint32_t version = 1;
    static constexpr Channel channel  = Channel::State;
    std::string text;
    template <class V>
    void visit(V& v)
    {
        v("text", text);
    }
};
struct Spawned
{
    static constexpr MessageId id     = 67;
    static constexpr const char* name = "Spawned";
    static constexpr uint32_t version = 1;
    uint32_t value                    = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};
constexpr MessageId kRaw = 66;
struct Echo
{
    static constexpr MessageId id     = 68;
    static constexpr const char* name = "Echo";
    static constexpr uint32_t version = 1;
    uint32_t value                    = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};

struct Received
{
    const World* world;
    uint8_t from;
    std::string what;
    uint64_t step;
};
std::vector<Received> g_received;
Entity g_created;
uint32_t g_poison = 0xFFFFFFFF;

void onPing(World& w, uint8_t from, const Ping& p)
{
    if (p.value == g_poison) throw std::runtime_error("poison");
    g_received.push_back({&w, from, "ping " + std::to_string(p.value), w.stepIndex()});
}
void onNote(World& w, uint8_t from, const Note& n)
{
    g_received.push_back({&w, from, "note " + n.text, w.stepIndex()});
}
void onRaw(World& w, uint8_t from, const uint8_t* data, size_t size)
{
    g_received.push_back(
        {&w, from, "raw " + std::to_string(size) + ":" + std::to_string(size ? data[0] : 0), w.stepIndex()});
}
// Replies with a Ping (value + 500)
void onEcho(World& w, uint8_t from, const Echo& e)
{
    w.send(from, Ping{e.value + 500});
}
// Creates an entity; the next handler of the batch can use it (the commands are flushed after each handler)
void onSpawned(World& w, uint8_t from, const Spawned& s)
{
    if (s.value == 0)
    {
        g_created = w.create();
        w.add<Health>(g_created);
    }
    else
        g_received.push_back(
            {&w, from, w.alive(g_created) && w.has<Health>(g_created) ? "uses it" : "missing", w.stepIndex()});
}

void testMessages(MessageList& l)
{
    l.add<Ping>(Reliability::Reliable, onPing);
    l.add<Note>(Reliability::Unreliable, onNote);
    l.addRaw(kRaw, "Raw", Reliability::Reliable, Channel::ChunkData, onRaw);
    l.add<Spawned>(Reliability::Reliable, onSpawned);
    l.add<Echo>(Reliability::Reliable, onEcho);
}

WorldSetup netSetup(Transport* t)
{
    WorldSetup s;
    s.transport = t;
    s.messages  = testMessages;
    return s;
}

std::vector<std::string> receivedBy(const World& w)
{
    std::vector<std::string> out;
    for (const auto& r : g_received)
        if (r.world == &w) out.push_back(r.what);
    return out;
}
}  // namespace

namespace
{
struct Huge
{
    static constexpr MessageId id     = 70;
    static constexpr const char* name = "Huge";
    static constexpr uint32_t version = 1;
    std::array<uint8_t, 2000> data{};
    template <class V>
    void visit(V& v)
    {
        v("data", data);
    }
};
}  // namespace

TEST(Messages, RegistrationErrors)
{
    auto handler = [](World&, uint8_t, const Ping&) {};
    EXPECT_DEATH(
        {
            MessageList l;
            l.beginGameMessages();
            l.add<Ping>(Reliability::Reliable, handler);
            l.add<Ping>(Reliability::Reliable, handler);
        },
        "is used by");
    EXPECT_DEATH(
        {
            MessageList l;
            l.add<Ping>(Reliability::Reliable, handler);  // 64 in the engine's list
        },
        "engine messages are below 64");
    EXPECT_DEATH(
        {
            MessageList l;
            l.beginGameMessages();
            l.addRaw(3, "Low", Reliability::Reliable, Channel::Control, onRaw);
        },
        "game messages start at 64");
    EXPECT_DEATH(
        {
            MessageList l;
            l.beginGameMessages();
            l.addRaw(80, "Raw", Reliability::Reliable, Channel::Control, onRaw);
            l.addRaw(81, "Raw", Reliability::Reliable, Channel::Control, onRaw);
        },
        "registered twice");
    EXPECT_DEATH(
        {
            MessageList l;
            l.beginGameMessages();
            l.add<Huge>(Reliability::Unreliable, [](World&, uint8_t, const Huge&) {});
        },
        "needs at least");
}

TEST(Messages, DeliveredInTheReceiversNextStep)
{
    g_received.clear();
    auto [ht, ct] = LoopbackTransport::createPair();
    World host(netSetup(ht.get())), client(netSetup(ct.get()));
    host.step();
    client.step();  // connected
    EXPECT_EQ(client.connectedPeers(), std::vector<uint8_t>{0});
    EXPECT_EQ(client.peersJoined(), std::vector<uint8_t>{0});
    // Between steps a send flushes at once; the client handles it in phase 1 of its next step
    host.send(1, Ping{5});
    EXPECT_TRUE(receivedBy(client).empty());
    client.step();
    ASSERT_EQ(receivedBy(client), std::vector<std::string>{"ping 5"});
    EXPECT_EQ(g_received.back().step, 1u);  // the client's second step (stepIndex counts completed steps)
    EXPECT_EQ(g_received.back().from, 0);
    EXPECT_TRUE(client.peersJoined().empty());  // replaced at the next receive
    // Raw messages; a typed message from the client to the host
    const auto data = bytes(5000, 3);
    host.sendRaw(1, kRaw, data.data(), data.size());
    client.send(0, Note{"hi"});
    client.step();
    host.step();
    EXPECT_EQ(receivedBy(client).back(), "raw 5000:" + std::to_string(int(data[0])));
    EXPECT_EQ(receivedBy(host), std::vector<std::string>{"note hi"});
    EXPECT_EQ(client.netStats().messagesIn, 2u);
}

TEST(Messages, CoalescedAndHandledInSendOrderAcrossChannels)
{
    g_received.clear();
    auto [ht, ct] = LoopbackTransport::createPair();
    World host(netSetup(ht.get())), client(netSetup(ct.get()));
    host.beginSendBatch();
    for (uint32_t i = 0; i < 100; ++i) host.send(1, Ping{i});
    host.endSendBatch();
    EXPECT_LE(ht->stats().packetsOut[size_t(Channel::Control)], 2u);  // 100 small messages in few packets
    // Interleaved channels and reliabilities: handled in send order
    host.beginSendBatch();
    host.send(1, Ping{1000});
    host.send(1, Note{"between"});
    host.send(1, Ping{1001});
    host.endSendBatch();
    client.step();
    const auto got = receivedBy(client);
    ASSERT_EQ(got.size(), 103u);
    EXPECT_EQ(got[0], "ping 0");
    EXPECT_EQ(got[99], "ping 99");
    EXPECT_EQ(got[100], "ping 1000");
    EXPECT_EQ(got[101], "note between");
    EXPECT_EQ(got[102], "ping 1001");
}

TEST(Messages, OrderedByPeerAndSequenceWithThreePeers)
{
    g_received.clear();
    auto net = LoopbackTransport::createNetwork(3);
    World host(netSetup(net[0].get())), a(netSetup(net[1].get())), b(netSetup(net[2].get()));
    b.send(0, Ping{20});
    a.send(0, Ping{10});
    b.send(0, Ping{21});
    a.send(0, Ping{11});
    host.step();
    EXPECT_EQ(receivedBy(host), (std::vector<std::string>{"ping 10", "ping 11", "ping 20", "ping 21"}));
    EXPECT_EQ(host.peersJoined(), (std::vector<uint8_t>{1, 2}));
}

TEST(Messages, AHandlersStructuralChangeIsAppliedBeforeTheNext)
{
    g_received.clear();
    auto [ht, ct] = LoopbackTransport::createPair();
    World host(netSetup(ht.get())), client(netSetup(ct.get()));
    host.beginSendBatch();
    host.send(1, Spawned{0});
    host.send(1, Spawned{1});
    host.endSendBatch();
    client.step();
    EXPECT_EQ(receivedBy(client), std::vector<std::string>{"uses it"});
}

TEST(Messages, MalformedInputIsDroppedAndCounted)
{
    g_received.clear();
    auto [ht, ct] = LoopbackTransport::createPair();
    World client(netSetup(ct.get()));
    // Garbage, an unknown ID, a payload that does not decode
    ht->send(1, Channel::Control, {0xFF, 0xFF, 0xFF}, Reliability::Reliable);
    std::vector<uint8_t> unknown;
    writeVarint(unknown, 0);
    writeVarint(unknown, 1);
    writeVarint(unknown, 0);
    unknown.push_back(99);
    unknown.push_back(0);
    writeVarint(unknown, 0);
    ht->send(1, Channel::Control, unknown, Reliability::Reliable);
    std::vector<uint8_t> shortPing;
    writeVarint(shortPing, 0);
    writeVarint(shortPing, 1);
    writeVarint(shortPing, 0);
    shortPing.push_back(64);
    shortPing.push_back(0);
    writeVarint(shortPing, 1);
    shortPing.push_back(1);  // Ping needs 4 bytes
    ht->send(1, Channel::Control, shortPing, Reliability::Reliable);
    client.step();
    EXPECT_TRUE(receivedBy(client).empty());
    EXPECT_EQ(client.netStats().rejected, 3u);
}

// Regression style: a handler that throws aborts the step; it and the earlier messages are consumed, the rest is
// handled first in the next step (exactly once); the peer lists stay until a step completes
TEST(Messages, AbortedStepKeepsTheUnhandledRest)
{
    g_received.clear();
    auto [ht, ct] = LoopbackTransport::createPair();
    World host(netSetup(ht.get())), client(netSetup(ct.get()));
    g_poison = 2;
    host.beginSendBatch();
    host.send(1, Ping{1});
    host.send(1, Ping{2});
    host.send(1, Ping{3});
    host.endSendBatch();
    EXPECT_THROW(client.step(), std::runtime_error);
    EXPECT_EQ(receivedBy(client), std::vector<std::string>{"ping 1"});
    EXPECT_EQ(client.peersJoined(), std::vector<uint8_t>{0});
    client.step();
    EXPECT_EQ(receivedBy(client), (std::vector<std::string>{"ping 1", "ping 3"}));
    EXPECT_EQ(client.netStats().handlerExceptions, 1u);
    EXPECT_EQ(client.peersJoined(), std::vector<uint8_t>{0});  // the join stayed visible in the completed step
    client.step();
    EXPECT_TRUE(client.peersJoined().empty());
    EXPECT_EQ(client.netStats().messagesIn, 2u);  // counted once per handled message, kept across the abort
    // Regression (S15 review): the reply of a handled message was dropped when a later handler of the batch threw
    host.beginSendBatch();
    host.send(1, Echo{7});
    host.send(1, Ping{2});
    host.endSendBatch();
    EXPECT_THROW(client.step(), std::runtime_error);
    client.step();  // the kept reply goes out with the next flush
    host.step();
    EXPECT_EQ(receivedBy(host), std::vector<std::string>{"ping 507"});
    g_poison = 0xFFFFFFFF;
}

namespace
{
int g_throwInPhase         = 0;
Transport* g_sendTransport = nullptr;
void sendInBehaviour(World& w, double)
{
    if (g_throwInPhase == 2) throw std::runtime_error("behaviour failed");
    if (g_throwInPhase) w.send(1, Ping{100});
}
void sendAndThrowInGameplay(World& w, double)
{
    if (!g_throwInPhase) return;
    w.send(1, Ping{200});
    throw std::runtime_error("gameplay failed");
}
void throwingSystems(SystemList& s)
{
    s.add(Phase::Behaviour, "send_in_behaviour", sendInBehaviour);
    s.add(Phase::Gameplay, "send_and_throw", sendAndThrowInGameplay);
}
}  // namespace

TEST(Messages, AbortedPhaseDropsItsSends)
{
    g_received.clear();
    auto [ht, ct] = LoopbackTransport::createPair();
    WorldSetup hs = netSetup(ht.get());
    hs.systems    = throwingSystems;
    World host(hs), client(netSetup(ct.get()));
    g_throwInPhase = 1;
    EXPECT_THROW(host.step(), std::runtime_error);
    // Regression (S15 review): the kept entry kept its phase tag, so a later abort in that phase dropped it
    g_throwInPhase = 2;
    EXPECT_THROW(host.step(), std::runtime_error);
    g_throwInPhase = 0;
    host.step();  // flushes the kept entry of the finished phase
    client.step();
    EXPECT_EQ(receivedBy(client), std::vector<std::string>{"ping 100"});
}

namespace
{
class SendsOnLoad final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass);
    static constexpr const char* name = "SendsOnLoad";
    void onLoad() override { world().send(1, Ping{9}); }
};
}  // namespace

TEST(Messages, LoadsAndTheOutbox)
{
    g_received.clear();
    auto [ht, ct] = LoopbackTransport::createPair();
    World host(netSetup(ht.get())), client(netSetup(ct.get()));
    const auto blob = host.save();
    // A load discards the outbox (encoded from the replaced state); its listeners' sends go out
    host.addLoadListener([&](const LoadInfo&) { host.send(1, Ping{77}); });
    host.beginSendBatch();
    host.send(1, Ping{1});
    ASSERT_TRUE(host.tryLoad(blob).ok());
    host.endSendBatch();
    EXPECT_EQ(host.netStats().droppedOnLoad, 1u);
    client.step();
    EXPECT_EQ(receivedBy(client), std::vector<std::string>{"ping 77"});
    // A refused load keeps the outbox
    host.beginSendBatch();
    host.send(1, Ping{2});
    EXPECT_FALSE(host.tryLoad({1, 2, 3}).ok());
    host.endSendBatch();
    client.step();
    EXPECT_EQ(receivedBy(client).back(), "ping 2");
}

#ifdef SAIGA_ASSERTS
TEST(MessagesDeathTest, SendFromOnLoad)
{
    auto run = []()
    {
        auto [ht, ct] = LoopbackTransport::createPair();
        WorldSetup s  = netSetup(ht.get());
        s.classes     = [](ClassList& c) { c.add<SendsOnLoad>(); };
        World w(s);
        w.spawn<SendsOnLoad>(Vec3::Zero());
        const auto blob = w.save();
        (void)w.tryLoad(blob);
    };
    EXPECT_DEATH(run(), "configure / onLoad");
}
#endif

TEST(MessagesDeathTest, SendErrors)
{
    auto noTransport = []()
    {
        World w(netSetup(nullptr));
        w.send(1, Ping{1});
    };
    EXPECT_DEATH(noTransport(), "no transport");
    auto ownPeer = []()
    {
        auto pair = LoopbackTransport::createPair();
        World w(netSetup(pair.first.get()));
        w.send(0, Ping{1});
    };
    EXPECT_DEATH(ownPeer(), "own peer");
    auto oversized = []()
    {
        auto pair = LoopbackTransport::createPair();
        World w(netSetup(pair.first.get()));
        const std::vector<uint8_t> small(16);
        w.sendRaw(1, kRaw, small.data(), kMaxReliablePayload + 1);  // checked before the buffer is read
    };
    EXPECT_DEATH(oversized(), "World::sendRaw");
}

TEST(Messages, WorldApiReportsTheNetwork)
{
    auto [ht, ct] = LoopbackTransport::createPair();
    World host(netSetup(ht.get())), client(netSetup(ct.get()));
    host.send(1, Ping{1});
    client.step();
    WorldApi api(client);
    const ApiResult s = api.call("world_status", Json::object{});
    ASSERT_TRUE(s.ok) << s.error;
    EXPECT_EQ(s.value["net"]["peer"].int_value(), 1) << s.value.dump();
    EXPECT_EQ(s.value["net"]["connected"][0].int_value(), 0);
    EXPECT_EQ(s.value["net"]["messages_in"].number_value(), 1);
    const ApiResult d = api.call("describe_world", Json::object{});
    ASSERT_TRUE(d.ok);
    bool found = false;
    for (const Json& m : d.value["messages"].array_items())
        found = found || (m["name"].string_value() == "Note" && m["channel"].string_value() == "state" &&
                          m["reliability"].string_value() == "unreliable");
    EXPECT_TRUE(found) << d.value["messages"].dump();
    // Without a transport there is no net block
    World alone{WorldSetup{}};
    WorldApi a2(alone);
    EXPECT_TRUE(a2.call("world_status", Json::object{}).value["net"].is_null());
}

// Two worlds exchanging unreliable messages with loss give the same hashes on both ends when run twice
TEST(Messages, DeterministicWithLoss)
{
    auto run = []()
    {
        LoopbackOptions o;
        o.dropRate = 0.3;
        o.seed     = 7;
        auto net   = LoopbackTransport::createNetwork(2, o);
        WorldSetup s;
        s.transport = net[0].get();
        s.messages  = [](MessageList& l)
        {
            l.add<Ping>(Reliability::Unreliable,
                        [](World& w, uint8_t, const Ping& p)
                        {
                            const Entity e = w.create();
                            Health h;
                            h.hp = double(p.value);
                            w.add<Health>(e, h);  // deferred in the step: applied after this handler
                        });
        };
        World host(s);
        s.transport = net[1].get();
        World client(s);
        for (uint32_t i = 0; i < 50; ++i)
        {
            host.send(1, Ping{i});
            client.send(0, Ping{1000 + i});
            host.step();
            client.step();
        }
        return std::make_pair(host.hash(), client.hash());
    };
    const auto a = run();
    EXPECT_EQ(a, run());
}

// Delays in steps between worlds, the peers' leave, packet-level rejects, the counters and the batching scopes of the
// world API and the listener dispatch
TEST(Messages, DelayLeavesRejectsCountersAndBatches)
{
    g_received.clear();
    LoopbackOptions o;
    o.delayRounds = 2;
    auto net      = LoopbackTransport::createNetwork(2, o);
    World host(netSetup(net[0].get())), client(netSetup(net[1].get()));
    host.send(1, Ping{1});
    client.step();
    client.step();
    EXPECT_TRUE(receivedBy(client).empty());
    client.step();  // sent before the client's first round: visible in its third (1 + delay 2)
    EXPECT_EQ(receivedBy(client), std::vector<std::string>{"ping 1"});
    // Packet-level rejects: count 0, a length beyond the packet
    net[0]->send(1, Channel::Control, {0, 0}, Reliability::Reliable);
    std::vector<uint8_t> beyond;
    writeVarint(beyond, 0);
    writeVarint(beyond, 1);
    writeVarint(beyond, 0);
    beyond.push_back(64);
    beyond.push_back(0);
    writeVarint(beyond, 100);
    beyond.push_back(1);
    net[0]->send(1, Channel::Control, beyond, Reliability::Reliable);
    for (int i = 0; i < 4; ++i) client.step();
    EXPECT_EQ(client.netStats().rejected, 2u);
    // The counters of the host's profile
    Profiler& p = host.profiler();
    host.send(1, Ping{2});
    host.step();
    EXPECT_GT(p.counterStats(p.counter("net/messages_out"), 1).max, 0);
    // A step listener's sends go out as one packet when the dispatch ends (also when it throws)
    const uint64_t before = net[0]->stats().packetsOut[0];
    host.addStepListener(
        [&]()
        {
            host.send(1, Ping{10});
            host.send(1, Ping{11});
            throw std::runtime_error("listener failed");
            return true;
        });
    EXPECT_THROW(host.step(), std::runtime_error);
    EXPECT_EQ(net[0]->stats().packetsOut[0], before + 1);
    // A world API call batches its sends too (an operation that sends nothing still flushes nothing)
    WorldApi api(host);
    EXPECT_TRUE(api.call("world_status", Json::object{}).ok);
    // The client leaves: peersLeft on the host
    net[1].reset();
    for (int i = 0; i < 4 && host.peersLeft().empty(); ++i) host.step();
    EXPECT_EQ(host.peersLeft(), std::vector<uint8_t>{1});
    EXPECT_TRUE(host.connectedPeers().empty());
}

#ifdef SAIGA_ASSERTS
TEST(MessagesDeathTest, SendInParallelEach)
{
    auto run = []()
    {
        auto pair       = LoopbackTransport::createPair();
        WorldSetup s    = netSetup(pair.first.get());
        s.workerThreads = 2;
        World w(s);
        for (int i = 0; i < 1000; ++i) w.add<Health>(w.create());
        w.view<Health>().parallelEach([&](Entity, Health&) { w.send(1, Ping{1}); });
    };
    EXPECT_DEATH(run(), "parallelEach");
}
#endif
