/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating benchmark of the message path (S15, R-TEST-2): many small messages per step through a pair and one large
// raw reliable message under a bandwidth budget; the replication (S16): snapshots of 10^4 moving entities and the
// HostReliable change detection of 10^3 entities with a 1 KB component; prints the numbers (reported in the PR).

#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <chrono>
#include <iostream>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);
struct Small
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "Small";
    static constexpr uint32_t version = 1;
    uint32_t a = 0, b = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
        v("b", b);
    }
};
uint64_t g_handled = 0;
void onSmall(World&, uint8_t, const Small&)
{
    g_handled++;
}
void onBig(World&, uint8_t, const uint8_t*, size_t size)
{
    g_handled += size;
}
// A 1 KB HostReliable component (S16)
struct Cargo
{
    static constexpr const char* name = "Cargo";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 1;
    std::vector<uint8_t> bytes;
    template <class V>
    void visit(V& v)
    {
        v("bytes", bytes);
    }
};
void cargoComponents(Registry& r)
{
    r.registerComponent<Cargo>(Replicate::HostReliable);
}
// Moves every root (the host's snapshots carry a changed pose for each)
void driftSystem(World& w, double dt)
{
    w.ownedBy<Transform>(w.localPeer())
        .each([dt](Entity, Transform& t) { t.setPosition(t.position() + Vec3(1, 0, 0.5) * dt); });
}
void driftSystems(SystemList& l)
{
    l.add(Phase::Gameplay, "drift", driftSystem);
}
double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
void messages(MessageList& l)
{
    l.add<Small>(Reliability::Reliable, onSmall);
    l.addRaw(65, "Big", 1, Reliability::Reliable, Channel::ChunkData, onBig);
}
}  // namespace

TEST(NetBenchmark, SmallMessagesAndALargeTransfer)
{
    {
        auto [ht, ct] = LoopbackTransport::createPair();
        WorldSetup s;
        s.messages  = messages;
        s.transport = ht.get();
        World host(s);
        s.transport = ct.get();
        s.role      = NetRole::Client;
        s.localPeer = 1;
        World client(s);
        const int steps = 20, perStep = 10000;
        g_handled     = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < steps; ++k)
        {
            host.beginSendBatch();
            for (int i = 0; i < perStep; ++i) host.send(1, Small{uint32_t(i), uint32_t(k)});
            host.endSendBatch();
            client.step();
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        EXPECT_EQ(g_handled, uint64_t(steps) * perStep);
        std::cout << "small messages: " << perStep << " per step, " << ms / steps << " ms per step, "
                  << double(ht->stats().packetsOut[0]) / steps << " packets per step" << std::endl;
    }
    {
        LoopbackOptions o;
        o.bytesPerRound = 1 << 20;
        auto net        = LoopbackTransport::createNetwork(2, o);
        WorldSetup s;
        s.messages  = messages;
        s.transport = net[0].get();
        World host(s);
        s.transport = net[1].get();
        s.role      = NetRole::Client;
        s.localPeer = 1;
        World client(s);
        std::vector<uint8_t> big(size_t(16) << 20, 1);
        g_handled     = 0;
        const auto t0 = std::chrono::steady_clock::now();
        host.sendRaw(1, 65, big.data(), big.size());
        int steps = 0;
        while (g_handled == 0 && steps < 100)
        {
            client.step();
            steps++;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        EXPECT_EQ(g_handled, big.size());
        std::cout << "16 MB raw message at 1 MB per round: " << steps << " steps, " << ms << " ms" << std::endl;
    }
}

// S16: full-state snapshots of 10^4 moving entities with Transform every step: the host's step with replication
// (serialization, zstd) against the same world without peers, and the client's step (decode, apply, buffers)
TEST(NetBenchmark, SnapshotsOfTenThousandEntities)
{
    const int n = 10000, steps = 20;
    auto make = [&](World& w)
    {
        for (int i = 0; i < n; ++i) w.get<Transform>(w.create()).setPosition(Vec3(i % 100, 0, i / 100));
    };
    WorldSetup s;
    s.systems = driftSystems;
    World alone(s);
    make(alone);
    alone.step(2);
    auto t0 = std::chrono::steady_clock::now();
    alone.step(steps);
    const double aloneMs = msSince(t0) / steps;

    auto [ht, ct] = LoopbackTransport::createPair();
    s.transport   = ht.get();
    World host(s);
    WorldSetup cs = s;
    cs.transport  = ct.get();
    cs.role       = NetRole::Client;
    cs.localPeer  = 1;
    World client(cs);
    make(host);
    for (int k = 0; k < 3; ++k)
    {
        host.step();
        client.step();
    }
    const uint64_t bytes0 = host.replication().stats().snapshotBytesSent;
    const uint64_t msgs0  = host.replication().stats().snapshotMessagesSent;
    double hostMs = 0, clientMs = 0;
    for (int k = 0; k < steps; ++k)
    {
        t0 = std::chrono::steady_clock::now();
        host.step();
        hostMs += msSince(t0);
        t0 = std::chrono::steady_clock::now();
        client.step();
        clientMs += msSince(t0);
    }
    const auto& st = host.replication().stats();
    std::cout << "snapshots of " << n << " moving entities: host step " << hostMs / steps << " ms (without peers "
              << aloneMs << " ms), client step " << clientMs / steps << " ms, "
              << double(st.snapshotBytesSent - bytes0) / steps / 1024 << " KB and "
              << double(st.snapshotMessagesSent - msgs0) / steps << " messages per step" << std::endl;
    EXPECT_EQ(client.replication().netIds().size(), size_t(n));
}

// S16: HostReliable change detection (the bytes of every component are hashed each step) for 10^3 entities with a
// 1 KB component, idle (nothing sent) and with 1 % changing per step
TEST(NetBenchmark, HostReliableChangeDetection)
{
    const int n = 1000, steps = 50;
    auto [ht, ct] = LoopbackTransport::createPair();
    WorldSetup s;
    s.components = cargoComponents;
    s.transport  = ht.get();
    World host(s);
    WorldSetup cs = s;
    cs.transport  = ct.get();
    cs.role       = NetRole::Client;
    cs.localPeer  = 1;
    World client(cs);
    std::vector<Entity> all;
    for (int i = 0; i < n; ++i)
    {
        all.push_back(host.create());
        host.add<Cargo>(all.back(), Cargo{std::vector<uint8_t>(1024, uint8_t(i))});
    }
    for (int k = 0; k < 3; ++k)
    {
        host.step();
        client.step();
    }
    const uint64_t changes0 = host.replication().stats().componentChangesSent;
    auto t0                 = std::chrono::steady_clock::now();
    for (int k = 0; k < steps; ++k) host.step();
    const double idleMs = msSince(t0) / steps;
    EXPECT_EQ(host.replication().stats().componentChangesSent, changes0);
    t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < steps; ++k)
    {
        for (int i = 0; i < n / 100; ++i) host.get<Cargo>(all[size_t((k * 10 + i) % n)]).bytes[0]++;
        host.step();
    }
    const double changingMs = msSince(t0) / steps;
    std::cout << "HostReliable change detection, " << n << " entities with 1 KB: host step " << idleMs << " ms idle, "
              << changingMs << " ms with 1 % changing per step" << std::endl;
    EXPECT_EQ(host.replication().stats().componentChangesSent, changes0 + uint64_t(steps * n / 100));
}
