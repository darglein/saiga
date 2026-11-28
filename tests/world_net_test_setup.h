/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The world-over-the-network test setup (docs/engine_design.md §13.2; S26): worldNetTestSetup(streaming) is S24's
// streamingTestSetup with a 16 x 16-chunk overworld at 2 m (its slots and sites 0 and 1), test_valley, test_dungeon,
// test_flat and test_small, streaming on or off, and the shared content cache of the process. NetPeers: a host and
// clients over LoopbackTransport, stepped host first, with the helpers of the S26 tests.

#pragma once

#include "saiga/engine/net/ClientChunkSync.h"
#include "saiga/engine/net/HostChunkSubscriptions.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/world/World.h"

#include <functional>
#include <memory>
#include <vector>

#include "streaming_test_setup.h"

namespace WorldNetTest
{
using namespace Saiga;
using namespace Saiga::Engine;

inline WorldSetup worldNetTestSetup(bool streaming, uint16_t maxVersion = 2,
                                    std::shared_ptr<ContentCache> cache = nullptr)
{
    WorldSetup s = StreamingTest::streamingTestSetup(cache ? cache : StreamingTest::sharedStreamingCache(), maxVersion);
    s.config.streaming                    = streaming;
    s.config.spaceTemplates[0].sizeChunks = 16;
    return s;
}

// A host (peer 0) and n clients over loopback; step() steps the host, then every client
struct NetPeers
{
    std::vector<std::unique_ptr<LoopbackTransport>> transports;
    std::unique_ptr<World> host;
    std::vector<std::unique_ptr<World>> clients;

    NetPeers(const WorldSetup& setup, int n = 1, LoopbackOptions options = {},
             const std::function<void(WorldSetup&, int peer)>& adjust = {})
    {
        transports   = LoopbackTransport::createNetwork(1 + n, options);
        WorldSetup h = setup;
        h.transport  = transports[0].get();
        h.role       = NetRole::Host;
        h.localPeer  = 0;
        if (adjust) adjust(h, 0);
        host = std::make_unique<World>(h);
        host->setStreamingSynchronous(true);
        for (int i = 1; i <= n; ++i)
        {
            WorldSetup c = setup;
            c.transport  = transports[size_t(i)].get();
            c.role       = NetRole::Client;
            c.localPeer  = uint8_t(i);
            if (adjust) adjust(c, i);
            clients.push_back(std::make_unique<World>(c));
            clients.back()->setStreamingSynchronous(true);
        }
    }
    World& client(int peer = 1) { return *clients[size_t(peer - 1)]; }
    void step(int n = 1)
    {
        for (int i = 0; i < n; ++i)
        {
            host->step();
            for (auto& c : clients) c->step();
        }
    }
    void stepNetworkOnly(int n = 1)
    {
        for (int i = 0; i < n; ++i)
        {
            host->stepNetworkOnly();
            for (auto& c : clients) c->stepNetworkOnly();
        }
    }
    // Steps until the predicate holds (checked before each step), at most maxSteps
    bool stepUntil(const std::function<bool()>& predicate, int maxSteps = 64)
    {
        for (int i = 0; i < maxSteps; ++i)
        {
            if (predicate()) return true;
            step();
        }
        return predicate();
    }
    bool allJoined() const
    {
        for (const auto& c : clients)
            if (c->joinState() != JoinState::Joined) return false;
        return true;
    }
};

inline TerrainBrush raiseBrush(double x, double z, double radius = 20, double strength = 3)
{
    TerrainBrush b;
    b.op       = TerrainBrush::Add;
    b.center   = Vec3(x, 0, z);
    b.radius   = radius;
    b.strength = strength;
    return b;
}

inline ChunkId chunkAt(SpaceId space, int32_t x, int32_t z)
{
    ChunkId c;
    c.space = space;
    c.x     = x;
    c.z     = z;
    return c;
}

}  // namespace WorldNetTest
