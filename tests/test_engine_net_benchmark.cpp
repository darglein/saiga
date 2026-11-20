/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating benchmark of the message path (S15, R-TEST-2): many small messages per step through a pair and one large
// raw reliable message under a bandwidth budget; prints the numbers (reported in the PR).

#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/LoopbackTransport.h"
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
void messages(MessageList& l)
{
    l.add<Small>(Reliability::Reliable, onSmall);
    l.addRaw(65, "Big", Reliability::Reliable, Channel::ChunkData, onBig);
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
