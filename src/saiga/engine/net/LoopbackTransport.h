/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// An in-process network of n peers (docs/engine_design.md §13.1; S15): a star (clients connect to peer 0, the host),
// delays in receive rounds, seeded loss of unreliable packets per link, a per-link bandwidth budget (token bucket) and
// fragmentation of large reliable packets. Deterministic for a seed (R-DET-1); single-threaded.

#pragma once

#include "saiga/engine/net/Transport.h"
#include "saiga/export.h"

#include <memory>
#include <utility>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct LoopbackOptions
{
    double dropRate      = 0;  // unreliable packets only
    int delayRounds      = 0;  // a packet sent before round r of the receiver starts is visible from round r + delay
    uint64_t seed        = 1;
    size_t bytesPerRound = 0;  // per link and round, 0: unlimited
};

class SAIGA_ENGINE_API LoopbackTransport : public Transport
{
   public:
    // n connected peers (peer 0 is the host; clients connect to it only)
    static std::vector<std::unique_ptr<LoopbackTransport>> createNetwork(int peers,
                                                                         const LoopbackOptions& options = {});
    // §13.1's pair: the host (peer 0) and one client (peer 1)
    static std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> createPair(double dropRate = 0,
                                                                                        int delaySteps  = 0,
                                                                                        uint64_t seed   = 1);
    ~LoopbackTransport() override;

    uint8_t localPeer() const override { return peer; }
    void send(uint8_t to, Channel channel, const std::vector<uint8_t>& packet, Reliability reliability) override;
    bool poll(TransportEvent& event) override;
    TransportStats stats() const override;

    // Test hook: a raw fragment (header included) into the reliable stream from -> to on channel
    void injectFragment(uint8_t from, uint8_t to, Channel channel, const std::vector<uint8_t>& fragment);

    struct Hub;

   private:
    LoopbackTransport(std::shared_ptr<Hub> hub, uint8_t peer);
    std::shared_ptr<Hub> hub;
    uint8_t peer;
};

}  // namespace Engine
}  // namespace Saiga
