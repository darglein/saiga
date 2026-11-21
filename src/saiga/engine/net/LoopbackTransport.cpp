/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/LoopbackTransport.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Rng.h"
#include "saiga/engine/net/Fragmenter.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <string>

namespace Saiga
{
namespace Engine
{
namespace
{
struct Unit
{
    Channel channel = Channel::Control;
    std::vector<uint8_t> bytes;
    bool fragment         = false;  // reliable: a fragment (else a whole packet)
    uint64_t visibleRound = 0;
};

// One direction (sender -> receiver)
struct Link
{
    bool connected = true;
    std::deque<Unit> unreliable;
    std::array<std::deque<Unit>, kChannelCount> reliable;
    std::array<Reassembler, kChannelCount> reassembler;  // on the receiver's side
    bool disconnectPending     = false;
    uint64_t disconnectVisible = 0;
    double budgetUnreliable    = 0;  // token buckets (bytesPerRound > 0)
    double budgetReliable      = 0;
    Rng loss;
    uint32_t nextFragmentSequence = 0;
    size_t nextChannel            = 0;  // round-robin start
};

struct Endpoint
{
    bool alive     = true;
    bool announced = false;             // Connected events delivered
    std::vector<uint8_t> initialPeers;  // announced as Connected by the first round (the topology at creation)
    std::vector<TransportEvent> pendingDisconnects;  // a break before the first round: after the Connected events
    bool inRound   = false;
    uint64_t round = 0;
    std::deque<TransportEvent> ready;
    TransportStats stats;
};
}  // namespace

struct LoopbackTransport::Hub
{
    LoopbackOptions options;
    std::vector<Endpoint> ends;
    std::map<std::pair<int, int>, Link> links;

    Link* link(int from, int to)
    {
        auto it = links.find({from, to});
        return it == links.end() ? nullptr : &it->second;
    }

    void deliver(int to, uint8_t from, Channel channel, std::vector<uint8_t> bytes)
    {
        Endpoint& e = ends[size_t(to)];
        e.stats.bytesIn[size_t(channel)] += bytes.size();
        e.stats.packetsIn[size_t(channel)]++;
        TransportEvent ev;
        ev.type    = TransportEvent::Packet;
        ev.peer    = from;
        ev.channel = channel;
        ev.packet  = std::move(bytes);
        e.ready.push_back(std::move(ev));
    }

    void disconnected(int to, int from)
    {
        TransportEvent ev;
        ev.type = TransportEvent::Disconnected;
        ev.peer = uint8_t(from);
        ends[size_t(to)].ready.push_back(std::move(ev));
    }

    // A broken reliable stream (a bad fragment): both directions end, both ends receive Disconnected
    void breakLink(int from, int to)
    {
        for (auto key : {std::make_pair(from, to), std::make_pair(to, from)})
        {
            Link* l = link(key.first, key.second);
            if (!l || !l->connected) continue;
            l->connected = false;
            l->unreliable.clear();
            for (auto& q : l->reliable) q.clear();
            for (auto& r : l->reassembler) r = Reassembler();  // frees incomplete packets
            l->disconnectPending = false;
            Endpoint& to         = ends[size_t(key.second)];
            if (!to.alive) continue;
            if (to.announced)
                disconnected(key.second, key.first);
            else
            {
                // After the Connected events of its first round
                TransportEvent ev;
                ev.type = TransportEvent::Disconnected;
                ev.peer = uint8_t(key.first);
                to.pendingDisconnects.push_back(std::move(ev));
            }
        }
    }

    void startRound(int r)
    {
        Endpoint& e = ends[size_t(r)];
        e.round++;
        e.inRound = true;
        if (!e.announced)
        {
            // Every peer of the topology at creation is announced, also one that went away since (its packets and
            // Disconnected follow), so no end sees a Disconnected or a packet of a peer it never saw connected
            e.announced = true;
            for (uint8_t p : e.initialPeers)
            {
                TransportEvent ev;
                ev.type = TransportEvent::Connected;
                ev.peer = p;
                e.ready.push_back(std::move(ev));
            }
            for (auto& d : e.pendingDisconnects) e.ready.push_back(std::move(d));
            e.pendingDisconnects.clear();
        }
        const size_t perRound = options.bytesPerRound;
        const bool unlimited  = perRound == 0;
        for (auto& kv : links)
        {
            if (kv.first.second != r) continue;
            const int from = kv.first.first;
            Link& l        = kv.second;
            if (!l.connected) continue;
            auto visibleReliable = [&]()
            {
                for (const auto& q : l.reliable)
                    if (!q.empty() && q.front().visibleRound <= e.round) return true;
                return false;
            };
            const bool hasUnreliable = !l.unreliable.empty() && l.unreliable.front().visibleRound <= e.round;
            const bool hasReliable   = visibleReliable();
            if (!unlimited)
            {
                // Two buckets: unreliable and reliable data share the link half and half (all of it for the one class
                // that has data), each capped at max(budget, kMaxPacket), so every packet of either class passes
                // eventually and neither class starves the other
                const double cap   = double(std::max(perRound, kMaxPacket));
                const double half  = double(perRound) / 2;
                const double toU   = hasUnreliable ? (hasReliable ? half : double(perRound)) : 0;
                const double toR   = hasReliable ? (hasUnreliable ? half : double(perRound)) : 0;
                l.budgetUnreliable = std::min(l.budgetUnreliable + toU, cap);
                l.budgetReliable   = std::min(l.budgetReliable + toR, cap);
            }
            auto fits = [&](double budget, size_t n) { return unlimited || double(n) <= budget; };
            while (!l.unreliable.empty() && l.unreliable.front().visibleRound <= e.round)
            {
                Unit u = std::move(l.unreliable.front());
                l.unreliable.pop_front();
                if (fits(l.budgetUnreliable, u.bytes.size()))
                {
                    if (!unlimited) l.budgetUnreliable -= double(u.bytes.size());
                    deliver(r, uint8_t(from), u.channel, std::move(u.bytes));
                }
                else
                    e.stats.droppedBandwidth++;
            }
            // Reliable data round-robin between the channels, one unit (packet or fragment) per turn, while its
            // bucket covers it. The channel whose turn it is keeps it until its unit fits: a stream of small packets on
            // another channel cannot starve a large fragment (the bucket is capped at >= kMaxPacket, so it fits)
            while (l.connected)
            {
                size_t c = kChannelCount;
                for (size_t k = 0; k < kChannelCount; ++k)
                {
                    const size_t x = (l.nextChannel + k) % kChannelCount;
                    if (!l.reliable[x].empty() && l.reliable[x].front().visibleRound <= e.round)
                    {
                        c = x;
                        break;
                    }
                }
                if (c == kChannelCount) break;
                auto& q = l.reliable[c];
                if (!fits(l.budgetReliable, q.front().bytes.size()))
                {
                    l.nextChannel = c;
                    break;
                }
                Unit u = std::move(q.front());
                q.pop_front();
                if (!unlimited) l.budgetReliable -= double(u.bytes.size());
                l.nextChannel = (c + 1) % kChannelCount;
                if (!u.fragment)
                {
                    deliver(r, uint8_t(from), u.channel, std::move(u.bytes));
                    continue;
                }
                e.stats.fragmentsIn++;
                std::vector<uint8_t> packet;
                std::string error;
                const auto result = l.reassembler[c].add(u.bytes, packet, error);
                if (result == Reassembler::Result::Complete)
                    deliver(r, uint8_t(from), u.channel, std::move(packet));
                else if (result == Reassembler::Result::Error)
                {
                    e.stats.rejectedInput++;
                    breakLink(from, r);
                }
            }
            // A sender that went away: Disconnected after its earlier packets
            bool empty = l.unreliable.empty();
            for (auto& q : l.reliable) empty = empty && q.empty();
            if (l.connected && l.disconnectPending && empty && l.disconnectVisible <= e.round)
            {
                l.connected = false;
                disconnected(r, from);
            }
        }
    }
};

std::vector<std::unique_ptr<LoopbackTransport>> LoopbackTransport::createNetwork(int peers,
                                                                                 const LoopbackOptions& options)
{
    if (peers < 1 || peers > 255) SAIGA_EXIT_ERROR("LoopbackTransport::createNetwork: 1..255 peers");
    auto hub     = std::make_shared<Hub>();
    hub->options = options;
    hub->ends.resize(size_t(peers));
    // A star: every client is linked with the host
    for (int c = 1; c < peers; ++c)
        for (auto key : {std::make_pair(0, c), std::make_pair(c, 0)})
        {
            Link l;
            l.loss = Rng(hashCombine(hashCombine(hashCombine(options.seed, uint64_t(key.first)), uint64_t(key.second)),
                                     fnv1a64("loss")));
            hub->links.emplace(key, std::move(l));
            hub->ends[size_t(key.second)].initialPeers.push_back(uint8_t(key.first));
        }
    std::vector<std::unique_ptr<LoopbackTransport>> out;
    for (int p = 0; p < peers; ++p)
        out.push_back(std::unique_ptr<LoopbackTransport>(new LoopbackTransport(hub, uint8_t(p))));
    return out;
}

std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> LoopbackTransport::createPair(double dropRate,
                                                                                                int delaySteps,
                                                                                                uint64_t seed)
{
    LoopbackOptions o;
    o.dropRate    = dropRate;
    o.delayRounds = delaySteps;
    o.seed        = seed;
    auto net      = createNetwork(2, o);
    return {std::move(net[0]), std::move(net[1])};
}

LoopbackTransport::LoopbackTransport(std::shared_ptr<Hub> h, uint8_t p) : hub(std::move(h)), peer(p) {}

LoopbackTransport::~LoopbackTransport()
{
    // The others receive Disconnected after the packets this end sent before; sends to it are dropped
    hub->ends[peer].alive = false;
    for (auto& kv : hub->links)
    {
        if (kv.first.first == peer && kv.second.connected)
        {
            kv.second.disconnectPending = true;
            kv.second.disconnectVisible =
                hub->ends[size_t(kv.first.second)].round + 1 + uint64_t(std::max(0, hub->options.delayRounds));
        }
        if (kv.first.second == peer)
        {
            // Its receive side is released (the outgoing queues stay: the others still receive them)
            Link& in      = kv.second;
            in.connected  = false;
            in.unreliable = {};
            for (auto& q : in.reliable) q = {};
            for (auto& ra : in.reassembler) ra = Reassembler{};
        }
    }
    Endpoint& self = hub->ends[peer];
    self.ready     = {};
    self.pendingDisconnects.clear();
    self.pendingDisconnects.shrink_to_fit();
}

void LoopbackTransport::send(uint8_t to, Channel channel, const std::vector<uint8_t>& packet, Reliability reliability)
{
    if (reliability == Reliability::Unreliable && packet.size() > kMaxPacket)
        SAIGA_EXIT_ERROR("LoopbackTransport: an unreliable packet of " + std::to_string(packet.size()) +
                         " bytes (at most " + std::to_string(kMaxPacket) + ")");
    if (packet.size() > kMaxReliablePacket)
        SAIGA_EXIT_ERROR("LoopbackTransport: a packet of " + std::to_string(packet.size()) + " bytes (at most " +
                         std::to_string(kMaxReliablePacket) + ")");
    Endpoint& self = hub->ends[peer];
    Link* l        = hub->link(peer, to);
    if (!l || !l->connected || l->disconnectPending || to >= hub->ends.size() || !hub->ends[to].alive)
    {
        self.stats.droppedNoPeer++;
        return;
    }
    self.stats.bytesOut[size_t(channel)] += packet.size();
    self.stats.packetsOut[size_t(channel)]++;
    // Visible from the receiver's next round + delay (a round being polled ends before it)
    const uint64_t visible = hub->ends[to].round + 1 + uint64_t(std::max(0, hub->options.delayRounds));
    if (reliability == Reliability::Unreliable)
    {
        if (hub->options.dropRate > 0 && l->loss.uniform() < hub->options.dropRate)
        {
            self.stats.droppedLoss++;
            return;
        }
        l->unreliable.push_back(Unit{channel, packet, false, visible});
        return;
    }
    auto& q = l->reliable[size_t(channel)];
    if (packet.size() <= kMaxPacket)
    {
        q.push_back(Unit{channel, packet, false, visible});
        return;
    }
    for (auto& f : fragmentPacket(l->nextFragmentSequence++, packet))
    {
        self.stats.fragmentsOut++;
        q.push_back(Unit{channel, std::move(f), true, visible});
    }
}

bool LoopbackTransport::poll(TransportEvent& event)
{
    Endpoint& e = hub->ends[peer];
    if (!e.inRound) hub->startRound(peer);
    if (e.ready.empty())
    {
        e.inRound = false;
        return false;
    }
    event = std::move(e.ready.front());
    e.ready.pop_front();
    return true;
}

TransportStats LoopbackTransport::stats() const
{
    return hub->ends[peer].stats;
}

void LoopbackTransport::injectFragment(uint8_t from, uint8_t to, Channel channel, const std::vector<uint8_t>& fragment)
{
    Link* l = hub->link(from, to);
    if (!l) SAIGA_EXIT_ERROR("injectFragment: no link from " + std::to_string(from) + " to " + std::to_string(to));
    l->reliable[size_t(channel)].push_back(
        Unit{channel, fragment, true, hub->ends[to].round + 1 + uint64_t(std::max(0, hub->options.delayRounds))});
}

}  // namespace Engine
}  // namespace Saiga
