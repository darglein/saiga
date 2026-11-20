/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The transport between worlds (docs/engine_design.md §13.1, §13.6; S15): packets per peer and channel, reliable
// (exactly once, in order per sender and channel) or unreliable (may be lost). LoopbackTransport is the in-process
// implementation; UDP / Steam transports come in S33.

#pragma once

#include "saiga/export.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Engine
{
enum class Reliability : uint8_t
{
    Unreliable,
    Reliable,
};

// Independent channels: a large ChunkData packet never delays Control
enum class Channel : uint8_t
{
    Control,
    State,
    ChunkData,
};
constexpr size_t kChannelCount = 3;

// Bytes of an unreliable packet and of a fragment, header included (§13.4)
constexpr size_t kMaxPacket = 1200;
// Worst-case framing of a message packet (varints at their maximum): a packet header and a message header
constexpr size_t kMaxPacketHeader  = 12;
constexpr size_t kMaxMessageHeader = 14;
// The payload of an unreliable message always fits a packet, at any sequence number
constexpr size_t kMaxMessagePayload = kMaxPacket - kMaxPacketHeader - kMaxMessageHeader;
// A reliable packet (fragmented by the transport) and the payload of a reliable message
constexpr size_t kMaxReliablePacket  = size_t(64) << 20;
constexpr size_t kMaxReliablePayload = kMaxReliablePacket - kMaxPacketHeader - kMaxMessageHeader;

struct TransportEvent
{
    enum Type : uint8_t
    {
        Connected,
        Disconnected,
        Packet,
    } type          = Packet;
    uint8_t peer    = 0;
    Channel channel = Channel::Control;
    std::vector<uint8_t> packet;
};

// Cumulative since the transport was created
struct TransportStats
{
    std::array<uint64_t, kChannelCount> bytesOut{}, bytesIn{}, packetsOut{},
        packetsIn{};  // per channel (payload bytes)
    uint64_t fragmentsOut     = 0;
    uint64_t fragmentsIn      = 0;
    uint64_t droppedLoss      = 0;  // unreliable packets lost by the link model (or the network)
    uint64_t droppedBandwidth = 0;  // unreliable packets over the link budget
    uint64_t droppedNoPeer    = 0;  // sent to a peer that is not connected
    uint64_t rejectedInput    = 0;  // malformed fragments or packets from the wire
};

class SAIGA_ENGINE_API Transport
{
   public:
    virtual ~Transport()              = default;
    virtual uint8_t localPeer() const = 0;
    // A packet to a peer: an unreliable one over kMaxPacket or a reliable one over kMaxReliablePacket stops the
    // program (every build); to a peer that is not connected it is dropped and counted
    virtual void send(uint8_t peer, Channel channel, const std::vector<uint8_t>& packet, Reliability reliability) = 0;
    // Non-blocking; false: nothing more in this round
    virtual bool poll(TransportEvent& event) = 0;
    virtual TransportStats stats() const     = 0;
};

}  // namespace Engine
}  // namespace Saiga
