/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The registered network messages (docs/engine_design.md §13.6, R-REG-1; S15): an ID, a name, the reliability, the
// channel and the handler. Typed messages are visit structs (binary visitors); raw messages carry a payload their
// sender serialized itself (S16 snapshots, S26 chunks). Engine messages are registered first (IDs 0..63), then the
// game's (from 64). An unreliable message's payload must fit kMaxMessagePayload: registration checks the minimum
// encoded size, and a larger payload at run time (strings, vectors) stops the program in send.

#pragma once

#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/engine/net/NetIds.h"
#include "saiga/engine/net/Transport.h"
#include "saiga/export.h"

#include <functional>
#include <string>
#include <vector>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
class World;

namespace MessageDetail
{
template <class M, class = void>
struct ChannelOf
{
    static constexpr Channel value = Channel::Control;
};
template <class M>
struct ChannelOf<M, std::void_t<decltype(M::channel)>>
{
    static constexpr Channel value = M::channel;
};
}  // namespace MessageDetail

class SAIGA_ENGINE_API MessageList
{
   public:
    // Decodes the payload and runs the handler; false (and error) for a payload that does not decode
    using Dispatch =
        std::function<bool(World&, uint8_t fromPeer, const uint8_t* data, size_t size, std::string& error)>;
    struct Entry
    {
        MessageId id = 0;
        std::string name;
        Reliability reliability = Reliability::Reliable;
        Channel channel         = Channel::Control;
        size_t minSize          = 0;  // minimum encoded payload (0 for raw messages)
        bool raw                = false;
        Dispatch dispatch;
    };

    // M: a visit struct (static constexpr MessageId id, const char* name, uint32_t version, visit); optional static
    // constexpr Channel channel (default Control)
    template <class M>
    void add(Reliability reliability, void (*handler)(World&, uint8_t fromPeer, const M&))
    {
        Entry e;
        e.id          = M::id;
        e.name        = M::name;
        e.reliability = reliability;
        e.channel     = MessageDetail::ChannelOf<M>::value;
        e.minSize     = minEncodedSize<M>();
        e.dispatch    = [handler](World& w, uint8_t from, const uint8_t* data, size_t size, std::string& error)
        {
            M m{};
            if (!fromBytes(m, data, size, &error)) return false;
            handler(w, from, m);
            return true;
        };
        addEntry(std::move(e));
    }
    void addRaw(MessageId id, const char* name, Reliability reliability, Channel channel,
                void (*handler)(World&, uint8_t fromPeer, const uint8_t* data, size_t size));

    const Entry* find(MessageId id) const;
    const std::vector<Entry>& entries() const { return list; }
    // The engine's list is done: the game's IDs must be >= 64 from now on
    void beginGameMessages() { gameMessages = true; }

   private:
    void addEntry(Entry e);
    std::vector<Entry> list;
    bool gameMessages = false;
};

// The engine's messages (S16: replication, EngineMessages.cpp)
SAIGA_ENGINE_API void registerEngineMessages(MessageList& list);

}  // namespace Engine
}  // namespace Saiga
