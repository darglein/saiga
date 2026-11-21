/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/MessageList.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Engine
{
void MessageList::addRaw(MessageId id, const char* name, Reliability reliability, Channel channel,
                         void (*handler)(World&, uint8_t fromPeer, const uint8_t* data, size_t size))
{
    Entry e;
    e.id          = id;
    e.name        = name ? name : "";
    e.reliability = reliability;
    e.channel     = channel;
    e.raw         = true;
    e.dispatch    = [handler](World& w, uint8_t from, const uint8_t* data, size_t size, std::string&)
    {
        handler(w, from, data, size);
        return true;
    };
    addEntry(std::move(e));
}

void MessageList::addEntry(Entry e)
{
    // Fatal at world construction in every build (R-REG-1)
    if (e.name.empty()) SAIGA_EXIT_ERROR("MessageList: message " + std::to_string(e.id) + " has no name");
    if (gameMessages && e.id < MessageId(EngineMessage::FirstGame))
        SAIGA_EXIT_ERROR("MessageList: the game message " + e.name + " has the ID " + std::to_string(e.id) +
                         " (game messages start at 64)");
    if (!gameMessages && e.id >= MessageId(EngineMessage::FirstGame))
        SAIGA_EXIT_ERROR("MessageList: the engine message " + e.name + " has the ID " + std::to_string(e.id) +
                         " (engine messages are below 64)");
    for (const Entry& x : list)
    {
        if (x.id == e.id)
            SAIGA_EXIT_ERROR("MessageList: the ID " + std::to_string(e.id) + " of " + e.name + " is used by " + x.name);
        if (x.name == e.name) SAIGA_EXIT_ERROR("MessageList: the name " + e.name + " is registered twice");
    }
    if (e.reliability == Reliability::Unreliable && e.minSize > kMaxMessagePayload)
        SAIGA_EXIT_ERROR("MessageList: the unreliable message " + e.name + " needs at least " +
                         std::to_string(e.minSize) + " bytes (at most " + std::to_string(kMaxMessagePayload) + ")");
    list.push_back(std::move(e));
}

const MessageList::Entry* MessageList::find(MessageId id) const
{
    for (const Entry& e : list)
        if (e.id == id) return &e;
    return nullptr;
}

}  // namespace Engine
}  // namespace Saiga
