/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's message list (§13.4, R-REG-1; S16): every handler forwards to the world's Replication.

#include "saiga/engine/net/EngineMessages.h"

#include "saiga/engine/net/MessageList.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void registerEngineMessages(MessageList& list)
{
    list.addRaw(MessageId(EngineMessage::Spawn), "Spawn", Reliability::Reliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onSpawn(from, d, n); });
    list.add<DestroyMessage>(Reliability::Reliable, [](World& w, uint8_t from, const DestroyMessage& m)
                             { w.replication().onDestroy(from, m); });
    list.add<HierarchyMessage>(Reliability::Reliable, [](World& w, uint8_t from, const HierarchyMessage& m)
                               { w.replication().onHierarchy(from, m); });
    list.addRaw(MessageId(EngineMessage::ComponentChange), "ComponentChange", Reliability::Reliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().onComponentChange(from, d, n); });
    list.addRaw(MessageId(EngineMessage::Snapshot), "Snapshot", Reliability::Unreliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onSnapshot(from, d, n); });
    list.addRaw(MessageId(EngineMessage::OwnerState), "OwnerState", Reliability::Unreliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onOwnerState(from, d, n); });
    list.add<OwnerCommandMessage>(Reliability::Reliable, [](World& w, uint8_t from, const OwnerCommandMessage& m)
                                  { w.replication().onOwnerCommand(from, m); });
    list.add<PlayerActionMessage>(Reliability::Reliable, [](World& w, uint8_t from, const PlayerActionMessage& m)
                                  { w.replication().onPlayerAction(from, m); });
    list.add<PingMessage>(Reliability::Unreliable,
                          [](World& w, uint8_t from, const PingMessage& m) { w.replication().onPing(from, m); });
    list.add<PongMessage>(Reliability::Unreliable,
                          [](World& w, uint8_t from, const PongMessage& m) { w.replication().onPong(from, m); });
    list.add<ResyncMessage>(Reliability::Reliable,
                            [](World& w, uint8_t from, const ResyncMessage& m) { w.replication().onResync(from, m); });
    list.addRaw(MessageId(EngineMessage::TerrainModification), "TerrainModification", Reliability::Reliable,
                Channel::State, [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().onTerrainModification(from, d, n); });
    list.addRaw(MessageId(EngineMessage::TerrainChunkState), "TerrainChunkState", Reliability::Reliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().onTerrainChunkState(from, d, n); });
}

}  // namespace Engine
}  // namespace Saiga
