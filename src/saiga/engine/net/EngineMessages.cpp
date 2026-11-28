/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's message list (§13.4, §13.6, R-REG-1; S16, S26): every handler forwards to the world's Replication or
// its chunk sync.

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
    // S26: the handshake (IDs and prefixes frozen, D3)
    list.addRaw(MessageId(EngineMessage::JoinRequest), "JoinRequest", 1, Reliability::Reliable, Channel::Control,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onJoinRequest(from, d, n); });
    list.addRaw(MessageId(EngineMessage::JoinReject), "JoinReject", 1, Reliability::Reliable, Channel::Control,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onJoinReject(from, d, n); });
    list.addRaw(MessageId(EngineMessage::JoinAccept), "JoinAccept", 1, Reliability::Reliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onJoinAccept(from, d, n); });
    list.addRaw(MessageId(EngineMessage::Spawn), "Spawn", 1, Reliability::Reliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onSpawn(from, d, n); });
    list.add<DestroyMessage>(Reliability::Reliable, [](World& w, uint8_t from, const DestroyMessage& m)
                             { w.replication().onDestroy(from, m); });
    list.add<HierarchyMessage>(Reliability::Reliable, [](World& w, uint8_t from, const HierarchyMessage& m)
                               { w.replication().onHierarchy(from, m); });
    list.addRaw(MessageId(EngineMessage::ComponentChange), "ComponentChange", 1, Reliability::Reliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().onComponentChange(from, d, n); });
    list.addRaw(MessageId(EngineMessage::Snapshot), "Snapshot", 1, Reliability::Unreliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onSnapshot(from, d, n); });
    list.addRaw(MessageId(EngineMessage::OwnerState), "OwnerState", 1, Reliability::Unreliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onOwnerState(from, d, n); });
    list.add<OwnerCommandMessage>(Reliability::Reliable, [](World& w, uint8_t from, const OwnerCommandMessage& m)
                                  { w.replication().onOwnerCommand(from, m); });
    list.add<PlayerActionMessage>(Reliability::Reliable, [](World& w, uint8_t from, const PlayerActionMessage& m)
                                  { w.replication().onPlayerAction(from, m); });
    list.add<PingMessage>(Reliability::Unreliable,
                          [](World& w, uint8_t from, const PingMessage& m) { w.replication().onPing(from, m); });
    list.add<PongMessage>(Reliability::Unreliable,
                          [](World& w, uint8_t from, const PongMessage& m) { w.replication().onPong(from, m); });
    // S26: version 2 (the epoch and the generation info)
    list.addRaw(MessageId(EngineMessage::Resync), "Resync", 2, Reliability::Reliable, Channel::State,
                [](World& w, uint8_t from, const uint8_t* d, size_t n) { w.replication().onResync(from, d, n); });
    list.add<DespawnMessage>(Reliability::Reliable, [](World& w, uint8_t from, const DespawnMessage& m)
                             { w.replication().onDespawn(from, m); });
    list.addRaw(MessageId(EngineMessage::TerrainModification), "TerrainModification", 1, Reliability::Reliable,
                Channel::State, [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().chunkSync().onTerrainModification(from, d, n); });
    list.addRaw(MessageId(EngineMessage::TombstoneModification), "TombstoneModification", 1, Reliability::Reliable,
                Channel::State, [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().chunkSync().onTombstoneModification(from, d, n); });
    // S26: the chunk sync
    list.addRaw(MessageId(EngineMessage::ChunkInterest), "ChunkInterest", 1, Reliability::Reliable, Channel::Control,
                [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().subscriptions().onInterest(from, d, n); });
    list.addRaw(MessageId(EngineMessage::ChunkSnapshot), "ChunkSnapshot", 1, Reliability::Reliable, Channel::ChunkData,
                [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().chunkSync().onSnapshot(from, d, n); });
    list.addRaw(MessageId(EngineMessage::ChunkIndex), "ChunkIndex", 1, Reliability::Reliable, Channel::ChunkData,
                [](World& w, uint8_t from, const uint8_t* d, size_t n)
                { w.replication().chunkSync().onIndex(from, d, n); });
}

}  // namespace Engine
}  // namespace Saiga
