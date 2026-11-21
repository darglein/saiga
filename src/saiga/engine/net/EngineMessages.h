/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's typed network messages (docs/engine_design.md §13.4; S16). Spawn, ComponentChange, Snapshot and
// OwnerState are raw messages serialized once with the network visitors (Replication.cpp); registerEngineMessages
// (MessageList.h) registers all of them.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/net/NetIds.h"
#include "saiga/engine/net/Transport.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// Host -> clients: the entity (and its subtree) is destroyed; the client calls onReplicatedDestroy
struct DestroyMessage
{
    static constexpr MessageId id     = MessageId(EngineMessage::Destroy);
    static constexpr const char* name = "Destroy";
    static constexpr uint32_t version = 1;
    static constexpr Channel channel  = Channel::State;
    NetId netId                       = 0;
    template <class V>
    void visit(V& v)
    {
        v("netId", netId);
    }
};

// Host -> clients: a reparenting (parent 0: the entity became a root) with the new LocalTransform
struct HierarchyMessage
{
    static constexpr MessageId id     = MessageId(EngineMessage::Hierarchy);
    static constexpr const char* name = "Hierarchy";
    static constexpr uint32_t version = 1;
    static constexpr Channel channel  = Channel::State;
    NetId netId                       = 0;
    NetId parent                      = 0;
    LocalTransform local;
    template <class V>
    void visit(V& v)
    {
        v("netId", netId);
        v("parent", parent);
        v("local", local);
    }
};

// Host -> owner: an effect on an owned root, applied in phase 1 (kind 0: teleport to the pose in the space; version 2,
// S21: the space); counter: the root's new command counter, which the owner stamps its owner states with
struct OwnerCommandMessage
{
    static constexpr MessageId id      = MessageId(EngineMessage::OwnerCommand);
    static constexpr const char* name  = "OwnerCommand";
    static constexpr uint32_t version  = 2;
    static constexpr uint8_t kTeleport = 0;
    NetId netId                        = 0;
    uint8_t kind                       = kTeleport;
    Vec3 position                      = Vec3::Zero();
    Quat rotation                      = Quat::Identity();
    uint8_t counter                    = 0;
    SpaceId space                      = 0;
    template <class V>
    void visit(V& v)
    {
        v("netId", netId);
        v("kind", kind);
        v("position", position);
        v("rotation", rotation);
        v("counter", counter);
        v("space", space);
    }
};

// Owner -> host: a one-shot action of an owned player (§9.3), queued on the host with (sender, sequence)
struct PlayerActionMessage
{
    static constexpr MessageId id     = MessageId(EngineMessage::PlayerAction);
    static constexpr const char* name = "PlayerAction";
    static constexpr uint32_t version = 1;
    NetId player                      = 0;
    uint32_t action                   = 0;
    Vec3 aim                          = Vec3::Zero();
    template <class V>
    void visit(V& v)
    {
        v("player", player);
        v("action", action);
        v("aim", aim);
    }
};

// Client -> host: the client's step; echoed by Pong with the host step (the round trip in local steps)
struct PingMessage
{
    static constexpr MessageId id     = MessageId(EngineMessage::Ping);
    static constexpr const char* name = "Ping";
    static constexpr uint32_t version = 1;
    uint64_t clientStep               = 0;
    template <class V>
    void visit(V& v)
    {
        v("clientStep", clientStep);
    }
};

struct PongMessage
{
    static constexpr MessageId id     = MessageId(EngineMessage::Pong);
    static constexpr const char* name = "Pong";
    static constexpr uint32_t version = 1;
    uint64_t clientStep               = 0;
    uint64_t hostStep                 = 0;
    template <class V>
    void visit(V& v)
    {
        v("clientStep", clientStep);
        v("hostStep", hostStep);
    }
};

// Host -> clients: a host load; the clients destroy every replicated entity without callbacks (not the bound
// world-state and space-state entities) and receive the spawns again; snapshots of other epochs are dropped
struct ResyncMessage
{
    static constexpr MessageId id     = MessageId(EngineMessage::Resync);
    static constexpr const char* name = "Resync";
    static constexpr uint32_t version = 1;
    static constexpr Channel channel  = Channel::State;
    uint32_t epoch                    = 0;
    template <class V>
    void visit(V& v)
    {
        v("epoch", epoch);
    }
};

}  // namespace Engine
}  // namespace Saiga
