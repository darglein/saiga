/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Network IDs (docs/engine_design.md §13.4, §13.6): message IDs (engine messages 0..63, game messages from 64), the
// replication NetIds and the peer roles (S16).

#pragma once

#include <cstdint>

namespace Saiga
{
namespace Engine
{
using MessageId = uint16_t;

// The engine's messages (S16, §13.4); game messages start at FirstGame
enum class EngineMessage : MessageId
{
    Spawn = 0,              // raw, host -> clients, State, reliable
    Destroy,                // host -> clients, State, reliable
    Hierarchy,              // host -> clients, State, reliable
    ComponentChange,        // raw, host -> clients, State, reliable
    Snapshot,               // raw, host -> clients, State, unreliable
    OwnerState,             // raw, owner -> host, State, unreliable
    OwnerCommand,           // host -> owner, Control, reliable
    PlayerAction,           // owner -> host, Control, reliable
    Ping,                   // client -> host, Control, unreliable
    Pong,                   // host -> client, Control, unreliable
    Resync,                 // host -> clients, State, reliable
    TerrainModification,    // raw, host -> clients, State, reliable (S22)
    TerrainChunkState,      // raw, host -> clients, State, reliable (S22)
    GenerationState,        // raw, host -> clients, State, reliable (S23a)
    TombstoneModification,  // raw, host -> clients, State, reliable (S23b)
    ChunkRecordState,       // raw, host -> clients, State, reliable (S23b)
    Count,
    FirstGame = 64,
};

// The replication ID of an entity (§13.4): 0 is none; 1 is the world-state entity, 2 + space the space-state entity
// of a space (SpaceId is 16 bit); dynamic NetIds start at kFirstDynamicNetId, monotonic for the life of the host
// process (never reused, not reset by a load)
using NetId                           = uint32_t;
constexpr NetId kWorldStateNetId      = 1;
constexpr NetId kFirstSpaceStateNetId = 2;
constexpr NetId kFirstDynamicNetId    = NetId(1) << 17;

// The role of a world (WorldSetup::role, §7.9)
enum class NetRole : uint8_t
{
    Host,
    Client,
};

}  // namespace Engine
}  // namespace Saiga
