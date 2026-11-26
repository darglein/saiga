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

// The engine's messages (S16, §13.4, §13.6); game messages start at FirstGame. S26 renumbered them: the handshake's
// IDs 0..2 are frozen (D3 of S26: a peer of another protocol version always reads a JoinReject)
enum class EngineMessage : MessageId
{
    JoinRequest = 0,        // raw, client -> host, Control, reliable (S26; frozen)
    JoinReject  = 1,        // raw, host -> client, Control, reliable (S26; frozen)
    JoinAccept  = 2,        // raw, host -> client, State, reliable (S26)
    Spawn,                  // raw, host -> clients, State, reliable
    Destroy,                // host -> clients, State, reliable
    Hierarchy,              // host -> clients, State, reliable
    ComponentChange,        // raw, host -> clients, State, reliable
    Snapshot,               // raw, host -> clients, State, unreliable
    OwnerState,             // raw, owner -> host, State, unreliable
    OwnerCommand,           // host -> owner, Control, reliable
    PlayerAction,           // owner -> host, Control, reliable
    Ping,                   // client -> host, Control, unreliable
    Pong,                   // host -> client, Control, unreliable
    Resync,                 // raw, host -> clients, State, reliable (S26: version 2 with the generation info)
    Despawn,                // host -> clients, State, reliable (S25: an unloaded entity, no callbacks)
    TerrainModification,    // raw, host -> clients, State, reliable (S22; S26: per peer, subscribed chunks only)
    TombstoneModification,  // raw, host -> clients, State, reliable (S23b; S26: per peer, subscribed chunks only)
    ChunkInterest,          // raw, client -> host, Control, reliable (S26)
    ChunkSnapshot,          // raw, host -> clients, ChunkData, reliable (S26)
    ChunkIndex,             // raw, host -> clients, ChunkData, reliable (S26: worlds without streaming)
    // TEMPORARY until step 3 of S26 (removed with their messages)
    TerrainChunkState,
    GenerationState,
    ChunkRecordState,
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

// The handshake state of a world (S26, §13.2): Local for a host or a world without transport; a client is Joining
// until the host's JoinAccept (Joined) or JoinReject, or a failed adoption of the host's generation (Rejected)
enum class JoinState : uint8_t
{
    Local,
    Joining,
    Joined,
    Rejected,
};

}  // namespace Engine
}  // namespace Saiga
