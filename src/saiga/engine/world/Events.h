/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/world/EventRegistry.h"

#include <cstdint>

// The engine's events (docs/engine_design.md §7.7, §9.3): IDs 0..63; game events start at 64. Emitted in step k (or
// between steps k and k+1), readable in all phases of step k+1. behaviourEventSystem delivers DamageEvent,
// InteractEvent and ContactEvent to the target's GameObject (phase 2).

namespace Saiga
{
namespace Engine
{
enum class EngineEvent : uint16_t
{
    Damage = 0,
    Interact,
    Contact,
    Action,
    SpaceTransition,  // S21
    TerrainEdited,    // S22
    IslandLimit,      // S24
    Count
};

// Damage to target (from source, may be invalid); the target's onDamage decides what happens (game policy)
struct DamageEvent
{
    static constexpr const char* name = "DamageEvent";
    static constexpr EventId id       = EventId(EngineEvent::Damage);
    static constexpr uint32_t version = 1;
    Entity target, source;
    double amount = 0;
    template <class V>
    void visit(V& v)
    {
        v("target", target);
        v("source", source);
        v("amount", amount);
    }
};

// actor interacts with target (game-defined action); delivered to the target's onInteract
struct InteractEvent
{
    static constexpr const char* name = "InteractEvent";
    static constexpr EventId id       = EventId(EngineEvent::Interact);
    static constexpr uint32_t version = 1;
    Entity target, actor;
    uint32_t action = 0;
    template <class V>
    void visit(V& v)
    {
        v("target", target);
        v("actor", actor);
        v("action", action);
    }
};

// A contact between a and b (physics, S28); delivered to the onContact of both
struct ContactEvent
{
    static constexpr const char* name = "ContactEvent";
    static constexpr EventId id       = EventId(EngineEvent::Contact);
    static constexpr uint32_t version = 1;
    Entity a, b;
    Vec3 point  = Vec3::Zero();
    Vec3 normal = Vec3::Zero();
    template <class V>
    void visit(V& v)
    {
        v("a", a);
        v("b", b);
        v("point", point);
        v("normal", normal);
    }
};

// A one-shot player action (PlayerAction delivered on the host): queued with world.queueAction (before S15) or received
// as a message, readable in the next step in (peer, sequence) order, exactly once
struct ActionEvent
{
    static constexpr const char* name = "ActionEvent";
    static constexpr EventId id       = EventId(EngineEvent::Action);
    static constexpr uint32_t version = 1;
    Entity player;
    uint32_t action = 0;
    Vec3 aim        = Vec3::Zero();
    template <class V>
    void visit(V& v)
    {
        v("player", player);
        v("action", action);
        v("aim", aim);
    }
};

// A transition through a SpaceLink completed on the host (spaceTransitionSystem, phase 1; S21): entity moved from one
// space to another (link may be dead by now). Readable in the next step (games: velocity fix-ups, quest triggers).
// Replicate::Never (events are host-side)
struct SpaceTransitionEvent
{
    static constexpr const char* name = "SpaceTransitionEvent";
    static constexpr EventId id       = EventId(EngineEvent::SpaceTransition);
    static constexpr uint32_t version = 1;
    Entity entity, link;
    SpaceId from = 0, to = 0;
    template <class V>
    void visit(V& v)
    {
        v("entity", entity);
        v("link", link);
        v("from", from);
        v("to", to);
    }
};

// The terrain of a space changed (§12.2; S22): one event per applied change (a host edit, a client's received
// TerrainModification or TerrainChunkState, a client's Resync clear of a chunk with deltas), on the host and on
// clients, with the inclusive bounds of the changed samples in metres. A change between steps is emitted at once, one
// inside a step at the step's commit (after an aborted step at the next step's start); readable in the next step. The
// broadcast for game reactions and later engine consumers (nav, region summaries, colliders). Replicate::Never
struct TerrainEditedEvent
{
    static constexpr const char* name = "TerrainEditedEvent";
    static constexpr EventId id       = EventId(EngineEvent::TerrainEdited);
    static constexpr uint32_t version = 1;
    SpaceId space                     = 0;
    double x0 = 0, z0 = 0, x1 = 0, z1 = 0;
    template <class V>
    void visit(V& v)
    {
        v("space", space);
        v("x0", x0);
        v("z0", z0);
        v("x1", x1);
        v("z1", z1);
    }
};

// The islands of players exceed WorldConfig::maxSimulationIslands (§12.3; S24, G14): emitted by the host's streaming
// system in a step whose count exceeds the limit while the last reported count did not, or exceeds it and is larger
// than the last reported one (IslandState). A soft warning for games (spread the players, refuse joins).
// Replicate::Never
struct IslandLimitEvent
{
    static constexpr const char* name = "IslandLimitEvent";
    static constexpr EventId id       = EventId(EngineEvent::IslandLimit);
    static constexpr uint32_t version = 1;
    uint32_t count                    = 0;
    template <class V>
    void visit(V& v)
    {
        v("count", count);
    }
};

inline void registerEngineEvents(EventRegistry& events)
{
    events.add<DamageEvent>();
    events.add<InteractEvent>();
    events.add<ContactEvent>();
    events.add<ActionEvent>();
    events.add<SpaceTransitionEvent>();
    events.add<TerrainEditedEvent>();
    events.add<IslandLimitEvent>();
}

}  // namespace Engine
}  // namespace Saiga
