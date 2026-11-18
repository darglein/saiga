/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/gameplay/GameObject.h"

#include <memory>
#include <tuple>
#include <utility>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
/**
 * The game object of an entity and its tick state (docs/engine_design.md §8.4, §9.2). Added by world.spawn<T> (never
 * directly). visit covers everything but the object, which saves and replication recreate through the class factory.
 * tickInterval (seconds; 0: every step) is set by the tick evaluation from the rings around the players; sinceTick
 * counts the seconds since the last update (tickOffset staggers the first tick of spawned objects).
 * alwaysTickEveryStep: every step regardless of the rings; alwaysSimulate: keeps ticking while its chunk is resident,
 * also outside the simulation radius (from S24).
 */
struct Behaviour
{
    static constexpr const char* name = "Behaviour";
    static constexpr ComponentId id   = ComponentId(EngineComponent::Behaviour);
    static constexpr uint32_t version = 1;

    std::unique_ptr<GameObject> object;
    ClassId classId     = 0;
    double tickInterval = 0;
    double sinceTick    = 0;
    // The stagger of a spawned object's first tick (seconds; it ticks when sinceTick + tickOffset reaches
    // tickInterval, then the offset is 0): sinceTick stays the real time since the last update
    double tickOffset        = 0;
    bool alwaysSimulate      = false;
    bool alwaysTickEveryStep = false;
    // Engine-internal: onDestroy is running (a destroy of the entity from inside it does not call it again)
    bool destroying = false;

    template <class V>
    void visit(V& v)
    {
        v("classId", classId);
        v("tickInterval", tickInterval);
        v("sinceTick", sinceTick);
        v("tickOffset", tickOffset);
        v("alwaysSimulate", alwaysSimulate);
        v("alwaysTickEveryStep", alwaysTickEveryStep);
    }
};

template <class T, class... C>
Entity World::spawn(const Vec3& position, C&&... initial)
{
    return spawnIn<T>(SpaceId(0), position, std::forward<C>(initial)...);
}

template <class T, class... C>
Entity World::spawnIn(SpaceId space, const Vec3& position, C&&... initial)
{
    static_assert(std::is_base_of_v<GameObject, T>, "spawn<T>: T is a GameObject class");
    static_assert(((std::decay_t<C>::id != Transform::id && std::decay_t<C>::id != Behaviour::id) && ...),
                  "spawn: Transform and Behaviour are set by spawn (pass the position)");
    const ClassList::ClassInfo& info = classes.info<T>();
    const Entity e                   = beginSpawn(space);
    // The initial components are moved into the command (applied at the flush during a step, at once between steps)
    auto apply = [e, &info, position,
                  components = std::make_tuple(std::decay_t<C>(std::forward<C>(initial))...)](World& w) mutable
    {
        // A command on an entity destroyed earlier in the same flush is dropped
        if (!w.alive(e)) return;
        try
        {
            w.finishSpawn(e, info, position);
            std::apply([&](auto&... c) { (w.add<std::decay_t<decltype(c)>>(e, std::move(c)), ...); }, components);
        }
        catch (...)
        {
            w.abandonSpawn(e);
            throw;
        }
        w.startObject(e);
    };
    if (deferring())
        record(makeCommand(std::move(apply)));
    else
        apply(*this);
    return e;
}

}  // namespace Engine
}  // namespace Saiga
