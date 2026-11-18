/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/gameplay/TickPolicy.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/World.h"
#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
/**
 * Behavior of one entity (docs/engine_design.md §9.2), owned by its Behaviour component and created by the class
 * factory (ClassList). All state lives in components (R-STATE-1): a subclass has no constructor parameters and only
 * members set in configure() (constant configuration) plus one member `cache` that onLoad() and onReplicatedSpawn()
 * rebuild. At most three inheritance levels below GameObject, `override` on every override, leaf classes final
 * (R-OO-1).
 *
 * Hooks run on the host (the client hooks from S16): configure() on spawn, load and replication; onSpawn() once after
 * spawning (the initial components are present); update(dtSinceLastTick) when the object's tick is due (§8.4, phase 3);
 * onDamage/onInteract/onContact for the previous step's events (phase 2); onDestroy() before the components are removed
 * (not on unloading). Structural changes inside the hooks follow the usual rules (commands during a step).
 */
class SAIGA_ENGINE_API GameObject
{
   public:
    virtual ~GameObject() = default;

    virtual void configure() {}
    virtual void onSpawn() {}
    virtual void onLoad() {}
    virtual void onCatchUp(double /*elapsedSeconds*/) {}
    virtual void onReplicatedSpawn() {}
    virtual void onReplicatedDestroy() {}
    virtual void update(double /*dtSinceLastTick*/) {}
    virtual void onDamage(const DamageEvent&) {}
    virtual void onInteract(const InteractEvent&) {}
    virtual void onContact(const ContactEvent&) {}
    virtual void onDestroy() {}
    // nullptr: the world's WorldConfig::ticking
    virtual const TickPolicy* tickPolicy() const { return nullptr; }

    Entity entity() const { return self; }
    World& world() const { return *owner; }

    template <class T>
    const T& read() const
    {
        return owner->read<T>(self);
    }
    template <class T>
    T& write()
    {
        return owner->get<T>(self);
    }
    template <class T>
    bool has() const
    {
        return owner->has<T>(self);
    }
    template <class T, class... A>
    void add(A&&... args)
    {
        owner->add<T>(self, std::forward<A>(args)...);
    }
    template <class T>
    void remove()
    {
        owner->remove<T>(self);
    }
    // In this object's space
    template <class T, class... C>
    Entity spawn(const Vec3& position, C&&... initial)
    {
        return owner->spawnIn<T>(transform().space(), position, std::forward<C>(initial)...);
    }
    template <class T, class... C>
    Entity spawnIn(SpaceId space, const Vec3& position, C&&... initial)
    {
        return owner->spawnIn<T>(space, position, std::forward<C>(initial)...);
    }
    void destroy(Entity e) { owner->destroy(e); }
    // Destroys this object's entity
    void destroy() { owner->destroy(self); }
    const Transform& transform() const { return owner->read<Transform>(self); }
    Transform& transformWrite() { return owner->get<Transform>(self); }

   private:
    friend class World;
    World* owner = nullptr;
    Entity self;
};

}  // namespace Engine
}  // namespace Saiga

// Behaviour and World::spawn need the complete GameObject
#include "saiga/engine/gameplay/Behaviour.h"
