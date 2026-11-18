/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/ecs/ComponentMask.h"
#include "saiga/engine/ecs/ComponentPool.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/Visit.h"
#include "saiga/export.h"

#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
template <class... C>
class View;

enum class SlotState : uint8_t
{
    Free,
    Reserved,  // created during a step, not yet applied (S08 commands): skipped by views
    Alive,
};

/**
 * The entity-component store (docs/engine_design.md §7.1-§7.5). The World (S08) owns one and forwards these calls,
 * adding commands, reserved entities and the Simulated filter.
 *
 * - Entities: slot index + generation; destroy increments the generation (old handles become stale) and the slot is
 *   reused in FIFO order. Slot 0 is never used, {0, 0} is never alive.
 * - Components: registered by T::id (enum per module, engine 0..63, game 64..255) with registerComponent<T>();
 *   Transform (ID 0) is built in, every alive entity has one.
 * - Structural changes (create, destroy, add, remove) are not allowed while a view iterates (asserted), and they may
 *   move components: references stay valid only until the next structural change.
 * - Not thread-safe; parallelEach (S08) reads and writes components of disjoint entities only.
 */
class SAIGA_ENGINE_API Registry
{
   public:
    struct ComponentInfo
    {
        const char* name    = nullptr;
        uint32_t version    = 0;
        Replicate replicate = Replicate::Never;
        Saved saved         = Saved::Yes;
        bool registered     = false;
    };

    Registry();
    ~Registry();
    Registry(const Registry&)            = delete;
    Registry& operator=(const Registry&) = delete;

    // ---- registration ----------------------------------------------------------------------------------------------

    // Asserts that T::id is unused; checks the visit field types at compile time
    template <class T>
    void registerComponent(Replicate mode = Replicate::Never, Saved saved = Saved::Yes);
    // The registered engine IDs are 0..n and the game IDs 64..m without gaps (every enum value up to the last
    // registered one is registered); asserts otherwise. The world calls it after the registration lists (S08).
    void validateRegistration() const;
    bool registered(ComponentId id) const { return infos[id].registered; }
    const ComponentInfo& componentInfo(ComponentId id) const { return infos[id]; }
    int registeredCount() const { return registeredComponents; }

    // ---- entities --------------------------------------------------------------------------------------------------

    Entity create();
    // Removes all components; the handle becomes stale
    void destroy(Entity e);
    bool alive(Entity e) const
    {
        return e.index != 0 && e.index < slots.size() && slots[e.index].generation == e.generation &&
               slots[e.index].state == SlotState::Alive;
    }
    uint32_t aliveCount() const { return aliveEntities; }
    // Slots including the unused slot 0 (an upper bound of the entity indices)
    uint32_t slotCount() const { return uint32_t(slots.size()); }
    // Flags of an alive entity (e.g. Simulated, §12.3); not component IDs
    uint8_t flags(Entity e) const;
    void setFlags(Entity e, uint8_t flags);
    const ComponentMask& entityMask(Entity e) const;

    // ---- components ------------------------------------------------------------------------------------------------

    // Asserts if e already has T (use get to change it). Aggregates are brace-initialized from args: the argument
    // types must match the fields exactly (add<Health>(e, 50.0, 100.0)); a narrowing conversion (50 for a double) is a
    // warning on GCC but an error on Clang and MSVC.
    template <class T, class... A>
    void add(Entity e, A&&... args);
    template <class T>
    void remove(Entity e);
    template <class T>
    bool has(Entity e) const;
    template <class T>
    T& get(Entity e);
    template <class T>
    T* tryGet(Entity e);
    template <class T>
    const T& read(Entity e) const;
    template <class T>
    const T* tryRead(Entity e) const;

    template <class... C>
    static ComponentMask mask();
    // View<const A, B>: A read-only, B writable
    template <class... C>
    View<C...> view();

    // True while a view iterates (structural changes assert)
    bool iterating() const { return iterationDepth > 0; }

    // Tests only: sets the generation of a free or alive slot (generation wrap-around)
    void debugSetGeneration(uint32_t index, uint32_t generation);

   private:
    template <class... C>
    friend class View;

    struct Slot
    {
        uint32_t generation = 0;
        SlotState state     = SlotState::Free;
        uint8_t flags       = 0;
        ComponentMask mask;
    };

    template <class T>
    ComponentPool<T>& pool() const;
    void checkAlive(Entity e, const char* operation) const;
    void checkNotIterating(const char* operation) const;
    std::string describe(Entity e) const;
    // Counts the registered types; logs a warning at 200 (prefer fields on existing components or tags)
    void countRegistration();

    std::vector<Slot> slots;
    std::vector<Transform> transforms;  // by entity index
    std::deque<uint32_t> freeSlots;     // FIFO reuse
    std::array<std::unique_ptr<PoolBase>, kMaxComponents> pools;
    std::array<ComponentInfo, kMaxComponents> infos;
    uint32_t aliveEntities   = 0;
    int registeredComponents = 0;
    int iterationDepth       = 0;
};

// ---- template definitions
// --------------------------------------------------------------------------------------------

template <class T>
void Registry::registerComponent(Replicate mode, Saved saved)
{
    static_assert(std::is_same_v<std::remove_cv_t<decltype(T::id)>, ComponentId>, "T::id must be a ComponentId");
    static_assert(std::is_default_constructible_v<T>, "components must be default constructible");
    checkVisitFields<T>();
    SAIGA_ASSERT(!infos[T::id].registered, std::string("registerComponent: ID ") + std::to_string(int(T::id)) + " of " +
                                               T::name + " is already used by " +
                                               (infos[T::id].name ? infos[T::id].name : "?"));
    // Names are unique (saves load components by name, §15.2)
    for (size_t i = 0; i < kMaxComponents; ++i)
        SAIGA_ASSERT(
            !infos[i].registered || std::strcmp(infos[i].name, T::name) != 0,
            std::string("registerComponent: the name ") + T::name + " is already used by ID " + std::to_string(i));
    ComponentInfo& info = infos[T::id];
    info.name           = T::name;
    info.version        = T::version;
    info.replicate      = mode;
    info.saved          = saved;
    info.registered     = true;
    if constexpr (T::id != Transform::id) pools[T::id] = std::make_unique<ComponentPool<T>>();
    countRegistration();
}

template <class T>
ComponentPool<T>& Registry::pool() const
{
    static_assert(T::id != Transform::id, "Transform has no pool");
    PoolBase* p = pools[T::id].get();
    SAIGA_ASSERT(p, std::string("component ") + T::name + " (ID " + std::to_string(int(T::id)) + ") is not registered");
    // The registered type must be T (e.g. a copy-pasted component that kept another's ID): the name, pointer equality
    // first (strcmp across DLLs, where the literal may have another address)
    SAIGA_ASSERT(p->name == T::name || std::strcmp(p->name, T::name) == 0,
                 std::string("component ID ") + std::to_string(int(T::id)) + " is registered as " + p->name + ", not " +
                     T::name);
    // A copy that kept both name and ID: at least the layout must match before the static_cast
    SAIGA_ASSERT(p->elementSize == sizeof(T),
                 std::string("component ") + T::name + " has another size than the registered type");
    return static_cast<ComponentPool<T>&>(*p);
}

template <class T, class... A>
void Registry::add(Entity e, A&&... args)
{
    static_assert(T::id != Transform::id, "every entity has a Transform; use get<Transform>");
    checkNotIterating("add");
    checkAlive(e, "add");
    auto& p = pool<T>();
    SAIGA_ASSERT(!p.has(e.index), "add: " + describe(e) + " already has " + T::name);
    p.emplace(e, std::forward<A>(args)...);
    slots[e.index].mask.set(T::id);
}

template <class T>
void Registry::remove(Entity e)
{
    static_assert(T::id != Transform::id, "Transform cannot be removed (destroy the entity)");
    checkNotIterating("remove");
    checkAlive(e, "remove");
    auto& p = pool<T>();
    SAIGA_ASSERT(p.has(e.index), "remove: " + describe(e) + " has no " + T::name);
    p.remove(e.index);
    slots[e.index].mask.reset(T::id);
}

template <class T>
bool Registry::has(Entity e) const
{
    checkAlive(e, "has");
    if constexpr (T::id == Transform::id)
        return true;
    else
    {
        SAIGA_ASSERT(infos[T::id].registered, std::string("component ") + T::name + " (ID " +
                                                  std::to_string(int(T::id)) + ") is not registered");
        // Like the typed pool access: a component that kept another one's ID is not answered with that one's bit
        SAIGA_ASSERT(infos[T::id].name == T::name || std::strcmp(infos[T::id].name, T::name) == 0,
                     std::string("component ID ") + std::to_string(int(T::id)) + " is registered as " +
                         infos[T::id].name + ", not " + T::name);
        return slots[e.index].mask.test(T::id);
    }
}

template <class T>
T& Registry::get(Entity e)
{
    checkAlive(e, "get");
    if constexpr (T::id == Transform::id)
        return transforms[e.index];
    else
    {
        auto& p = pool<T>();
        SAIGA_ASSERT(p.has(e.index), "get: " + describe(e) + " has no " + T::name);
        return p.at(e.index);
    }
}

template <class T>
T* Registry::tryGet(Entity e)
{
    checkAlive(e, "tryGet");
    if constexpr (T::id == Transform::id)
        return &transforms[e.index];
    else
        return pool<T>().find(e.index);
}

template <class T>
const T& Registry::read(Entity e) const
{
    return const_cast<Registry*>(this)->get<T>(e);
}

template <class T>
const T* Registry::tryRead(Entity e) const
{
    return const_cast<Registry*>(this)->tryGet<T>(e);
}

template <class... C>
ComponentMask Registry::mask()
{
    ComponentMask m;
    (m.set(std::remove_cv_t<C>::id), ...);
    return m;
}

}  // namespace Engine
}  // namespace Saiga

#include "saiga/engine/ecs/View.h"

namespace Saiga
{
namespace Engine
{
template <class... C>
View<C...> Registry::view()
{
    return View<C...>(*this);
}

}  // namespace Engine
}  // namespace Saiga
