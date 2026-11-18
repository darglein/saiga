/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/ecs/ComponentMask.h"
#include "saiga/engine/ecs/ComponentPool.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/PagedArray.h"
#include "saiga/engine/ecs/Visit.h"
#include "saiga/export.h"

#include <array>
#include <atomic>
#include <cstring>
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
 *   move components: references stay valid only until the next structural change. Slots and Transforms live in
 *   stable pages: reserve() (allowed during a view) never moves a Transform.
 * - Not thread-safe; parallelEach reads and writes components of disjoint entities only.
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
    // A Reserved slot (the World's create during a step, §7.6): not alive, skipped by views, allowed while a view
    // iterates. activate() makes it an alive entity with a Transform; release() returns it unused.
    Entity reserve();
    void activate(Entity e);
    void release(Entity e);
    // Capacity for n releases (or destroys) without allocation: the World calls it when it reserves an entity during a
    // step, so releasing it on an aborted step cannot fail
    void reserveRelease(size_t n);
    bool reserved(Entity e) const
    {
        return e.index != 0 && e.index < slots.size() && slots[e.index].generation == e.generation &&
               slots[e.index].state == SlotState::Reserved;
    }
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
    // Flags of an alive entity, not component IDs. kSimulated (§12.3) is set on every new entity (streaming clears it
    // from S24); views in phases 2-5 skip entities without it.
    static constexpr uint8_t kSimulated = 1;
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
    // requiredFlags: only entities that have all these flags (the World passes kSimulated in phases 2-5)
    template <class... C>
    View<C...> view(uint8_t requiredFlags = 0);

    // True while a view iterates (structural changes assert); atomic: views nested in parallelEach run on workers
    bool iterating() const { return iterationDepth.load(std::memory_order_relaxed) > 0; }
    // True while a parallelEach body runs (structural changes, commands, events and random() assert)
    bool inParallelEach() const { return parallelDepth.load(std::memory_order_relaxed) > 0; }
    // The pool of parallelEach (the World's frame JobPool); without one parallelEach runs serially
    void setJobPool(JobPool* pool) { jobPool = pool; }

    // End of a world step for every entity (§7.4): changedLastStep = changedThisStep, teleportedLastStep = teleported,
    // changedThisStep = teleported = 0
    void shiftTransformFlags();

    // world.hash(): every alive entity in index order (index, generation, the saved components in ID order, Transform
    // first), then the free list
    void hashInto(Hasher& h) const;
    // Saved components only (Saved::No components are derived)
    bool savedComponent(ComponentId id) const { return infos[id].registered && infos[id].saved == Saved::Yes; }

    // Tests only: sets the generation of a free slot (generation wrap-around)
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

    Entity reserveSlot();

    PagedArray<Slot> slots;
    PagedArray<Transform> transforms;  // by entity index
    // FIFO reuse: a vector with a head index (reserveRelease can promise capacity, so release does not allocate)
    std::vector<uint32_t> freeSlots;
    size_t freeHead = 0;
    bool freeEmpty() const { return freeHead == freeSlots.size(); }
    void pushFree(uint32_t index);
    std::array<std::unique_ptr<PoolBase>, kMaxComponents> pools;
    std::array<ComponentInfo, kMaxComponents> infos;
    uint32_t aliveEntities   = 0;
    int registeredComponents = 0;
    std::atomic<int> iterationDepth{0};
    std::atomic<int> parallelDepth{0};
    JobPool* jobPool = nullptr;
    bool destroying  = false;  // structural changes from component destructors during destroy assert
};

// ---- template definitions
// --------------------------------------------------------------------------------------------

template <class T>
void Registry::registerComponent(Replicate mode, Saved saved)
{
    static_assert(std::is_same_v<std::remove_cv_t<decltype(T::id)>, ComponentId>, "T::id must be a ComponentId");
    static_assert(std::is_default_constructible_v<T>, "components must be default constructible");
    // Removal (swap-and-pop) and destroy must not throw halfway: moves must be noexcept
    static_assert(std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>,
                  "components must have noexcept move construction and assignment");
    checkVisitFields<T>();
    // In every build: a second registration would replace the first type's pool (or describe the wrong type)
    if (infos[T::id].registered)
        SAIGA_EXIT_ERROR(std::string("registerComponent: ID ") + std::to_string(int(T::id)) + " of " + T::name +
                         " is already used by " + (infos[T::id].name ? infos[T::id].name : "?"));
    // Names are unique (saves load components by name, §15.2)
    for (size_t i = 0; i < kMaxComponents; ++i)
        SAIGA_ASSERT(
            !infos[i].registered || std::strcmp(infos[i].name, T::name) != 0,
            std::string("registerComponent: the name ") + T::name + " is already used by ID " + std::to_string(i));
    // Allocate first, publish afterwards: a failed allocation leaves the ID unregistered
    std::unique_ptr<PoolBase> pool;
    if constexpr (T::id != Transform::id) pool = std::make_unique<ComponentPool<T>>();
    ComponentInfo& info = infos[T::id];
    info.name           = T::name;
    info.version        = T::version;
    info.replicate      = mode;
    info.saved          = saved;
    info.registered     = true;
    pools[T::id]        = std::move(pool);
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
    // The registered C++ type itself (a copy that kept name, ID and size is still another type)
    SAIGA_ASSERT(p->holds<T>(), std::string("component ") + T::name + " is another C++ type than the registered one");
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
View<C...> Registry::view(uint8_t requiredFlags)
{
    return View<C...>(*this, requiredFlags);
}

}  // namespace Engine
}  // namespace Saiga
