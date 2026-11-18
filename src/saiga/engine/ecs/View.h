/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Registry.h"

#include <cstddef>
#include <tuple>
#include <utility>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
/**
 * Iteration over the entities that have all components C... (docs/engine_design.md §7.5). `const C` gives read-only
 * access. The view walks the dense entity array of its smallest pool (or the entity slots for a Transform-only view)
 * and tests the masks, so the order follows from the history of structural changes: two registries with the same
 * history iterate in the same order. Entities that are not alive (free or reserved slots) are skipped, and entities
 * without the required flags (the World requires Simulated in phases 2-5). Structural changes during each() assert.
 */
template <class... C>
class View
{
    static_assert(sizeof...(C) >= 1, "a view needs at least one component");

   public:
    explicit View(Registry& registry, uint8_t requiredFlags = 0) : reg(&registry), flags(requiredFlags) {}

    // fn(Entity, C&...)
    template <class F>
    void each(F&& fn)
    {
        IterationGuard guard(*reg);
        auto access              = std::make_tuple(Accessor<C>(*reg)...);
        const ComponentMask mask = Registry::mask<C...>();
        const Traversal tr       = traversal();
        for (size_t k = 0; k < tr.count; ++k)
        {
            Entity e;
            if (accept(tr, k, mask, e)) call(fn, e, access, std::index_sequence_for<C...>{});
        }
    }

    /**
     * Like each(), on the registry's JobPool (serially without one) in fixed chunks of the candidate list, so the
     * result does not depend on the thread count. The body may write only components of its own entity and read
     * components of other entities only if the view does not write that type; it makes no structural changes, commands,
     * events or random() calls (asserted by the World, R-PAR-1). No reductions: do them in a serial pass.
     */
    template <class F>
    void parallelEach(F&& fn, int64_t chunk = 256)
    {
        IterationGuard guard(*reg);
        ParallelGuard parallel(*reg);
        const ComponentMask mask = Registry::mask<C...>();
        const Traversal tr       = traversal();
        const int64_t n          = int64_t(tr.count);
        auto body                = [&](int64_t begin, int64_t end)
        {
            auto access = std::make_tuple(Accessor<C>(*reg)...);
            for (int64_t k = begin; k < end; ++k)
            {
                Entity e;
                if (accept(tr, size_t(k), mask, e)) call(fn, e, access, std::index_sequence_for<C...>{});
            }
        };
        if (reg->jobPool)
            reg->jobPool->parallelFor(0, n, chunk, body);
        else if (n > 0)
            body(0, n);
    }

    // Number of entities each() would visit
    size_t count()
    {
        size_t n = 0;
        each([&n](Entity, C&...) { n++; });
        return n;
    }

   private:
    static constexpr bool onlyTransform = ((std::remove_cv_t<C>::id == Transform::id) && ...);

    struct IterationGuard
    {
        Registry& r;
        explicit IterationGuard(Registry& r) : r(r) { r.iterationDepth++; }
        ~IterationGuard() { r.iterationDepth--; }
        IterationGuard(const IterationGuard&)            = delete;
        IterationGuard& operator=(const IterationGuard&) = delete;
    };
    struct ParallelGuard
    {
        Registry& r;
        explicit ParallelGuard(Registry& r) : r(r) { r.parallelDepth++; }
        ~ParallelGuard() { r.parallelDepth--; }
        ParallelGuard(const ParallelGuard&)            = delete;
        ParallelGuard& operator=(const ParallelGuard&) = delete;
    };

    // The storage of one component type, looked up once per each()
    template <class Q>
    struct Accessor
    {
        using T = std::remove_cv_t<Q>;
        explicit Accessor(Registry& r)
        {
            if constexpr (T::id == Transform::id)
                transforms = &r.transforms;
            else
                pool = &r.pool<T>();
        }
        Q& get(uint32_t index) const
        {
            if constexpr (T::id == Transform::id)
                return (*transforms)[index];
            else
                return pool->at(index);
        }
        PagedArray<Transform>* transforms = nullptr;
        ComponentPool<T>* pool            = nullptr;
    };

    template <class F, class Tuple, size_t... I>
    static void call(F& fn, Entity e, Tuple& access, std::index_sequence<I...>)
    {
        fn(e, std::get<I>(access).get(e.index)...);
    }

    // The candidates of one traversal: the slots (Transform-only view) or the dense entities of the smallest pool.
    // Local to each each()/parallelEach() call (no shared state: nested calls on the same View from worker threads do
    // not race). Their number does not change during the traversal (no structural changes; reserving adds slots that
    // are not alive yet).
    struct Traversal
    {
        PoolBase* pool = nullptr;
        size_t count   = 0;
    };
    Traversal traversal() const
    {
        Traversal t;
        if constexpr (onlyTransform)
            t.count = reg->slots.size();
        else
        {
            t.pool  = smallestPool();
            t.count = t.pool->size();
        }
        return t;
    }
    bool accept(const Traversal& tr, size_t k, const ComponentMask& mask, Entity& e) const
    {
        if constexpr (onlyTransform)
        {
            (void)tr;
            if (k == 0) return false;
            const Registry::Slot& s = reg->slots[k];
            if (s.state != SlotState::Alive || (s.flags & flags) != flags) return false;
            e = Entity{uint32_t(k), s.generation};
            return true;
        }
        else
        {
            e                       = tr.pool->denseEntities()[k];
            const Registry::Slot& s = reg->slots[e.index];
            return s.state == SlotState::Alive && s.mask.containsAll(mask) && (s.flags & flags) == flags;
        }
    }

    PoolBase* smallestPool() const
    {
        PoolBase* best = nullptr;
        auto consider  = [&](PoolBase* p)
        {
            if (p && (!best || p->size() < best->size())) best = p;
        };
        (consider(poolOf<C>()), ...);
        return best;
    }
    template <class Q>
    PoolBase* poolOf() const
    {
        using T = std::remove_cv_t<Q>;
        if constexpr (T::id == Transform::id)
            return nullptr;
        else
            return &reg->pool<T>();
    }

    Registry* reg;
    uint8_t flags;
};

}  // namespace Engine
}  // namespace Saiga
