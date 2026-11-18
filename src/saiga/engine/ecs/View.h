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
 * history iterate in the same order. Entities that are not alive (free or reserved slots) are skipped. Structural
 * changes during each() assert.
 */
template <class... C>
class View
{
    static_assert(sizeof...(C) >= 1, "a view needs at least one component");

   public:
    explicit View(Registry& registry) : reg(&registry) {}

    // fn(Entity, C&...)
    template <class F>
    void each(F&& fn)
    {
        IterationGuard guard(*reg);
        auto access              = std::make_tuple(Accessor<C>(*reg)...);
        const ComponentMask mask = Registry::mask<C...>();
        if constexpr (onlyTransform)
        {
            for (uint32_t i = 1; i < uint32_t(reg->slots.size()); ++i)
            {
                const Registry::Slot& s = reg->slots[i];
                if (s.state != SlotState::Alive) continue;
                call(fn, Entity{i, s.generation}, access, std::index_sequence_for<C...>{});
            }
        }
        else
        {
            const std::vector<Entity>& entities = smallestPool()->denseEntities();
            for (size_t k = 0; k < entities.size(); ++k)
            {
                Entity e                = entities[k];
                const Registry::Slot& s = reg->slots[e.index];
                if (s.state != SlotState::Alive || !s.mask.containsAll(mask)) continue;
                call(fn, e, access, std::index_sequence_for<C...>{});
            }
        }
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
        std::vector<Transform>* transforms = nullptr;
        ComponentPool<T>* pool             = nullptr;
    };

    template <class F, class Tuple, size_t... I>
    static void call(F& fn, Entity e, Tuple& access, std::index_sequence<I...>)
    {
        fn(e, std::get<I>(access).get(e.index)...);
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
};

}  // namespace Engine
}  // namespace Saiga
