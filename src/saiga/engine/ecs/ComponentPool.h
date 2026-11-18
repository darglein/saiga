/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/VisitBinary.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
/**
 * Storage of one component type (docs/engine_design.md §7.3): the components and their entities in dense arrays, and
 * sparse pages (4096 entries) from the entity index to the dense index. Removal moves the last element into the gap,
 * so the dense order depends only on the sequence of adds and removes (deterministic views). A structural change may
 * move components: references are valid until the next structural change (R-STATE-2).
 */
class PoolBase
{
   public:
    static constexpr uint32_t kNone     = 0xFFFFFFFFu;
    static constexpr uint32_t kPageSize = 4096;

    PoolBase(ComponentId id, const char* name, size_t elementSize, const char* typeName)
        : id(id), name(name), elementSize(elementSize), typeName(typeName)
    {
    }

    // The registered C++ type is T (typeid name: pointer equality first, strcmp across DLLs, where the pointers
    // differ). Checked before every downcast: two types with the same ID, name and size are told apart.
    template <class T>
    bool holds() const
    {
        const char* n = typeid(T).name();
        return n == typeName || std::strcmp(n, typeName) == 0;
    }
    virtual ~PoolBase()                  = default;
    PoolBase(const PoolBase&)            = delete;
    PoolBase& operator=(const PoolBase&) = delete;

    // The dense index of the entity's component, kNone if it has none
    uint32_t denseIndex(uint32_t entityIndex) const
    {
        size_t page = entityIndex / kPageSize;
        if (page >= pages.size() || !pages[page]) return kNone;
        return (*pages[page])[entityIndex % kPageSize];
    }
    bool has(uint32_t entityIndex) const { return denseIndex(entityIndex) != kNone; }
    size_t size() const { return entities.size(); }
    // The entities in dense order
    const std::vector<Entity>& denseEntities() const { return entities; }

    virtual void remove(uint32_t entityIndex) = 0;
    // Appends the binary form (VisitBinary.h) of the entity's component; it must have one. Type-erased access for the
    // world hash (S08) and saves (S11).
    virtual void appendBytes(uint32_t entityIndex, std::vector<uint8_t>& out) const = 0;

    const ComponentId id;
    const char* const name;
    const size_t elementSize;
    const char* const typeName;

   protected:
    // Allocates the sparse page of the entity if needed (may throw; no other state changes)
    void ensurePage(uint32_t entityIndex)
    {
        size_t page = entityIndex / kPageSize;
        if (page >= pages.size()) pages.resize(page + 1);
        if (!pages[page])
        {
            auto p = std::make_unique<std::array<uint32_t, kPageSize>>();
            p->fill(kNone);
            pages[page] = std::move(p);
        }
    }
    void setSparse(uint32_t entityIndex, uint32_t dense)
    {
        ensurePage(entityIndex);
        (*pages[entityIndex / kPageSize])[entityIndex % kPageSize] = dense;
    }

    std::vector<Entity> entities;
    std::vector<std::unique_ptr<std::array<uint32_t, kPageSize>>> pages;
};

template <class T>
class ComponentPool final : public PoolBase
{
   public:
    ComponentPool() : PoolBase(T::id, T::name, sizeof(T), typeid(T).name()) {}

    // Aggregates are brace-initialized (add<Health>(e, 50.0, 100.0)), other types use their constructor
    template <class... A>
    T& emplace(Entity e, A&&... args)
    {
        SAIGA_ASSERT(!has(e.index));
        // The sparse page and the entity array are prepared first; the append to data is the commit point. It has
        // no effect if it throws (bad_alloc or the constructor; components have noexcept moves), so data, entities and
        // the sparse pages stay consistent. data is not reserved ahead: the arguments may refer to another component
        // of this pool (add<Health>(b, read<Health>(a))), and the insertion reads them before it reallocates.
        ensurePage(e.index);
        if (entities.size() == entities.capacity()) entities.reserve(2 * entities.size() + 1);
        uint32_t dense = uint32_t(data.size());
        if constexpr (std::is_constructible_v<T, A...>)
            data.emplace_back(std::forward<A>(args)...);
        else
            data.push_back(T{std::forward<A>(args)...});
        entities.push_back(e);      // no allocation (reserved)
        setSparse(e.index, dense);  // no allocation (page exists)
        return data.back();
    }

    // noexcept: components have noexcept moves (registerComponent), the sparse page exists
    void remove(uint32_t entityIndex) override
    {
        uint32_t dense = denseIndex(entityIndex);
        SAIGA_ASSERT(dense != kNone);
        uint32_t last = uint32_t(data.size() - 1);
        if (dense != last)
        {
            data[dense]     = std::move(data[last]);
            entities[dense] = entities[last];
            setSparse(entities[dense].index, dense);
        }
        data.pop_back();
        entities.pop_back();
        setSparse(entityIndex, kNone);
    }

    void appendBytes(uint32_t entityIndex, std::vector<uint8_t>& out) const override
    {
        BinaryWriter w(out);
        w.write(at(entityIndex));
        // Registered components pass the field check, so only the reader's limits can fail here (S07)
        SAIGA_ASSERT(w.ok(), std::string("appendBytes: ") + name + ": " + w.error());
    }

    // The entity must have the component (checked by the registry)
    T& at(uint32_t entityIndex) { return data[denseIndex(entityIndex)]; }
    const T& at(uint32_t entityIndex) const { return data[denseIndex(entityIndex)]; }
    T* find(uint32_t entityIndex)
    {
        uint32_t d = denseIndex(entityIndex);
        return d == kNone ? nullptr : &data[d];
    }

   private:
    std::vector<T> data;
};

}  // namespace Engine
}  // namespace Saiga
