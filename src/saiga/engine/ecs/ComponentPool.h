/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/EntityRemap.h"
#include "saiga/engine/ecs/FieldList.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/TypeCheck.h"
#include "saiga/engine/ecs/TypeSchema.h"
#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/engine/net/NetVisitors.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
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

    PoolBase(ComponentId id, const char* name, size_t elementSize, const std::type_info& type)
        : id(id), name(name), elementSize(elementSize), type(&type)
    {
    }

    // The registered C++ type is T (isRegisteredType: across the engine DLL, and types in unnamed namespaces of
    // different translation units are told apart). Checked before every downcast.
    template <class T>
    bool holds() const
    {
        return isRegisteredType(*type, typeid(T));
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
    // Type-erased access by the world API (S10): a default component for the entity (registry bookkeeping is the
    // caller's), the JSON form (VisitJson.h), a dry-run check of a JSON document, a merge (false and unchanged on an
    // error), and the field list (schemas for agents)
    virtual void emplaceDefault(Entity e)                   = 0;
    virtual json11::Json toJson(uint32_t entityIndex) const = 0;
    // finiteOnly: nan and inf are errors (writes from the world API)
    virtual bool checkJson(const json11::Json& json, std::string& error, bool finiteOnly = true) const = 0;
    virtual bool mergeJson(uint32_t entityIndex, const json11::Json& json, std::string& error,
                           bool finiteOnly = true)                                                     = 0;
    virtual std::vector<FieldInfo> fields() const                                                      = 0;

    // Saves and loads (S11): the schema of the type and its nested structs; staged values of a load (decoded from the
    // binary form or merged from JSON into a default component), committed to the pool after the whole save was
    // checked; removing every component at once
    virtual SchemaType describe() const                                                        = 0;
    virtual void describeNested(std::vector<SchemaType>& out) const                            = 0;
    virtual std::unique_ptr<StagedValues> newStage() const                                     = 0;
    virtual bool stageBinary(StagedValues& stage, uint32_t entityIndex, BinaryReader& r) const = 0;
    virtual bool stageJson(StagedValues& stage, uint32_t entityIndex, const json11::Json& json,
                           std::string& error) const                                           = 0;
    // Adds the staged components of the entities that are alive after the load (generation[index] >= 0: the entity's
    // generation; -1: dropped); they must not have one yet
    virtual void commitStage(StagedValues& stage, const std::vector<int64_t>& generation) = 0;
    virtual void clearAll()                                                               = 0;
    // Constructs and drops a default value (a load checks that the defaults it will add cannot throw)
    // A default value built before a load commits (a throwing constructor rejects the load), moved in by
    // emplaceStaged (no second construction after the world was cleared)
    virtual std::unique_ptr<StagedValues> stageDefault() const = 0;
    virtual void emplaceStaged(Entity e, StagedValues& stage)  = 0;

    // Entity groups (S25): the binary form with every Entity field mapped through remap (the live value is mapped in
    // place and restored afterwards: no copy, R-PAR-1: main thread, no view running); the Entity fields of staged
    // values mapped; staged value i added to e
    virtual void appendRemapped(uint32_t entityIndex, std::vector<uint8_t>& out, const EntityRemap& remap) const = 0;
    virtual void remapStaged(StagedValues& stage, const EntityRemap& remap) const                                = 0;
    virtual void emplaceStagedAt(Entity e, StagedValues& stage, size_t i)                                        = 0;

    // Replication (S16): the network form (NetVisitors.h) of the entity's component; decoding into the entity's
    // existing component (false and unchanged on malformed input); the leaves for comparing replicas
    virtual void netAppend(uint32_t entityIndex, NetWriter& w) const                                       = 0;
    virtual bool netReadInto(uint32_t entityIndex, NetReader& r)                                           = 0;
    virtual void netFlatten(uint32_t entityIndex, std::vector<NetLeaf>& out, const NetIdResolver* r) const = 0;

    const ComponentId id;
    const char* const name;
    const size_t elementSize;
    const std::type_info* const type;

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
    ComponentPool() : PoolBase(T::id, T::name, sizeof(T), typeid(T)) {}

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
        // Registered components pass the field check, so only the reader's limits can fail here (S07, e.g. an enum
        // value without a name): an error in every build, never a block the reader rejects
        if (!w.ok()) throw std::runtime_error(std::string("save: ") + name + ": " + w.error());
    }

    void emplaceDefault(Entity e) override { emplace(e); }
    json11::Json toJson(uint32_t entityIndex) const override { return Engine::toJson(at(entityIndex)); }
    bool checkJson(const json11::Json& json, std::string& error, bool finiteOnly = true) const override
    {
        T scratch{};
        JsonReader check(json, false, finiteOnly);
        scratch.visit(check);
        check.finish();
        error = check.errorMessage();
        return check.ok();
    }
    bool mergeJson(uint32_t entityIndex, const json11::Json& json, std::string& error, bool finiteOnly = true) override
    {
        return fromJson(json, at(entityIndex), &error, finiteOnly);
    }
    std::vector<FieldInfo> fields() const override { return describeFields<T>(); }

    SchemaType describe() const override { return describeType<T>(SchemaKind::Component, T::id); }
    void describeNested(std::vector<SchemaType>& out) const override { collectNestedTypes<T>(out); }
    std::unique_ptr<StagedValues> newStage() const override { return std::make_unique<StagedOf<T>>(); }
    bool stageBinary(StagedValues& stage, uint32_t entityIndex, BinaryReader& r) const override
    {
        auto& s = static_cast<StagedOf<T>&>(stage);
        T value{};
        r.read(value);
        if (!r.ok()) return false;
        s.indices.push_back(entityIndex);
        s.values.push_back(std::move(value));
        return true;
    }
    bool stageJson(StagedValues& stage, uint32_t entityIndex, const json11::Json& json,
                   std::string& error) const override
    {
        auto& s = static_cast<StagedOf<T>&>(stage);
        T value{};
        // Saved values may be non-finite (finiteOnly is for agents' writes)
        if (!fromJson(json, value, &error, false)) return false;
        s.indices.push_back(entityIndex);
        s.values.push_back(std::move(value));
        return true;
    }
    void commitStage(StagedValues& stage, const std::vector<int64_t>& generation) override
    {
        auto& s = static_cast<StagedOf<T>&>(stage);
        data.reserve(data.size() + s.values.size());
        entities.reserve(entities.size() + s.values.size());
        for (size_t i = 0; i < s.indices.size(); ++i)
        {
            const uint32_t index = s.indices[i];
            if (index >= generation.size() || generation[index] < 0) continue;
            SAIGA_ASSERT(!has(index), std::string("commitStage: ") + name + " is staged twice for one entity");
            ensurePage(index);
            data.push_back(std::move(s.values[i]));
            entities.push_back(Entity{index, uint32_t(generation[index])});
            setSparse(index, uint32_t(data.size() - 1));
        }
    }
    std::unique_ptr<StagedValues> stageDefault() const override
    {
        auto s = std::make_unique<StagedOf<T>>();
        s->values.emplace_back();
        return s;
    }
    void emplaceStaged(Entity e, StagedValues& stage) override
    {
        auto& s = static_cast<StagedOf<T>&>(stage);
        SAIGA_ASSERT(!s.values.empty(), std::string("emplaceStaged: no staged value of ") + name);
        emplace(e, std::move(s.values.back()));
        s.values.pop_back();
    }
    void appendRemapped(uint32_t entityIndex, std::vector<uint8_t>& out, const EntityRemap& remap) const override
    {
        T& value = const_cast<T&>(at(entityIndex));
        // The originals first (an allocation failure here changes nothing), then the remap; every exit restores the
        // live handles (also when the write throws), so a retried unload starts from the original references
        std::vector<Entity> original;
        forEachEntityField(value, [&](Entity& e) { original.push_back(e); });
        size_t k = 0;
        forEachEntityField(value, [&](Entity& e) { e = remap(original[k++]); });
        auto restore = [&]()
        {
            size_t j = 0;
            forEachEntityField(value, [&](Entity& e) { e = original[j++]; });
        };
        bool ok = true;
        std::string error;
        try
        {
            BinaryWriter w(out);
            w.write(value);
            ok = w.ok();
            if (!ok) error = w.error();
        }
        catch (...)
        {
            restore();
            throw;
        }
        restore();
        if (!ok) throw std::runtime_error(std::string("entity group: ") + name + ": " + error);
    }
    void remapStaged(StagedValues& stage, const EntityRemap& remap) const override
    {
        auto& s = static_cast<StagedOf<T>&>(stage);
        for (T& v : s.values) forEachEntityField(v, [&](Entity& e) { e = remap(e); });
    }
    void emplaceStagedAt(Entity e, StagedValues& stage, size_t i) override
    {
        auto& s = static_cast<StagedOf<T>&>(stage);
        SAIGA_ASSERT(i < s.values.size(), std::string("emplaceStagedAt: no staged value of ") + name);
        emplace(e, std::move(s.values[i]));
    }
    void netAppend(uint32_t entityIndex, NetWriter& w) const override { w.write(at(entityIndex)); }
    bool netReadInto(uint32_t entityIndex, NetReader& r) override
    {
        T value{};
        r.read(value);
        if (!r.ok()) return false;
        at(entityIndex) = std::move(value);
        return true;
    }
    void netFlatten(uint32_t entityIndex, std::vector<NetLeaf>& out, const NetIdResolver* r) const override
    {
        NetFlattener f(out, r);
        const_cast<T&>(at(entityIndex)).visit(f);
    }
    void clearAll() override
    {
        data.clear();
        entities.clear();
        pages.clear();
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
