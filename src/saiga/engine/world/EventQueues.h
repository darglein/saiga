/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/ecs/FieldList.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/TypeCheck.h"
#include "saiga/engine/ecs/TypeSchema.h"
#include "saiga/engine/ecs/Visit.h"
#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/export.h"

#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
class Registry;

// The queue of one event type (type-erased base for the world: step begin, hash)
class EventQueueBase
{
   public:
    EventQueueBase(EventId id, const char* name, Replicate mode, const std::type_info& type)
        : id(id), name(name), replicate(mode), type(&type)
    {
    }
    // The registered C++ type is E (isRegisteredType: across the engine DLL, and types in unnamed namespaces of
    // different translation units are told apart)
    template <class E>
    bool holds() const
    {
        return isRegisteredType(*type, typeid(E));
    }
    virtual ~EventQueueBase() = default;
    // Pending -> readable (the previous readable events are dropped)
    virtual void beginStep() = 0;
    // Undoes beginStep of an aborted step: the readable events are pending again, the aborted step's own events are
    // dropped (no allocation)
    virtual void abortStep() = 0;
    // The pending events (emitted since the last beginStep) in emit order
    virtual void hashPending(Hasher& h) const = 0;
    virtual size_t pendingCount() const       = 0;
    // Type-erased access by the world API (S10): emit from JSON (false and nothing emitted on an error), the readable
    // events as JSON, the field list
    virtual bool emitJson(const json11::Json& json, std::string& error) = 0;
    virtual json11::Json readableJson() const                           = 0;
    virtual std::vector<FieldInfo> fields() const                       = 0;
    // Saves and loads (S11): the schema, the pending events in the binary form (concatenated, emit order), staged
    // events of a load, and the restore (pending = the staged events, nothing readable; null: none)
    virtual SchemaType describe() const                                                             = 0;
    virtual void describeNested(std::vector<SchemaType>& out) const                                 = 0;
    virtual void appendPending(std::vector<uint8_t>& out) const                                     = 0;
    virtual std::unique_ptr<StagedValues> newStage() const                                          = 0;
    virtual bool stageBinary(StagedValues& stage, BinaryReader& r) const                            = 0;
    virtual bool stageJson(StagedValues& stage, const json11::Json& json, std::string& error) const = 0;
    virtual void restore(StagedValues* stage)                                                       = 0;

    const EventId id;
    const char* const name;
    const Replicate replicate;
    const std::type_info* const type;  // the registered C++ type (checked before the downcast)
};

template <class E>
class EventQueue final : public EventQueueBase
{
   public:
    explicit EventQueue(Replicate mode) : EventQueueBase(E::id, E::name, mode, typeid(E)) {}

    void beginStep() override
    {
        readable.swap(pending);
        pending.clear();
    }
    void abortStep() override
    {
        // The readable events become pending again; the events the aborted step emitted are dropped (like its
        // commands). A swap: no allocation.
        pending.clear();
        pending.swap(readable);
    }
    void hashPending(Hasher& h) const override
    {
        std::vector<uint8_t> bytes;
        for (const E& e : pending)
        {
            bytes.clear();
            BinaryWriter w(bytes);
            w.write(e);
            SAIGA_ASSERT(w.ok(), std::string("event ") + name + ": " + w.error());
            h.addU64(uint64_t(bytes.size())).add(bytes.data(), bytes.size());
        }
    }
    size_t pendingCount() const override { return pending.size(); }
    bool emitJson(const json11::Json& json, std::string& error) override
    {
        E e{};
        if (!fromJson(json, e, &error, true)) return false;
        pending.push_back(std::move(e));
        return true;
    }
    json11::Json readableJson() const override
    {
        json11::Json::array a;
        for (const E& e : readable) a.push_back(toJson(e));
        return a;
    }
    std::vector<FieldInfo> fields() const override { return describeFields<E>(); }

    SchemaType describe() const override { return describeType<E>(SchemaKind::Event, E::id); }
    void describeNested(std::vector<SchemaType>& out) const override { collectNestedTypes<E>(out); }
    void appendPending(std::vector<uint8_t>& out) const override
    {
        BinaryWriter w(out);
        for (const E& e : pending) w.write(e);
        // An error in every build (e.g. an enum value without a name): never a block the reader rejects
        if (!w.ok()) throw std::runtime_error(std::string("save: event ") + name + ": " + w.error());
        if (pending.size() > kMaxEmptyElements && minEncodedSize<E>() == 0)
            throw std::runtime_error(std::string("save: more than ") + std::to_string(kMaxEmptyElements) +
                                     " pending events " + name + " without bytes");
    }
    std::unique_ptr<StagedValues> newStage() const override { return std::make_unique<StagedOf<E>>(); }
    bool stageBinary(StagedValues& stage, BinaryReader& r) const override
    {
        E e{};
        r.read(e);
        if (!r.ok()) return false;
        static_cast<StagedOf<E>&>(stage).values.push_back(std::move(e));
        return true;
    }
    bool stageJson(StagedValues& stage, const json11::Json& json, std::string& error) const override
    {
        E e{};
        if (!fromJson(json, e, &error, false)) return false;
        static_cast<StagedOf<E>&>(stage).values.push_back(std::move(e));
        return true;
    }
    void restore(StagedValues* stage) override
    {
        pending.clear();
        readable.clear();
        if (stage) pending = std::move(static_cast<StagedOf<E>*>(stage)->values);
    }

    std::vector<E> pending, readable;
};

/**
 * world.events (docs/engine_design.md §7.7): emit<E>(e) appends to the pending events; at the start of every step the
 * pending events become readable (read<E>(), emit order) and the previously readable ones are dropped. So events
 * emitted in step k, or between steps k and k+1, are readable in all phases of step k+1, then cleared. Pending events
 * are part of the hash (and of saves, S11). Not inside parallelEach (asserted).
 */
class SAIGA_ENGINE_API EventQueues
{
   public:
    // A pointer, so the owner can pass a member that is constructed later (only the address is taken here)
    explicit EventQueues(const Registry* registry) : registry(*registry) {}
    EventQueues(const EventQueues&)            = delete;
    EventQueues& operator=(const EventQueues&) = delete;

    template <class E>
    void emit(const E& e);
    template <class E>
    const std::vector<E>& read() const;

    // Engine-internal (EventRegistry, World)
    template <class E>
    void registerEvent(Replicate mode);
    // Contiguous IDs: engine events 0..n, game events 64..m (like components)
    void validateRegistration() const;
    void beginStep();
    void abortStep();
    void hashPending(Hasher& h) const;
    bool registered(EventId id) const { return id < queues.size() && queues[id] != nullptr; }
    // Engine-internal (world API): the queue by name or ID, nullptr if none
    EventQueueBase* find(const std::string& name) const;
    EventQueueBase* find(EventId id) const { return registered(id) ? queues[id].get() : nullptr; }
    size_t registeredCount() const;
    // All queues by ID (null: not registered), for saves (S11)
    const std::vector<std::unique_ptr<EventQueueBase>>& all() const { return queues; }

   private:
    template <class E>
    EventQueue<E>& queue() const;
    void checkNotParallel() const;

    const Registry& registry;
    std::vector<std::unique_ptr<EventQueueBase>> queues;  // by EventId
};

template <class E>
void EventQueues::registerEvent(Replicate mode)
{
    static_assert(std::is_same_v<std::remove_cv_t<decltype(E::id)>, EventId>, "E::id must be an EventId");
    static_assert(std::is_nothrow_move_constructible_v<E>, "events must have noexcept move construction");
    checkVisitFields<E>();
    for (auto& q : queues)
        SAIGA_ASSERT(!q || std::strcmp(q->name, E::name) != 0,
                     std::string("registerEvent: the name ") + E::name + " is already used");
    if (queues.size() <= E::id) queues.resize(size_t(E::id) + 1);
    // In every build: replacing a registered queue would hand its events to another type
    if (queues[E::id])
        SAIGA_EXIT_ERROR(std::string("registerEvent: ID ") + std::to_string(int(E::id)) + " of " + E::name +
                         " is already used by " + queues[E::id]->name);
    queues[E::id] = std::make_unique<EventQueue<E>>(mode);
}

template <class E>
EventQueue<E>& EventQueues::queue() const
{
    SAIGA_ASSERT(registered(E::id), std::string("event ") + E::name + " is not registered");
    EventQueueBase* q = queues[E::id].get();
    SAIGA_ASSERT(
        q->name == E::name || std::strcmp(q->name, E::name) == 0,
        std::string("event ID ") + std::to_string(int(E::id)) + " is registered as " + q->name + ", not " + E::name);
    SAIGA_ASSERT(q->holds<E>(), std::string("event ") + E::name + " is another C++ type than the registered one");
    return static_cast<EventQueue<E>&>(*q);
}

template <class E>
void EventQueues::emit(const E& e)
{
    checkNotParallel();
    queue<E>().pending.push_back(e);
}

template <class E>
const std::vector<E>& EventQueues::read() const
{
    return queue<E>().readable;
}

}  // namespace Engine
}  // namespace Saiga
