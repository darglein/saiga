/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/export.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * Every applied structural change (docs/engine_design.md §7.6), also between steps: (entity, component, added or
 * removed). Creating an entity appends a Transform record, destroying it one removed record per component in ID order
 * (Transform first). Engine-internal consumers (render sync, physics, network) each keep a cursor and consume at fixed
 * points; at the end of every step the log is truncated up to the smallest cursor (everything without consumers). Not
 * an observer API.
 */
class SAIGA_ENGINE_API StructuralLog
{
   public:
    struct Record
    {
        Entity entity;
        ComponentId component = 0;
        bool added            = false;
        // S25: a removal by the engine's unload (the entity went into an entity group of its chunk, no onDestroy);
        // consumers that do not care treat it as a removal (replication sends Despawn instead of Destroy)
        bool unloaded = false;
    };

    // Records reserved by prepare: append publishes one without allocation; the slots not appended are given back
    // when it goes out of scope (a mutation that failed after prepare leaves no reservation behind)
    class Prepared
    {
       public:
        Prepared(const Prepared&)            = delete;
        Prepared& operator=(const Prepared&) = delete;
        ~Prepared() { log.reserved -= left; }
        void append(Entity e, ComponentId component, bool added, bool unloaded = false)
        {
            SAIGA_ASSERT(left > 0, "StructuralLog: more records appended than prepared");
            left--;
            log.reserved--;
            log.records.push_back(Record{e, component, added, unloaded});
        }

       private:
        friend class StructuralLog;
        Prepared(StructuralLog& log, size_t n) : log(log), left(n) {}
        StructuralLog& log;
        size_t left;
    };

    // Reserves room for n more records: a mutation calls prepare before it changes anything and appends through the
    // result after it succeeded, so the record cannot be lost to an allocation failure after the change (append does
    // not allocate then). Reservations are counted: a nested change (e.g. from a component's constructor) reserves and
    // uses its own slot, never the one promised to the outer change. If the reserve throws, nothing is reserved.
    Prepared prepare(size_t n)
    {
        const size_t needed = records.size() + reserved + n;
        // Geometric growth: reserving exactly the needed size made every prepare reallocate (quadratic)
        if (records.capacity() < needed) records.reserve(std::max(needed, 2 * records.capacity()));
        reserved += n;
        return Prepared(*this, n);
    }
    // Slots promised by prepare and not yet appended or given back
    size_t reservedCount() const { return reserved; }

    // A new consumer starts at the current end of the log
    int addConsumer();
    // A consumer that stops reading (a replaced render sink, a disconnected peer) must be removed, or the log grows
    void removeConsumer(int consumer);
    // Calls fn for every record after the consumer's cursor, in order. The cursor moves past a record after fn
    // returned: a throwing fn leaves the record unread (delivered again by the next consume; callbacks must tolerate
    // that). Stops if fn removes the consumer.
    void consume(int consumer, const std::function<void(const Record&)>& fn);
    // Number of records the consumer has not read yet
    size_t unread(int consumer) const;
    // The absolute number (appendedTotal numbering) of the consumer's next record
    uint64_t cursor(int consumer) const { return cursors[size_t(consumer)]; }

    // Moves the consumer back to an earlier cursor (an aborted step re-reads what its replication consumed, S16). Safe
    // because the log is truncated only after a completed step: the records from cursor on are still there.
    void rewind(int consumer, uint64_t cursor);

    // End of a step: drops the records every consumer has read
    void truncate();
    size_t size() const { return records.size(); }
    // Records appended since the log was created (truncation does not change it)
    uint64_t appendedTotal() const { return firstRecord + records.size(); }

   private:
    std::vector<Record> records;    // a vector: prepare can reserve
    uint64_t firstRecord = 0;       // absolute number of records[0]
    std::vector<uint64_t> cursors;  // absolute position of each consumer
    std::vector<bool> active;
    std::vector<bool> consuming;  // consume() of the consumer runs (reentry asserts)
    size_t reserved = 0;          // slots promised by prepare and not yet appended or given back
};

}  // namespace Engine
}  // namespace Saiga
