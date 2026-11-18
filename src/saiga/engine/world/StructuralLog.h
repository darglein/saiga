/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

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
    };

    // Reserves room for n more records: a mutation calls prepare before it changes anything and append after it
    // succeeded, so the record cannot be lost to an allocation failure after the change (append does not allocate then)
    // Reserves room for n more records: a mutation calls prepare before it changes anything and append after it
    // succeeded, so the record cannot be lost to an allocation failure after the change (append does not allocate
    // then). Reservations are counted: a nested change (e.g. from a component's constructor) reserves and uses its own
    // slot, never the one promised to the outer change.
    void prepare(size_t n)
    {
        reserved += n;
        const size_t needed = records.size() + reserved;
        // Geometric growth: reserving exactly the needed size made every prepare reallocate (quadratic)
        if (records.capacity() < needed) records.reserve(std::max(needed, 2 * records.capacity()));
    }
    void append(Entity e, ComponentId component, bool added)
    {
        if (reserved > 0) reserved--;
        records.push_back(Record{e, component, added});
    }

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

    // End of a step: drops the records every consumer has read
    void truncate();
    size_t size() const { return records.size(); }

   private:
    std::vector<Record> records;    // a vector: prepare can reserve
    uint64_t firstRecord = 0;       // absolute number of records[0]
    std::vector<uint64_t> cursors;  // absolute position of each consumer
    std::vector<bool> active;
    std::vector<bool> consuming;  // consume() of the consumer runs (reentry asserts)
    size_t reserved = 0;  // slots promised by prepare and not yet appended       // removed consumers keep their index,
                          // not their cursor
};

}  // namespace Engine
}  // namespace Saiga
