/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/export.h"

#include <cstdint>
#include <deque>
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

    void append(Entity e, ComponentId component, bool added) { records.push_back(Record{e, component, added}); }

    // A new consumer starts at the current end of the log
    int addConsumer();
    // Calls fn for every record after the consumer's cursor, in order, and moves the cursor to the end
    void consume(int consumer, const std::function<void(const Record&)>& fn);
    // Number of records the consumer has not read yet
    size_t unread(int consumer) const;

    // End of a step: drops the records every consumer has read
    void truncate();
    size_t size() const { return records.size(); }

   private:
    std::deque<Record> records;
    uint64_t firstRecord = 0;       // absolute number of records[0]
    std::vector<uint64_t> cursors;  // absolute position of each consumer
};

}  // namespace Engine
}  // namespace Saiga
