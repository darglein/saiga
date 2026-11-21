/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// StructuralLog: consumer cursors and truncation.

#include "saiga/engine/world/StructuralLog.h"

#include "saiga/core/util/assert.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
int StructuralLog::addConsumer()
{
    cursors.push_back(firstRecord + records.size());
    active.push_back(true);
    consuming.push_back(false);
    return int(cursors.size()) - 1;
}

void StructuralLog::removeConsumer(int consumer)
{
    SAIGA_ASSERT(consumer >= 0 && size_t(consumer) < cursors.size() && active[consumer],
                 "StructuralLog::removeConsumer: unknown consumer");
    active[consumer] = false;
}

void StructuralLog::consume(int consumer, const std::function<void(const Record&)>& fn)
{
    SAIGA_ASSERT(consumer >= 0 && size_t(consumer) < cursors.size() && active[consumer],
                 "StructuralLog::consume: unknown consumer");
    // Records appended by fn are consumed in this call too
    // A callback that consumes the same consumer again would get the current record twice and move the cursor past
    // the end afterwards
    SAIGA_ASSERT(!consuming[consumer], "StructuralLog::consume: reentrant call for the same consumer");
    if (consuming[consumer]) return;
    struct Guard
    {
        std::vector<bool>& flags;
        int i;
        ~Guard() { flags[i] = false; }
    } guard{consuming, consumer};
    consuming[consumer] = true;
    while (active[consumer] && cursors[consumer] < firstRecord + records.size())
    {
        const Record r = records[size_t(cursors[consumer] - firstRecord)];
        fn(r);
        cursors[consumer]++;
    }
}

size_t StructuralLog::unread(int consumer) const
{
    SAIGA_ASSERT(consumer >= 0 && size_t(consumer) < cursors.size() && active[consumer],
                 "StructuralLog::unread: unknown consumer");
    return size_t(firstRecord + records.size() - cursors[consumer]);
}

void StructuralLog::rewind(int consumer, uint64_t cursor)
{
    SAIGA_ASSERT(consumer >= 0 && size_t(consumer) < cursors.size() && active[consumer],
                 "StructuralLog::rewind: unknown consumer");
    SAIGA_ASSERT(cursor >= firstRecord && cursor <= cursors[consumer], "StructuralLog::rewind: the records are gone");
    cursors[consumer] = std::max(firstRecord, std::min(cursor, cursors[consumer]));
}

void StructuralLog::truncate()
{
    uint64_t end = firstRecord + records.size();
    for (size_t i = 0; i < cursors.size(); ++i)
        if (active[i]) end = std::min(end, cursors[i]);
    size_t drop = size_t(end - firstRecord);
    records.erase(records.begin(), records.begin() + std::ptrdiff_t(drop));
    firstRecord = end;
}

}  // namespace Engine
}  // namespace Saiga
