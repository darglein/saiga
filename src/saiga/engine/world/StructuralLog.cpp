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
    return int(cursors.size()) - 1;
}

void StructuralLog::consume(int consumer, const std::function<void(const Record&)>& fn)
{
    SAIGA_ASSERT(consumer >= 0 && size_t(consumer) < cursors.size(), "StructuralLog::consume: unknown consumer");
    // Records appended by fn are consumed in this call too
    while (cursors[consumer] < firstRecord + records.size())
    {
        const Record r = records[size_t(cursors[consumer] - firstRecord)];
        cursors[consumer]++;
        fn(r);
    }
}

size_t StructuralLog::unread(int consumer) const
{
    SAIGA_ASSERT(consumer >= 0 && size_t(consumer) < cursors.size(), "StructuralLog::unread: unknown consumer");
    return size_t(firstRecord + records.size() - cursors[consumer]);
}

void StructuralLog::truncate()
{
    uint64_t end = firstRecord + records.size();
    for (uint64_t c : cursors) end = std::min(end, c);
    size_t drop = size_t(end - firstRecord);
    records.erase(records.begin(), records.begin() + std::ptrdiff_t(drop));
    firstRecord = end;
}

}  // namespace Engine
}  // namespace Saiga
