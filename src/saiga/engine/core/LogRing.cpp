/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/core/LogRing.h"

#include "saiga/core/util/assert.h"

#include <cstddef>
#include <utility>

namespace Saiga
{
namespace Engine
{
LogRing::LogRing(size_t capacity) : cap(capacity)
{
    SAIGA_ASSERT(capacity > 0);
}

uint64_t LogRing::add(std::string category, std::string message)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (lines.size() == cap) lines.pop_front();
    Line line;
    line.sequence = ++last;
    line.category = std::move(category);
    line.message  = std::move(message);
    lines.push_back(std::move(line));
    return last;
}

std::vector<LogRing::Line> LogRing::since(uint64_t sequence) const
{
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<Line> result;
    if (lines.empty() || sequence >= last) return result;
    // The sequence numbers of the kept lines are consecutive: index of the first newer line
    uint64_t first = lines.front().sequence;
    size_t start   = sequence < first ? 0 : size_t(sequence - first + 1);
    result.assign(lines.begin() + std::ptrdiff_t(start), lines.end());
    return result;
}

uint64_t LogRing::lastSequence() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return last;
}

void LogRing::clear()
{
    std::lock_guard<std::mutex> lock(mutex);
    lines.clear();
}

}  // namespace Engine
}  // namespace Saiga
