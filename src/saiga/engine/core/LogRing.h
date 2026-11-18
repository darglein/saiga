/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * Thread-safe ring buffer of log lines (docs/engine_design.md §17.1). Each line gets a sequence number (1, 2, ...);
 * readers ask for the lines after the last sequence number they saw (the MCP tool logs(since), S14). When full, the
 * oldest line is dropped.
 */
class SAIGA_ENGINE_API LogRing
{
   public:
    struct Line
    {
        uint64_t sequence = 0;
        std::string category;
        std::string message;
    };

    // capacity > 0 (std::invalid_argument otherwise)
    explicit LogRing(size_t capacity = 4096);

    // Returns the sequence number of the new line
    uint64_t add(std::string category, std::string message);

    // The kept lines with a sequence number > sequence, oldest first
    std::vector<Line> since(uint64_t sequence) const;

    // The sequence number of the newest line (0 before the first)
    uint64_t lastSequence() const;
    size_t capacity() const { return cap; }

    // Drops all lines; the sequence numbers continue
    void clear();

   private:
    const size_t cap;
    mutable std::mutex mutex;
    std::deque<Line> lines;
    uint64_t last = 0;
};

}  // namespace Engine
}  // namespace Saiga
