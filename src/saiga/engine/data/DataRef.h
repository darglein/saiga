/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace Saiga
{
namespace Engine
{
class DataStore;

/**
 * A reference from a component to a data entry (docs/engine_design.md §14, S11): v("model", model) in visit. Saved and
 * shown as {"entry": name}; world.data(ref) resolves it once per data store (a cached pointer, not saved) and then
 * reads without a string lookup, so systems read tunables every step (R-REG-2) and a reload (entries updated in place)
 * reaches them at once.
 */
template <class T>
struct DataRef
{
    static constexpr const char* name = "DataRef";
    static constexpr uint32_t version = 1;

    std::string entry;

    DataRef() = default;
    explicit DataRef(std::string e) : entry(std::move(e)) {}
    DataRef(const DataRef& o) : entry(o.entry) {}
    DataRef(DataRef&& o) noexcept : entry(std::move(o.entry)) {}
    DataRef& operator=(const DataRef& o)
    {
        entry = o.entry;
        return *this;
    }
    DataRef& operator=(DataRef&& o) noexcept
    {
        entry = std::move(o.entry);
        return *this;
    }

    template <class V>
    void visit(V& v)
    {
        v("entry", entry);
    }

    // Engine-internal cache (World::data): the entry resolved for this store (by its unique ID, not its address) and
    // this name (by a hash of it: changing entry resolves again). Atomic: the first resolution may happen in
    // parallelEach on several threads (they store the same values).
    mutable std::atomic<const T*> cached{nullptr};
    mutable std::atomic<uint64_t> resolvedStore{0};
    mutable std::atomic<uint64_t> resolvedName{0};
};

}  // namespace Engine
}  // namespace Saiga
