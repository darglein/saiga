/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

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
        if (entry != o.entry) reset();
        entry = o.entry;
        return *this;
    }
    DataRef& operator=(DataRef&& o) noexcept
    {
        if (entry != o.entry) reset();
        entry = std::move(o.entry);
        return *this;
    }

    template <class V>
    void visit(V& v)
    {
        v("entry", entry);
        // A changed name (a load, an API write) resolves again
        reset();
    }

    // Engine-internal cache (World::data)
    mutable const T* cached             = nullptr;
    mutable const DataStore* resolvedIn = nullptr;
    void reset() const
    {
        cached     = nullptr;
        resolvedIn = nullptr;
    }
};

}  // namespace Engine
}  // namespace Saiga
