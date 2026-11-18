/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * An array that grows in pages of PageSize elements: growing never moves existing elements, so references stay valid
 * (the registry's slots and Transforms: a view callback may reserve an entity while it holds a Transform&). Elements
 * are default constructed when their page is allocated.
 */
template <class T, size_t PageSize = 4096>
class PagedArray
{
   public:
    size_t size() const { return count; }

    T& operator[](size_t i) { return (*pages[i / PageSize])[i % PageSize]; }
    const T& operator[](size_t i) const { return (*pages[i / PageSize])[i % PageSize]; }

    // Appends the next element (default constructed with its page; pages never shrink) and returns it. Allocates at
    // most one page and the page table; on failure nothing changes.
    T& push_back()
    {
        size_t page = count / PageSize;
        if (page == pages.size())
        {
            auto p = std::make_unique<std::array<T, PageSize>>();
            pages.push_back(std::move(p));
        }
        T& e = (*pages[page])[count % PageSize];
        count++;
        return e;
    }

    // Undoes the last push_back (the element and its page stay allocated; the next push_back returns it again, the
    // caller resets it)
    void pop_back() { count--; }

   private:
    std::vector<std::unique_ptr<std::array<T, PageSize>>> pages;
    size_t count = 0;
};

}  // namespace Engine
}  // namespace Saiga
