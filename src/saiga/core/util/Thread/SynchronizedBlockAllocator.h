/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include <vector>

namespace Saiga
{
template <typename T>
class SAIGA_TEMPLATE SynchronizedBlockAllocator
{
   protected:
    std::vector<char> buffer;
    std::vector<size_t> freeList;
    size_t capacity = 0;

    static constexpr size_t invalidIndex = -1;


   public:
    SynchronizedBlockAllocator(size_t capacity) : buffer(capacity * sizeof(T)), freeList(capacity), capacity(capacity)
    {
        for (auto i = 0; i < capacity; ++i)
        {
            freeList[i] = capacity - i - 1;
        }
    }
    ~SynchronizedBlockAllocator() { freeAll(); }


    void free(int id)
    {
        SAIGA_ASSERT(isConstructed(id));
        get(id).~T();
        constructed[id] = false;
    }


    void freeAll()
    {
        for (unsigned int i = 0; i < capacity; ++i)
        {
            if (isConstructed(i)) free(i);
        }
    }

   private:
    size_t nextFreeIndex()
    {
        if (freeList.empty()) return invalidIndex;
        auto i = freeList.back();
        freeList.pop_back();
        return i;
    }


    T* getPtr(int id)
    {
        char* ptr = buffer.data() + id * sizeof(T);
        return reinterpret_cast<T*>(ptr);
    }

    T& get(int id) { return *getPtr(id); }

    const T* getPtr(int id) const
    {
        const char* ptr = buffer.data() + id * sizeof(T);
        return reinterpret_cast<const T*>(ptr);
    }

    const T& get(int id) const { return *getPtr(id); }
};

}  // namespace Saiga
