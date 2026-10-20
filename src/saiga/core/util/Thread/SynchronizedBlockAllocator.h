/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace Saiga
{
/**
 * A thread safe pool of 'capacity' objects of type T in one contiguous block of memory.
 *
 * allocate() constructs an object in a free slot and returns its id, or invalidIndex if all slots are used.
 * free(id) destroys the object and returns the slot to the pool. The objects are not moved, so a reference returned by
 * get(id) stays valid until free(id). Only the bookkeeping is synchronized: accessing the same object from multiple
 * threads requires external synchronization.
 */
template <typename T>
class SAIGA_TEMPLATE SynchronizedBlockAllocator
{
   public:
    static constexpr int invalidIndex = -1;

    explicit SynchronizedBlockAllocator(int capacity)
        : storage(new Storage[capacity]), constructed(capacity, false), capacity(capacity)
    {
        SAIGA_ASSERT(capacity >= 0);
        freeList.reserve(capacity);
        // The lowest ids are allocated first.
        for (int i = capacity - 1; i >= 0; --i)
        {
            freeList.push_back(i);
        }
    }
    ~SynchronizedBlockAllocator() { freeAll(); }

    SynchronizedBlockAllocator(const SynchronizedBlockAllocator&)            = delete;
    SynchronizedBlockAllocator& operator=(const SynchronizedBlockAllocator&) = delete;

    template <typename... Args>
    int allocate(Args&&... args)
    {
        int id;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (freeList.empty()) return invalidIndex;
            id = freeList.back();
            freeList.pop_back();
        }
        // The slot is owned by this thread now, so the constructor can run without the lock.
        try
        {
            new (getPtr(id)) T(std::forward<Args>(args)...);
        }
        catch (...)
        {
            std::unique_lock<std::mutex> lock(mutex);
            freeList.push_back(id);
            throw;
        }
        std::unique_lock<std::mutex> lock(mutex);
        constructed[id] = true;
        return id;
    }

    void free(int id)
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            SAIGA_ASSERT(isConstructedNoLock(id));
            constructed[id] = false;
        }
        get(id).~T();
        std::unique_lock<std::mutex> lock(mutex);
        freeList.push_back(id);
    }

    void freeAll()
    {
        for (int i = 0; i < capacity; ++i)
        {
            if (isConstructed(i)) free(i);
        }
    }

    bool isConstructed(int id)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return isConstructedNoLock(id);
    }

    // Number of currently allocated objects.
    int size()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return capacity - int(freeList.size());
    }

    int getCapacity() const { return capacity; }

    T& get(int id)
    {
        SAIGA_DEBUG_ASSERT(id >= 0 && id < capacity);
        return *getPtr(id);
    }
    const T& get(int id) const
    {
        SAIGA_DEBUG_ASSERT(id >= 0 && id < capacity);
        return *getPtr(id);
    }

   private:
    struct Storage
    {
        alignas(T) unsigned char data[sizeof(T)];
    };

    std::unique_ptr<Storage[]> storage;
    std::vector<bool> constructed;
    std::vector<int> freeList;
    int capacity;
    std::mutex mutex;

    bool isConstructedNoLock(int id) const { return id >= 0 && id < capacity && constructed[id]; }

    T* getPtr(int id) { return std::launder(reinterpret_cast<T*>(storage[id].data)); }
    const T* getPtr(int id) const { return std::launder(reinterpret_cast<const T*>(storage[id].data)); }
};

}  // namespace Saiga
