/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/assert.h"
#include "saiga/export.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <utility>
#include <vulkan/vulkan.hpp>

// The handle of a VMA allocation (vk_mem_alloc.h declares it the same way). The saiga headers do not include VMA.
typedef struct VmaAllocation_T* VmaAllocation;

namespace Saiga::Vulkan::Memory
{
/**
 * Locks the mutex of the given locations while their handles are read. The defragmentation and
 * VulkanMemory::mark_dynamic swap the handles of a location under this lock. It is locked for static locations as well:
 * mark_dynamic() may make a location dynamic concurrently, so checking is_dynamic() first would race with the swap.
 */
template <typename T>
struct SafeAccessor final
{
   private:
    std::unique_lock<std::mutex> lock1, lock2;

   public:
    explicit SafeAccessor(T& _location) : lock1(_location.mutex) {}

    SafeAccessor(T& _location, T& _location2)
        : lock1(_location.mutex, std::defer_lock), lock2(_location2.mutex, std::defer_lock)
    {
        if (&_location == &_location2)
        {
            lock1.lock();
            return;
        }
        std::lock(lock1, lock2);
    }

    SafeAccessor(const SafeAccessor&)            = delete;
    SafeAccessor(SafeAccessor&&)                 = delete;
    SafeAccessor& operator=(const SafeAccessor&) = delete;
    SafeAccessor& operator=(SafeAccessor&&)      = delete;
};

/**
 * A buffer or image allocated by VulkanMemory (through VMA).
 *
 * The location object stays valid until it is deallocated. A dynamic location (see VulkanMemory::mark_dynamic) can be
 * moved by the defragmentation: the handles in data (the buffer, or the image, view and sampler) are then replaced.
 * Read them again each time they are used (or through SafeAccessor) instead of caching them.
 */
template <typename Data>
struct SAIGA_VULKAN_API BaseMemoryLocation
{
    using Clock     = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    Data data;
    VmaAllocation allocation = nullptr;
    // The requested size in bytes (buffers) or the size of the memory (images).
    vk::DeviceSize size = 0;
    // Persistently mapped pointer, if the location was requested with host visible memory.
    void* mappedPointer = nullptr;
    std::mutex mutex;
    // The last time the handles in data were changed (by the defragmentation). Guarded by mutex, see modifiedTime().
    TimePoint modified_time = Clock::now();

   private:
    // Atomic: read without the mutex by is_dynamic() / is_static().
    std::atomic<bool> static_mem{true};

   public:
    explicit BaseMemoryLocation(Data _data = nullptr, VmaAllocation _allocation = nullptr, vk::DeviceSize _size = 0,
                                void* _mappedPointer = nullptr)
        : data(std::move(_data)), allocation(_allocation), size(_size), mappedPointer(_mappedPointer)
    {
    }

    BaseMemoryLocation(const BaseMemoryLocation& other)            = delete;
    BaseMemoryLocation& operator=(const BaseMemoryLocation& other) = delete;

    explicit operator bool() { return allocation != nullptr; }

    inline bool is_dynamic() const { return !is_static(); }
    inline bool is_static() const { return static_mem; }

    TimePoint modifiedTime()
    {
        std::scoped_lock lock(mutex);
        return modified_time;
    }

    // Only VulkanMemory changes this (it moves the location into a defragmentable pool).
    inline void set_dynamic(bool dynamic) { static_mem = !dynamic; }

    // Copies data_size bytes into the mapped memory. Only for host visible locations.
    void upload(vk::Device, const void* src, size_t data_size)
    {
        SAIGA_ASSERT(data_size <= size, "data_size is too large");
        SAIGA_ASSERT(mappedPointer, "The memory is not host visible, use a staged upload");
        std::memcpy(mappedPointer, src, data_size);
    }

    // Copies size bytes from the mapped memory. Only for host visible locations.
    void download(vk::Device, void* dst)
    {
        SAIGA_ASSERT(mappedPointer, "The memory is not host visible, use a staged download");
        std::memcpy(dst, mappedPointer, size);
    }

    // Destroys the Vulkan objects in data (not the memory).
    inline void destroy_owned_data(const vk::Device& device)
    {
        if (data)
        {
            data.destroy_owned_data(device);
            data = nullptr;
        }
    }

    void modified() { modified_time = Clock::now(); }

    friend std::ostream& operator<<(std::ostream& os, const BaseMemoryLocation& location)
    {
        os << "{" << location.allocation << ", " << location.data << ", " << location.size;
        if (location.mappedPointer)
        {
            os << ", " << location.mappedPointer;
        }
        os << "}";
        return os;
    }
};

}  // namespace Saiga::Vulkan::Memory
