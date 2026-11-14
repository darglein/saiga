//
// Created by Peter Eichinger on 14.11.18.
//

#pragma once

#include "saiga/core/util/singleton.h"

#include <mutex>
#include <vulkan/vulkan.hpp>
namespace Saiga::Vulkan::Memory
{
using vk::AllocationCallbacks;
using vk::Device;
using vk::DeviceMemory;
using vk::MemoryAllocateInfo;
using vk::Optional;
using vk::Result;

class SafeAllocator : public Singleton<SafeAllocator>
{
   private:
    std::mutex allocMutex;

   public:
    SafeAllocator() = default;

    SafeAllocator(const SafeAllocator&)            = delete;
    SafeAllocator& operator=(const SafeAllocator&) = delete;

    template <typename Dispatch = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>
    inline Result allocateMemory(const Device device, const MemoryAllocateInfo* pAllocateInfo,
                                 const AllocationCallbacks* pAllocator, DeviceMemory* pMemory,
                                 Dispatch const& d = Dispatch())
    {
        allocMutex.lock();
        auto result = device.allocateMemory(pAllocateInfo, pAllocator, pMemory, d);
        allocMutex.unlock();
        return result;
    }

    template <typename Dispatch = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>
    DeviceMemory allocateMemory(const Device device, const MemoryAllocateInfo& allocateInfo,
                                Optional<const AllocationCallbacks> allocator = nullptr, Dispatch const& d = Dispatch())
    {
        allocMutex.lock();
        auto memory = device.allocateMemory(allocateInfo, allocator, d);
        allocMutex.unlock();
        return memory;
    }
};

}  // namespace Saiga::Vulkan::Memory