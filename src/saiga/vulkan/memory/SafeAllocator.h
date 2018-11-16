//
// Created by Peter Eichinger on 14.11.18.
//

#pragma once

#include "saiga/util/singleton.h"
#include <mutex>
#include <vulkan/vulkan.hpp>
namespace Saiga{
namespace Vulkan{
namespace Memory{

using vk::Device;
using vk::Result;
using vk::Optional;
using vk::MemoryAllocateInfo;
using vk::AllocationCallbacks;
using vk::DeviceMemory;
using vk::DispatchLoaderStatic;

class SafeAllocator : public Singleton<SafeAllocator> {
private:
    std::mutex allocMutex;
public:

    template <typename Dispatch = DispatchLoaderStatic>
    inline Result allocateMemory(const Device device, const MemoryAllocateInfo* pAllocateInfo, const AllocationCallbacks* pAllocator, DeviceMemory* pMemory, Dispatch const &d = Dispatch()) {
        allocMutex.lock();
        auto result = device.allocateMemory(pAllocateInfo,pAllocator,pMemory, d);
        allocMutex.unlock();
        return result;
    }

    template <typename Dispatch = DispatchLoaderStatic>
    DeviceMemory allocateMemory(const Device device, const MemoryAllocateInfo & allocateInfo, Optional<const AllocationCallbacks> allocator = nullptr, Dispatch const &d = Dispatch()) {
        allocMutex.lock();
        auto memory = device.allocateMemory(allocateInfo, allocator, d);
        allocMutex.unlock();
        return memory;
    }
};

}
}
}