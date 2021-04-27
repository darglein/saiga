/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
/**
 * Helper objects to syncronize rendering with multiple buffers in the swap chain.
 *
 */
class SAIGA_VULKAN_API FrameSync
{
   public:
    vk::Semaphore imageAvailable = nullptr;
    vk::Semaphore renderComplete = nullptr;
    vk::Semaphore defragMayStart = nullptr;
    vk::Fence frameFence         = nullptr;

    ~FrameSync() { destroy(); }
    void create(vk::Device device);
    void destroy();

    void wait();

   private:
    vk::Device device;
};

}  // namespace Vulkan
}  // namespace Saiga
