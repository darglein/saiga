/**
 * Copyright (c) 2017 Darius Rückert
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
 *
 */
class SAIGA_GLOBAL FrameSync
{
   public:
    vk::Semaphore imageVailable;
    vk::Semaphore renderComplete;
    vk::Fence frameFence;

    void create(vk::Device device);
    void destroy(vk::Device device);

    void wait(vk::Device device);
};

}  // namespace Vulkan
}  // namespace Saiga
