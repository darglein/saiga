/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

#include "Buffer.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_GLOBAL DepthBuffer
{
   private:
    VulkanBase* base;

   public:
    MemoryLocation location;
    //    vk::DeviceMemory depthmem;

    // depth image
    vk::Format depthFormat;
    vk::Image depthimage;
    vk::ImageView depthview;

    void init(Saiga::Vulkan::VulkanBase& base, int width, int height);
    void destroy();
};

}  // namespace Vulkan
}  // namespace Saiga
