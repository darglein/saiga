/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/svulkan.h"

struct VulkanCameraData;

namespace Saiga
{
namespace Vulkan
{
/**
 * A uniform buffer + descriptor set for the camera data.
 * Many different render modules require these parameters, therefore
 * it is efficient to share the uniform buffer.
 */
class SAIGA_VULKAN_API CameraData
{
   public:
    /**
     * Allocates the buffer and descriptor set.
     */
    void create(VulkanBase& vulkanDevice);

   private:
    UniformBuffer buffer;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    std::unique_ptr<VulkanCameraData> data;
};


}  // namespace Vulkan
}  // namespace Saiga
