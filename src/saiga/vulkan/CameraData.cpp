/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CameraData.h"

#include "saiga/vulkan/VulkanCamera.h"
namespace Saiga
{
namespace Vulkan
{
void CameraData::create(VulkanBase& vulkanDevice)
{
    buffer.init(vulkanDevice, &data, sizeof(VulkanCameraData));

    std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;

    //    addDescriptorSetLayout({{7, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex}});


    vk::DescriptorSetLayoutCreateInfo descriptorLayout(vk::DescriptorSetLayoutCreateFlags(), setLayoutBindings.size(),
                                                       setLayoutBindings.data());
    descriptorSetLayout = vulkanDevice.device.createDescriptorSetLayout(descriptorLayout);
    SAIGA_ASSERT(descriptorSetLayout);


    //    descriptorSet                           = createDescriptorSet();
    //    vk::DescriptorBufferInfo descriptorInfo = uniformBufferVS.getDescriptorInfo();
    //    device.updateDescriptorSets(
    //        {
    //            vk::WriteDescriptorSet(descriptorSet, 7, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr,
    //            &descriptorInfo,
    //                                   nullptr),
    //        },
    //        nullptr);
}

}  // namespace Vulkan
}  // namespace Saiga
