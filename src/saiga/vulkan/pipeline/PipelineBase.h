/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/svulkan.h"

#include "DescriptorSet.h"
#include "DescriptorSetLayout.h"
namespace Saiga
{
namespace Vulkan
{
/**
 * Base class for both a graphics and compute pipeline
 */

class SAIGA_VULKAN_API PipelineBase
{
   public:
    // ==== Initialization ====
    PipelineBase(vk::PipelineBindPoint type);
    virtual ~PipelineBase() { destroy(); }

    void init(VulkanBase& base, uint32_t numDescriptorSetLayouts);
    void destroy();

    void addDescriptorSetLayout(const DescriptorSetLayout& layout, uint32_t id = 0);
    void addPushConstantRange(vk::PushConstantRange pcr);

    // ==== Runtime ====
    vk::DescriptorSet createRawDescriptorSet(uint32_t id = 0);
    StaticDescriptorSet createDescriptorSet(uint32_t id = 0);
    DynamicDescriptorSet createDynamicDescriptorSet(uint32_t id = 0);

    SAIGA_WARN_UNUSED_RESULT bool bind(vk::CommandBuffer cmd);

    template <typename SetType>
    void bindDescriptorSet(vk::CommandBuffer cmd, SetType& descriptorSet, uint32_t firstSet = 0,
                           vk::ArrayProxy<const uint32_t> dynamicOffsets = nullptr)
    {
        descriptorSet.update();
        cmd.bindDescriptorSets(type, pipelineLayout, firstSet, static_cast<vk::DescriptorSet>(descriptorSet),
                               dynamicOffsets);
    }

    void bindRawDescriptorSet(vk::CommandBuffer cmd, vk::DescriptorSet& descriptorSet, uint32_t firstSet = 0,
                              vk::ArrayProxy<const uint32_t> dynamicOffsets = nullptr)
    {
        cmd.bindDescriptorSets(type, pipelineLayout, firstSet, (descriptorSet), dynamicOffsets);
    }


    void pushConstant(vk::CommandBuffer cmd, vk::ShaderStageFlags stage, size_t size, const void* data,
                      size_t offset = 0);

   protected:
    VulkanBase* base = nullptr;
    vk::Device device;
    vk::PipelineBindPoint type;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    std::vector<DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    bool isInitialized() { return base; }
    void createPipelineLayout();

    virtual bool checkShader() = 0;
};


}  // namespace Vulkan
}  // namespace Saiga
