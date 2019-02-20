/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/svulkan.h"

#include <optional>
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
    ~PipelineBase() { destroy(); }

    void init(VulkanBase& base, uint32_t numDescriptorSetLayouts);
    void destroy();

    void addDescriptorSetLayout(std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings, uint32_t id = 0);
    void addPushConstantRange(vk::PushConstantRange pcr);

    // ==== Runtime ====
    vk::DescriptorSet createDescriptorSet(uint32_t id = 0);


    SAIGA_WARN_UNUSED_RESULT bool bind(vk::CommandBuffer cmd);

    void bindDescriptorSets(vk::CommandBuffer cmd, vk::ArrayProxy<const vk::DescriptorSet> descriptorSets,
                            uint32_t firstSet = 0, vk::ArrayProxy<const uint32_t> dynamicOffsets = nullptr);
    void pushConstant(vk::CommandBuffer cmd, vk::ShaderStageFlags stage, size_t size, const void* data,
                      size_t offset = 0);

    std::optional<vk::ShaderStatisticsInfoAMD> get_statistics(vk::ShaderStageFlagBits stage_flags)
    {
        auto statistics = device.getShaderInfoAMD(pipeline, stage_flags, vk::ShaderInfoTypeAMD::eStatistics);

        return *reinterpret_cast<vk::ShaderStatisticsInfoAMD*>(statistics.data());
    }

   protected:
    VulkanBase* base = nullptr;
    vk::Device device;
    vk::PipelineBindPoint type;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    bool isInitialized() { return base; }
    void createPipelineLayout();

    virtual bool checkShader() = 0;
};


}  // namespace Vulkan
}  // namespace Saiga
