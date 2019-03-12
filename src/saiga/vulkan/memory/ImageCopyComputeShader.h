//
// Created by Peter Eichinger on 2019-02-14.
//

#pragma once

#include "ImageMemoryLocation.h"
namespace Saiga::Vulkan
{
struct VulkanBase;
class ComputePipeline;
}  // namespace Saiga::Vulkan

namespace Saiga::Vulkan::Memory
{
class VulkanMemory;


class ImageCopyComputeShader
{
   private:
    VulkanBase* base;
    ComputePipeline* pipeline;
    bool initialized = false;


   public:
    inline bool is_initialized() const { return initialized; }
    void init(VulkanBase* _base);

    void destroy();

    virtual ~ImageCopyComputeShader() { destroy(); }

    std::optional<vk::DescriptorSet> copy_image(vk::CommandBuffer cmd, ImageMemoryLocation* target,
                                                ImageMemoryLocation* source);
};

}  // namespace Saiga::Vulkan::Memory
