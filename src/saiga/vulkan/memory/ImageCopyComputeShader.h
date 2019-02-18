//
// Created by Peter Eichinger on 2019-02-14.
//

#pragma once

//#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "MemoryLocation.h"
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

   public:
    void init(VulkanBase* _base);

    void destroy();

    virtual ~ImageCopyComputeShader() {destroy();};

    bool copy_image(ImageMemoryLocation* target, ImageMemoryLocation* source);
};

}  // namespace Saiga::Vulkan::Memory
