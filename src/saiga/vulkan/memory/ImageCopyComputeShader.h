//
// Created by Peter Eichinger on 2019-02-14.
//

#pragma once

#include "ImageMemoryLocation.h"

#include <mutex>
#include <optional>
namespace Saiga::Vulkan
{
struct VulkanBase;
class ComputePipeline;
}  // namespace Saiga::Vulkan

namespace Saiga::Vulkan::Memory
{
class VulkanMemory;

/**
 * Copies an image into a new image with the same create info (used by the image defragmenter). The source is sampled
 * in its current layout, so frames may read it concurrently.
 */
class ImageCopyComputeShader
{
   private:
    VulkanBase* base          = nullptr;
    ComputePipeline* pipeline = nullptr;
    vk::Sampler sampler       = nullptr;
    bool initialized          = false;
    // The shader is shared by the defragger threads of all image allocators.
    std::mutex mutex;

   public:
    ImageCopyComputeShader()                                         = default;
    ImageCopyComputeShader(const ImageCopyComputeShader&)            = delete;
    ImageCopyComputeShader& operator=(const ImageCopyComputeShader&) = delete;

    inline bool is_initialized() const { return initialized; }
    void init(VulkanBase* _base);

    void destroy();

    virtual ~ImageCopyComputeShader();

    /**
     * True if copy_image supports the image: a 2D R8G8B8A8_UNORM color image (matches the rgba8 storage image of
     * img_copy.comp) with one mip level and one layer, sampled + storage usage, in a layout it can be sampled in.
     */
    static bool can_copy(const ImageData& data);

    /**
     * Records the copy. The target must be a new image (layout undefined). After the copy it is in the layout of the
     * source. Returns the descriptor set used by the copy, which must be freed after the copy completed, or nullopt if
     * the copy could not be recorded.
     */
    std::optional<vk::DescriptorSet> copy_image(vk::CommandBuffer cmd, ImageMemoryLocation* target,
                                                ImageMemoryLocation* source);
};

}  // namespace Saiga::Vulkan::Memory
