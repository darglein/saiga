//
// Created by Peter Eichinger on 2019-01-21.
//

#include "Defragger.h"

#include "saiga/core/util/threadName.h"

#include "ImageCopyComputeShader.h"

#include <optional>
namespace Saiga::Vulkan::Memory
{
std::optional<BufferDefragger::FreeOperation> BufferDefragger::execute_defrag_operation(
    const BufferDefragger::DefragOperation& op)
{
    VLOG(1) << "DEFRAG" << *(op.source) << "->" << op.targetMemory << "," << op.target.offset << " " << op.target.size;

    BufferMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);
    auto defrag_cmd                     = allocator->queue->commandPool.createAndBeginOneTimeBuffer();


    copy_buffer(defrag_cmd, reserve_space, op.source);

    defrag_cmd.end();

    allocator->queue->submitAndWait(defrag_cmd);

    allocator->queue->commandPool.freeCommandBuffer(defrag_cmd);

    // allocator->move_allocation(reserve_space, op.source);
    return std::optional<BufferDefragger::FreeOperation>(
        BufferDefragger::FreeOperation{reserve_space, op.source, dealloc_delay});
}

std::optional<ImageDefragger::FreeOperation> ImageDefragger::execute_defrag_operation(
    const ImageDefragger::DefragOperation& op)
{
    ImageMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);



    auto new_data = op.source->data;
    new_data.create_image(device);

    bind_image_data(device, reserve_space, std::move(new_data));
    reserve_space->data.create_view(device);
    reserve_space->data.create_sampler(device);

    VLOG(1) << "IMAGE DEFRAG" << *(op.source) << "->" << *reserve_space;

    bool didcopy = img_copy_shader->copy_image(reserve_space, op.source);

    if (!didcopy)
    {
        return std::optional<ImageDefragger::FreeOperation>();
    }

    return std::optional<ImageDefragger::FreeOperation>(
        ImageDefragger::FreeOperation{reserve_space, op.source, dealloc_delay});
}

ImageDefragger::ImageDefragger(VulkanBase* base, vk::Device device, BaseChunkAllocator<ImageMemoryLocation>* allocator,
                               uint32_t dealloc_delay, ImageCopyComputeShader* _img_copy_shader)
    : Defragger(base, device, allocator, dealloc_delay), img_copy_shader(_img_copy_shader)
{
    if (!img_copy_shader->is_initialized())
    {
        LOG(ERROR) << "Image copy shader could not be loaded. Image defragmentation is not possible.";
        valid = false;
    }
}

}  // namespace Saiga::Vulkan::Memory
