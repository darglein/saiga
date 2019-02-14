//
// Created by Peter Eichinger on 2019-01-21.
//

#include "Defragger.h"

#include "saiga/core/util/threadName.h"

namespace Saiga::Vulkan::Memory
{
void BufferDefragger::execute_defrag_operation(const BufferDefragger::DefragOperation& op)
{
    LOG(INFO) << "DEFRAG" << *(op.source) << "->" << op.targetMemory << "," << op.target.offset << " "
              << op.target.size;

    BufferMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);
    auto defrag_cmd                     = allocator->queue->commandPool.createAndBeginOneTimeBuffer();


    copy_buffer(defrag_cmd, reserve_space, op.source);

    defrag_cmd.end();

    allocator->queue->submitAndWait(defrag_cmd);

    allocator->queue->commandPool.freeCommandBuffer(defrag_cmd);

    allocator->move_allocation(reserve_space, op.source);
}

void ImageDefragger::execute_defrag_operation(const ImageDefragger::DefragOperation& op)
{
    LOG(INFO) << "IMAGE DEFRAG" << *(op.source) << "->" << op.targetMemory << "," << op.target.offset << " "
              << op.target.size;

    ImageMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);

    auto new_data = op.source->data;

    new_data.create_image(device);

    bind_image_data(device, reserve_space, new_data);

    reserve_space->data.create_view(device);

    auto defrag_cmd = allocator->queue->commandPool.createAndBeginOneTimeBuffer();
    copy_image(defrag_cmd, reserve_space, op.source);
    defrag_cmd.end();
    allocator->queue->submitAndWait(defrag_cmd);
    allocator->queue->commandPool.freeCommandBuffer(defrag_cmd);


    allocator->move_allocation(reserve_space, op.source);
}

}  // namespace Saiga::Vulkan::Memory
