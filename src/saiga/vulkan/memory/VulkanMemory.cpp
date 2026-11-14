/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanMemory.h"

#include "saiga/core/util/easylogging++.h"
#include "saiga/vulkan/Base.h"

#include <algorithm>
#include <limits>

#include <vk_mem_alloc.h>

namespace Saiga::Vulkan::Memory
{
namespace
{
// Memory barrier for all commands before -> the copies of a move.
void barrierBeforeCopy(vk::CommandBuffer cmd)
{
    vk::MemoryBarrier2 barrier(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllTransfer,
                               vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, barrier, nullptr, nullptr));
}

// The copies of a move -> all later commands, which use the new place.
void barrierAfterCopy(vk::CommandBuffer cmd)
{
    vk::MemoryBarrier2 barrier(vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
                               vk::PipelineStageFlagBits2::eAllCommands,
                               vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, barrier, nullptr, nullptr));
}

VmaAllocationCreateInfo allocationInfo(vk::MemoryPropertyFlags flags, bool map)
{
    VmaAllocationCreateInfo info = {};
    info.usage                   = VMA_MEMORY_USAGE_UNKNOWN;
    info.requiredFlags           = static_cast<VkMemoryPropertyFlags>(flags);
    if (map)
    {
        info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    return info;
}

bool isMovableImage(const ImageData& data)
{
    const auto usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    return data.image && (data.image_create_info.usage & usage) == usage;
}

}  // namespace

VulkanMemory::~VulkanMemory()
{
    destroy();
}

void VulkanMemory::init(VulkanBase* _base, uint32_t, const VulkanParameters& parameters)
{
    SAIGA_ASSERT(!allocator, "VulkanMemory is already initialized");
    base                  = _base;
    device                = base->device;
    enableDefragmentation = parameters.enableDefragmentation;

    VmaAllocatorCreateInfo info = {};
    info.vulkanApiVersion       = base->apiVersion;
    info.physicalDevice         = base->physicalDevice;
    info.device                 = base->device;
    info.instance               = base->getInstance();
    auto result                 = vmaCreateAllocator(&info, &allocator);
    SAIGA_ASSERT(result == VK_SUCCESS, "vmaCreateAllocator failed");
}

void VulkanMemory::destroy()
{
    if (!allocator) return;
    std::scoped_lock lock(mutex);

    if (pass.active)
    {
        if (!finishPass()) endDefragmentation();
    }
    if (defragContext)
    {
        endDefragmentation();
    }
    defragQueue.clear();

    if (!buffers.empty() || !images.empty())
    {
        VLOG(1) << "VulkanMemory::destroy: freeing " << buffers.size() << " buffers and " << images.size()
                << " images that were not deallocated";
    }
    std::vector<BufferMemoryLocation*> remainingBuffers;
    for (auto& entry : buffers) remainingBuffers.push_back(entry.first);
    for (auto* location : remainingBuffers) freeLocation(location);

    std::vector<ImageMemoryLocation*> remainingImages;
    for (auto& entry : images) remainingImages.push_back(entry.first);
    for (auto* location : remainingImages) freeLocation(location);

    for (auto& entry : movablePools)
    {
        vmaDestroyPool(allocator, entry.second);
    }
    movablePools.clear();

    vmaDestroyAllocator(allocator);
    allocator = nullptr;
}

BufferMemoryLocation* VulkanMemory::allocate(const BufferType& type, vk::DeviceSize size)
{
    SAIGA_ASSERT(size > 0, "Allocations must have a size > 0");
    SAIGA_ASSERT(allocator, "VulkanMemory is not initialized");

    // Transfer src/dst: staged uploads/downloads and the defragmentation copies.
    const auto usage = type.usageFlags | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::BufferCreateInfo bufferInfo({}, size, usage, vk::SharingMode::eExclusive);
    VkBufferCreateInfo vkBufferInfo = bufferInfo;

    const bool map         = type.is_mappable();
    auto createInfo        = allocationInfo(type.memoryFlags, map);
    VkBuffer buffer        = VK_NULL_HANDLE;
    VmaAllocation alloc    = nullptr;
    VmaAllocationInfo info = {};
    auto result            = vmaCreateBuffer(allocator, &vkBufferInfo, &createInfo, &buffer, &alloc, &info);
    if (result != VK_SUCCESS)
    {
        LOG(ERROR) << "Buffer allocation of " << size << " bytes " << type
                   << " failed: " << vk::to_string(vk::Result(result));
        return nullptr;
    }

    auto location = std::make_unique<BufferMemoryLocation>(BufferData(buffer, usage), alloc, size,
                                                           map ? info.pMappedData : nullptr);
    SAIGA_ASSERT(!map || location->mappedPointer, "The host visible memory is not mapped");
    auto* ptr = location.get();
    vmaSetAllocationUserData(allocator, alloc, ptr);

    std::scoped_lock lock(mutex);
    buffers.emplace(ptr, std::move(location));
    return ptr;
}

ImageMemoryLocation* VulkanMemory::allocate(const ImageType& type, ImageData& image_data)
{
    SAIGA_ASSERT(!image_data.image, "Image must not be created beforehand.");
    SAIGA_ASSERT(allocator, "VulkanMemory is not initialized");

    VkImageCreateInfo imageInfo = image_data.image_create_info;
    auto createInfo             = allocationInfo(type.memoryFlags, false);
    VkImage image               = VK_NULL_HANDLE;
    VmaAllocation alloc         = nullptr;
    VmaAllocationInfo info      = {};
    auto result                 = vmaCreateImage(allocator, &imageInfo, &createInfo, &image, &alloc, &info);
    if (result != VK_SUCCESS)
    {
        LOG(ERROR) << "Image allocation " << type << " failed: " << vk::to_string(vk::Result(result));
        return nullptr;
    }

    image_data.image                  = image;
    image_data.view_create_info.image = image;
    image_data.layout                 = image_data.image_create_info.initialLayout;
    image_data.create_view(device);
    image_data.create_sampler(device);

    auto location = std::make_unique<ImageMemoryLocation>(std::move(image_data), alloc, info.size, nullptr);
    auto* ptr     = location.get();
    vmaSetAllocationUserData(allocator, alloc, ptr);

    std::scoped_lock lock(mutex);
    images.emplace(ptr, std::move(location));
    return ptr;
}

void VulkanMemory::deallocate(BufferMemoryLocation* location)
{
    if (!location) return;
    std::scoped_lock lock(mutex);
    if (!allocator) return;
    SAIGA_ASSERT(buffers.count(location), "The buffer was not allocated by VulkanMemory");

    if (location->is_dynamic()) defragRequested = true;

    if (pass.active)
    {
        auto* moves = static_cast<VmaDefragmentationMove*>(pass.pMoves);
        for (uint32_t i = 0; i < pass.moveCount; ++i)
        {
            if (pass.moves[i].buffer == location)
            {
                // Must not be freed before the pass ends: VMA frees both places then.
                moves[i].operation      = VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY;
                pass.moves[i].destroyed = true;
                return;
            }
        }
    }
    freeLocation(location);
}

void VulkanMemory::deallocate(ImageMemoryLocation* location)
{
    if (!location) return;
    std::scoped_lock lock(mutex);
    if (!allocator) return;
    SAIGA_ASSERT(images.count(location), "The image was not allocated by VulkanMemory");

    if (location->is_dynamic()) defragRequested = true;

    if (pass.active)
    {
        auto* moves = static_cast<VmaDefragmentationMove*>(pass.pMoves);
        for (uint32_t i = 0; i < pass.moveCount; ++i)
        {
            if (pass.moves[i].image == location)
            {
                moves[i].operation      = VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY;
                pass.moves[i].destroyed = true;
                return;
            }
        }
    }
    freeLocation(location);
}

void VulkanMemory::freeLocation(BufferMemoryLocation* location)
{
    location->destroy_owned_data(device);
    vmaFreeMemory(allocator, location->allocation);
    buffers.erase(location);
}

void VulkanMemory::freeLocation(ImageMemoryLocation* location)
{
    location->destroy_owned_data(device);
    vmaFreeMemory(allocator, location->allocation);
    images.erase(location);
}

VmaPool VulkanMemory::getMovablePool(uint32_t memoryTypeIndex)
{
    auto found = movablePools.find(memoryTypeIndex);
    if (found != movablePools.end()) return found->second;

    VmaPoolCreateInfo info = {};
    info.memoryTypeIndex   = memoryTypeIndex;
    VmaPool pool           = nullptr;
    auto result            = vmaCreatePool(allocator, &info, &pool);
    if (result != VK_SUCCESS)
    {
        LOG(ERROR) << "vmaCreatePool failed for memory type " << memoryTypeIndex << ": "
                   << vk::to_string(vk::Result(result));
        return nullptr;
    }
    movablePools.emplace(memoryTypeIndex, pool);
    return pool;
}

bool VulkanMemory::mark_dynamic(BufferMemoryLocation* location)
{
    SAIGA_ASSERT(location);
    std::scoped_lock lock(mutex);
    if (location->is_dynamic()) return true;
    if (location->mappedPointer)
    {
        VLOG(1) << "mark_dynamic: host visible buffers can not be moved by the defragmentation";
        return false;
    }

    VmaAllocationInfo info = {};
    vmaGetAllocationInfo(allocator, location->allocation, &info);
    VmaPool pool = getMovablePool(info.memoryType);
    if (!pool) return false;

    // A copy of the buffer in the pool.
    vk::BufferCreateInfo bufferInfo({}, location->size, location->data.usage, vk::SharingMode::eExclusive);
    VkBufferCreateInfo vkBufferInfo    = bufferInfo;
    VmaAllocationCreateInfo createInfo = {};
    createInfo.pool                    = pool;
    VkBuffer buffer                    = VK_NULL_HANDLE;
    VmaAllocation alloc                = nullptr;
    auto result = vmaCreateBuffer(allocator, &vkBufferInfo, &createInfo, &buffer, &alloc, nullptr);
    if (result != VK_SUCCESS)
    {
        VLOG(1) << "mark_dynamic: allocation in the defragmentable pool failed: " << vk::to_string(vk::Result(result));
        return false;
    }

    // Copies after all previous submissions on the main queue.
    auto& queue = base->mainQueue;
    auto cmd    = queue.commandPool.createAndBeginOneTimeBuffer();
    barrierBeforeCopy(cmd);
    cmd.copyBuffer(location->data.buffer, buffer, vk::BufferCopy(0, 0, location->size));
    barrierAfterCopy(cmd);
    cmd.end();
    queue.submitAndWait(cmd);
    queue.commandPool.freeCommandBuffer(cmd);

    vk::Buffer oldBuffer;
    VmaAllocation oldAlloc;
    {
        std::scoped_lock locationLock(location->mutex);
        oldBuffer             = location->data.buffer;
        oldAlloc              = location->allocation;
        location->data.buffer = buffer;
        location->allocation  = alloc;
        location->set_dynamic(true);
        location->modified();
    }
    vmaSetAllocationUserData(allocator, alloc, location);

    // Frames in flight may still use the old buffer.
    auto dev = device;
    auto vma = allocator;
    base->deferDelete(
        [dev, vma, oldBuffer, oldAlloc]()
        {
            dev.destroy(oldBuffer);
            vmaFreeMemory(vma, oldAlloc);
        });
    return true;
}

bool VulkanMemory::mark_dynamic(ImageMemoryLocation* location)
{
    SAIGA_ASSERT(location);
    std::scoped_lock lock(mutex);
    if (location->is_dynamic()) return true;
    if (!isMovableImage(location->data))
    {
        VLOG(1) << "mark_dynamic: the image needs the usage transfer src and dst to be moved by the defragmentation";
        return false;
    }

    VmaAllocationInfo info = {};
    vmaGetAllocationInfo(allocator, location->allocation, &info);
    VmaPool pool = getMovablePool(info.memoryType);
    if (!pool) return false;

    // A copy of the image in the pool.
    ImageData newData                  = unboundImageCopy(location->data);
    VmaAllocationCreateInfo createInfo = {};
    createInfo.pool                    = pool;
    VmaAllocation alloc                = nullptr;
    auto result = vmaAllocateMemoryForImage(allocator, newData.image, &createInfo, &alloc, nullptr);
    if (result != VK_SUCCESS)
    {
        VLOG(1) << "mark_dynamic: allocation in the defragmentable pool failed: " << vk::to_string(vk::Result(result));
        device.destroy(newData.image);
        return false;
    }
    bindImageCopy(newData, alloc);

    auto& queue       = base->mainQueue;
    auto cmd          = queue.commandPool.createAndBeginOneTimeBuffer();
    ImageData oldData = nullptr;
    {
        std::scoped_lock locationLock(location->mutex);
        oldData = location->data;
    }
    recordImageCopy(cmd, oldData, newData);
    cmd.end();
    queue.submitAndWait(cmd);
    queue.commandPool.freeCommandBuffer(cmd);

    VmaAllocation oldAlloc;
    {
        std::scoped_lock locationLock(location->mutex);
        oldAlloc = location->allocation;
        // The handles of the old data are owned by oldData now.
        location->data       = std::move(newData);
        location->allocation = alloc;
        location->set_dynamic(true);
        location->modified();
    }
    vmaSetAllocationUserData(allocator, alloc, location);

    // Frames in flight may still use the old image, view and sampler.
    auto dev = device;
    auto vma = allocator;
    base->deferDelete(
        [dev, vma, oldData, oldAlloc]() mutable
        {
            oldData.destroy_owned_data(dev);
            vmaFreeMemory(vma, oldAlloc);
        });
    return true;
}

void VulkanMemory::flush(BufferMemoryLocation* location, vk::DeviceSize offset, vk::DeviceSize size)
{
    SAIGA_ASSERT(location && offset <= location->size);
    if (size == VK_WHOLE_SIZE) size = location->size - offset;
    auto result = vmaFlushAllocation(allocator, location->allocation, offset, size);
    SAIGA_ASSERT(result == VK_SUCCESS, "vmaFlushAllocation failed");
}

ImageData VulkanMemory::unboundImageCopy(const ImageData& src)
{
    ImageData dst = src;
    // The copied handles belong to src.
    dst.image   = nullptr;
    dst.view    = nullptr;
    dst.sampler = nullptr;

    dst.image                  = device.createImage(dst.image_create_info);
    dst.view_create_info.image = dst.image;
    dst.layout                 = vk::ImageLayout::eUndefined;
    return dst;
}

void VulkanMemory::bindImageCopy(ImageData& dst, VmaAllocation allocation)
{
    auto bound = vmaBindImageMemory(allocator, allocation, dst.image);
    SAIGA_ASSERT(bound == VK_SUCCESS, "vmaBindImageMemory failed");
    dst.create_view(device);
    dst.create_sampler(device);
}

void VulkanMemory::recordBufferCopy(vk::CommandBuffer cmd, vk::Buffer src, vk::Buffer dst, vk::DeviceSize size)
{
    barrierBeforeCopy(cmd);
    cmd.copyBuffer(src, dst, vk::BufferCopy(0, 0, size));
    barrierAfterCopy(cmd);
}

void VulkanMemory::recordImageCopy(vk::CommandBuffer cmd, ImageData& src, ImageData& dst)
{
    const auto layout = src.layout;
    if (layout == vk::ImageLayout::eUndefined)
    {
        // No content.
        return;
    }

    // The transitions wait for all previous accesses of the source.
    src.transitionImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
    dst.transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

    const auto& info  = src.image_create_info;
    const auto aspect = ImageData::aspectFlags(info.format);
    std::vector<vk::ImageCopy> regions;
    for (uint32_t mip = 0; mip < info.mipLevels; ++mip)
    {
        vk::ImageSubresourceLayers subresource(aspect, mip, 0, info.arrayLayers);
        vk::Extent3D extent(std::max(1u, info.extent.width >> mip), std::max(1u, info.extent.height >> mip),
                            std::max(1u, info.extent.depth >> mip));
        regions.emplace_back(subresource, vk::Offset3D(), subresource, vk::Offset3D(), extent);
    }
    cmd.copyImage(src.image, vk::ImageLayout::eTransferSrcOptimal, dst.image, vk::ImageLayout::eTransferDstOptimal,
                  regions);

    // Back to the layout of the source, visible to all later commands.
    dst.transitionImageLayout(cmd, layout == vk::ImageLayout::ePreinitialized ? vk::ImageLayout::eGeneral : layout);
}

bool VulkanMemory::beginDefragmentation(bool full)
{
    SAIGA_ASSERT(!defragContext && !pass.active);
    while (!defragQueue.empty())
    {
        VmaPool pool = defragQueue.back();
        defragQueue.pop_back();

        VmaDefragmentationInfo info = {};
        info.pool                   = pool;
        if (full)
        {
            info.flags = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT;
        }
        else
        {
            info.flags           = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT;
            info.maxBytesPerPass = defragBytesPerFrame;
        }
        auto result = vmaBeginDefragmentation(allocator, &info, &defragContext);
        if (result == VK_SUCCESS) return true;
        LOG(ERROR) << "vmaBeginDefragmentation failed: " << vk::to_string(vk::Result(result));
        defragContext = nullptr;
    }
    return false;
}

bool VulkanMemory::beginPass(bool wait)
{
    SAIGA_ASSERT(defragContext && !pass.active);

    VmaDefragmentationPassMoveInfo info = {};
    auto result                         = vmaBeginDefragmentationPass(allocator, defragContext, &info);
    if (result == VK_SUCCESS)
    {
        // Nothing to move.
        return false;
    }
    SAIGA_ASSERT(result == VK_INCOMPLETE, "vmaBeginDefragmentationPass failed");

    pass.pMoves    = info.pMoves;
    pass.moveCount = info.moveCount;
    pass.moves.assign(info.moveCount, Move());

    auto& queue = base->mainQueue;
    pass.cmd    = queue.commandPool.createAndBeginOneTimeBuffer();

    for (uint32_t i = 0; i < info.moveCount; ++i)
    {
        auto& vmaMove        = info.pMoves[i];
        auto& move           = pass.moves[i];
        VmaAllocationInfo ai = {};
        vmaGetAllocationInfo(allocator, vmaMove.srcAllocation, &ai);

        auto* buffer = static_cast<BufferMemoryLocation*>(ai.pUserData);
        auto* image  = static_cast<ImageMemoryLocation*>(ai.pUserData);
        if (buffers.count(buffer))
        {
            move.buffer = buffer;
            vk::BufferCreateInfo bufferInfo({}, buffer->size, buffer->data.usage, vk::SharingMode::eExclusive);
            vk::Buffer newBuffer = device.createBuffer(bufferInfo);
            auto bound           = vmaBindBufferMemory(allocator, vmaMove.dstTmpAllocation, newBuffer);
            SAIGA_ASSERT(bound == VK_SUCCESS, "vmaBindBufferMemory failed");

            recordBufferCopy(pass.cmd, buffer->data.buffer, newBuffer, buffer->size);

            std::scoped_lock locationLock(buffer->mutex);
            move.oldBuffer      = buffer->data;
            buffer->data.buffer = newBuffer;
            buffer->modified();
        }
        else
        {
            SAIGA_ASSERT(images.count(image), "Unknown allocation in a defragmentation pass");
            move.image        = image;
            ImageData newData = unboundImageCopy(image->data);
            bindImageCopy(newData, vmaMove.dstTmpAllocation);
            {
                std::scoped_lock locationLock(image->mutex);
                move.oldImage = image->data;
            }
            recordImageCopy(pass.cmd, move.oldImage, newData);

            std::scoped_lock locationLock(image->mutex);
            // The old handles are owned by move.oldImage.
            image->data = std::move(newData);
            image->modified();
        }
    }

    pass.cmd.end();
    pass.fence = device.createFence({});
    vk::CommandBufferSubmitInfo cmdInfo(pass.cmd);
    vk::SubmitInfo2 submit;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos    = &cmdInfo;
    // After all frames submitted so far. The frames submitted later use the new places and run after the copies.
    queue.submit2(submit, pass.fence);
    pass.active = true;

    VLOG(3) << "Defragmentation pass with " << info.moveCount << " moves submitted";

    if (wait)
    {
        return finishPass();
    }
    return true;
}

bool VulkanMemory::finishPass()
{
    SAIGA_ASSERT(pass.active);
    auto result = device.waitForFences(pass.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    SAIGA_ASSERT(result == vk::Result::eSuccess);
    return endPass();
}

bool VulkanMemory::endPass()
{
    SAIGA_ASSERT(pass.active);

    // The copies are complete. The fence also covers all frames submitted before the copies, the only ones that
    // could use the old handles.
    for (auto& move : pass.moves)
    {
        move.oldBuffer.destroy_owned_data(device);
        move.oldImage.destroy_owned_data(device);
    }

    VmaDefragmentationPassMoveInfo info = {};
    info.moveCount                      = pass.moveCount;
    info.pMoves                         = static_cast<VmaDefragmentationMove*>(pass.pMoves);
    auto result                         = vmaEndDefragmentationPass(allocator, defragContext, &info);

    // Destroyed during the pass: VMA has freed the memory (operation DESTROY).
    for (auto& move : pass.moves)
    {
        if (!move.destroyed) continue;
        if (move.buffer)
        {
            move.buffer->destroy_owned_data(device);
            buffers.erase(move.buffer);
        }
        if (move.image)
        {
            move.image->destroy_owned_data(device);
            images.erase(move.image);
        }
    }

    base->mainQueue.commandPool.freeCommandBuffer(pass.cmd);
    device.destroy(pass.fence);
    pass = Pass();

    return result == VK_INCOMPLETE;
}

void VulkanMemory::endDefragmentation()
{
    SAIGA_ASSERT(defragContext && !pass.active);
    VmaDefragmentationStats stats = {};
    vmaEndDefragmentation(allocator, defragContext, &stats);
    defragContext = nullptr;

    defragStats.bytesMoved += stats.bytesMoved;
    defragStats.bytesFreed += stats.bytesFreed;
    defragStats.allocationsMoved += stats.allocationsMoved;
    defragStats.deviceMemoryBlocksFreed += stats.deviceMemoryBlocksFreed;
    VLOG(3) << "Defragmentation: " << stats.allocationsMoved << " allocations moved (" << stats.bytesMoved
            << " bytes), " << stats.deviceMemoryBlocksFreed << " blocks freed";
}

void VulkanMemory::update()
{
    std::scoped_lock lock(mutex);
    if (!allocator) return;

    if (fullDefragRequested)
    {
        fullDefragLocked();
        return;
    }

    // One step per frame.
    if (pass.active)
    {
        if (device.getFenceStatus(pass.fence) == vk::Result::eSuccess)
        {
            if (!endPass()) endDefragmentation();
        }
        return;
    }

    if (!enableDefragmentation) return;

    if (!defragContext)
    {
        if (defragQueue.empty())
        {
            if (!defragRequested) return;
            defragRequested = false;
            for (auto& entry : movablePools) defragQueue.push_back(entry.second);
        }
        if (!beginDefragmentation(false)) return;
    }

    if (!beginPass(false))
    {
        endDefragmentation();
    }
}

void VulkanMemory::full_defrag()
{
    std::scoped_lock lock(mutex);
    if (!allocator) return;
    fullDefragLocked();
}

void VulkanMemory::fullDefragLocked()
{
    fullDefragRequested = false;
    if (pass.active)
    {
        if (!finishPass()) endDefragmentation();
    }
    // An incremental defragmentation (limited passes) is completed.
    if (defragContext)
    {
        while (beginPass(true))
        {
        }
        endDefragmentation();
    }

    defragQueue.clear();
    for (auto& entry : movablePools) defragQueue.push_back(entry.second);
    while (beginDefragmentation(true))
    {
        while (beginPass(true))
        {
        }
        endDefragmentation();
    }
    defragRequested = false;
}

DefragmentationStats VulkanMemory::getDefragmentationStats()
{
    std::scoped_lock lock(mutex);
    return defragStats;
}

uint32_t VulkanMemory::pendingDefragmentationMoves()
{
    std::scoped_lock lock(mutex);
    return pass.active ? static_cast<uint32_t>(pass.moves.size()) : 0;
}

std::pair<vk::DeviceSize, vk::DeviceSize> VulkanMemory::dynamicMemoryUsage()
{
    std::scoped_lock lock(mutex);
    vk::DeviceSize blocks = 0, used = 0;
    for (auto& entry : movablePools)
    {
        VmaStatistics stats = {};
        vmaGetPoolStatistics(allocator, entry.second, &stats);
        blocks += stats.blockBytes;
        used += stats.allocationBytes;
    }
    return {blocks, used};
}

}  // namespace Saiga::Vulkan::Memory
