//
// Created by Peter Eichinger on 2018-11-30.
//

#include "VulkanMemory.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Base.h"

#include "ImageCopyComputeShader.h"

#include <memory>
namespace Saiga::Vulkan::Memory
{
void VulkanMemory::init(VulkanBase* base, uint32_t swapchain_frames, const VulkanParameters& parameters)
{
    this->base                  = base;
    this->enableDefragmentation = parameters.enableDefragmentation;
    this->enableChunkAllocator  = parameters.enableChunkAllocator;
    m_pDevice                   = base->physicalDevice;
    m_device                    = base->device;
    m_queue                     = base->transferQueue;
    m_swapchain_images          = swapchain_frames;
    strategy                    = std::make_unique<FirstFitStrategy<BufferMemoryLocation>>();
    image_strategy              = std::make_unique<FirstFitStrategy<ImageMemoryLocation>>();

    img_copy_shader = std::make_unique<ImageCopyComputeShader>();
    img_copy_shader->init(base);

    auto props = m_pDevice.getMemoryProperties();
    memoryTypes.resize(props.memoryTypeCount);
    for (auto i = 0U; i < props.memoryTypeCount; ++i)
    {
        memoryTypes[i] = props.memoryTypes[i];
    }

    auto vertIndexType = BufferType{vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                                        vk::BufferUsageFlagBits::eTransferDst,
                                    vk::MemoryPropertyFlagBits::eDeviceLocal};
    auto vertIndexHostType =
        BufferType{vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                       vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};

    get_allocator_exact(vertIndexType);
    get_allocator_exact(vertIndexHostType);

    auto stagingType = BufferType{vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};

    auto effectiveFlags   = getEffectiveFlags(stagingType.memoryFlags);
    auto effectiveStaging = BufferType{stagingType.usageFlags, effectiveFlags};
    get_allocator_exact(effectiveStaging);

    fallbackAllocator = std::make_unique<UniqueAllocator>(m_device, m_pDevice);
}

VulkanMemory::BufferIter VulkanMemory::createNewBufferAllocator(VulkanMemory::BufferMap& map,
                                                                const VulkanMemory::BufferDefaultMap& defaultSizes,
                                                                const BufferType& type)
{
    auto effectiveFlags = getEffectiveFlags(type.memoryFlags);



    auto effectiveType = BufferType{type.usageFlags, effectiveFlags};

    bool allow_defragger =
        (effectiveType.usageFlags & vk::BufferUsageFlagBits::eUniformBuffer) != vk::BufferUsageFlagBits::eUniformBuffer;

    if (enableDefragmentation && allow_defragger)
    {
        effectiveType.usageFlags |= vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
    }

    // Different requested types can map to the same effective type, e.g. device local and host visible memory on an
    // integrated GPU, where all memory is both.
    auto existing = map.find(effectiveType);
    if (existing != map.end())
    {
        return existing;
    }

    auto found = find_default_size<BufferDefaultMap, BufferType>(default_buffer_chunk_sizes, effectiveType);


    auto chunk_alloc =
        std::make_unique<BufferChunkAllocator>(m_pDevice, m_device, effectiveType, *strategy, m_queue, found->second);

    std::unique_ptr<BufferDefragger> defragger;
    if (enableDefragmentation && allow_defragger)
    {
        defragger = std::make_unique<BufferDefragger>(base, m_device, chunk_alloc.get());
    }
    auto new_alloc = map.emplace(effectiveType, BufferContainer{std::move(chunk_alloc), std::move(defragger)});
    SAIGA_ASSERT(new_alloc.second, "Allocator was already present.");


    return new_alloc.first;
}

VulkanMemory::ImageIter VulkanMemory::createNewImageAllocator(VulkanMemory::ImageMap& map,
                                                              const VulkanMemory::ImageDefaultMap& defaultSizes,
                                                              const ImageType& type)
{
    auto effectiveFlags = getEffectiveFlags(type.memoryFlags);

    auto effectiveType = ImageType{type.usageFlags, effectiveFlags};

    auto found = find_default_size<ImageDefaultMap, ImageType>(default_image_chunk_sizes, type);


    bool allow_defragger = (effectiveType.usageFlags & vk::ImageUsageFlagBits::eDepthStencilAttachment) !=
                           vk::ImageUsageFlagBits::eDepthStencilAttachment;

    if (enableDefragmentation && allow_defragger)
    {
        effectiveType.usageFlags |= vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
    }

    // See createNewBufferAllocator.
    auto existing = map.find(effectiveType);
    if (existing != map.end())
    {
        return existing;
    }

    std::unique_ptr<ImageDefragger> defragger;

    auto chunk_alloc = std::make_unique<ImageChunkAllocator>(m_pDevice, m_device, effectiveType, *image_strategy,
                                                             m_queue, found->second);
    if (enableDefragmentation && allow_defragger)
    {
        defragger = std::make_unique<ImageDefragger>(base, m_device, chunk_alloc.get(), img_copy_shader.get());
    }
    auto emplaced = map.emplace(effectiveType, ImageContainer{std::move(chunk_alloc), std::move(defragger)});
    SAIGA_ASSERT(emplaced.second, "Allocator was already present.");
    return emplaced.first;
}

VulkanMemory::~VulkanMemory()
{
    destroy();
}

VulkanMemory::BufferContainer& VulkanMemory::getAllocatorLocked(const BufferType& type)
{
    auto foundAllocator = findAllocator<BufferMap, vk::BufferUsageFlags>(bufferAllocators, type);
    if (foundAllocator == bufferAllocators.end())
    {
        foundAllocator = createNewBufferAllocator(bufferAllocators, default_buffer_chunk_sizes, type);
    }
    return foundAllocator->second;
}

VulkanMemory::ImageContainer& VulkanMemory::getImageAllocatorLocked(const ImageType& type)
{
    auto foundAllocator = findAllocator<ImageMap, vk::ImageUsageFlags>(imageAllocators, type);

    if (foundAllocator == imageAllocators.end())
    {
        foundAllocator = createNewImageAllocator(imageAllocators, default_image_chunk_sizes, type);
    }

    return foundAllocator->second;
}

VulkanMemory::BufferContainer& VulkanMemory::getAllocator(const BufferType& type)
{
    std::scoped_lock lock(allocatorMutex);
    return getAllocatorLocked(type);
}

VulkanMemory::ImageContainer& VulkanMemory::getImageAllocator(const ImageType& type)
{
    std::scoped_lock lock(allocatorMutex);
    return getImageAllocatorLocked(type);
}

VulkanMemory::BufferContainer& VulkanMemory::get_allocator_exact(const BufferType& type)
{
    std::scoped_lock lock(allocatorMutex);
    auto foundAllocator = find_allocator_exact<BufferMap, vk::BufferUsageFlags>(bufferAllocators, type);
    if (foundAllocator == bufferAllocators.end())
    {
        foundAllocator = createNewBufferAllocator(bufferAllocators, default_buffer_chunk_sizes, type);
    }
    return foundAllocator->second;
}


VulkanMemory::ImageContainer& VulkanMemory::get_image_allocator_exact(const ImageType& type)
{
    std::scoped_lock lock(allocatorMutex);
    auto foundAllocator = find_allocator_exact<ImageMap, vk::ImageUsageFlags>(imageAllocators, type);

    if (foundAllocator == imageAllocators.end())
    {
        foundAllocator = createNewImageAllocator(imageAllocators, default_image_chunk_sizes, type);
    }

    return foundAllocator->second;
}

std::vector<BufferDefragger*> VulkanMemory::bufferDefraggers()
{
    std::scoped_lock lock(allocatorMutex);
    std::vector<BufferDefragger*> result;
    for (auto& allocator : bufferAllocators)
    {
        if (allocator.second.defragger)
        {
            result.push_back(allocator.second.defragger.get());
        }
    }
    return result;
}

std::vector<ImageDefragger*> VulkanMemory::imageDefraggers()
{
    std::scoped_lock lock(allocatorMutex);
    std::vector<ImageDefragger*> result;
    for (auto& allocator : imageAllocators)
    {
        if (allocator.second.defragger)
        {
            result.push_back(allocator.second.defragger.get());
        }
    }
    return result;
}

void VulkanMemory::destroy()
{
    std::scoped_lock lock(allocatorMutex);
    // Exit all defraggers first: they free their pending allocations.
    for (auto& allocator : bufferAllocators)
    {
        if (allocator.second.defragger)
        {
            allocator.second.defragger->exit();
        }
    }
    for (auto& allocator : imageAllocators)
    {
        if (allocator.second.defragger)
        {
            allocator.second.defragger->exit();
        }
    }

    for (auto& allocator : bufferAllocators)
    {
        allocator.second.allocator->destroy();
    }
    for (auto& allocator : imageAllocators)
    {
        allocator.second.allocator->destroy();
    }

    bufferAllocators.clear();
    imageAllocators.clear();
    bufferOwners.clear();
    imageOwners.clear();

    if (fallbackAllocator)
    {
        fallbackAllocator->destroy();
    }

    if (img_copy_shader)
    {
        img_copy_shader->destroy();
        img_copy_shader = nullptr;
    }
}


BufferMemoryLocation* VulkanMemory::allocate(const BufferType& type, vk::DeviceSize size)
{
    BufferContainer* container;
    {
        std::scoped_lock lock(allocatorMutex);
        container = &getAllocatorLocked(type);
    }

    BufferMemoryLocation* location = nullptr;
    if (enableChunkAllocator && container->allocator->can_allocate(size))
    {
        if (container->defragger)
        {
            container->defragger->pause();
        }

        location = container->allocator->allocate(size);

        if (container->defragger)
        {
            container->defragger->resume();
        }
    }
    else
    {
        container = nullptr;
        location  = fallbackAllocator->allocate(type, size);
    }

    std::scoped_lock lock(allocatorMutex);
    bufferOwners[location] = container;
    return location;
}

ImageMemoryLocation* VulkanMemory::allocate(const ImageType& type, ImageData& image_data)
{
    SAIGA_ASSERT(!image_data.image, "Image must not be created beforehand.");
    ImageContainer* container;
    {
        std::scoped_lock lock(allocatorMutex);
        container = &getImageAllocatorLocked(type);
    }

    image_data.create_image(m_device);

    ImageMemoryLocation* location = nullptr;
    if (enableChunkAllocator)
    {
        if (container->defragger)
        {
            container->defragger->pause();
        }

        // nullptr if the allocator can not hold the image (size, alignment or memory type).
        location = container->allocator->allocate(image_data);

        if (container->defragger)
        {
            container->defragger->resume();
        }
    }

    if (!location)
    {
        container = nullptr;
        location  = fallbackAllocator->allocate(type, image_data);
    }

    std::scoped_lock lock(allocatorMutex);
    imageOwners[location] = container;
    return location;
}

void VulkanMemory::deallocateBuffer(const BufferType& type, BufferMemoryLocation* location)
{
    BufferContainer* container;
    {
        std::scoped_lock lock(allocatorMutex);
        auto found = bufferOwners.find(location);
        SAIGA_ASSERT(found != bufferOwners.end(), "The buffer was not allocated by VulkanMemory");
        container = found->second;
        bufferOwners.erase(found);
    }

    if (!container)
    {
        fallbackAllocator->deallocate(location);
        return;
    }

    if (container->defragger)
    {
        container->defragger->pause();
        container->defragger->invalidate(location);
    }

    container->allocator->deallocate(location);

    if (container->defragger)
    {
        container->defragger->resume();
    }
}

void VulkanMemory::deallocateImage(const ImageType& type, ImageMemoryLocation* location)
{
    ImageContainer* container;
    {
        std::scoped_lock lock(allocatorMutex);
        auto found = imageOwners.find(location);
        SAIGA_ASSERT(found != imageOwners.end(), "The image was not allocated by VulkanMemory");
        container = found->second;
        imageOwners.erase(found);
    }

    if (!container)
    {
        fallbackAllocator->deallocate(location);
        return;
    }

    if (container->defragger)
    {
        container->defragger->pause();
        container->defragger->invalidate(location);
    }

    container->allocator->deallocate(location);

    if (container->defragger)
    {
        container->defragger->resume();
    }
}

void VulkanMemory::enable_defragmentation(const BufferType& type, bool enable)
{
    auto& container = getAllocator(type);
    if (container.defragger)
    {
        container.defragger->setEnabled(enable);
    }
}

void VulkanMemory::start_defrag(const BufferType& type)
{
    auto& container = getAllocator(type);
    if (container.defragger)
    {
        container.defragger->start();
    }
}

void VulkanMemory::stop_defrag(const BufferType& type)
{
    auto& container = getAllocator(type);
    if (container.defragger)
    {
        container.defragger->stop();
    }
}

void VulkanMemory::enable_defragmentation(const ImageType& type, bool enable)
{
    auto& container = getImageAllocator(type);
    if (container.defragger)
    {
        container.defragger->setEnabled(enable);
    }
}

void VulkanMemory::start_defrag(const ImageType& type)
{
    auto& container = getImageAllocator(type);
    if (container.defragger)
    {
        container.defragger->start();
    }
}

void VulkanMemory::stop_defrag(const ImageType& type)
{
    auto& container = getImageAllocator(type);
    if (container.defragger)
    {
        container.defragger->stop();
    }
}

bool VulkanMemory::performTimedDefrag(int64_t time, vk::Semaphore frameTimeline, uint64_t waitValue)
{
    auto remainingTime = time;

    bool performed = false;
    for (auto* defragger : bufferDefraggers())
    {
        if (remainingTime < 0)
        {
            break;
        }
        auto [perf, remTime] = defragger->perform_defrag(remainingTime, frameTimeline, waitValue);
        performed |= perf;
        remainingTime = remTime;
    }

    for (auto* defragger : imageDefraggers())
    {
        if (remainingTime < 0)
        {
            break;
        }
        auto [perf, remTime] = defragger->perform_defrag(remainingTime, frameTimeline, waitValue);
        performed |= perf;
        remainingTime = remTime;
    }
    return performed;
}

void VulkanMemory::full_defrag()
{
    if (!enableDefragmentation)
    {
        LOG(WARNING) << "Cannot perform defragmentation. It is disabled.";
        return;
    }

    // Allocators without defragger (uniform buffers, depth images) are skipped.
    for (auto* defragger : bufferDefraggers())
    {
        defragger->full_defrag();
    }
    for (auto* defragger : imageDefraggers())
    {
        defragger->full_defrag();
    }
}


}  // namespace Saiga::Vulkan::Memory
