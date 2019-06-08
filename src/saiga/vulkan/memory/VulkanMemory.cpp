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

    auto found = find_default_size<BufferDefaultMap, BufferType>(default_buffer_chunk_sizes, effectiveType);


    auto chunk_alloc =
        std::make_unique<BufferChunkAllocator>(m_pDevice, m_device, effectiveType, *strategy, m_queue, found->second);

    std::unique_ptr<BufferDefragger> defragger;
    if (enableDefragmentation && allow_defragger)
    {
        defragger = std::make_unique<BufferDefragger>(base, m_device, chunk_alloc.get(), m_swapchain_images);
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

    std::unique_ptr<ImageDefragger> defragger;

    auto chunk_alloc = std::make_unique<ImageChunkAllocator>(m_pDevice, m_device, effectiveType, *image_strategy,
                                                             m_queue, found->second);
    if (enableDefragmentation && allow_defragger)
    {
        defragger = std::make_unique<ImageDefragger>(base, m_device, chunk_alloc.get(), m_swapchain_images,
                                                     img_copy_shader.get());
    }
    auto emplaced = map.emplace(effectiveType, ImageContainer{std::move(chunk_alloc), std::move(defragger)});
    SAIGA_ASSERT(emplaced.second, "Allocator was already present.");
    return emplaced.first;
}

VulkanMemory::BufferContainer& VulkanMemory::getAllocator(const BufferType& type)
{
    auto foundAllocator = findAllocator<BufferMap, vk::BufferUsageFlags>(bufferAllocators, type);
    if (foundAllocator == bufferAllocators.end())
    {
        foundAllocator = createNewBufferAllocator(bufferAllocators, default_buffer_chunk_sizes, type);
    }
    return foundAllocator->second;
}



VulkanMemory::ImageContainer& VulkanMemory::getImageAllocator(const ImageType& type)
{
    auto foundAllocator = findAllocator<ImageMap, vk::ImageUsageFlags>(imageAllocators, type);

    if (foundAllocator == imageAllocators.end())
    {
        foundAllocator = createNewImageAllocator(imageAllocators, default_image_chunk_sizes, type);
    }

    return foundAllocator->second;
}

VulkanMemory::BufferContainer& VulkanMemory::get_allocator_exact(const BufferType& type)
{
    auto foundAllocator = find_allocator_exact<BufferMap, vk::BufferUsageFlags>(bufferAllocators, type);
    if (foundAllocator == bufferAllocators.end())
    {
        foundAllocator = createNewBufferAllocator(bufferAllocators, default_buffer_chunk_sizes, type);
    }
    return foundAllocator->second;
}


VulkanMemory::ImageContainer& VulkanMemory::get_image_allocator_exact(const ImageType& type)
{
    auto foundAllocator = find_allocator_exact<ImageMap, vk::ImageUsageFlags>(imageAllocators, type);

    if (foundAllocator == imageAllocators.end())
    {
        foundAllocator = createNewImageAllocator(imageAllocators, default_image_chunk_sizes, type);
    }

    return foundAllocator->second;
}

void VulkanMemory::destroy()
{
    for (auto& allocator : bufferAllocators)
    {
        if (allocator.second.defragger)
        {
            allocator.second.defragger->exit();
        }
        allocator.second.allocator->destroy();
    }

    for (auto& allocator : imageAllocators)
    {
        if (allocator.second.defragger)
        {
            allocator.second.defragger->exit();
        }
        allocator.second.allocator->destroy();
    }

    bufferAllocators.clear();
    imageAllocators.clear();

    if (img_copy_shader)
    {
        img_copy_shader->destroy();
        img_copy_shader = nullptr;
    }
}


BufferMemoryLocation* VulkanMemory::allocate(const BufferType& type, vk::DeviceSize size)
{
    auto& allocator = getAllocator(type);

    if (!enableChunkAllocator || !allocator.allocator->can_allocate(size))
    {
        return fallbackAllocator->allocate(type, size);
    }

    if (allocator.defragger)
    {
        allocator.defragger->stop();
    }

    auto memoryLocation = allocator.allocator->allocate(size);
    if (allocator.defragger)
    {
        allocator.defragger->start();
    }


    return memoryLocation;
}

ImageMemoryLocation* VulkanMemory::allocate(const ImageType& type, ImageData& image_data)
{
    SAIGA_ASSERT(!image_data.image, "Image must not be created beforehand.");
    auto& allocator = getImageAllocator(type);

    image_data.create_image(m_device);
    if (!enableChunkAllocator || !allocator.allocator->can_allocate(image_data.image_requirements.size))
    {
        return fallbackAllocator->allocate(type, image_data);
    }

    if (allocator.defragger)
    {
        allocator.defragger->stop();
    }
    auto location = allocator.allocator->allocate(image_data);

    if (allocator.defragger)
    {
        allocator.defragger->start();
    }
    return location;
}

void VulkanMemory::deallocateBuffer(const BufferType& type, BufferMemoryLocation* location)
{
    auto& allocator = getAllocator(type);
    if (!enableChunkAllocator || !allocator.allocator->can_allocate(location->size))
    {
        fallbackAllocator->deallocate(location);

        return;
    }

    if (allocator.defragger)
    {
        allocator.defragger->stop();

        allocator.defragger->invalidate(location);
    }

    allocator.allocator->deallocate(location);

    if (allocator.defragger)
    {
        allocator.defragger->start();
    }
}

void VulkanMemory::deallocateImage(const ImageType& type, ImageMemoryLocation* location)
{
    auto& allocator = getImageAllocator(type);
    if (!enableChunkAllocator || !allocator.allocator->can_allocate(location->size))
    {
        fallbackAllocator->deallocate(location);
        return;
    }
    if (allocator.defragger)
    {
        allocator.defragger->stop();
        allocator.defragger->invalidate(location);
    }

    allocator.allocator->deallocate(location);

    if (allocator.defragger)
    {
        allocator.defragger->start();
    }
}

void VulkanMemory::enable_defragmentation(const BufferType& type, bool enable)
{
    if (getAllocator(type).defragger)
    {
        getAllocator(type).defragger->setEnabled(enable);
    }
}

void VulkanMemory::start_defrag(const BufferType& type)
{
    if (getAllocator(type).defragger)
    {
        getAllocator(type).defragger->start();
    }
}

void VulkanMemory::stop_defrag(const BufferType& type)
{
    if (getAllocator(type).defragger)
    {
        getAllocator(type).defragger->stop();
    }
}

void VulkanMemory::enable_defragmentation(const ImageType& type, bool enable)
{
    if (getImageAllocator(type).defragger)
    {
        getImageAllocator(type).defragger->setEnabled(enable);
    }
}

void VulkanMemory::start_defrag(const ImageType& type)
{
    if (getImageAllocator(type).defragger)
    {
        getImageAllocator(type).defragger->start();
    }
}

void VulkanMemory::stop_defrag(const ImageType& type)
{
    if (getImageAllocator(type).defragger)
    {
        getImageAllocator(type).defragger->stop();
    }
}

bool VulkanMemory::performTimedDefrag(int64_t time, vk::Semaphore semaphore)
{
    auto remainingTime = time;

    bool performed = false;
    for (auto& allocator : bufferAllocators)
    {
        if (remainingTime < 0)
        {
            break;
        }
        auto* defragger = allocator.second.defragger.get();
        if (defragger)
        {
            auto [perf, remTime] = defragger->perform_defrag(remainingTime, semaphore);
            if (perf)
            {
                semaphore = nullptr;
            }
            performed |= perf;
            remainingTime = remTime;
        }
    }

    for (auto& allocator : imageAllocators)
    {
        if (remainingTime < 0)
        {
            break;
        }
        auto* defragger = allocator.second.defragger.get();
        if (defragger)
        {
            auto [perf, remTime] = defragger->perform_defrag(remainingTime, semaphore);
            if (perf)
            {
                semaphore = nullptr;
            }
            performed |= perf;
            remainingTime = remTime;
        }
    }
    return performed;
}
void VulkanMemory::full_defrag()
{
    if (enableDefragmentation)
    {
        for (auto& allocator : bufferAllocators)
        {
            auto* defragger = allocator.second.defragger.get();

            if (defragger)
            {
                defragger->setEnabled(true);
                defragger->start();
            }
        }

        for (auto& allocator : imageAllocators)
        {
            auto* defragger = allocator.second.defragger.get();

            if (defragger)
            {
                defragger->setEnabled(true);
                defragger->start();
            }
        }

        for (auto& allocator : bufferAllocators)
        {
            auto* defragger = allocator.second.defragger.get();

            while (defragger->running)
            {
                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(100us);
                }
            }
        }

        for (auto& allocator : imageAllocators)
        {
            auto* defragger = allocator.second.defragger.get();

            while (defragger->running)
            {
                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(100us);
                }
            }
        }
    }
    else
    {
        LOG(WARNING) << "Cannot perform defragmentation. It is disabled.";
    }
}


}  // namespace Saiga::Vulkan::Memory
