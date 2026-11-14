/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Base.h"

#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Shader/GLSL.h"

#include "Debug.h"

#include <array>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
VulkanBase::VulkanBase()
{
    Vulkan::GLSLANG::init();
}

VulkanBase::~VulkanBase()
{
    destroy();
    Vulkan::GLSLANG::quit();
}

void VulkanBase::setPhysicalDevice(Instance& instance, vk::PhysicalDevice physicalDevice)
{
    SAIGA_ASSERT(physicalDevice);
    this->instance       = &instance;
    this->physicalDevice = physicalDevice;
    apiVersion           = instance.getApiVersion();
    SAIGA_ASSERT(physicalDevice.getProperties().apiVersion >= SAIGA_VULKAN_API_VERSION);

    // Queue family properties, used for setting up requested queues upon device creation
    queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
}

void VulkanBase::destroy()
{
    if (!device) return;

    VLOG(3) << "Destroying VulkanBase with device: " << device;

    flushDeferredDeletes();

    vkDestroyPipelineCache(device, pipelineCache, nullptr);

    // Before the queues: the defragger frees command buffers of a pool created on the main queue.
    memory.destroy();
    // memory.destroy() defers the destruction of its own objects (e.g. the image copy pipeline).
    flushDeferredDeletes();

    if (dedicated_compute_queue)
    {
        dedicated_compute_queue->destroy();
        dedicated_compute_queue.reset();
    }
    if (dedicated_transfer_queue)
    {
        dedicated_transfer_queue->destroy();
        dedicated_transfer_queue.reset();
    }
    mainQueue.destroy();

    descriptorPool.destroy();

    device.destroy(frameTimeline);
    frameTimeline = nullptr;

    device.destroy();
    device = nullptr;

    VLOG(3) << "Vulkan device destroyed.";
}

uint64_t VulkanBase::completedFrameValue() const
{
    return device.getSemaphoreCounterValue(frameTimeline);
}

void VulkanBase::waitFrameValue(uint64_t value)
{
    if (value == 0) return;
    vk::SemaphoreWaitInfo info;
    info.semaphoreCount = 1;
    info.pSemaphores    = &frameTimeline;
    info.pValues        = &value;
    auto result         = device.waitSemaphores(info, std::numeric_limits<uint64_t>::max());
    SAIGA_ASSERT(result == vk::Result::eSuccess);
}

void VulkanBase::deferDelete(std::function<void()> f)
{
    std::scoped_lock lock(deletionMutex);
    deletionQueue.emplace_back(currentFrameSignalValue(), std::move(f));
}

void VulkanBase::processDeferredDeletes()
{
    if (!device) return;
    auto completed = completedFrameValue();

    // Run outside of the lock, the functions may defer further deletions.
    std::vector<std::function<void()>> ready;
    {
        std::scoped_lock lock(deletionMutex);
        while (!deletionQueue.empty() && deletionQueue.front().first <= completed)
        {
            ready.push_back(std::move(deletionQueue.front().second));
            deletionQueue.pop_front();
        }
    }
    for (auto& f : ready) f();
}

void VulkanBase::flushDeferredDeletes()
{
    if (!device) return;
    device.waitIdle();
    while (true)
    {
        std::deque<std::pair<uint64_t, std::function<void()>>> all;
        {
            std::scoped_lock lock(deletionMutex);
            all.swap(deletionQueue);
        }
        if (all.empty()) break;
        for (auto& f : all) f.second();
    }
}

void VulkanBase::createLogicalDevice(VulkanParameters& parameters, vk::SurfaceKHR surface)
{
    m_parameters = parameters;
    std::vector<uint32_t> queueCounts(queueFamilyProperties.size(), 0);

    // The main queue does graphics, compute, transfer and presents.
    uint32_t main_idx = 0;
    bool found_main   = false;
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
    {
        auto required = vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics;
        if ((queueFamilyProperties[i].queueFlags & required) == required &&
            (!surface || physicalDevice.getSurfaceSupportKHR(i, surface)))
        {
            main_idx   = i;
            found_main = true;
            break;
        }
    }
    SAIGA_ASSERT(found_main, "A main queue with compute, graphics (and present) capabilities is required");

    uint32_t transfer_idx = main_idx, compute_idx = main_idx;
    int offset = 0;
    if (!findDedicatedQueueFamily(vk::QueueFlagBits::eCompute, compute_idx))
    {
        findQueueFamily(vk::QueueFlagBits::eCompute, compute_idx, ++offset);
    }

    if (!findDedicatedQueueFamily(vk::QueueFlagBits::eTransfer, transfer_idx))
    {
        findQueueFamily(vk::QueueFlagBits::eTransfer, transfer_idx, ++offset);
    }

    std::pair<uint32_t, uint32_t> main_queue_info, transfer_info, compute_info;

    main_queue_info = std::make_pair(main_idx, queueCounts[main_idx] % queueFamilyProperties[main_idx].queueCount);
    queueCounts[main_idx]++;
    compute_info =
        std::make_pair(compute_idx, queueCounts[compute_idx] % queueFamilyProperties[compute_idx].queueCount);
    queueCounts[compute_idx]++;
    transfer_info =
        std::make_pair(transfer_idx, queueCounts[transfer_idx] % queueFamilyProperties[transfer_idx].queueCount);
    queueCounts[transfer_idx]++;

    if (main_queue_info == compute_info)
    {
        VLOG(1) << "Main queue and compute queue are the same";
    }

    if (main_queue_info == transfer_info)
    {
        VLOG(1) << "Main queue and transfer queue are the same";
    }

    for (uint32_t i = 0; i < queueCounts.size(); ++i)
    {
        queueCounts[i] = std::min(queueCounts[i], queueFamilyProperties[i].queueCount);
    }

    auto maxCount = std::max_element(queueCounts.begin(), queueCounts.end());
    std::vector<float> prios(*maxCount, 1.0f);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
    for (uint32_t i = 0; i < queueCounts.size(); ++i)
    {
        auto queueCount = queueCounts[i];
        if (queueCount > 0)
        {
            queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), i, queueCount, prios.data());
        }
    }

    std::vector<const char*> deviceExtensions(parameters.deviceExtensions);
    if (surface)
    {
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    // Core features: enable the requested ones that are available.
    auto featuresToEnable  = parameters.physicalDeviceFeatures;
    auto availableFeatures = physicalDevice.getFeatures();
    auto disableIfMissing  = [&](vk::Bool32& requested, vk::Bool32 available, const char* name)
    {
        if (requested && !available)
        {
            requested = VK_FALSE;
            LOG(WARNING) << "Vulkan feature " << name << " requested but not available on this device";
        }
    };
    disableIfMissing(featuresToEnable.wideLines, availableFeatures.wideLines, "wideLines");
    disableIfMissing(featuresToEnable.fillModeNonSolid, availableFeatures.fillModeNonSolid, "fillModeNonSolid");
    disableIfMissing(featuresToEnable.largePoints, availableFeatures.largePoints, "largePoints");
    disableIfMissing(featuresToEnable.samplerAnisotropy, availableFeatures.samplerAnisotropy, "samplerAnisotropy");

    // Vulkan 1.2/1.3 features saiga requires (checked in Instance::pickPhysicalDevice).
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                       vk::PhysicalDeviceVulkan13Features>
        features;
    features.get<vk::PhysicalDeviceFeatures2>().features                 = featuresToEnable;
    features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore = VK_TRUE;
    features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering  = VK_TRUE;
    features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2  = VK_TRUE;

    vk::DeviceCreateInfo deviceCreateInfo    = {};
    deviceCreateInfo.pNext                   = &features.get<vk::PhysicalDeviceFeatures2>();
    deviceCreateInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos       = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures        = nullptr;  // in VkPhysicalDeviceFeatures2
    deviceCreateInfo.enabledExtensionCount   = (uint32_t)deviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    device          = physicalDevice.createDevice(deviceCreateInfo);
    enabledFeatures = featuresToEnable;

    vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCache                                       = device.createPipelineCache(pipelineCacheCreateInfo);
    SAIGA_ASSERT(pipelineCache);

    {
        vk::SemaphoreTypeCreateInfo timelineInfo(vk::SemaphoreType::eTimeline, 0);
        vk::SemaphoreCreateInfo semaphoreInfo;
        semaphoreInfo.pNext = &timelineInfo;
        frameTimeline       = device.createSemaphore(semaphoreInfo);
    }

    mainQueue.create(device, main_queue_info.first, main_queue_info.second,
                     vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    if (main_queue_info != compute_info)
    {
        dedicated_compute_queue = std::make_unique<Saiga::Vulkan::Queue>();
        dedicated_compute_queue->create(device, compute_info.first, compute_info.second);
        computeQueue = dedicated_compute_queue.get();
    }
    else
    {
        computeQueue = &mainQueue;
    }

    if (main_queue_info != transfer_info)
    {
        dedicated_transfer_queue = std::make_unique<Saiga::Vulkan::Queue>();
        dedicated_transfer_queue->create(device, transfer_info.first, transfer_info.second);
        transferQueue = dedicated_transfer_queue.get();
    }
    else
    {
        transferQueue = &mainQueue;
    }

    descriptorPool.create(
        device, parameters.maxDescriptorSets,
        {
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, parameters.descriptorCounts[0]},
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, parameters.descriptorCounts[1]},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, parameters.descriptorCounts[2]},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, parameters.descriptorCounts[3]},
            vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, parameters.descriptorCounts[3]},
        });

    current_frame = 0;
}

bool VulkanBase::findQueueFamily(vk::QueueFlags flags, uint32_t& family, uint32_t offset)
{
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
    {
        auto offset_index = (i + offset) % queueFamilyProperties.size();
        auto& prop        = queueFamilyProperties[offset_index];

        if ((prop.queueFlags & flags) == flags)
        {
            family = offset_index;
            return true;
        }
    }

    return false;
}

bool VulkanBase::findDedicatedQueueFamily(vk::QueueFlags flags, uint32_t& family)
{
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
    {
        auto& prop = queueFamilyProperties[i];

        if (prop.queueFlags == flags)
        {
            family = i;
            return true;
        }
    }

    return false;
}

}  // namespace Vulkan
}  // namespace Saiga
