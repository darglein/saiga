#include "Base.h"

#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Shader/GLSL.h"

#include "Debug.h"

#include <array>
namespace Saiga
{
namespace Vulkan
{
VulkanBase::VulkanBase()
{
    Vulkan::GLSLANG::init();
}

void VulkanBase::setPhysicalDevice(Instance& instance, vk::PhysicalDevice physicalDevice)
{
    assert(physicalDevice);
    this->instance       = &instance;
    this->physicalDevice = physicalDevice;

    // Queue family properties, used for setting up requested queues upon device creation
    queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
}

void VulkanBase::destroy()
{
    if (!device) return;

    VLOG(3) << "Destroying VulkanBase with device: " << device;

    vkDestroyPipelineCache(device, pipelineCache, nullptr);


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

    memory.destroy();

    if (device)
    {
        device.destroy();
        device = nullptr;
    }

    VLOG(3) << "Vulkan device destroyed.";
}


void VulkanBase::createLogicalDevice(VulkanParameters& parameters, bool useSwapChain)
{
    m_parameters = parameters;
    std::vector<uint32_t> queueCounts(queueFamilyProperties.size(), 0);

    uint32_t main_idx, transfer_idx, compute_idx;
    bool found_main = findQueueFamily(vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics,
                                      main_idx);  // A queue with compute or graphics can always be used for transfer
    SAIGA_ASSERT(found_main, "A main queue with compute and graphics capabilities is required");

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
        LOG(WARNING) << "Main queue and compute queue are the same";
    }

    if (main_queue_info == transfer_info)
    {
        LOG(WARNING) << "Main queue and transfer queue are the same";
    }

    queueCounts[main_idx]     = std::min(queueCounts[main_idx], queueFamilyProperties[main_idx].queueCount);
    queueCounts[compute_idx]  = std::min(queueCounts[compute_idx], queueFamilyProperties[compute_idx].queueCount);
    queueCounts[transfer_idx] = std::min(queueCounts[transfer_idx], queueFamilyProperties[transfer_idx].queueCount);

    auto maxCount = std::max_element(queueCounts.begin(), queueCounts.end());
    std::vector<float> prios(*maxCount, 1.0f);


    // Desired queues need to be requested upon logical device creation
    // Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially
    // if the application requests different queue types
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};


    for (uint32_t i = 0; i < queueCounts.size(); ++i)
    {
        auto queueCount = queueCounts[i];
        if (queueCount > 0)
        {
            queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), i, queueCount, prios.data());
        }
    }

    // Create the logical device representation
    std::vector<const char*> deviceExtensions(parameters.deviceExtensions);
    if (useSwapChain)
    {
        // If the device will be used for presenting to a display via a swapchain we need to request the swapchain
        // extension
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    auto featuresToEnable = parameters.physicalDeviceFeatures;

    auto availableFeatures = physicalDevice.getFeatures();

    if (!availableFeatures.wideLines)
    {
        featuresToEnable.wideLines = VK_FALSE;
        LOG(ERROR) << "Wide lines requested but not available on this device";
    }



    vk::DeviceCreateInfo deviceCreateInfo = {};
    //    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures  = &featuresToEnable;


    if (!deviceExtensions.empty())
    {
        deviceCreateInfo.enabledExtensionCount   = (uint32_t)deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    }


    // just use the same layers as the instance
    auto layers                          = instance->getEnabledLayers();
    deviceCreateInfo.enabledLayerCount   = layers.size();
    deviceCreateInfo.ppEnabledLayerNames = layers.data();

#if 0
    std::cout << "Device extensions" << std::endl;
    for (auto de : deviceExtensions) std::cout << de << std::endl;

    std::cout << "Device layers" << std::endl;
    for (auto de : layers) std::cout << de << std::endl;
#endif

    device = physicalDevice.createDevice(deviceCreateInfo);

    enabledFeatures = featuresToEnable;

#if 0
    std::vector<vk::ExtensionProperties> extprops = physicalDevice.enumerateDeviceExtensionProperties();
    for (auto e : extprops)
    {
        std::cout << e.specVersion << " " << e.extensionName << std::endl;
    }
#endif


    vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCache                                       = device.createPipelineCache(pipelineCacheCreateInfo);
    SAIGA_ASSERT(pipelineCache);


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

    if (main_queue_info != compute_info)
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

void VulkanBase::init(VulkanParameters params) {}


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

;

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
