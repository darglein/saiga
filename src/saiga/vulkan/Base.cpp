#include "Base.h"
#include "saiga/util/table.h"
#include "saiga/util/tostring.h"

#include "Debug.h"

#include <array>
namespace Saiga
{
namespace Vulkan
{
void VulkanBase::setPhysicalDevice(vk::PhysicalDevice physicalDevice)
{
    assert(physicalDevice);
    this->physicalDevice = physicalDevice;

    // Memory properties are used regularly for creating all kinds of buffers
    memoryProperties = physicalDevice.getMemoryProperties();

    // Queue family properties, used for setting up requested queues upon device creation
    queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
}

void VulkanBase::destroy()
{
    vkDestroyPipelineCache(device, pipelineCache, nullptr);


    mainQueue.destroy();
    computeQueue.destroy();
    transferQueue.destroy();
    descriptorPool.destroy();

    memory.destroy();

    if (device)
    {
        device.destroy();
    }
}

uint32_t VulkanBase::getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags properties, VkBool32* memTypeFound)
{
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
    {
        if ((typeBits & 1) == 1)
        {
            if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                if (memTypeFound)
                {
                    *memTypeFound = true;
                }
                return i;
            }
        }
        typeBits >>= 1;
    }

    if (memTypeFound)
    {
        *memTypeFound = false;
        return 0;
    }
    else
    {
        throw std::runtime_error("Could not find a matching memory type");
    }
}


void VulkanBase::createLogicalDevice(vk::SurfaceKHR surface, vk::PhysicalDeviceFeatures requestedFeatures,
                                     std::vector<const char*> enabledExtensions, bool useSwapChain,
                                     vk::QueueFlags requestedQueueTypes, bool createSecondaryTransferQueue)
{
    // createDedicatedTransferQueue = createSecondaryTransferQueue;

    std::vector<uint32_t> queueCounts(queueFamilyProperties.size(), 0);

    uint32_t main_idx, transfer_idx, compute_idx;
    bool found_main = findQueueFamily(vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics,
                                      main_idx);  // A queue with compute or graphics can always be used for transfer
    SAIGA_ASSERT(found_main, "A main queue with compute and graphics capabilities is required");

    if (!findDedicatedQueueFamily(vk::QueueFlagBits::eCompute, compute_idx))
    {
        findQueueFamily(vk::QueueFlagBits::eCompute, compute_idx);
    }

    if (!findDedicatedQueueFamily(vk::QueueFlagBits::eTransfer, transfer_idx))
    {
        findQueueFamily(vk::QueueFlagBits::eTransfer, transfer_idx);
    }

    main_queue_info = std::make_pair(main_idx, queueCounts[main_idx]);
    queueCounts[main_idx]++;
    compute_info = std::make_pair(compute_idx, queueCounts[compute_idx]);
    queueCounts[compute_idx]++;
    transfer_info = std::make_pair(transfer_idx, queueCounts[transfer_idx]);
    queueCounts[transfer_idx]++;


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
            // vk::DeviceQueueCreateInfo qci{};
            queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), i, queueCount, prios.data());
        }
    }

    // Create the logical device representation
    std::vector<const char*> deviceExtensions(enabledExtensions);
    if (useSwapChain)
    {
        // If the device will be used for presenting to a display via a swapchain we need to request the swapchain
        // extension
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    auto featuresToEnable = requestedFeatures;

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


    if (deviceExtensions.size() > 0)
    {
        deviceCreateInfo.enabledExtensionCount   = (uint32_t)deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    }

    auto layers = Saiga::Vulkan::Debug::getDebugValidationLayers();

    deviceCreateInfo.enabledLayerCount   = layers.size();
    deviceCreateInfo.ppEnabledLayerNames = layers.data();

#if 0
    cout << "Device extensions" << endl;
    for (auto de : deviceExtensions) cout << de << endl;

    cout << "Device layers" << endl;
    for (auto de : layers) cout << de << endl;
#endif

    device = physicalDevice.createDevice(deviceCreateInfo);

    enabledFeatures = featuresToEnable;

#if 0
    std::vector<vk::ExtensionProperties> extprops = physicalDevice.enumerateDeviceExtensionProperties();
    for (auto e : extprops)
    {
        cout << e.specVersion << " " << e.extensionName << endl;
    }
#endif


    return;
}

void VulkanBase::init(VulkanParameters params)
{
    memory.init(physicalDevice, device);

    vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCache                                       = device.createPipelineCache(pipelineCacheCreateInfo);
    SAIGA_ASSERT(pipelineCache);


    mainQueue.create(device, main_queue_info.first, main_queue_info.second, vk::CommandPoolCreateFlagBits::eTransient);

    computeQueue.create(device, compute_info.first, compute_info.second);

    transferQueue.create(device, transfer_info.first, transfer_info.second);

    descriptorPool.create(
        device, params.maxDescriptorSets,
        {
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, params.descriptorCounts[0]},
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, params.descriptorCounts[1]},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, params.descriptorCounts[2]},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, params.descriptorCounts[3]},
        });
}


vk::CommandBuffer VulkanBase::createAndBeginTransferCommand()
{
    auto cmd = mainQueue.commandPool.allocateCommandBuffer();
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    return cmd;
}


void VulkanBase::submitAndWait(vk::CommandBuffer commandBuffer, vk::Queue queue)
{
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &commandBuffer;
    vk::FenceCreateInfo fenceInfo;
    vk::Fence fence = device.createFence(fenceInfo);
    SAIGA_ASSERT(fence);
    queue.submit(submitInfo, fence);
    device.waitForFences(fence, true, 100000000000);
    device.destroyFence(fence);
}


void VulkanBase::endTransferWait(vk::CommandBuffer commandBuffer)
{
    commandBuffer.end();
    submitAndWait(commandBuffer, mainQueue);
    mainQueue.commandPool.freeCommandBuffer(commandBuffer);
}


bool VulkanBase::findQueueFamily(vk::QueueFlags flags, uint32_t& family)
{
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
    {
        auto& prop = queueFamilyProperties[i];

        if ((prop.queueFlags & flags) == flags)
        {
            family = i;
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
