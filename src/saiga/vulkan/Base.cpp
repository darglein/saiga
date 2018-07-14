#include "Base.h"
#include "Debug.h"

#include "saiga/util/table.h"
namespace Saiga{
namespace Vulkan{

void VulkanBase::setPhysicalDevice(vk::PhysicalDevice physicalDevice)
{
    assert(physicalDevice);
    this->physicalDevice = physicalDevice;

    // Memory properties are used regularly for creating all kinds of buffers
    //    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    memoryProperties = physicalDevice.getMemoryProperties();
    // Queue family properties, used for setting up requested queues upon device creation

    queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    //    uint32_t queueFamilyCount;
    //    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    //    assert(queueFamilyCount > 0);
    //    queueFamilyProperties.resize(queueFamilyCount);
    //    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
}

void VulkanBase::destroy()
{
    vkDestroyPipelineCache(device, pipelineCache, nullptr);

    transferQueue.destroy();
    commandPool.destroy();
    descriptorPool.destroy();
    //    if (commandPool)
    {
        //        vkDestroyCommandPool(device, commandPool, nullptr);
    }
    if (device)
    {
        vkDestroyDevice(device, nullptr);
    }


}

uint32_t VulkanBase::getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags properties, VkBool32 *memTypeFound)
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

void VulkanBase::printAvailableMemoryTypes()
{
    {
        cout << endl;
        cout << "Heaps:" << endl;
        Table tab({10,20,20,20,20,20});
        tab << "id" << "size" << "size (GB)" << "DeviceLocal" << "MultiInstance" <<"MultiInstanceKHR";
        for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++)
        {
            vk::MemoryHeap mt = memoryProperties.memoryHeaps[i];
            tab << i
                << mt.size
                << double(mt.size) / (1000*1000*1000)
                << (bool)(mt.flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                << (bool)(mt.flags & vk::MemoryHeapFlagBits::eMultiInstance)
                << (bool)(mt.flags & vk::MemoryHeapFlagBits::eMultiInstanceKHR);

        }
        cout << endl;
    }

    {
        cout << "Memory Types:" << endl;
        Table tab({10,20,20,20,20,20,20,20});
        tab << "id" << "heapIndex" << "DeviceLocal" << "HostCached" <<"HostCoherent" <<"HostVisible" <<"LazilyAllocated" <<"Protected";
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        {
            vk::MemoryType mt = memoryProperties.memoryTypes[i];
            tab << i
                << mt.heapIndex
                << (bool)(mt.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)
                << (bool)(mt.propertyFlags & vk::MemoryPropertyFlagBits::eHostCached)
                << (bool)(mt.propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)
                << (bool)(mt.propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)
                << (bool)(mt.propertyFlags & vk::MemoryPropertyFlagBits::eLazilyAllocated)
                << (bool)(mt.propertyFlags & vk::MemoryPropertyFlagBits::eProtected);

        }
        cout << endl;
    }
}


uint32_t VulkanBase::getQueueFamilyIndex(vk::QueueFlags queueFlags)
{
    return 0;
#if 0
    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if (queueFlags & vk::QueueFlagBits::eCompute)
    {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
        {
            if ((queueFamilyProperties[i].queueFlags & queueFlags) && ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) == 0))
            {
                return i;
                break;
            }
        }
    }

    // Dedicated queue for transfer
    // Try to find a queue family index that supports transfer but not graphics and compute
    if (queueFlags & vk::QueueFlagBits::eTransfer)
    {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
        {
            if ((queueFamilyProperties[i].queueFlags & queueFlags) && ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) == 0) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
            {
                return i;
                break;
            }
        }
    }

    // For other queue types or if no separate compute queue is present, return the first one to support the requested flags
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
    {
        if (queueFamilyProperties[i].queueFlags & queueFlags)
        {
            return i;
            break;
        }
    }
#endif

    throw std::runtime_error("Could not find a matching queue family index");
}

uint32_t VulkanBase::getPresentQueue(vk::SurfaceKHR surface)
{
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
    {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

        if (queueFamilyProperties[i].queueCount > 0 && presentSupport) {
            return i;
        }

    }
    throw std::runtime_error("Could not find a matching queue family index");
}

void VulkanBase::createLogicalDevice(vk::SurfaceKHR surface,
                                     vk::PhysicalDeviceFeatures enabledFeatures,
                                     std::vector<const char*> enabledExtensions,
                                     bool useSwapChain,
                                     vk::QueueFlags requestedQueueTypes)
{
    printAvailableMemoryTypes();
    // Desired queues need to be requested upon logical device creation
    // Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
    // requests different queue types

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};

    // Get queue family indices for the requested queue family types
    // Note that the indices may overlap depending on the implementation

    const float defaultQueuePriority(1.0f);

    // Graphics queue
    if (requestedQueueTypes &  vk::QueueFlagBits::eGraphics )
    {
        queueFamilyIndices.graphics = getQueueFamilyIndex( vk::QueueFlagBits::eGraphics );
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndices.graphics;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &defaultQueuePriority;
        queueCreateInfos.push_back(queueInfo);
    }
    else
    {
        queueFamilyIndices.graphics = VK_NULL_HANDLE;
    }

    // Dedicated compute queue
    if (requestedQueueTypes &  vk::QueueFlagBits::eCompute  && false)
    {
        queueFamilyIndices.compute = getQueueFamilyIndex(vk::QueueFlagBits::eCompute);
        if (queueFamilyIndices.compute != queueFamilyIndices.graphics)
        {
            // If compute family index differs, we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = queueFamilyIndices.compute;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &defaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    }
    else
    {
        // Else we use the same queue
        queueFamilyIndices.compute = queueFamilyIndices.graphics;
    }

    // Dedicated transfer queue
    if (requestedQueueTypes & vk::QueueFlagBits::eTransfer && false)
    {
        queueFamilyIndices.transfer = getQueueFamilyIndex(vk::QueueFlagBits::eTransfer);
        if ((queueFamilyIndices.transfer != queueFamilyIndices.graphics) && (queueFamilyIndices.transfer != queueFamilyIndices.compute))
        {
            // If compute family index differs, we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = queueFamilyIndices.transfer;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &defaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    }
    else
    {
        // Else we use the same queue
        queueFamilyIndices.transfer = queueFamilyIndices.graphics;
    }

    queueFamilyIndices.present = getPresentQueue(surface);


    cout << "Device Queues:" << endl;
    cout << "   graphics " << queueFamilyIndices.graphics << endl;
    cout << "   compute  " << queueFamilyIndices.compute << endl;
    cout << "   transfer " << queueFamilyIndices.transfer << endl;
    cout << "   present  " << queueFamilyIndices.present << endl;


    for(VkDeviceQueueCreateInfo i : queueCreateInfos)
    {
        cout << "Create Queue " << i.queueFamilyIndex << " " << i.queueCount << endl;
    }

    // Create the logical device representation
    std::vector<const char*> deviceExtensions(enabledExtensions);
    if (useSwapChain)
    {
        // If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo deviceCreateInfo = {};
    //    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures = &enabledFeatures;


    if (deviceExtensions.size() > 0)
    {
        deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    }

    auto layers = Saiga::Vulkan::Debug::getDebugValidationLayers();

    deviceCreateInfo.enabledLayerCount = layers.size();
    deviceCreateInfo.ppEnabledLayerNames = layers.data();

    cout << "Device extensions" << endl;
    for(auto de : deviceExtensions)
        cout << de << endl;

    cout << "Device layers" << endl;
    for(auto de : layers)
        cout << de << endl;

    //    VkResult result = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    device = physicalDevice.createDevice(deviceCreateInfo);



    std::vector<vk::ExtensionProperties> extprops = physicalDevice.enumerateDeviceExtensionProperties();
    for(auto e : extprops)
    {
        cout << e.specVersion << " " << e.extensionName << endl;
    }


    return;
}

void VulkanBase::init(VulkanParameters params)
{

    vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCache = device.createPipelineCache(pipelineCacheCreateInfo);
    SAIGA_ASSERT(pipelineCache);

    //    commandPool = createCommandPool(queueFamilyIndices.graphics);
    commandPool.create(device,queueFamilyIndices.transfer,vk::CommandPoolCreateFlagBits::eTransient);

    transferQueue.create(device,queueFamilyIndices.transfer);


    descriptorPool.create(
                device,
                params.maxDescriptorSets,{
                    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer,          params.descriptorCounts[0]},
                    vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler,   params.descriptorCounts[1]},
                    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer,   params.descriptorCounts[2]},
                    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage,   params.descriptorCounts[3]},
                });


}


vk::CommandBuffer VulkanBase::createAndBeginTransferCommand()
{
    auto cmd = commandPool.allocateCommandBuffer();
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    return cmd;
}


void VulkanBase::submitAndWait(vk::CommandBuffer commandBuffer, vk::Queue queue)
{
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vk::FenceCreateInfo fenceInfo;
    vk::Fence fence = device.createFence(fenceInfo);
    SAIGA_ASSERT(fence);
    queue.submit(submitInfo,fence);
    device.waitForFences(fence,true,100000000000);
    device.destroyFence(fence);
}


void VulkanBase::endTransferWait(vk::CommandBuffer commandBuffer)
{
    commandBuffer.end();
    submitAndWait(commandBuffer,transferQueue);
    commandPool.freeCommandBuffer(commandBuffer);
}



}
}
