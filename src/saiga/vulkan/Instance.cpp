/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Instance.h"

#include "saiga/core/util/easylogging++.h"

#include <algorithm>
#include <iostream>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
static std::string versionString(uint32_t version)
{
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." + std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

void Instance::destroy()
{
    if (instance)
    {
        VLOG(3) << "Destroying Vulkan Instance: " << instance;
        debug.destroy();
        instance.destroy();
        instance = nullptr;
    }
}

void Instance::create(const std::vector<std::string>& _instanceExtensions, bool enableValidation)
{
    SAIGA_ASSERT(!instance);

    uint32_t loaderVersion = vk::enumerateInstanceVersion();
    if (loaderVersion < SAIGA_VULKAN_API_VERSION)
    {
        SAIGA_EXIT_ERROR("Saiga requires Vulkan " + versionString(SAIGA_VULKAN_API_VERSION) +
                         ", but the Vulkan loader only supports " + versionString(loaderVersion));
    }
    // The highest version the application uses. Devices with a lower version are rejected in pickPhysicalDevice.
    apiVersion = SAIGA_VULKAN_API_VERSION;

    std::vector<std::string> instanceExtensions = _instanceExtensions;
    std::vector<std::string> instanceLayers;

    vk::ApplicationInfo appInfo;
    appInfo.pApplicationName = "Saiga Application";
    appInfo.pEngineName      = "Saiga";
    appInfo.apiVersion       = apiVersion;

    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    if (enableValidation)
    {
        // We require both, the debug utils extension and the validation layer
        std::string ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        std::string val = Debug::getDebugValidationLayers();
        if (hasExtension(ext) && hasLayer(val))
        {
            std::cout << "Vulkan Validation layer enabled!" << std::endl;
            instanceExtensions.push_back(ext);
            instanceLayers.push_back(val);
        }
        else
        {
            std::cerr << "Vulkan Warning: You tried to enable the validation layer, but "
                      << (hasExtension(ext) ? "the layer " + val : "the extension " + ext)
                      << " was not found. Starting without validation layer..." << std::endl;
            enableValidation = false;
        }
    }

    for (auto& s : instanceExtensions) _extensions.push_back(s);
    for (auto& s : instanceLayers) _layers.push_back(s);

    auto extensions = getEnabledExtensions();
    auto layers     = getEnabledLayers();

    instanceCreateInfo.enabledExtensionCount   = (uint32_t)extensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = extensions.data();
    instanceCreateInfo.enabledLayerCount       = (uint32_t)layers.size();
    instanceCreateInfo.ppEnabledLayerNames     = layers.data();

    // Also report messages of vkCreateInstance/vkDestroyInstance.
    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo = Debug::createInfo();
    if (enableValidation) instanceCreateInfo.pNext = &debugCreateInfo;

    instance = vk::createInstance(instanceCreateInfo);
    SAIGA_ASSERT(instance);

    if (enableValidation)
    {
        debug.init(instance);
    }
    std::cout << "Vulkan instance created (API " << versionString(apiVersion) << ")." << std::endl;
}

int Instance::deviceScore(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    auto properties = device.getProperties();
    if (properties.apiVersion < SAIGA_VULKAN_API_VERSION) return -1;

    auto features = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                                        vk::PhysicalDeviceVulkan13Features>();
    auto& f12     = features.get<vk::PhysicalDeviceVulkan12Features>();
    auto& f13     = features.get<vk::PhysicalDeviceVulkan13Features>();
    if (!f12.timelineSemaphore || !f13.dynamicRendering || !f13.synchronization2) return -1;

    // The main queue (see VulkanBase::createLogicalDevice) needs graphics + compute and must be able to present.
    auto families    = device.getQueueFamilyProperties();
    bool has_present = false;
    for (uint32_t i = 0; i < families.size(); ++i)
    {
        auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
        if ((families[i].queueFlags & required) == required && (!surface || device.getSurfaceSupportKHR(i, surface)))
        {
            has_present = true;
            break;
        }
    }
    if (!has_present) return -1;

    switch (properties.deviceType)
    {
        case vk::PhysicalDeviceType::eDiscreteGpu:
            return 3;
        case vk::PhysicalDeviceType::eIntegratedGpu:
            return 2;
        case vk::PhysicalDeviceType::eVirtualGpu:
            return 1;
        default:
            return 0;  // e.g. CPU (llvmpipe)
    }
}

vk::PhysicalDevice Instance::pickPhysicalDevice(vk::SurfaceKHR surface, int preferredDevice)
{
    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

    int selected   = -1;
    int best_score = -1;
    for (int i = 0; i < (int)physicalDevices.size(); ++i)
    {
        auto properties = physicalDevices[i].getProperties();
        int score       = deviceScore(physicalDevices[i], surface);
        std::cout << "Vulkan Device [" << i << "] : " << properties.deviceName.data() << ", API "
                  << versionString(properties.apiVersion) << (score < 0 ? " (not suitable)" : "") << std::endl;

        if (score < 0) continue;
        if (i == preferredDevice)
        {
            selected   = i;
            best_score = std::numeric_limits<int>::max();
        }
        else if (score > best_score)
        {
            selected   = i;
            best_score = score;
        }
    }

    if (selected < 0)
    {
        SAIGA_EXIT_ERROR("Could not find a suitable Vulkan device. Saiga requires Vulkan " +
                         versionString(SAIGA_VULKAN_API_VERSION) +
                         " with dynamic rendering, synchronization2 and timeline semaphores.");
    }

    std::cout << "Selected Vulkan Device [" << selected
              << "] : " << physicalDevices[selected].getProperties().deviceName.data() << std::endl;
    return physicalDevices[selected];
}

bool Instance::hasLayer(const std::string& name)
{
    auto prop = vk::enumerateInstanceLayerProperties();
    for (auto l : prop)
    {
        if (std::string(l.layerName.data()) == name) return true;
    }
    return false;
}

bool Instance::hasExtension(const std::string& name)
{
    auto prop = vk::enumerateInstanceExtensionProperties();
    for (auto l : prop)
    {
        if (std::string(l.extensionName.data()) == name) return true;
    }
    return false;
}

}  // namespace Vulkan
}  // namespace Saiga
