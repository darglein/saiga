/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Instance.h"

#include "saiga/core/util/easylogging++.h"
//#include "saiga/vulkan/VulkanTools.h"

namespace Saiga
{
namespace Vulkan
{
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

    std::vector<std::string> instanceExtensions = _instanceExtensions;
    std::vector<std::string> instanceLayers;

    vk::ApplicationInfo appInfo;
    appInfo.pApplicationName = "Saiga Application";
    appInfo.pEngineName      = "Saiga";
    appInfo.apiVersion       = VK_API_VERSION_1_1;


    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    if (enableValidation)
    {
        // We require both, the debug report extension as well as a validation layer
        std::string ext = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
        std::string val = Debug::getDebugValidationLayers();


        if (hasExtension(ext) && hasLayer(val))
        {
            std::cout << "Vulkan Validation layer enabled!" << std::endl;
            instanceExtensions.push_back(ext);
            instanceLayers.push_back(val);
        }
        else if (!hasExtension(ext))
        {
            std::cerr << "Vulkan Warning: You tried to enable the validation layer, but the extension " << ext
                 << " was not found. Starting without valdiation layer..." << std::endl;
            enableValidation = false;
        }
        else if (!hasLayer(val))
        {
            std::cerr << "Vulkan Warning: You tried to enable the validation layer, but the layer " << val
                 << " was not found. Starting without valdiation layer..." << std::endl;
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

    instance = vk::createInstance(instanceCreateInfo);
    SAIGA_ASSERT(instance);

    // If requested, we enable the default validation layers for debugging
    if (enableValidation)
    {
        debug.init(instance);
    }
    std::cout << "Vulkan instance created." << std::endl;
}

vk::PhysicalDevice Instance::pickPhysicalDevice()
{
    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    if (physicalDevices.size() == 0)
    {
        SAIGA_EXIT_ERROR("Could not find a vulkan capable device.");
    }

    uint32_t selectedDevice = 0;
    auto physicalDevice     = physicalDevices[selectedDevice];

    {
        vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();

        std::cout << "Selected Vulkan Device [" << selectedDevice << "] : " << deviceProperties.deviceName << ", ";
        std::cout << " API: " << (deviceProperties.apiVersion >> 22) << "."
                  << ((deviceProperties.apiVersion >> 12) & 0x3ff) << "." << (deviceProperties.apiVersion & 0xfff)
                  << std::endl;
    }
    return physicalDevice;
}

bool Instance::hasLayer(const std::string& name)
{
    auto prop = vk::enumerateInstanceLayerProperties();
    for (auto l : prop)
    {
        if (std::string(l.layerName) == name) return true;
    }
    return false;
}

bool Instance::hasExtension(const std::string& name)
{
    auto prop = vk::enumerateInstanceExtensionProperties();
    for (auto l : prop)
    {
        if (std::string(l.extensionName) == name) return true;
    }
    return false;
}

}  // namespace Vulkan
}  // namespace Saiga
