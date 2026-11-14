/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/svulkan.h"

#include "Debug.h"

namespace Saiga
{
namespace Vulkan
{
/**
 * The Vulkan module uses exactly Vulkan 1.3 core: instance and device are created with API version 1.3 (also on 1.4
 * drivers), and the only extensions are the window system ones (surface, swapchain) and VK_EXT_debug_utils for the
 * validation layer. Used core features: dynamic rendering, synchronization2, timeline semaphores.
 */
constexpr uint32_t SAIGA_VULKAN_API_VERSION = VK_API_VERSION_1_3;

class SAIGA_VULKAN_API Instance
{
   public:
    ~Instance() { destroy(); }

    void destroy();

    void create(const std::vector<std::string>& instanceExtensions, bool enableValidation);

    /**
     * Selects a device with Vulkan >= 1.3, the required features and a queue family with graphics, compute and
     * present support for the surface. Discrete GPUs are preferred.
     *
     * @param preferredDevice If >= 0 and suitable, the device with this index (in vkEnumeratePhysicalDevices order)
     *                        is used.
     */
    vk::PhysicalDevice pickPhysicalDevice(vk::SurfaceKHR surface, int preferredDevice = -1);

    /**
     * Returns true if a Vulkan driver is installed that provides a device saiga can render with (without a surface,
     * see HeadlessWindow). Creating a renderer without such a device exits the program, so e.g. tests check this first.
     */
    static bool suitableDeviceAvailable();

    // Check if the layer is available
    bool hasLayer(const std::string& name);
    bool hasExtension(const std::string& name);

    std::vector<const char*> getEnabledExtensions()
    {
        std::vector<const char*> v;
        for (auto& s : _extensions) v.push_back(s.c_str());
        return v;
    }

    std::vector<const char*> getEnabledLayers()
    {
        std::vector<const char*> v;
        for (auto& s : _layers) v.push_back(s.c_str());
        return v;
    }

    // The API version the instance was created with.
    uint32_t getApiVersion() const { return apiVersion; }

    operator vk::Instance() const { return instance; }
    operator VkInstance() const { return instance; }

   private:
    // Vulkan instance, stores all per-application states
    vk::Instance instance = nullptr;
    uint32_t apiVersion   = 0;

    Debug debug;

    // the extensions and layers which have been enabled for this instance
    std::vector<std::string> _extensions;
    std::vector<std::string> _layers;

    // Returns a score < 0 if the device can not be used.
    static int deviceScore(vk::PhysicalDevice device, vk::SurfaceKHR surface);
};

}  // namespace Vulkan
}  // namespace Saiga
