/**
 * Copyright (c) 2017 Darius Rückert
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
class SAIGA_VULKAN_API Instance
{
   public:
    ~Instance() { destroy(); }
    void destroy();

    void create(const std::vector<std::string>& instanceExtensions, bool enableValidation);

    vk::PhysicalDevice pickPhysicalDevice();

    // Check if the layer is available
    bool hasLayer(const std::string& name);
    bool hasExtension(const std::string& name);

    auto getEnabledExtensions() { return extensions; }
    auto getEnabledLayers() { return layers; }

    operator vk::Instance() const { return instance; }
    operator VkInstance() const { return instance; }

   private:
    // Vulkan instance, stores all per-application states
    vk::Instance instance = nullptr;
    Debug debug;

    // the extensions and layers which have been enabled for this instance
    std::vector<const char*> extensions;
    std::vector<const char*> layers;
};


}  // namespace Vulkan
}  // namespace Saiga
