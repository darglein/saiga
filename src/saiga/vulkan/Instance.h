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

    operator vk::Instance() const { return instance; }
    operator VkInstance() const { return instance; }

   private:
    // Vulkan instance, stores all per-application states
    vk::Instance instance = nullptr;
    Debug debug;

    // the extensions and layers which have been enabled for this instance
    std::vector<std::string> _extensions;
    std::vector<std::string> _layers;
};


}  // namespace Vulkan
}  // namespace Saiga
