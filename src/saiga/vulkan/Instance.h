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

    void create(std::vector<const char*> instanceExtensions, bool enableValidation);

    vk::PhysicalDevice pickPhysicalDevice();

    operator vk::Instance() const { return instance; }
    operator VkInstance() const { return instance; }

   private:
    // Vulkan instance, stores all per-application states
    vk::Instance instance = nullptr;
    Debug debug;
};


}  // namespace Vulkan
}  // namespace Saiga
