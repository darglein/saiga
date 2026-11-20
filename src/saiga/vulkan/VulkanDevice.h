/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Parameters.h"

#include <memory>

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;

/**
 * The instance, the window's surface (none for a HeadlessWindow) and the device (docs/engine_design.md §8.3, S13).
 * Owns the surface: the swap chain does not destroy it. Destroyed in reverse order: device, surface, instance. Not
 * movable (VulkanBase holds a mutex): created by createVulkanDevice.
 */
struct SAIGA_VULKAN_API VulkanDevice
{
    Instance instance;
    VulkanBase base;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VulkanDevice() = default;
    ~VulkanDevice();
    VulkanDevice(const VulkanDevice&)            = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
};

/**
 * Creates the instance (with the window's instance extensions), the window's surface, picks the physical device,
 * enables the features the engine's renderers use (non-solid fill, wide lines, large points, anisotropic sampling,
 * multi-draw indirect, first instance in indirect draws; dropped if unsupported), creates the logical device and the
 * memory allocator for framesInFlight frames. VulkanRenderer and the engine app use it.
 */
SAIGA_VULKAN_API std::unique_ptr<VulkanDevice> createVulkanDevice(VulkanWindow& window, VulkanParameters parameters,
                                                                  int framesInFlight);

}  // namespace Vulkan
}  // namespace Saiga
