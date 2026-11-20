/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/VulkanDevice.h"

#include "saiga/vulkan/window/Window.h"

namespace Saiga
{
namespace Vulkan
{
VulkanDevice::~VulkanDevice()
{
    // The device first (its memory, queues and pools), then the surface (the swap chain that used it is gone), then
    // the instance (a member, destroyed after this body)
    if (base.device) base.device.waitIdle();
    base.destroy();
    if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
    surface = VK_NULL_HANDLE;
}

std::unique_ptr<VulkanDevice> createVulkanDevice(VulkanWindow& window, VulkanParameters parameters, int framesInFlight)
{
    auto d = std::make_unique<VulkanDevice>();
    d->instance.create(window.getRequiredInstanceExtensions(), parameters.enableValidationLayer);
    window.createSurface(d->instance, &d->surface);
    d->base.setPhysicalDevice(d->instance, d->instance.pickPhysicalDevice(d->surface, parameters.preferredDevice));

    parameters.physicalDeviceFeatures.fillModeNonSolid = VK_TRUE;
    parameters.physicalDeviceFeatures.wideLines        = VK_TRUE;
    parameters.physicalDeviceFeatures.largePoints      = VK_TRUE;
    // For Sampler (anisotropic filtering) and IndirectBuffer. Dropped in createLogicalDevice if not available.
    parameters.physicalDeviceFeatures.samplerAnisotropy         = VK_TRUE;
    parameters.physicalDeviceFeatures.multiDrawIndirect         = VK_TRUE;
    parameters.physicalDeviceFeatures.drawIndirectFirstInstance = VK_TRUE;
    d->base.createLogicalDevice(parameters, d->surface);
    d->base.initMemory(framesInFlight);
    return d;
}

}  // namespace Vulkan
}  // namespace Saiga
