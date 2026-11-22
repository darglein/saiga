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
 * movable (VulkanBase holds a mutex): created by createVulkanDevice. The instance and the base are on the heap (core),
 * so that a reused headless device (setHeadlessDeviceReuse) can return them to the pool when it is destroyed.
 */
struct SAIGA_VULKAN_API VulkanDevice
{
    struct Core
    {
        Instance instance;
        VulkanBase base;
    };
    std::unique_ptr<Core> core;
    Instance& instance;
    VulkanBase& base;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    // The pool key of a reusable device (set by createVulkanDevice while the reuse is on; empty: destroyed)
    std::string reuseKey;

    VulkanDevice();
    explicit VulkanDevice(std::unique_ptr<Core> core);
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

/**
 * Reuse of headless devices, for test executables (off by default; the tests turn it on, tests/CMakeLists.txt): each
 * test creates its own renderer, and creating and destroying the instance and the device took longer than many tests
 * (e.g. 0.25 s on an NVIDIA driver). While the reuse is on, a destroyed VulkanDevice of a HeadlessWindow keeps its
 * instance and logical device in a process-wide pool, and createVulkanDevice for a HeadlessWindow takes a pooled one
 * that was created with the same instance and device options (validation, preferred device, features, device
 * extensions). The rest is new for every VulkanDevice: on the return the deferred deletions run and the objects on
 * the device are destroyed (memory allocator, queues and command pools, descriptor pool, frame timeline;
 * VulkanBase::destroyDeviceObjects), and they are created again on the reuse. Objects a renderer leaked stay on the
 * device until the pool is released: the validation layer reports them then (releasePooledDevices, at exit) instead of
 * when the renderer is destroyed. Windows with a surface never reuse a device.
 */
SAIGA_VULKAN_API void setHeadlessDeviceReuse(bool enable);
SAIGA_VULKAN_API bool headlessDeviceReuse();
// Destroys the pooled devices (also at exit; turning the reuse off releases them too)
SAIGA_VULKAN_API void releasePooledDevices();

}  // namespace Vulkan
}  // namespace Saiga
