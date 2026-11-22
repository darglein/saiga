/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/VulkanDevice.h"

#include "saiga/vulkan/window/HeadlessWindow.h"
#include "saiga/vulkan/window/Window.h"

#include <map>
#include <mutex>
#include <string>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// The pool of reusable headless devices (setHeadlessDeviceReuse): the instance and the logical device with the
// pipeline cache, without the objects on the device (VulkanBase::destroyDeviceObjects)
struct DevicePool
{
    std::mutex mutex;
    bool enabled = false;
    std::multimap<std::string, std::unique_ptr<VulkanDevice::Core>> devices;

    std::unique_ptr<VulkanDevice::Core> take(const std::string& key)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = devices.find(key);
        if (it == devices.end()) return nullptr;
        auto core = std::move(it->second);
        devices.erase(it);
        return core;
    }

    // Destroys the pooled devices (outside the lock: the validation layer may report leaked objects)
    void release()
    {
        std::multimap<std::string, std::unique_ptr<VulkanDevice::Core>> all;
        {
            std::lock_guard<std::mutex> lock(mutex);
            all.swap(devices);
        }
        all.clear();
    }

    ~DevicePool() { release(); }
};
DevicePool& devicePool()
{
    static DevicePool pool;
    return pool;
}

// Everything the instance and the logical device are created from (the objects on the device are created again)
std::string deviceKey(const VulkanParameters& parameters)
{
    std::string key = parameters.enableValidationLayer ? "v1" : "v0";
    key += ":" + std::to_string(parameters.preferredDevice) + ":";
    key.append(reinterpret_cast<const char*>(&parameters.physicalDeviceFeatures),
               sizeof(parameters.physicalDeviceFeatures));
    for (const char* e : parameters.deviceExtensions) key += std::string(":") + e;
    return key;
}
}  // namespace

VulkanDevice::VulkanDevice() : VulkanDevice(std::make_unique<Core>()) {}

VulkanDevice::VulkanDevice(std::unique_ptr<Core> c) : core(std::move(c)), instance(core->instance), base(core->base) {}

VulkanDevice::~VulkanDevice()
{
    // The device first (its memory, queues and pools), then the surface (the swap chain that used it is gone), then
    // the instance (with the core, after this body)
    if (base.device) base.device.waitIdle();
    if (!reuseKey.empty() && surface == VK_NULL_HANDLE && base.device && headlessDeviceReuse())
    {
        base.destroyDeviceObjects();
        auto& pool = devicePool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.devices.emplace(reuseKey, std::move(core));
        return;
    }
    base.destroy();
    if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
    surface = VK_NULL_HANDLE;
}

std::unique_ptr<VulkanDevice> createVulkanDevice(VulkanWindow& window, VulkanParameters parameters, int framesInFlight)
{
    parameters.physicalDeviceFeatures.fillModeNonSolid = VK_TRUE;
    parameters.physicalDeviceFeatures.wideLines        = VK_TRUE;
    parameters.physicalDeviceFeatures.largePoints      = VK_TRUE;
    // For Sampler (anisotropic filtering) and IndirectBuffer. Dropped in createLogicalDevice if not available.
    parameters.physicalDeviceFeatures.samplerAnisotropy         = VK_TRUE;
    parameters.physicalDeviceFeatures.multiDrawIndirect         = VK_TRUE;
    parameters.physicalDeviceFeatures.drawIndirectFirstInstance = VK_TRUE;

    const bool reusable   = headlessDeviceReuse() && dynamic_cast<HeadlessWindow*>(&window) != nullptr;
    const std::string key = reusable ? deviceKey(parameters) : std::string();
    if (reusable)
    {
        if (auto core = devicePool().take(key))
        {
            auto d = std::make_unique<VulkanDevice>(std::move(core));
            window.createSurface(d->instance, &d->surface);
            SAIGA_ASSERT(d->surface == VK_NULL_HANDLE);
            d->base.recreateDeviceObjects(parameters);
            d->base.initMemory(framesInFlight);
            d->reuseKey            = key;
            d->base.reusePipelines = true;
            return d;
        }
    }

    auto d = std::make_unique<VulkanDevice>();
    d->instance.create(window.getRequiredInstanceExtensions(), parameters.enableValidationLayer);
    window.createSurface(d->instance, &d->surface);
    d->base.setPhysicalDevice(d->instance, d->instance.pickPhysicalDevice(d->surface, parameters.preferredDevice));
    d->base.createLogicalDevice(parameters, d->surface);
    d->base.initMemory(framesInFlight);
    if (reusable && d->surface == VK_NULL_HANDLE)
    {
        d->reuseKey            = key;
        d->base.reusePipelines = true;
    }
    return d;
}

void setHeadlessDeviceReuse(bool enable)
{
    auto& pool = devicePool();
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.enabled = enable;
    }
    if (!enable) pool.release();
}

bool headlessDeviceReuse()
{
    auto& pool = devicePool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    return pool.enabled;
}

void releasePooledDevices()
{
    devicePool().release();
}

}  // namespace Vulkan
}  // namespace Saiga
