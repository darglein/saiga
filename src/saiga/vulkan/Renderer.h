/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

//#include "saiga/vulkan/memory/ChunkAllocator.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/SwapChain.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/window/Interfaces.h"

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;

class SAIGA_GLOBAL VulkanRenderer : public RendererBase
{
   public:
    Saiga::Vulkan::VulkanBase base;

    VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters);
    virtual ~VulkanRenderer();

    virtual void render(Camera* cam) {}
    virtual void bindCamera(Camera* cam) {}

    virtual float getTotalRenderTime();

    void renderImGui(bool *p_open) override;

protected:
    /**
     * Shared Member variables common for all vulkan render engines.
     */

    Saiga::Vulkan::VulkanWindow& window;


    uint32_t width  = 1280;
    uint32_t height = 720;

    Saiga::Vulkan::Instance instance;


    std::vector<const char*> enabledDeviceExtensions;
    std::vector<const char*> enabledInstanceExtensions;
    VulkanSwapChain swapChain;
    VulkanParameters vulkanParameters;

   private:
    void initInstanceDevice();
};



}  // namespace Vulkan
}  // namespace Saiga
