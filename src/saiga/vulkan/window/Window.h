/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/core/window/WindowBase.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/svulkan.h"


typedef struct SDL_Window SDL_Window;


namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API VulkanWindow : public WindowBase
{
   public:
    VulkanWindow(WindowParameters windowParameters);
    virtual ~VulkanWindow();



    void renderImGui(bool* p_open = nullptr) override;
    virtual void swap() override;

    virtual std::unique_ptr<ImGuiVulkanRenderer> createImGui(size_t frameCount) { return nullptr; }

    virtual std::vector<std::string> getRequiredInstanceExtensions()       = 0;
    virtual void createSurface(VkInstance instance, VkSurfaceKHR* surface) = 0;
};


}  // namespace Vulkan
}  // namespace Saiga
