/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/window/WindowBase.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"

typedef struct SDL_Window SDL_Window;


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL VulkanWindow : public WindowBase
{
public:
    VulkanWindow(WindowParameters windowParameters);
    virtual ~VulkanWindow();



    void renderImGui(bool* p_open = nullptr) override;
    virtual void swap() override;

    virtual std::shared_ptr<ImGuiVulkanRenderer> createImGui() { return nullptr; }

    virtual std::vector<const char*> getRequiredInstanceExtensions() = 0;
    virtual void createSurface(VkInstance instance, VkSurfaceKHR* surface) = 0;
};


}
}
