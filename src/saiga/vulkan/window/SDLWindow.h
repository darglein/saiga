/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/window/Window.h"

#ifndef SAIGA_USE_SDL
#    error Saiga was compiled without SDL2.
#endif

typedef struct SDL_Window SDL_Window;


namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API SDLWindow : public VulkanWindow
{
   public:
    SDL_Window* sdl_window = nullptr;
    SDLWindow(WindowParameters _windowParameters);

    ~SDLWindow();
    virtual std::unique_ptr<ImGuiVulkanRenderer> createImGui(size_t frameCount) override;

    std::vector<std::string> getRequiredInstanceExtensions() override;
    void createSurface(VkInstance instance, VkSurfaceKHR* surface) override;
    virtual void update(float dt) override;

   private:
    void create();
};


}  // namespace Vulkan
}  // namespace Saiga
