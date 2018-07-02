/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/window/Window.h"

typedef struct SDL_Window SDL_Window;


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL SDLWindow : public VulkanWindow
{
public:
    SDL_Window *sdl_window = nullptr;
    SDLWindow(WindowParameters _windowParameters);

    std::vector<const char*> getRequiredInstanceExtensions();
    void createSurface(VkInstance instance, VkSurfaceKHR* surface);
    virtual void update(float dt) override;
private:
    void create();
};


}
}
