/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/window/window.h"
#include "saiga/vulkan/svulkan.h"

typedef struct SDL_Window SDL_Window;


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL SDLWindow
{
public:
    int width, height;

    SDL_Window *sdl_window = nullptr;
    WindowParameters windowParameters;
    std::vector<const char*> getRequiredInstanceExtensions();

    SDLWindow(WindowParameters _windowParameters);

    void setupWindow();

    void createSurface(VkInstance instance, VkSurfaceKHR* surface);
};


}
}
