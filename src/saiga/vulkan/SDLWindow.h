/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "saiga/vulkan/base/vulkanexamplebase.h"

typedef struct SDL_Window SDL_Window;


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL SDLWindow : public VulkanExampleBase
{
public:

    SDL_Window *sdl_window = nullptr;

    std::vector<const char*> getRequiredInstanceExtensions() override;

    void setupWindow() override;

    void createSurface(VkInstance instance, VkSurfaceKHR* surface);
};


}
}
