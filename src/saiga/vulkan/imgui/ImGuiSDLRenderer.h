/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
/*
 * UI overlay class using ImGui
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/util/math.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"

#ifndef SAIGA_USE_SDL
#    error Saiga was compiled without SDL2.
#endif

typedef struct SDL_Window SDL_Window;

namespace Saiga
{
namespace Vulkan
{
class SAIGA_GLOBAL ImGuiSDLRenderer : public SDL_EventListener, public ImGuiVulkanRenderer
{
   public:
    ImGuiSDLRenderer(size_t frameCount, const ImGuiParameters& params) : ImGuiVulkanRenderer(frameCount, params) {}
    // Initialize styles, keys, etc.
    void init(SDL_Window* window, float width, float height);

    void beginFrame() override;

    ~ImGuiSDLRenderer() override;

   protected:
    SDL_Window* window;

    virtual bool processEvent(const SDL_Event& event) override;
};

}  // namespace Vulkan
}  // namespace Saiga
