/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_SDL
#    include "saiga/core/sdl/saiga_sdl.h"
#    include "saiga/core/sdl/sdl_eventhandler.h"
#    include "saiga/vulkan/imgui/ImGuiSDLRenderer.h"

#    include "SDLWindow.h"

#    include "SDL_vulkan.h"
#    if defined(SAIGA_OPENGL_INCLUDED)
#        error OpenGL was included somewhere.
#    endif


namespace Saiga
{
namespace Vulkan
{
SDLWindow::SDLWindow(WindowParameters _windowParameters) : VulkanWindow(_windowParameters)
{
    Saiga::initSaiga(windowParameters.saigaParameters);
    create();
}

SDLWindow::~SDLWindow()
{
    SDL_StopTextInput();

    // Destroy window
    SDL_DestroyWindow(sdl_window);
    sdl_window = nullptr;

    // Quit SDL subsystems
    SDL_Quit();
}

std::unique_ptr<ImGuiVulkanRenderer> SDLWindow::createImGui(size_t frameCount)
{
    if (windowParameters.imguiParameters.enable)
    {
        auto imGui = std::make_unique<Saiga::Vulkan::ImGuiSDLRenderer>(frameCount, windowParameters.imguiParameters);
        imGui->init(sdl_window, (float)windowParameters.width, (float)windowParameters.height);
        return std::move(imGui);
    }
    return {};
}

std::vector<std::string> SDLWindow::getRequiredInstanceExtensions()
{
    unsigned int count = 0;
    const char** names = nullptr;
    auto res           = SDL_Vulkan_GetInstanceExtensions(sdl_window, &count, nullptr);
    std::cout << SDL_GetError() << std::endl;
    SAIGA_ASSERT(res);
    // now count is (probably) 2. Now you can make space:
    names = new const char*[count];

    // now call again with that not-NULL array you just allocated.
    res = SDL_Vulkan_GetInstanceExtensions(sdl_window, &count, names);
    std::cout << SDL_GetError() << std::endl;
    SAIGA_ASSERT(res);
    // Now names should have (count) strings in it:

    std::vector<std::string> extensions;
    for (unsigned int i = 0; i < count; i++)
    {
        extensions.push_back(names[i]);
    }

    // use it for VkInstanceCreateInfo and when you're done, free it:

    delete[] names;

    return extensions;
}



void SDLWindow::create()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cout << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
        SAIGA_ASSERT(0);
    }

    Uint32 window_flags = SDL_WINDOW_VULKAN;

    if (windowParameters.fullscreen()) window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    if (windowParameters.borderLess()) window_flags |= SDL_WINDOW_BORDERLESS;
    if (windowParameters.resizeAble) window_flags |= SDL_WINDOW_RESIZABLE;
    if (windowParameters.hidden) window_flags |= SDL_WINDOW_HIDDEN;

    sdl_window = SDL_CreateWindow(windowParameters.name.c_str(),
                                  SDL_WINDOWPOS_CENTERED_DISPLAY(windowParameters.selected_display),
                                  SDL_WINDOWPOS_CENTERED_DISPLAY(windowParameters.selected_display),
                                  windowParameters.width, windowParameters.height, window_flags);
    if (!sdl_window)
    {
        std::cout << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
        SAIGA_ASSERT(0);
    }
}

void SDLWindow::createSurface(VkInstance instance, VkSurfaceKHR* surface)
{
    auto create_surface_success = SDL_Vulkan_CreateSurface(sdl_window, instance, surface);
    SAIGA_ASSERT(create_surface_success);
}

void SDLWindow::update(float dt)
{
    Saiga::SDL_EventHandler::update();

    if (Saiga::SDL_EventHandler::shouldQuit())
    {
        close();
    }

    if (updating) updating->update(dt);
}



}  // namespace Vulkan
}  // namespace Saiga

#endif
