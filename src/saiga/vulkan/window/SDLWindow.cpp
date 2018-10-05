/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_SDL
#include "SDLWindow.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_vulkan.h"
#include "saiga/sdl/sdl_eventhandler.h"
#include "saiga/vulkan/imgui/ImGuiSDLRenderer.h"
#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif


namespace Saiga {
namespace Vulkan {

SDLWindow::SDLWindow(WindowParameters _windowParameters)
    :VulkanWindow(_windowParameters)
{
    Saiga::initSaiga(windowParameters.saigaParameters);
    create();
}

SDLWindow::~SDLWindow()
{
        SDL_StopTextInput();

        //Destroy window
        SDL_DestroyWindow( sdl_window );
        sdl_window = nullptr;

        //Quit SDL subsystems
        SDL_Quit();
}

std::shared_ptr<ImGuiVulkanRenderer> SDLWindow::createImGui()
{

    auto imGui = std::make_shared<Saiga::Vulkan::ImGuiSDLRenderer>();
    imGui->init(sdl_window,(float)windowParameters.width, (float)windowParameters.height);

    return imGui;
}

std::vector<const char *> SDLWindow::getRequiredInstanceExtensions()
{
    unsigned int count = 0;
    const char **names = NULL;
    auto res = SDL_Vulkan_GetInstanceExtensions(sdl_window, &count, NULL);
    cout << SDL_GetError() << endl;
    SAIGA_ASSERT(res);
    // now count is (probably) 2. Now you can make space:
    names = new const char *[count];

    // now call again with that not-NULL array you just allocated.
    res = SDL_Vulkan_GetInstanceExtensions(sdl_window, &count, names);
    cout << SDL_GetError() << endl;
    SAIGA_ASSERT(res);
    cout << "num extensions " << count << endl;
    // Now names should have (count) strings in it:

    std::vector<const char *> extensions;
    for (unsigned int i = 0; i < count; i++) {
        printf("Extension %d: %s\n", i, names[i]);
        extensions.push_back(names[i]);
    }

    // use it for VkInstanceCreateInfo and when you're done, free it:

    delete[] names;

    return extensions;
}



void SDLWindow::create()
{
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
    {
        std::cout << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
        SAIGA_ASSERT(0);
    }


    sdl_window = SDL_CreateWindow(windowParameters.name.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowParameters.width, windowParameters.height, SDL_WINDOW_VULKAN );
    if(!sdl_window)
    {
        std::cout << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
        SAIGA_ASSERT(0);

    }
}

void SDLWindow::createSurface(VkInstance instance, VkSurfaceKHR *surface)
{
    auto asdf = SDL_Vulkan_CreateSurface(sdl_window,instance,surface);
    SAIGA_ASSERT(asdf);
}

void SDLWindow::update(float dt)
{
    Saiga::SDL_EventHandler::update();

    if(Saiga::SDL_EventHandler::shouldQuit())
    {
        close();
    }

    if(updating)
        updating->update(dt);
}



}
}

#endif
