/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "message_box.h"

#include <iostream>
#ifdef SAIGA_USE_SDL
#    include "saiga/core/sdl/saiga_sdl.h"
#endif
#ifdef SAIGA_USE_GLFW
#    include <GLFW/glfw3.h>
#endif
namespace Saiga
{
void MessageBox(const std::string& title, const std::string& content)
{
#ifdef SAIGA_USE_SDL
    SDL_ShowSimpleMessageBox(0, title.c_str(), content.c_str(), nullptr);
#else
    std::cout << ">> " << title << std::endl;
    std::cout << ">> " << content << std::endl;
    std::cout << "Press enter to continue..." << std::endl;
    std::cin.get();
#endif
}

}  // namespace Saiga
