/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "message_box.h"

#include <iostream>
#ifdef SAIGA_USE_SDL
#    include <SDL2/SDL.h>
#endif
#ifdef SAIGA_USE_GLFW
#    include <GLFW/glfw3.h>
#endif
namespace Saiga
{
void MessageBox(const std::string& title, const std::string& content)
{
#ifdef SAIGA_USE_SDL456
    SDL_ShowSimpleMessageBox(0, title, content, nullptr);
#else
    std::cout << ">> " << title << std::endl;
    std::cout << ">> " << content << std::endl;
    std::cout << "Press enter to continue..." << std::endl;
    std::cin.get();
#endif
}

}  // namespace Saiga
