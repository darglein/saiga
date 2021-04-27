/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "message_box.h"

#include <iostream>

#ifdef SAIGA_USE_GLFW
#    include "saiga/core/glfw/saiga_glfw.h"
#endif
namespace Saiga
{
void MessageBox(const std::string& title, const std::string& content)
{
    std::cout << ">> " << title << std::endl;
    std::cout << ">> " << content << std::endl;
    std::cout << "Press enter to continue..." << std::endl;
    std::cin.get();
}

}  // namespace Saiga
