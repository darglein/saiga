/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"


using namespace Saiga;

int main()
{
    initSaigaSampleNoWindow();

    std::cout << ConsoleColor::RED << "Hello World! :D" << std::endl;
    std::cout << ConsoleColor::GREEN << "Hello World! :D" << ConsoleColor::RESET << std::endl;
    return 0;
}
