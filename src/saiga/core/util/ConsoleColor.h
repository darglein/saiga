/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <iostream>
namespace Saiga
{
/**
 * Colored console output based on the ANSI escape codes:
 * https://en.wikipedia.org/wiki/ANSI_escape_code
 *
 * This may not work for all terminals. From my experience, most linux
 * shells support these colors.
 *
 * Usage:
 * std::cout << ConsoleColor::RED << "Hello World! :D" << ConsoleColor::RESET << std::endl;
 *
 */
enum class ConsoleColor : int
{
    RESET      = 0,
    BLACK      = 30,
    RED        = 31,
    GREEN      = 32,
    YELLOW     = 33,
    BLUE       = 34,
    MAGENTA    = 35,
    CYAN       = 36,
    WHITE      = 37,
    BG_BLACK   = 40,
    BG_RED     = 41,
    BG_GREEN   = 42,
    BG_YELLOW  = 43,
    BG_BLUE    = 44,
    BG_MAGENTA = 45,
    BG_CYAN    = 46,
    BG_WHITE   = 47,
};

inline std::ostream& operator<<(std::ostream& strm, ConsoleColor color)
{
    strm << "\033[" << static_cast<int>(color) << "m";
    return strm;
}


}  // namespace Saiga
