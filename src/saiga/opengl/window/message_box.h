/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>

namespace Saiga
{
// Create a message box with title and content.
// The control flow waits until "ok" has been pressed and then returns.
// If no windowing library (like SDL) is found, the message will be written to the console.
SAIGA_OPENGL_API void MessageBox(const std::string& title, const std::string& content);


}  // namespace Saiga
