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
// Set the value of an enviroment variable for this process.
// This value is reset after the program exits.
SAIGA_CORE_API int SetEnv(const std::string& name, const std::string& value, int replace);

// Read the current value of an enviroment variable.
SAIGA_CORE_API std::string GetEnv(const std::string& name);

}  // namespace Saiga
