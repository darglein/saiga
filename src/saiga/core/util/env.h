/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <optional>
#include <string>

namespace Saiga
{
// Set the value of an enviroment variable for this process.
// This value is reset after the program exits.
SAIGA_CORE_API int SetEnv(const std::string& name, const std::string& value, int replace);

// Read the current value of an enviroment variable. Returns an empty string if it is not set.
SAIGA_CORE_API std::string GetEnv(const std::string& name);

// The value of an environment variable, or nullopt if it is not set (unlike GetEnv, an empty value is not unset).
// Use it instead of std::getenv, which the MSVC CRT deprecates (C4996).
SAIGA_CORE_API std::optional<std::string> GetEnvIfSet(const std::string& name);

}  // namespace Saiga
