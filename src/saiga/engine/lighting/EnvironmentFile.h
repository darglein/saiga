/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The environment as a data file (docs/engine_design.md §10.7, §14; S36): data/<game>/environment.json, optional, one
// JSON object with the keys celestial, exposure, ambient, clouds, atmosphere, lighting (R-DATA-1: missing keys keep the
// defaults, unknown keys and wrong types are fatal, naming the file and the key path). The field names are the
// structs' (camelCase); clouds also takes windScale (EnvironmentSetup::cloudWindScale), atmosphere starIntensity
// (EnvironmentSettings::starIntensity). EnvironmentSetup::enabled is not read (the game enables the environment).

#pragma once

#include "saiga/engine/lighting/EnvironmentSetup.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"

#include <string>

namespace Saiga
{
namespace Engine
{
struct EnvironmentFile
{
    EnvironmentSetup setup;
    Vulkan::EnvironmentSettings settings;
    Vulkan::LightingParameters lighting;
};

// false and an error naming the file and the key on any error (out unchanged)
SAIGA_ENGINE_API bool tryReadEnvironmentFile(const std::string& path, EnvironmentFile& out, std::string& error);
// The file merged into the defaults; an error stops the program in every build (R-DATA-1)
SAIGA_ENGINE_API EnvironmentFile readEnvironmentFile(const std::string& path);

}  // namespace Engine
}  // namespace Saiga
