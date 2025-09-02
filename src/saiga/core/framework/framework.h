/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>
#include <vector>
#include <filesystem>

namespace Saiga
{
struct SAIGA_CORE_API SaigaParameters
{
    // share/ directory where saiga has been installed.
    std::vector<std::filesystem::path> shaderDirectory  = {"shader/", SAIGA_SHADER_PREFIX};
    std::vector<std::filesystem::path> textureDirectory = {"textures/"};
    std::vector<std::filesystem::path> modelDirectory   = {"models/"};
    std::vector<std::filesystem::path> fontDirectory    = {"fonts/"};
    // for everything else
    std::vector<std::filesystem::path> dataDirectory = {"data/"};

    std::string mainThreadName = "Saiga::main";

    bool logging_enabled = false;
    int verbose_logging  = false;

    bool ensureSaigaShadersExist = true;
    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};



SAIGA_CORE_API extern void printSaigaInfo();

SAIGA_CORE_API extern bool findShaders(const SaigaParameters& params);

SAIGA_CORE_API extern void initSaigaSampleNoWindow(bool createConfig = true);
SAIGA_CORE_API extern void initSaigaSample();
SAIGA_CORE_API extern void initSaiga(const SaigaParameters& params);
SAIGA_CORE_API extern void cleanupSaiga();

}  // namespace Saiga
