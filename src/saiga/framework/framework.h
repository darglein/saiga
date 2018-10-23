/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include <vector>

namespace Saiga {


struct SAIGA_GLOBAL SaigaParameters
{
    // share/ directory where saiga has been installed.
    std::vector<std::string> shaderDirectory    =  {"shader", SAIGA_INSTALL_PREFIX  "/share/saiga/shader"};
    std::vector<std::string> textureDirectory   = {"textures/"};
    std::vector<std::string> modelDirectory     = {"models/"};
    std::vector<std::string> fontDirectory      = {"fonts/"};

    std::string mainThreadName = "Saiga::main";

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};

SAIGA_GLOBAL extern void writeExtensions();

SAIGA_GLOBAL extern void initSample(SaigaParameters& saigaParameters);
SAIGA_GLOBAL extern void initSaiga(const SaigaParameters& params);
SAIGA_GLOBAL extern void cleanupSaiga();

}
