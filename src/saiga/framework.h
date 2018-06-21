/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>

namespace Saiga {


struct SAIGA_GLOBAL SaigaParameters
{
    // share/ directory where saiga has been installed.
    std::string shaderDirectory      = SAIGA_INSTALL_PREFIX  "/share/saiga/shader";
    std::string textureDirectory    = "textures/";

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};

SAIGA_GLOBAL extern void writeExtensions();

SAIGA_GLOBAL extern void initSaiga(const SaigaParameters& params);
SAIGA_GLOBAL extern void cleanupSaiga();

}
