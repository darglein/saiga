/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>

namespace Saiga {


SAIGA_GLOBAL extern void writeExtensions();

SAIGA_GLOBAL extern void initSaiga(const std::string& configFile);
SAIGA_GLOBAL extern void cleanupSaiga();

}
