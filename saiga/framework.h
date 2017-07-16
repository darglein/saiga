#pragma once

#include <saiga/config.h>

namespace Saiga {

class OpenGLWindow;

SAIGA_LOCAL extern void readConfigFile();
SAIGA_GLOBAL extern void writeExtensions();

SAIGA_GLOBAL extern void initSaiga();
SAIGA_GLOBAL extern void cleanupSaiga();

}
