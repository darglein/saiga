#pragma once

#include <saiga/config.h>

class Window;



SAIGA_LOCAL extern void readConfigFile();
SAIGA_LOCAL extern void writeExtensions();

SAIGA_GLOBAL extern void initFramework(Window* window);
