#pragma once

#include "saiga/config.h"
#include <functional>

SAIGA_GLOBAL extern void catchSegFaults();

SAIGA_GLOBAL extern void addCustomSegfaultHandler(std::function<void()> fnc);


SAIGA_GLOBAL extern void printCurrentStack();