#pragma once

#include "saiga/config.h"
#include <functional>

/**
 * Adds a signal handler for the SIGSEGV signal.
 * When a SIGSEGV is caught the current stack trace is printed to std::cout
 * and a assertion is triggered. If a custom SegfaultHandler is set this function
 * is called after printing the stack trace.
 */
SAIGA_GLOBAL extern void catchSegFaults();

SAIGA_GLOBAL extern void addCustomSegfaultHandler(std::function<void()> fnc);

SAIGA_GLOBAL extern void printCurrentStack();
