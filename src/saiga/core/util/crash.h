/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <functional>

namespace Saiga
{
/**
 * Adds a signal handler for the SIGSEGV signal.
 * When a SIGSEGV is caught the current stack trace is printed to std::cout
 * and a assertion is triggered. If a custom SegfaultHandler is set this function
 * is called after printing the stack trace.
 */
SAIGA_CORE_API extern void catchSegFaults();

SAIGA_CORE_API extern void addCustomSegfaultHandler(std::function<void()> fnc);

SAIGA_CORE_API extern void printCurrentStack();

}  // namespace Saiga
