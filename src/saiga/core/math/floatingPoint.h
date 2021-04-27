/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

namespace Saiga
{
namespace FP
{
// Note:
// This only sets the flags for the SSE control register.
// You have to make sure, that the compiler actually uses the sse unit for floating point operations.
// GCC flags: -msse2 -mfpmath=sse
// Visual Studio flags: /arch:SSE2 /fp:strict


// resets the control register to the default rounding modes
// is called once in initSaiga
SAIGA_CORE_API extern void resetSSECSR();

// returns true if the flags of the control register still matches the default values
SAIGA_CORE_API extern bool checkSSECSR();

// set a different floating point rounding mode
// use this only for tests
SAIGA_CORE_API extern void breakSSECSR();


SAIGA_CORE_API extern void printCPUInfo();
}  // namespace FP
}  // namespace Saiga
