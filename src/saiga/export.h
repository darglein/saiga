/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga_buildconfig.h"


//source: https://gcc.gnu.org/wiki/Visibility

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
  #define SAIGA_HELPER_DLL_IMPORT __declspec(dllimport)
  #define SAIGA_HELPER_DLL_EXPORT __declspec(dllexport)
  #define SAIGA_HELPER_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define SAIGA_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
    #define SAIGA_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
    #define SAIGA_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define SAIGA_HELPER_DLL_IMPORT
    #define SAIGA_HELPER_DLL_EXPORT
    #define SAIGA_HELPER_DLL_LOCAL
  #endif
#endif

// Now we use the generic helper definitions above to define SAIGA_API and SAIGA_LOCAL.
// SAIGA_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// SAIGA_LOCAL is used for non-api symbols.

#ifdef SAIGA_BUILD_SHARED // defined if SAIGA is compiled as a DLL
  #ifdef SAIGA_DLL_EXPORTS // defined if we are building the SAIGA DLL (instead of using it)
    #define SAIGA_GLOBAL SAIGA_HELPER_DLL_EXPORT
  #else
    #define SAIGA_GLOBAL SAIGA_HELPER_DLL_IMPORT
  #endif // SAIGA_DLL_EXPORTS
  #define SAIGA_LOCAL SAIGA_HELPER_DLL_LOCAL
#else // SAIGA_DLL is not defined: this means SAIGA is a static lib.
  #define SAIGA_GLOBAL
  #define SAIGA_LOCAL
#endif // SAIGA_BUILD_SHARED

//don't use any specifiers on templates
#define SAIGA_TEMPLATE
