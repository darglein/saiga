/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/saiga_buildconfig.h"


// source: https://gcc.gnu.org/wiki/Visibility

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
#    define SAIGA_HELPER_DLL_IMPORT __declspec(dllimport)
#    define SAIGA_HELPER_DLL_EXPORT __declspec(dllexport)
#    define SAIGA_HELPER_DLL_LOCAL
#else
#    if __GNUC__ >= 4  // Note: Clang also defines GNUC
#        define SAIGA_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#        define SAIGA_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#        define SAIGA_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#    else
#        error Unknown import/export defines.
#        define SAIGA_HELPER_DLL_IMPORT
#        define SAIGA_HELPER_DLL_EXPORT
#        define SAIGA_HELPER_DLL_LOCAL
#    endif
#endif

// Now we use the generic helper definitions above to define SAIGA_API and SAIGA_LOCAL.
// SAIGA_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// SAIGA_LOCAL is used for non-api symbols.

#ifdef SAIGA_BUILD_SHARED  // defined if SAIGA is compiled as a DLL

#    ifdef saiga_core_EXPORTS
#        define SAIGA_CORE_API SAIGA_HELPER_DLL_EXPORT
#    else
#        define SAIGA_CORE_API SAIGA_HELPER_DLL_IMPORT
#    endif

#    ifdef saiga_opengl_EXPORTS
#        define SAIGA_OPENGL_API SAIGA_HELPER_DLL_EXPORT
#    else
#        define SAIGA_OPENGL_API SAIGA_HELPER_DLL_IMPORT
#    endif

#    ifdef saiga_vulkan_EXPORTS
#        define SAIGA_VULKAN_API SAIGA_HELPER_DLL_EXPORT
#    else
#        define SAIGA_VULKAN_API SAIGA_HELPER_DLL_IMPORT
#    endif

#    ifdef saiga_vision_EXPORTS
#        define SAIGA_VISION_API SAIGA_HELPER_DLL_EXPORT
#    else
#        define SAIGA_VISION_API SAIGA_HELPER_DLL_IMPORT
#    endif

#    ifdef saiga_cuda_EXPORTS
#        define SAIGA_CUDA_API SAIGA_HELPER_DLL_EXPORT
#    else
#        define SAIGA_CUDA_API SAIGA_HELPER_DLL_IMPORT
#    endif

#    define SAIGA_LOCAL SAIGA_HELPER_DLL_LOCAL
#else  // SAIGA_DLL is not defined: this means SAIGA is a static lib.
#    define SAIGA_CORE_API
#    define SAIGA_OPENGL_API
#    define SAIGA_VULKAN_API
#    define SAIGA_VISION_API
#    define SAIGA_CUDA_API

#    define SAIGA_LOCAL
#endif  // SAIGA_BUILD_SHARED



// don't use any specifiers on templates
#define SAIGA_TEMPLATE
