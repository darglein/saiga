/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The implementation of the Vulkan Memory Allocator (submodules/VulkanMemoryAllocator).

#include "saiga/core/util/assert.h"

// VMA's checks (e.g. the use of a freed allocation) as SAIGA_ASSERT, so they are enabled with SAIGA_ASSERTS also in
// release builds (NDEBUG disables VMA's default assert()).
#if defined(SAIGA_ASSERTS)
#    define VMA_ASSERT(expr) SAIGA_ASSERT_MSG((expr), "Vulkan Memory Allocator")
#endif

// saiga_vulkan links the Vulkan loader, so the functions are imported statically.
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
