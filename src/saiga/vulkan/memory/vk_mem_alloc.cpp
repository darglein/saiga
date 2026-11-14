/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The implementation of the Vulkan Memory Allocator (submodules/VulkanMemoryAllocator).
// saiga_vulkan links the Vulkan loader, so the functions are imported statically.
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
