/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanHelper.h"



namespace Saiga {
namespace Vulkan {

bool memory_type_from_properties(const vk::PhysicalDeviceMemoryProperties& memory_properties, int32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex) {
    // Search memtypes to find first index with those properties
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((memory_properties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                *typeIndex = i;
                return true;
            }
        }
        typeBits >>= 1;
    }
    // No memory types matched, return failure
    return false;
}


}
}
