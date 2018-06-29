/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkan.h"
#include "saiga/vulkan/vulkanBase.h"

namespace Saiga {
namespace Vulkan {

bool memory_type_from_properties(const vk::PhysicalDeviceMemoryProperties& memory_properties, int32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex);


}
}
