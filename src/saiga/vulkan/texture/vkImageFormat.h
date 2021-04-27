/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/imageFormat.h"
#include "saiga/core/math/math.h"
#include "saiga/vulkan/svulkan.h"



namespace Saiga
{
namespace Vulkan
{
SAIGA_VULKAN_API vk::Format getvkFormat(ImageType type);

}
}  // namespace Saiga
