/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"
#include "saiga/image/imageFormat.h"
#include "saiga/vulkan/svulkan.h"



namespace Saiga {
namespace Vulkan {

SAIGA_GLOBAL vk::Format getvkFormat(ImageType type);

}
}
