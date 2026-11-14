/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vkImageFormat.h"

#include "saiga/core/util/assert.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

namespace Saiga
{
namespace Vulkan
{
static const vk::Format ImageTypeInternalVK[] = {

    vk::Format::eR8Snorm,   vk::Format::eR8G8Snorm,    vk::Format::eR8G8B8Snorm,     vk::Format::eR8G8B8A8Snorm,
    vk::Format::eR8Unorm,   vk::Format::eR8G8Unorm,    vk::Format::eR8G8B8Unorm,     vk::Format::eR8G8B8A8Unorm,

    vk::Format::eR16Snorm,  vk::Format::eR16G16Snorm,  vk::Format::eR16G16B16Snorm,  vk::Format::eR16G16B16A16Snorm,
    vk::Format::eR16Unorm,  vk::Format::eR16G16Unorm,  vk::Format::eR16G16B16Unorm,  vk::Format::eR16G16B16A16Unorm,

    vk::Format::eR32Sint,   vk::Format::eR32G32Sint,   vk::Format::eR32G32B32Sint,   vk::Format::eR32G32B32A32Sint,
    vk::Format::eR32Uint,   vk::Format::eR32G32Uint,   vk::Format::eR32G32B32Uint,   vk::Format::eR32G32B32A32Uint,

    vk::Format::eR32Sfloat, vk::Format::eR32G32Sfloat, vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32A32Sfloat,
    vk::Format::eR64Sfloat, vk::Format::eR64G64Sfloat, vk::Format::eR64G64B64Sfloat, vk::Format::eR64G64B64A64Sfloat};


static_assert(sizeof(ImageTypeInternalVK) / sizeof(ImageTypeInternalVK[0]) == TYPE_UNKNOWN,
              "The table must have one format per image type");

vk::Format getvkFormat(ImageType type)
{
    SAIGA_ASSERT(type >= 0 && type < TYPE_UNKNOWN, "getvkFormat: unknown image type");
    return ImageTypeInternalVK[type];
}

}  // namespace Vulkan
}  // namespace Saiga
