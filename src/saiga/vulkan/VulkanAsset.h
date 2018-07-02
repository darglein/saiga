/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Asset.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Device.h"
#include "saiga/vulkan/VulkanBuffer.hpp"

namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL VulkanVertexColoredAsset : public VertexColoredAsset
{
public:
    VkDevice device = nullptr;
    vks::Buffer vertices;
    vks::Buffer indices;
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;

    void render(VkCommandBuffer cmd);

    void updateBuffer(vks::VulkanDevice *device, VkQueue copyQueue);
};

}
}
