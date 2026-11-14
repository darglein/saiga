/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/VulkanAsset.h"
#include "saiga/vulkan/VulkanBuffer.hpp"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/texture/Texture.h"

#include <optional>

namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API TextureDisplay : public Pipeline
{
   public:
    using VertexType = VertexNC;


    ~TextureDisplay() { destroy(); }
    void destroy();


    /**
     * Render the texture at the given pixel position and size (the pipeline must be bound).
     *
     * This sets the dynamic viewport to the rectangle. If restoreViewport is given, it is set again afterwards;
     * otherwise the caller must reset the viewport before drawing anything else.
     */
    void renderTexture(vk::CommandBuffer cmd, DescriptorSet& descriptor, vec2 position, vec2 size,
                       std::optional<vk::Viewport> restoreViewport = {});



    void init(Saiga::Vulkan::VulkanBase& vulkanDevice, const RenderingFormats& formats);


    StaticDescriptorSet createAndUpdateDescriptorSet(Texture& texture);

   private:
    Saiga::Vulkan::VulkanVertexColoredAsset blitMesh;
};



}  // namespace Vulkan
}  // namespace Saiga
