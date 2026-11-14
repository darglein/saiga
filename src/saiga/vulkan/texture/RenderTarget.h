/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"
#include "saiga/vulkan/texture/Texture.h"

namespace Saiga
{
namespace Vulkan
{
struct SAIGA_VULKAN_API RenderTargetParameters
{
    // One color attachment per format (multiple render targets). Can be empty for depth only passes.
    std::vector<vk::Format> colorFormats = {vk::Format::eR8G8B8A8Unorm};

    // A depth format without stencil (eD32Sfloat, eD16Unorm, eX8D24UnormPack32), or eUndefined for no depth buffer.
    vk::Format depthFormat = vk::Format::eD32Sfloat;

    // > 1: the scene is rendered into multisampled images, which are resolved into the color textures by end().
    // The multisampled depth buffer is not resolved, so depthTexture() can not be sampled then.
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;

    // Usage of the color textures in addition to color attachment + sampled (e.g. eStorage).
    vk::ImageUsageFlags colorUsage = {};
};

/**
 * Offscreen color and depth images to render into with dynamic rendering, e.g. for shadow maps, post processing or
 * G-buffers.
 *
 *   RenderTarget target;
 *   target.create(base, 512, 512);
 *   pipeline.create(target.renderingFormats(), pipelineInfo);   // MSAA: set
 *                                                               // pipelineInfo.multisampleState.rasterizationSamples
 *   target.begin(cmd);
 *   ... draw ...
 *   target.end(cmd);
 *   // target.colorTexture() is now in eShaderReadOnlyOptimal and can be sampled or downloaded.
 *
 * begin()/end() must be recorded outside of another rendering pass (e.g. in
 * VulkanForwardRenderingInterface::transfer). The images are shared by all frames in flight, which is fine as long as
 * every frame renders the target before using it.
 */
class SAIGA_VULKAN_API RenderTarget
{
   public:
    RenderTarget() = default;
    ~RenderTarget() { destroy(); }
    RenderTarget(const RenderTarget&)            = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;

    void create(VulkanBase& base, uint32_t width, uint32_t height,
                const RenderTargetParameters& parameters = RenderTargetParameters());

    // Releases the images (deferred, see Texture::destroy).
    void destroy();

    uint32_t width() const { return w; }
    uint32_t height() const { return h; }
    vk::SampleCountFlagBits samples() const { return params.samples; }

    // The attachment formats, for the pipelines that render into this target.
    RenderingFormats renderingFormats() const;

    size_t colorCount() const { return color.size(); }
    // The (resolved) color image. After end() it is in eShaderReadOnlyOptimal.
    Texture2D& colorTexture(size_t index = 0);

    bool hasDepth() const { return params.depthFormat != vk::Format::eUndefined; }
    // Without multisampling the depth buffer can be sampled (after end(): eShaderReadOnlyOptimal).
    Texture2D& depthTexture();

    /**
     * Transitions the images and begins the rendering pass with the full viewport and scissor.
     * clear: clears all color attachments with clearColor and the depth buffer with clearDepth, otherwise the previous
     * content is kept.
     */
    void begin(vk::CommandBuffer cmd, bool clear = true, const vec4& clearColor = vec4(0, 0, 0, 1),
               float clearDepth = 1.0f);

    // Ends the rendering pass (resolving multisampled images) and makes the color and depth textures readable.
    void end(vk::CommandBuffer cmd);

   private:
    VulkanBase* base = nullptr;
    RenderTargetParameters params;
    uint32_t w = 0, h = 0;

    std::vector<Texture2D> color;
    // Only with multisampling
    std::vector<Texture2D> multisampledColor;
    Texture2D depth;

    bool multisampled() const { return params.samples != vk::SampleCountFlagBits::e1; }
};

}  // namespace Vulkan
}  // namespace Saiga
