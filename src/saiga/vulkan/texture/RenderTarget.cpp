/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RenderTarget.h"

#include <cstring>

#include <vulkan/vulkan_format_traits.hpp>

namespace Saiga
{
namespace Vulkan
{
namespace
{
enum class NumericType
{
    Float,
    SignedInt,
    UnsignedInt
};

// How the clear color of an attachment is interpreted.
NumericType numericType(vk::Format format)
{
    const char* type = vk::componentNumericFormat(format, 0);
    if (std::strcmp(type, "UINT") == 0) return NumericType::UnsignedInt;
    if (std::strcmp(type, "SINT") == 0) return NumericType::SignedInt;
    return NumericType::Float;
}

// The sample counts the device supports for an attachment of this format. They depend on the format, e.g. integer
// color formats are only covered by framebufferIntegerColorSampleCounts, not by framebufferColorSampleCounts.
vk::SampleCountFlags attachmentSampleCounts(vk::PhysicalDevice physicalDevice, vk::Format format,
                                            vk::ImageUsageFlags usage)
{
    VkImageFormatProperties properties{};
    auto result = vkGetPhysicalDeviceImageFormatProperties(physicalDevice, static_cast<VkFormat>(format),
                                                           VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                           static_cast<VkImageUsageFlags>(usage), 0, &properties);
    if (result != VK_SUCCESS) return {};
    return vk::SampleCountFlags(properties.sampleCounts);
}
}  // namespace

void RenderTarget::destroy()
{
    color.clear();
    multisampledColor.clear();
    depth.destroy();
}

void RenderTarget::create(VulkanBase& _base, uint32_t width, uint32_t height, const RenderTargetParameters& parameters)
{
    destroy();
    base   = &_base;
    params = parameters;
    w      = width;
    h      = height;

    SAIGA_ASSERT(w > 0 && h > 0, "RenderTarget: empty size");
    SAIGA_ASSERT(!params.colorFormats.empty() || hasDepth(), "RenderTarget: no attachments");

    auto limits = base->physicalDevice.getProperties().limits;
    SAIGA_ASSERT(params.colorFormats.size() <= limits.maxColorAttachments, "RenderTarget: too many color attachments");
    if (multisampled())
    {
        for (auto format : params.colorFormats)
        {
            SAIGA_ASSERT(
                attachmentSampleCounts(base->physicalDevice, format, vk::ImageUsageFlagBits::eColorAttachment) &
                    params.samples,
                "RenderTarget: the sample count is not supported for the color format " + vk::to_string(format));
        }
        SAIGA_ASSERT(!hasDepth() || (attachmentSampleCounts(base->physicalDevice, params.depthFormat,
                                                            vk::ImageUsageFlagBits::eDepthStencilAttachment) &
                                     params.samples),
                     "RenderTarget: the sample count is not supported for the depth format " +
                         vk::to_string(params.depthFormat));
    }

    color.resize(params.colorFormats.size());
    for (size_t i = 0; i < color.size(); ++i)
    {
        color[i].create(
            *base, w, h, params.colorFormats[i],
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | params.colorUsage);
    }

    if (multisampled())
    {
        multisampledColor.resize(color.size());
        for (size_t i = 0; i < color.size(); ++i)
        {
            multisampledColor[i].create(*base, w, h, params.colorFormats[i], vk::ImageUsageFlagBits::eColorAttachment,
                                        1, params.samples);
        }
    }

    if (hasDepth())
    {
        auto aspect = Memory::ImageData::aspectFlags(params.depthFormat);
        SAIGA_ASSERT(aspect == vk::ImageAspectFlagBits::eDepth,
                     "RenderTarget: a depth format without stencil is required");
        auto usage = vk::ImageUsageFlags(vk::ImageUsageFlagBits::eDepthStencilAttachment);
        if (!multisampled()) usage |= vk::ImageUsageFlagBits::eSampled;
        depth.create(*base, w, h, params.depthFormat, usage, 1, params.samples);
    }
}

RenderingFormats RenderTarget::renderingFormats() const
{
    RenderingFormats formats;
    formats.color   = params.colorFormats;
    formats.depth   = params.depthFormat;
    formats.samples = params.samples;
    return formats;
}

Texture2D& RenderTarget::colorTexture(size_t index)
{
    SAIGA_ASSERT(index < color.size());
    return color[index];
}

Texture2D& RenderTarget::depthTexture()
{
    SAIGA_ASSERT(hasDepth());
    return depth;
}

void RenderTarget::begin(vk::CommandBuffer cmd, bool clear, const vec4& clearColor, float clearDepth)
{
    SAIGA_ASSERT(base, "RenderTarget: not created");

    // The transitions wait for all previous uses of the images (e.g. the sampling of the previous frame).
    for (auto& t : color) t.transitionImageLayout(cmd, vk::ImageLayout::eColorAttachmentOptimal);
    for (auto& t : multisampledColor) t.transitionImageLayout(cmd, vk::ImageLayout::eColorAttachmentOptimal);
    if (hasDepth()) depth.transitionImageLayout(cmd, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    auto loadOp = clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;

    std::vector<vk::RenderingAttachmentInfo> colorAttachments(color.size());
    for (size_t i = 0; i < color.size(); ++i)
    {
        auto& a       = colorAttachments[i];
        a.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        a.loadOp      = loadOp;
        a.storeOp     = vk::AttachmentStoreOp::eStore;

        auto type = numericType(params.colorFormats[i]);
        if (multisampled())
        {
            // Store as well: the next pass may continue with clear = false.
            a.imageView = multisampledColor[i].view();
            // Integer formats can not be averaged.
            a.resolveMode =
                type == NumericType::Float ? vk::ResolveModeFlagBits::eAverage : vk::ResolveModeFlagBits::eSampleZero;
            a.resolveImageView   = color[i].view();
            a.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }
        else
        {
            a.imageView = color[i].view();
        }

        switch (type)
        {
            case NumericType::Float:
                a.clearValue.color = vk::ClearColorValue(
                    std::array<float, 4>{clearColor[0], clearColor[1], clearColor[2], clearColor[3]});
                break;
            case NumericType::SignedInt:
                a.clearValue.color = vk::ClearColorValue(std::array<int32_t, 4>{
                    int32_t(clearColor[0]), int32_t(clearColor[1]), int32_t(clearColor[2]), int32_t(clearColor[3])});
                break;
            case NumericType::UnsignedInt:
                a.clearValue.color =
                    vk::ClearColorValue(std::array<uint32_t, 4>{uint32_t(clearColor[0]), uint32_t(clearColor[1]),
                                                                uint32_t(clearColor[2]), uint32_t(clearColor[3])});
                break;
        }
    }

    vk::RenderingAttachmentInfo depthAttachment;
    if (hasDepth())
    {
        depthAttachment.imageView                     = depth.view();
        depthAttachment.imageLayout                   = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depthAttachment.loadOp                        = loadOp;
        depthAttachment.storeOp                       = vk::AttachmentStoreOp::eStore;
        depthAttachment.clearValue.depthStencil.depth = clearDepth;
    }

    vk::RenderingInfo renderingInfo;
    renderingInfo.renderArea           = vk::Rect2D({0, 0}, {w, h});
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = (uint32_t)colorAttachments.size();
    renderingInfo.pColorAttachments    = colorAttachments.data();
    renderingInfo.pDepthAttachment     = hasDepth() ? &depthAttachment : nullptr;
    cmd.beginRendering(renderingInfo);

    cmd.setViewport(0, vk::Viewport(0, 0, (float)w, (float)h, 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {w, h}));
}

void RenderTarget::end(vk::CommandBuffer cmd)
{
    cmd.endRendering();

    for (auto& t : color) t.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    if (hasDepth() && !multisampled()) depth.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

}  // namespace Vulkan
}  // namespace Saiga
