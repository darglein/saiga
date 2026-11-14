/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
struct SAIGA_VULKAN_API SamplerParameters
{
    vk::Filter magFilter             = vk::Filter::eLinear;
    vk::Filter minFilter             = vk::Filter::eLinear;
    vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eLinear;
    vk::SamplerAddressMode addressU  = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressV  = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressW  = vk::SamplerAddressMode::eRepeat;
    vk::BorderColor borderColor      = vk::BorderColor::eFloatOpaqueBlack;
    float mipLodBias                 = 0.0f;
    float minLod                     = 0.0f;
    // All mip levels of the sampled image
    float maxLod = VK_LOD_CLAMP_NONE;

    // > 1 enables anisotropic filtering. Clamped to the device limit and ignored if the samplerAnisotropy feature is
    // not enabled (VulkanParameters::physicalDeviceFeatures).
    float maxAnisotropy = 1.0f;

    // Depth comparison (e.g. sampler2DShadow), disabled if not set.
    std::optional<vk::CompareOp> compareOp;

    void setFilter(vk::Filter filter)
    {
        magFilter  = filter;
        minFilter  = filter;
        mipmapMode = filter == vk::Filter::eNearest ? vk::SamplerMipmapMode::eNearest : vk::SamplerMipmapMode::eLinear;
    }

    void setAddressMode(vk::SamplerAddressMode mode)
    {
        addressU = mode;
        addressV = mode;
        addressW = mode;
    }
};

/**
 * A sampler that is independent of a texture (move-only). Every Texture also has a default sampler; use this one to
 * sample a texture differently (Texture::getDescriptorInfo(sampler), DescriptorSet::assign(index, texture, sampler))
 * or to share one sampler between many textures.
 */
class SAIGA_VULKAN_API Sampler
{
   public:
    Sampler() = default;
    ~Sampler() { destroy(); }
    Sampler(const Sampler&)            = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&& other) noexcept;
    Sampler& operator=(Sampler&& other) noexcept;

    void create(VulkanBase& base, const SamplerParameters& parameters = SamplerParameters());
    void create(VulkanBase& base, const vk::SamplerCreateInfo& createInfo);

    // Destroys the sampler once all frames submitted so far (including the one being recorded) have completed.
    void destroy();

    operator vk::Sampler() const { return sampler; }
    explicit operator bool() const { return bool(sampler); }

   private:
    VulkanBase* base = nullptr;
    vk::Sampler sampler;
};

}  // namespace Vulkan
}  // namespace Saiga
