/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Sampler.h"

#include <algorithm>

namespace Saiga
{
namespace Vulkan
{
Sampler::Sampler(Sampler&& other) noexcept : base(other.base), sampler(other.sampler)
{
    other.sampler = nullptr;
}

Sampler& Sampler::operator=(Sampler&& other) noexcept
{
    if (this != &other)
    {
        destroy();
        base          = other.base;
        sampler       = other.sampler;
        other.sampler = nullptr;
    }
    return *this;
}

void Sampler::create(VulkanBase& _base, const SamplerParameters& p)
{
    vk::SamplerCreateInfo info;
    info.magFilter    = p.magFilter;
    info.minFilter    = p.minFilter;
    info.mipmapMode   = p.mipmapMode;
    info.addressModeU = p.addressU;
    info.addressModeV = p.addressV;
    info.addressModeW = p.addressW;
    info.borderColor  = p.borderColor;
    info.mipLodBias   = p.mipLodBias;
    info.minLod       = p.minLod;
    info.maxLod       = p.maxLod;

    info.maxAnisotropy = 1.0f;
    if (p.maxAnisotropy > 1.0f && _base.enabledFeatures.samplerAnisotropy)
    {
        float limit           = _base.physicalDevice.getProperties().limits.maxSamplerAnisotropy;
        info.anisotropyEnable = VK_TRUE;
        info.maxAnisotropy    = std::min(p.maxAnisotropy, limit);
    }

    info.compareEnable = p.compareOp.has_value();
    info.compareOp     = p.compareOp.value_or(vk::CompareOp::eNever);
    create(_base, info);
}

void Sampler::create(VulkanBase& _base, const vk::SamplerCreateInfo& createInfo)
{
    destroy();
    base    = &_base;
    sampler = base->device.createSampler(createInfo);
    SAIGA_ASSERT(sampler);
}

void Sampler::destroy()
{
    if (!sampler) return;
    // The device is gone after VulkanBase::destroy().
    if (base && base->device)
    {
        // Frames in flight may still use the sampler.
        auto device = base->device;
        auto s      = sampler;
        base->deferDelete([device, s]() { device.destroySampler(s); });
    }
    sampler = nullptr;
}

}  // namespace Vulkan
}  // namespace Saiga
