/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/SceneLightingSet.h"

#include <cstring>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

enum class Kind
{
    Uniform,
    Storage,
    Sampler
};
Kind bindingKind(uint32_t b)
{
    if (b == 0) return Kind::Uniform;
    if (b == 2 || b == 10 || b == 11 || b == 12 || b == 14) return Kind::Storage;
    return Kind::Sampler;
}
}  // namespace

DescriptorSetLayout SceneLightingSet::layout()
{
    DescriptorSetLayout l;
    const auto stages =
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute;
    for (uint32_t b = 0; b < kBindingCount; ++b)
    {
        const Kind k               = bindingKind(b);
        const vk::DescriptorType t = k == Kind::Uniform   ? vk::DescriptorType::eUniformBuffer
                                     : k == Kind::Storage ? vk::DescriptorType::eStorageBuffer
                                                          : vk::DescriptorType::eCombinedImageSampler;
        l.addBinding(b, vk::DescriptorSetLayoutBinding(b, t, 1, stages));
    }
    return l;
}

DescriptorSetLayout SceneLightingSet::shadowLayout()
{
    DescriptorSetLayout l;
    const auto stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, stages));
    l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, stages));
    return l;
}

void SceneLightingSet::init(VulkanBase& b)
{
    destroy();
    base      = &b;
    ownLayout = layout();
    ownLayout.create(base);
    ownShadowLayout = shadowLayout();
    ownShadowLayout.create(base);
    for (Frame& f : frames)
    {
        f.buffer.createBuffer(*base, sizeof(SceneLighting), vk::BufferUsageFlagBits::eUniformBuffer,
                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        std::memcpy(f.buffer.getMappedPointer(), &f.block, sizeof(SceneLighting));
        f.set       = ownLayout.createRawDescriptorSet();
        f.shadowSet = ownShadowLayout.createRawDescriptorSet();
    }

    // The dummies, cleared to 0 (depth: 1, lit) and in their sampled layouts
    const auto usage = vk::ImageUsageFlagBits::eSampled;
    dummyArray.create(*base, 1, 1, 1, vk::Format::eR8G8B8A8Unorm, usage);
    dummy2D.create(*base, 1, 1, vk::Format::eR8G8B8A8Unorm, usage);
    dummyCube.create(*base, 1, vk::Format::eR8G8B8A8Unorm, usage);
    dummyDepth.create(*base, 1, 1, 1, vk::Format::eD32Sfloat, usage);
    dummyDepth16.create(*base, 1, 1, 2, vk::Format::eD16Unorm, usage);
    // The cascades' comparison sampler (S36a2): linear (a 2 x 2 bilinear compare per tap) where the format supports
    // linear filtering, else nearest (shadows.glsl then gathers and weights itself)
    const auto features = base->physicalDevice.getFormatProperties(vk::Format::eD32Sfloat).optimalTilingFeatures;
    linearCompare       = bool(features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);
    SamplerParameters sp;
    sp.setFilter(linearCompare ? vk::Filter::eLinear : vk::Filter::eNearest);
    sp.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    sp.compareOp = vk::CompareOp::eLessOrEqual;
    compareSampler.create(*base, sp);
    // The atlas's (S36b, D16)
    const auto features16 = base->physicalDevice.getFormatProperties(vk::Format::eD16Unorm).optimalTilingFeatures;
    linearAtlasCompare    = bool(features16 & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);
    sp.setFilter(linearAtlasCompare ? vk::Filter::eLinear : vk::Filter::eNearest);
    atlasCompareSampler.create(*base, sp);
    dummyStorage.createBuffer(*base, 256, vk::BufferUsageFlagBits::eStorageBuffer,
                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    std::memset(dummyStorage.getMappedPointer(), 0, 256);

    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    std::vector<vk::ImageMemoryBarrier2> toDst, toRead;
    auto add = [&](Texture& t, vk::ImageAspectFlags aspect, vk::ImageLayout readLayout)
    {
        vk::ImageSubresourceRange range(aspect, 0, 1, 0, t.layerCount);
        toDst.emplace_back(Stage::eNone, Access::eNone, Stage::eClear, Access::eTransferWrite,
                           vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED,
                           VK_QUEUE_FAMILY_IGNORED, t.image(), range);
        toRead.emplace_back(Stage::eClear, Access::eTransferWrite, Stage::eAllCommands, Access::eShaderSampledRead,
                            vk::ImageLayout::eTransferDstOptimal, readLayout, VK_QUEUE_FAMILY_IGNORED,
                            VK_QUEUE_FAMILY_IGNORED, t.image(), range);
    };
    const auto color = vk::ImageAspectFlagBits::eColor;
    add(dummyArray, color, vk::ImageLayout::eShaderReadOnlyOptimal);
    add(dummy2D, color, vk::ImageLayout::eShaderReadOnlyOptimal);
    add(dummyCube, color, vk::ImageLayout::eShaderReadOnlyOptimal);
    add(dummyDepth, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eShaderReadOnlyOptimal);
    add(dummyDepth16, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eShaderReadOnlyOptimal);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toDst));
    const vk::ClearColorValue zero(std::array<float, 4>{0, 0, 0, 0});
    for (Texture* t : std::initializer_list<Texture*>{&dummyArray, &dummy2D, &dummyCube})
        cmd.clearColorImage(t->image(), vk::ImageLayout::eTransferDstOptimal, zero,
                            vk::ImageSubresourceRange(color, 0, 1, 0, t->layerCount));
    cmd.clearDepthStencilImage(dummyDepth.image(), vk::ImageLayout::eTransferDstOptimal,
                               vk::ClearDepthStencilValue(1.0f, 0),
                               vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
    cmd.clearDepthStencilImage(dummyDepth16.image(), vk::ImageLayout::eTransferDstOptimal,
                               vk::ClearDepthStencilValue(1.0f, 0),
                               vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 2));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    for (Texture* t : std::initializer_list<Texture*>{&dummyArray, &dummy2D, &dummyCube, &dummyDepth, &dummyDepth16})
        t->setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    // Every binding valid from the start (the dummies)
    for (uint32_t i = 0; i < frames.size(); ++i) write(i, frames[i].block, SceneLightingBindings());
}

void SceneLightingSet::destroy()
{
    if (!base) return;
    auto* pool = &base->descriptorPool;
    for (Frame& f : frames)
    {
        f.buffer.destroy();
        for (vk::DescriptorSet* s : {&f.set, &f.shadowSet})
        {
            auto set = *s;
            if (set) base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
            *s = nullptr;
        }
    }
    // The layouts after the sets (deferred as well, in order)
    DescriptorSetLayout l = ownLayout, sl = ownShadowLayout;
    base->deferDelete(
        [l, sl]() mutable
        {
            l.destroy();
            sl.destroy();
        });
    ownLayout       = DescriptorSetLayout();
    ownShadowLayout = DescriptorSetLayout();
    dummyArray.destroy();
    dummyDepth.destroy();
    dummyDepth16.destroy();
    dummy2D.destroy();
    dummyCube.destroy();
    compareSampler.destroy();
    atlasCompareSampler.destroy();
    dummyStorage.destroy();
    base = nullptr;
}

void SceneLightingSet::write(uint32_t frameIndex, const SceneLighting& block, const SceneLightingBindings& in)
{
    Frame& f = frames[frameIndex];
    f.block  = block;
    std::memcpy(f.buffer.getMappedPointer(), &block, sizeof(SceneLighting));

    std::array<vk::DescriptorImageInfo, kBindingCount> images;
    std::array<vk::DescriptorBufferInfo, kBindingCount> buffers;
    std::vector<vk::WriteDescriptorSet> writes;
    const vk::DescriptorImageInfo array = dummyArray.getDescriptorInfo();
    const vk::DescriptorImageInfo tex2D = dummy2D.getDescriptorInfo();
    const vk::DescriptorImageInfo cube  = dummyCube.getDescriptorInfo();
    const vk::DescriptorImageInfo depth = dummyDepth.getDescriptorInfo(compareSampler);
    const vk::DescriptorBufferInfo storage(handle(dummyStorage), 0, VK_WHOLE_SIZE);
    for (uint32_t b = 0; b < kBindingCount; ++b)
    {
        switch (bindingKind(b))
        {
            case Kind::Uniform:
                buffers[b] = vk::DescriptorBufferInfo(handle(f.buffer), 0, sizeof(SceneLighting));
                writes.emplace_back(f.set, b, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffers[b]);
                break;
            case Kind::Storage:
                buffers[b] =
                    b == 2 && in.ambient.sh ? vk::DescriptorBufferInfo(in.ambient.sh, 0, in.ambient.shBytes) : storage;
                // S36b: the lights, the clusters and the atlas's tile table
                if (b == 10 && in.lights) buffers[b] = *in.lights;
                if (b == 11 && in.clusterCounts) buffers[b] = *in.clusterCounts;
                if (b == 12 && in.clusterIndices) buffers[b] = *in.clusterIndices;
                if (b == 14 && in.atlasTiles) buffers[b] = *in.atlasTiles;
                writes.emplace_back(f.set, b, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffers[b]);
                break;
            case Kind::Sampler:
            {
                vk::DescriptorImageInfo info = tex2D;
                if (b == 1) info = in.foliage ? *in.foliage : array;
                if (b == 3 && in.cloudCoverage) info = *in.cloudCoverage;
                if (b == 4) info = in.ambient.environment.imageView ? in.ambient.environment : cube;
                if (b == 5 && in.brdfLut) info = *in.brdfLut;
                if (b == 6 && in.transmittanceLut) info = *in.transmittanceLut;
                if (b == 7 && in.skyViewLut) info = *in.skyViewLut;
                if (b == 8) info = depth;
                if (b == 13) info = in.atlas ? *in.atlas : dummyDepth16.getDescriptorInfo(atlasCompareSampler);
                if (b == 8 && in.cascades) info = *in.cascades;
                if (b == 9) info = in.horizon ? *in.horizon : array;
                images[b] = info;
                writes.emplace_back(f.set, b, 0, 1, vk::DescriptorType::eCombinedImageSampler, &images[b]);
                break;
            }
        }
    }
    // The shadow set: the block and the foliage texture
    writes.emplace_back(f.shadowSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffers[0]);
    writes.emplace_back(f.shadowSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &images[1]);
    base->device.updateDescriptorSets(writes, nullptr);
}

}  // namespace Vulkan
}  // namespace Saiga
