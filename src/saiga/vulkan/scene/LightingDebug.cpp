/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/LightingDebug.h"

#include "saiga/core/util/assert.h"
#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
struct PreviewPush
{
    int32_t size[2];
    uint32_t pass;
    uint32_t kind;
    float scale;
    uint32_t useLog;
    uint32_t autoRange;
    uint32_t colorMap;
    float rangeMin;
    float rangeMax;
    uint32_t groups;
    uint32_t pad;
    uint32_t step;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
};
static_assert(sizeof(PreviewPush) == 64, "Push of debug_preview.comp");

// The Statistics buffer of debug_preview.comp: stats (min, max, mean, count), range, histogram[16]
constexpr size_t kStatsBytes = 32 + 16 * 4;

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

double nowSeconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool sameMapping(const PreviewMapping& a, const PreviewMapping& b)
{
    return a.autoRange == b.autoRange && a.min == b.min && a.max == b.max && a.log == b.log && a.colorMap == b.colorMap;
}

void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::MemoryBarrier2 b(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
}
}  // namespace

float LightingDebug::halfToFloat(uint16_t h)
{
    const uint32_t sign = uint32_t(h >> 15) & 1u, exponent = uint32_t(h >> 10) & 0x1Fu, mantissa = h & 0x3FFu;
    float v;
    if (exponent == 0)
        v = std::ldexp(float(mantissa), -24);
    else if (exponent == 31)
        v = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else
        v = std::ldexp(float(mantissa | 0x400u), int(exponent) - 25);
    return sign ? -v : v;
}

void LightingDebug::init(VulkanBase& b)
{
    destroy();
    base = &b;
    DescriptorSetLayout layout;
    const auto c = vk::ShaderStageFlagBits::eCompute;
    layout.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1, c));
    layout.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageImage, 1, c));
    layout.addBinding(2, vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, c));
    layout.addBinding(3, vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, c));
    pipeline.init(*base, 1);
    pipeline.addDescriptorSetLayout(layout);
    pipeline.addPushConstantRange({c, 0, sizeof(PreviewPush)});
    pipeline.shaderPipeline.loadCompute(base->device, "vulkan/scene/lighting/debug_preview.comp");
    pipeline.create();
    SamplerParameters sp;
    sp.setFilter(vk::Filter::eNearest);
    sp.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    nearestSampler.create(*base, sp);
    for (auto& s : sets) s = pipeline.createRawDescriptorSet();
}

void LightingDebug::destroy()
{
    if (!base) return;
    pipeline.destroy();
    nearestSampler.destroy();
    partials.destroy();
    auto* pool = &base->descriptorPool;
    for (auto& s : sets)
    {
        auto set = s;
        if (set) base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
        s = nullptr;
    }
    for (Image& i : images)
    {
        if (i.texture) i.texture->destroy();
        i.texture.reset();
        for (Buffer& b : i.statsBuffers) b.destroy();
        for (Buffer& b : i.gpuStats) b.destroy();
    }
    for (Buffer& b : pixelBuffers) b.destroy();
    base = nullptr;
}

void LightingDebug::request(LightingImage image, const PreviewMapping& mapping, CaptureCallback callback,
                            bool withImage)
{
    Image& i         = images[size_t(image)];
    i.requested      = true;
    i.fullResolution = true;
    i.mapping        = mapping;
    if (callback) i.callbacks.emplace_back(std::move(callback), withImage);
}

void LightingDebug::show(LightingImage image, const PreviewMapping& mapping)
{
    Image& i = images[size_t(image)];
    // A request() of this frame takes precedence (its conversion is shown); a waiting show() takes the new mapping
    if (i.requested)
    {
        if (!i.fullResolution) i.mapping = mapping;
        return;
    }
    // The preview is recent enough (the same mapping, converted less than the interval ago)
    if (i.texture && sameMapping(mapping, i.shownMapping) && nowSeconds() - i.lastShown < previewIntervalSeconds)
        return;
    i.requested      = true;
    i.fullResolution = false;
    i.mapping        = mapping;
}

bool LightingDebug::requested() const
{
    for (const Image& i : images)
        if (i.requested) return true;
    return false;
}

void LightingDebug::requestPixels(const std::vector<ivec2>& pixels, PixelCallback callback)
{
    pixelRequests.emplace_back(pixels, std::move(callback));
}

Texture2D* LightingDebug::preview(LightingImage image)
{
    return images[size_t(image)].texture.get();
}

uint64_t LightingDebug::memoryBytes() const
{
    uint64_t bytes = partials ? uint64_t(partials.size()) : 0;
    for (const Image& i : images)
    {
        if (i.texture) bytes += uint64_t(i.texture->width) * i.texture->height * 4;
        for (const Buffer& b : i.statsBuffers)
            if (b) bytes += b.size();
        for (const Buffer& b : i.gpuStats)
            if (b) bytes += b.size();
    }
    return bytes;
}

PreviewStatistics LightingDebug::readStatistics(Buffer& b)
{
    PreviewStatistics s;
    float head[8];
    std::memcpy(head, b.getMappedPointer(), sizeof(head));
    s.min      = head[0];
    s.max      = head[1];
    s.mean     = head[2];
    s.count    = uint32_t(head[3]);
    s.rangeMin = head[4];
    s.rangeMax = head[5];
    std::memcpy(s.histogram.data(), static_cast<const char*>(b.getMappedPointer()) + 32, 16 * 4);
    return s;
}

void LightingDebug::harvest(uint32_t frameIndex)
{
    for (Image& i : images)
    {
        if (i.pendingFrame[frameIndex] == 0) continue;
        if (i.pendingFrame[frameIndex] > i.statsFrame)
        {
            i.stats      = readStatistics(i.statsBuffers[frameIndex]);
            i.statsFrame = i.pendingFrame[frameIndex];
        }
        i.pendingFrame[frameIndex] = 0;
    }
}

uint32_t LightingDebug::recordPreviews(vk::CommandBuffer cmd, uint32_t frameIndex, const Sources& sources)
{
    dispatches = 0;
    // The requests first, then the shown images by the age of their preview (the oldest first)
    std::vector<size_t> order;
    for (size_t k = 0; k < images.size(); ++k)
        if (images[k].requested) order.push_back(k);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b)
                     {
                         const Image &x = images[a], &y = images[b];
                         if (x.fullResolution != y.fullResolution) return x.fullResolution;
                         return x.lastShown < y.lastShown;
                     });
    const double now     = nowSeconds();
    uint64_t shownTexels = 0;
    for (size_t k : order)
    {
        Image& i                  = images[k];
        const LightingImage image = LightingImage(k);
        Texture2D* source         = image == LightingImage::Luminance ? sources.hdr : sources.brdfLut;
        // A shown preview at most kPreviewMaxSize per side: every step-th source texel
        uint32_t step = 1;
        if (source && !i.fullResolution)
            step = std::max(1u, (std::max(source->width, source->height) + previewMaxSize - 1) / previewMaxSize);
        const uint32_t w = source ? (source->width + step - 1) / step : 0;
        const uint32_t h = source ? (source->height + step - 1) / step : 0;
        // The texel budget of the shown images (the first conversion of the frame is always made)
        if (source && !i.fullResolution)
        {
            if (dispatches > 0 && shownTexels + uint64_t(w) * h > previewTexelBudget) continue;
            shownTexels += uint64_t(w) * h;
            i.lastShown    = now;
            i.shownMapping = i.mapping;
        }
        i.requested = false;
        i.recorded[frameIndex].insert(i.recorded[frameIndex].end(), std::make_move_iterator(i.callbacks.begin()),
                                      std::make_move_iterator(i.callbacks.end()));
        i.callbacks.clear();
        if (!source) continue;
        if (!i.texture || i.texture->width != w || i.texture->height != h)
        {
            if (i.texture) i.texture->destroy();
            i.texture = std::make_unique<Texture2D>();
            i.texture->create(*base, w, h, vk::Format::eR8G8B8A8Unorm,
                              vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);
            i.version++;
        }
        for (Buffer& b : i.statsBuffers)
            if (!b)
            {
                b.createBuffer(*base, kStatsBytes, vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
                std::memset(b.getMappedPointer(), 0, kStatsBytes);
            }
        // The shader's statistics in device memory: the histogram's atomics on a host-visible buffer (system memory on
        // a discrete GPU) cost about 150 ms for a 1080p frame on an RTX 5080 Laptop GPU
        for (Buffer& b : i.gpuStats)
            if (!b)
                b.createBuffer(*base, kStatsBytes,
                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);
        const uint32_t gx = (w + 15) / 16, gy = (h + 15) / 16, groups = gx * gy;
        const size_t partialBytes = size_t(groups) * 16;
        if (!partials || partials.size() < partialBytes)
            partials.createBuffer(*base, std::max<size_t>(partialBytes, 4096), vk::BufferUsageFlagBits::eStorageBuffer,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal);

        vk::DescriptorSet set = sets[k * 2 + frameIndex];
        vk::DescriptorImageInfo src(vk::Sampler(nearestSampler), source->view(),
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo dst(nullptr, i.texture->view(), vk::ImageLayout::eGeneral);
        vk::DescriptorBufferInfo part(handle(partials), 0, VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo stats(handle(i.gpuStats[frameIndex]), 0, kStatsBytes);
        std::array<vk::WriteDescriptorSet, 4> writes = {
            vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &src),
            vk::WriteDescriptorSet(set, 1, 0, 1, vk::DescriptorType::eStorageImage, &dst),
            vk::WriteDescriptorSet(set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &part),
            vk::WriteDescriptorSet(set, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &stats)};
        base->device.updateDescriptorSets(writes, nullptr);

        // The preview: the previous frame's ImGui read or capture copy before the writes (its content is replaced)
        vk::ImageMemoryBarrier2 toGeneral(
            Stage::eFragmentShader | Stage::eAllTransfer | Stage::eComputeShader,
            Access::eShaderSampledRead | Access::eTransferRead | Access::eShaderStorageWrite, Stage::eComputeShader,
            Access::eShaderStorageWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, i.texture->image(),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toGeneral));
        if (!pipeline.bind(cmd)) continue;
        pipeline.bindRawDescriptorSet(cmd, set);
        PreviewPush push;
        push.size[0]   = int32_t(w);
        push.size[1]   = int32_t(h);
        push.kind      = image == LightingImage::Luminance ? 0u : 1u;
        push.scale     = image == LightingImage::Luminance ? sources.luminanceScale : 1.0f;
        push.useLog    = i.mapping.log ? 1u : 0u;
        push.autoRange = i.mapping.autoRange ? 1u : 0u;
        push.colorMap  = i.mapping.colorMap == ColorMap::Turbo ? 1u : 0u;
        push.rangeMin  = i.mapping.min;
        push.rangeMax  = i.mapping.max;
        push.groups    = groups;
        push.pad       = 0;
        push.step      = step;
        push.pad1 = push.pad2 = push.pad3 = 0;
        auto pass                         = [&](uint32_t p, uint32_t x, uint32_t y)
        {
            push.pass = p;
            pipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
            cmd.dispatch(x, y, 1);
            memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite | Access::eShaderStorageRead,
                          Stage::eComputeShader, Access::eShaderStorageRead | Access::eShaderStorageWrite);
        };
        pass(0, gx, gy);
        pass(1, 1, 1);
        pass(2, gx, gy);
        // The preview -> ImGui's fragment read and a capture's copy; the statistics -> the host
        vk::ImageMemoryBarrier2 toRead(
            Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eFragmentShader | Stage::eAllTransfer,
            Access::eShaderSampledRead | Access::eTransferRead, vk::ImageLayout::eGeneral,
            vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            i.texture->image(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        vk::MemoryBarrier2 toCopy(Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eAllTransfer,
                                  Access::eTransferRead);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, toCopy, nullptr, toRead));
        cmd.copyBuffer(handle(i.gpuStats[frameIndex]), handle(i.statsBuffers[frameIndex]),
                       vk::BufferCopy(0, 0, kStatsBytes));
        memoryBarrier(cmd, Stage::eAllTransfer, Access::eTransferWrite, Stage::eHost, Access::eHostRead);
        i.texture->setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        i.pendingFrame[frameIndex] = currentFrame;
        dispatches++;
    }
    return dispatches;
}

void LightingDebug::recordPixels(vk::CommandBuffer cmd, uint32_t frameIndex, vk::Image hdr, vk::Image depth, ivec2 size)
{
    recordedPixels[frameIndex].insert(recordedPixels[frameIndex].end(), std::make_move_iterator(pixelRequests.begin()),
                                      std::make_move_iterator(pixelRequests.end()));
    pixelRequests.clear();
    std::vector<ivec2> all;
    for (auto& r : recordedPixels[frameIndex]) all.insert(all.end(), r.first.begin(), r.first.end());
    pixelSize[frameIndex] = size;
    if (all.empty()) return;
    const size_t n     = all.size();
    const size_t bytes = n * 12;
    Buffer& b          = pixelBuffers[frameIndex];
    if (!b || b.size() < bytes)
        b.createBuffer(*base, std::max<size_t>(bytes, 256), vk::BufferUsageFlagBits::eTransferDst,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    std::memset(b.getMappedPointer(), 0, bytes);
    std::vector<vk::BufferImageCopy> colors, depths;
    for (size_t k = 0; k < n; ++k)
    {
        const ivec2 p = all[k];
        if (p(0) < 0 || p(1) < 0 || p(0) >= size(0) || p(1) >= size(1)) continue;
        colors.emplace_back(vk::DeviceSize(k * 8), 0, 0,
                            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                            vk::Offset3D(p(0), p(1), 0), vk::Extent3D(1, 1, 1));
        depths.emplace_back(vk::DeviceSize(n * 8 + k * 4), 0, 0,
                            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eDepth, 0, 0, 1),
                            vk::Offset3D(p(0), p(1), 0), vk::Extent3D(1, 1, 1));
    }
    if (colors.empty()) return;
    cmd.copyImageToBuffer(hdr, vk::ImageLayout::eTransferSrcOptimal, handle(b), colors);
    cmd.copyImageToBuffer(depth, vk::ImageLayout::eTransferSrcOptimal, handle(b), depths);
    memoryBarrier(cmd, Stage::eAllTransfer, Access::eTransferWrite, Stage::eHost, Access::eHostRead);
}

void LightingDebug::finishFrame(uint32_t frameIndex, const std::function<void()>& waitFrame)
{
    bool captures = !recordedPixels[frameIndex].empty();
    for (const Image& i : images) captures = captures || !i.recorded[frameIndex].empty();
    if (!captures) return;
    waitFrame();
    for (Image& i : images)
    {
        if (i.recorded[frameIndex].empty()) continue;
        auto callbacks = std::move(i.recorded[frameIndex]);
        i.recorded[frameIndex].clear();
        Capture c;
        if (i.texture && i.pendingFrame[frameIndex] != 0)
        {
            c.statistics = readStatistics(i.statsBuffers[frameIndex]);
            c.size       = ivec2(int(i.texture->width), int(i.texture->height));
            i.stats      = c.statistics;
            i.statsFrame = i.pendingFrame[frameIndex];
            bool image   = false;
            for (auto& cb : callbacks) image = image || cb.second;
            if (image)
            {
                std::vector<uint8_t> bytes = i.texture->download();
                c.image.create(c.size(1), c.size(0));
                if (bytes.size() == size_t(c.size(0)) * size_t(c.size(1)) * 4)
                    std::memcpy(c.image.data(), bytes.data(), bytes.size());
            }
        }
        for (auto& cb : callbacks) cb.first(c);
    }
    auto requests = std::move(recordedPixels[frameIndex]);
    recordedPixels[frameIndex].clear();
    if (requests.empty()) return;
    size_t total = 0;
    for (auto& r : requests) total += r.first.size();
    const ivec2 size = pixelSize[frameIndex];
    size_t offset    = 0;
    for (auto& r : requests)
    {
        std::vector<PixelValue> values;
        for (const ivec2& p : r.first)
        {
            PixelValue v;
            v.pixel = p;
            v.valid = p(0) >= 0 && p(1) >= 0 && p(0) < size(0) && p(1) < size(1) && pixelBuffers[frameIndex];
            if (v.valid)
            {
                const char* data = static_cast<const char*>(pixelBuffers[frameIndex].getMappedPointer());
                uint16_t h[4];
                std::memcpy(h, data + offset * 8, 8);
                v.color = vec4(halfToFloat(h[0]), halfToFloat(h[1]), halfToFloat(h[2]), halfToFloat(h[3]));
                std::memcpy(&v.depth, data + total * 8 + offset * 4, 4);
            }
            values.push_back(v);
            offset++;
        }
        r.second(values);
    }
}

}  // namespace Vulkan
}  // namespace Saiga
