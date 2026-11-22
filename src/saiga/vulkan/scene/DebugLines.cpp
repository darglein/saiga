/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/DebugLines.h"

#include <algorithm>
#include <cstring>

namespace Saiga
{
namespace Vulkan
{
namespace
{
struct LinePush
{
    mat4 viewProj;
    float scale;
};

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}
}  // namespace

void DebugLines::init(VulkanBase& b, const RenderingFormats& formats)
{
    destroy();
    base        = &b;
    auto create = [&](Pipeline& p, bool test)
    {
        PipelineInfo info;
        info.inputAssemblyState.topology        = vk::PrimitiveTopology::eLineList;
        info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthTestEnable  = test ? VK_TRUE : VK_FALSE;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        info.depthStencilState.depthCompareOp   = vk::CompareOp::eGreaterOrEqual;  // reversed-Z
        info.addVertexBinding(0, sizeof(Vertex), vk::VertexInputRate::eVertex,
                              {{0, 0, vk::Format::eR32G32B32Sfloat, uint32_t(offsetof(Vertex, position))},
                               {1, 0, vk::Format::eR32G32B32A32Sfloat, uint32_t(offsetof(Vertex, color))}});
        p.init(*base, 0);
        p.addPushConstantRange(
            {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(LinePush)});
        p.shaderPipeline.load(base->device,
                              {"vulkan/scene/lighting/debug_lines.vert", "vulkan/scene/lighting/debug_lines.frag"});
        p.create(formats, info);
    };
    create(depthTested, true);
    create(onTop, false);
}

void DebugLines::destroy()
{
    if (!base) return;
    depthTested.destroy();
    onTop.destroy();
    for (Buffer& v : vertices) v.destroy();
    lines.clear();
    base = nullptr;
}

void DebugLines::addLine(const Vec3& a, const Vec3& b, const vec3& color, bool top)
{
    lines.push_back(Line{a, b, color, top});
}

uint64_t DebugLines::memoryBytes() const
{
    uint64_t bytes = 0;
    for (const Buffer& v : vertices)
        if (v) bytes += v.size();
    return bytes;
}

size_t DebugLines::record(vk::CommandBuffer cmd, uint32_t frameIndex, const Vec3& camera, const mat4& viewProj,
                          float colorScale)
{
    if (lines.empty()) return 0;
    // Depth-tested lines first, then the ones on top
    std::stable_sort(lines.begin(), lines.end(), [](const Line& x, const Line& y) { return !x.onTop && y.onTop; });
    const size_t bytes = lines.size() * 2 * sizeof(Vertex);
    Buffer& buffer     = vertices[frameIndex];
    if (!buffer || buffer.size() < bytes)
        buffer.createBuffer(*base, std::max<size_t>(bytes * 2, 4096), vk::BufferUsageFlagBits::eVertexBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto* v         = static_cast<Vertex*>(buffer.getMappedPointer());
    uint32_t tested = 0;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        const Line& l = lines[i];
        const vec4 c(l.color(0), l.color(1), l.color(2), 1);
        // Relative to the camera in double (R-MATH-1)
        v[2 * i]     = Vertex{(l.a - camera).cast<float>(), c};
        v[2 * i + 1] = Vertex{(l.b - camera).cast<float>(), c};
        if (!l.onTop) tested++;
    }
    const uint32_t total = uint32_t(lines.size());
    LinePush push{viewProj, colorScale};
    const auto stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    auto draw         = [&](Pipeline& p, uint32_t first, uint32_t count)
    {
        if (count == 0 || !p.bind(cmd)) return;
        p.pushConstant(cmd, stages, sizeof(push), &push);
        cmd.bindVertexBuffers(0, handle(buffer), vk::DeviceSize(0));
        cmd.draw(count * 2, 1, first * 2, 0);
    };
    draw(depthTested, 0, tested);
    draw(onTop, tested, total - tested);
    lines.clear();
    return total;
}

}  // namespace Vulkan
}  // namespace Saiga
