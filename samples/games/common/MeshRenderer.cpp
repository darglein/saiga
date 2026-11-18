/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Transitional renderer of flat colored meshes for the tech demo games (camera relative, forward renderer).

#include "games/common/MeshRenderer.h"

#include "saiga/vulkan/world/WorldCommon.h"

namespace Saiga
{
namespace Game
{
namespace
{
// The push constants of shader/vulkan/games/mesh_common.glsl (128 bytes, the guaranteed minimum)
struct MeshPush
{
    mat4 modelRel;
    vec4 normal0, normal1, normal2;
    vec4 tint;
};
static_assert(sizeof(MeshPush) == 128, "push constants");
}  // namespace

void MeshRenderer::init(Vulkan::VulkanForwardRenderer& _renderer)
{
    destroy();
    renderer = &_renderer;
    base     = &renderer->base();

    uniformBuffer.init(*base, nullptr, sizeof(Uniforms));

    auto vf = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    Vulkan::DescriptorSetLayout layout({{0, {0, vk::DescriptorType::eUniformBuffer, 1, vf}}});
    Vulkan::PipelineInfo info = Vulkan::worldPipelineInfo();
    // Some builders are not consistently wound and the meshes are small: no back face culling (two sided lighting)
    info.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    info.addVertexBinding(0, sizeof(MeshVertex), vk::VertexInputRate::eVertex,
                          {{0, 0, vk::Format::eR32G32B32Sfloat, uint32_t(offsetof(MeshVertex, position))},
                           {1, 0, vk::Format::eR32G32B32Sfloat, uint32_t(offsetof(MeshVertex, normal))},
                           {2, 0, vk::Format::eR32G32B32Sfloat, uint32_t(offsetof(MeshVertex, color))}});
    Vulkan::createWorldPipeline(pipeline, *base, layout, sizeof(MeshPush),
                                {"vulkan/games/mesh.vert", "vulkan/games/mesh.frag"}, renderer->renderingFormats(),
                                info);
    set = pipeline.createDescriptorSet();
    set.assign(0, &uniformBuffer);
    initialized = true;
}

void MeshRenderer::destroy()
{
    if (!initialized) return;
    // Release the descriptor set while the renderer (its VulkanBase) still exists; a later destructor would read a
    // dangling base pointer
    set = Vulkan::StaticDescriptorSet();
    pipeline.destroy();
    for (auto& m : meshes)
    {
        m.vertices.destroy();
        m.indices.destroy();
    }
    meshes.clear();
    uniformBuffer.destroy();
    initialized = false;
}

int MeshRenderer::addMesh(const MeshData& mesh)
{
    SAIGA_ASSERT(initialized && !mesh.indices.empty());
    meshes.emplace_back();
    GpuMesh& g    = meshes.back();
    g.indexCount  = uint32_t(mesh.indices.size());
    size_t vbytes = mesh.vertices.size() * sizeof(MeshVertex);
    size_t ibytes = mesh.indices.size() * sizeof(uint32_t);
    g.vertices.createBuffer(*base, vbytes,
                            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
    g.vertices.stagedUpload(*base, vbytes, mesh.vertices.data());
    g.indices.createBuffer(*base, ibytes, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                           vk::MemoryPropertyFlagBits::eDeviceLocal);
    g.indices.stagedUpload(*base, ibytes, mesh.indices.data());
    return int(meshes.size()) - 1;
}

void MeshRenderer::transfer(vk::CommandBuffer cmd, const mat4& view, const Vec3& _cameraPosition, const mat4& proj,
                            const Lighting& l)
{
    SAIGA_ASSERT(initialized);
    cameraPosition = _cameraPosition;

    // The view without the translation: the camera is at the origin of the relative positions
    mat4 viewRel  = view;
    viewRel(0, 3) = 0;
    viewRel(1, 3) = 0;
    viewRel(2, 3) = 0;

    auto colorFormat = renderer->getColorFormat();
    bool srgbTarget  = colorFormat == vk::Format::eB8G8R8A8Srgb || colorFormat == vk::Format::eR8G8B8A8Srgb;
    Vec3 sun         = l.sunDirection.normalized();

    Uniforms u;
    u.viewProjRel = proj * viewRel;
    u.cameraPos   = vec4(float(cameraPosition(0)), float(cameraPosition(1)), float(cameraPosition(2)), 0);
    u.sunDir      = vec4(float(sun(0)), float(sun(1)), float(sun(2)), float(l.sunIntensity));
    u.shading     = vec4(float(l.ambient), 0, 0, srgbTarget ? 0 : 1);
    u.fog         = vec4(float(l.fogDensity), float(l.fogFalloff), l.fog ? 1 : 0, float(l.fogBaseHeight));
    u.fogColor    = vec4(float(l.fogColor(0)), float(l.fogColor(1)), float(l.fogColor(2)), 1);

    using Stage  = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    // The draws of the previous frame read the uniforms
    vk::MemoryBarrier2 war(Stage::eVertexShader | Stage::eFragmentShader, {}, Stage::eAllTransfer, {});
    cmd.pipelineBarrier2(vk::DependencyInfo({}, war, nullptr, nullptr));
    uniformBuffer.update(cmd, sizeof(Uniforms), &u);
    vk::MemoryBarrier2 raw(Stage::eAllTransfer, Access::eTransferWrite, Stage::eVertexShader | Stage::eFragmentShader,
                           Access::eUniformRead);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, raw, nullptr, nullptr));
}

void MeshRenderer::draw(int mesh, const Mat4& model, const Vec3& tint)
{
    SAIGA_ASSERT(mesh >= 0 && mesh < int(meshes.size()));
    draws.push_back({mesh, model, tint});
}

void MeshRenderer::render(vk::CommandBuffer cmd)
{
    lastTriangles = 0;
    if (draws.empty() || !pipeline.bind(cmd))
    {
        draws.clear();
        return;
    }
    pipeline.bindDescriptorSet(cmd, set);
    auto stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    for (auto& d : draws)
    {
        // Relative to the camera in double, then float
        Mat4 rel = d.model;
        for (int i = 0; i < 3; ++i) rel(i, 3) -= cameraPosition(i);

        Mat3 A;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) A(i, j) = d.model(i, j);
        Mat3 N = A.inverse().transpose();

        MeshPush push;
        push.modelRel = rel.cast<float>();
        push.normal0  = vec4(float(N(0, 0)), float(N(1, 0)), float(N(2, 0)), 0);
        push.normal1  = vec4(float(N(0, 1)), float(N(1, 1)), float(N(2, 1)), 0);
        push.normal2  = vec4(float(N(0, 2)), float(N(1, 2)), float(N(2, 2)), 0);
        push.tint     = vec4(float(d.tint(0)), float(d.tint(1)), float(d.tint(2)), 1);
        pipeline.pushConstant(cmd, stages, sizeof(push), &push);

        auto& m = meshes[d.mesh];
        cmd.bindVertexBuffers(0, vk::Buffer(m.vertices.memoryLocation()->data), vk::DeviceSize(0));
        cmd.bindIndexBuffer(m.indices.memoryLocation()->data, 0, vk::IndexType::eUint32);
        cmd.drawIndexed(m.indexCount, 1, 0, 0, 0);
        lastTriangles += int(m.indexCount / 3);
    }
    draws.clear();
}

}  // namespace Game
}  // namespace Saiga
