/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"

#include "games/common/Mesh.h"

#include <vector>

namespace Saiga
{
namespace Game
{
/**
 * Draws flat colored low-poly meshes (MeshData) in RenderScene's scene pass (MSAA, reversed depth), lit like the world
 * prototype (sun + hemisphere ambient, the same height fog). Positions are camera relative: draw() takes the double
 * model matrix, render() subtracts the camera position in double before converting to float.
 *
 *   meshes.init(scene.base(), scene.formats());  int plane = meshes.addMesh(MeshBuilder::airplane());
 *   // transfer(): meshes.setOutputLinear(scene.sceneWritesLinear(), scene.preExposure()) (as WorldRenderer)
 *   // transfer(): meshes.transfer(cmd, view, cameraPosition, projection, lighting);
 *   // render():   meshes.draw(plane, model); ... meshes.render(cmd);
 *
 * Transitional (docs/engine_design.md §18.4): replaced by the RenderScene with instance sets (S17-S19).
 */
class MeshRenderer
{
   public:
    struct Lighting
    {
        Vec3 sunDirection   = Vec3(0.5, 0.6, -0.6);
        double sunIntensity = 1.0;
        double ambient      = 0.25;
        bool fog            = false;
        double fogDensity = 0.00015, fogFalloff = 0.0012, fogBaseHeight = 0;
        Vec3 fogColor = Vec3(0.42, 0.42, 0.42);
    };

    ~MeshRenderer() { destroy(); }

    // formats: the pass the meshes are drawn in (reversed depth)
    void init(Vulkan::VulkanBase& base, const Vulkan::RenderingFormats& formats);
    void destroy();

    // As WorldRenderer::setOutputLinear: false (default) display values, true linear values times preExposure (from
    // the next transfer() on)
    void setOutputLinear(bool linear, float preExposure = 1)
    {
        outputLinear      = linear;
        outputPreExposure = preExposure;
    }

    // Uploads a mesh, returns its id (call before the first frame or after waiting for the device)
    int addMesh(const MeshData& mesh);

    // Per frame, outside of the rendering pass: the camera (view of the camera at cameraPosition, the projection) and
    // the lighting
    void transfer(vk::CommandBuffer cmd, const mat4& view, const Vec3& cameraPosition, const mat4& projection,
                  const Lighting& lighting);
    // Queues a draw for render()
    void draw(int mesh, const Mat4& model, const Vec3& tint = Vec3(1, 1, 1));
    // Records the queued draws (inside of the rendering pass) and clears the queue
    void render(vk::CommandBuffer cmd);


   private:
    struct GpuMesh
    {
        Vulkan::Buffer vertices, indices;
        uint32_t indexCount = 0;
    };
    struct Draw
    {
        int mesh;
        Mat4 model;
        Vec3 tint;
    };
    // std140, see shader/vulkan/games/mesh_common.glsl
    struct Uniforms
    {
        mat4 viewProjRel;
        vec4 cameraPos;
        vec4 sunDir;
        vec4 shading;
        vec4 fog;
        vec4 fogColor;
    };

    Vulkan::VulkanBase* base = nullptr;
    bool outputLinear        = false;
    float outputPreExposure  = 1;
    Vulkan::UniformBuffer uniformBuffer;
    Vulkan::Pipeline pipeline;
    Vulkan::StaticDescriptorSet set;
    std::vector<GpuMesh> meshes;
    std::vector<Draw> draws;
    Vec3 cameraPosition = Vec3(0, 0, 0);
    bool initialized    = false;
};

}  // namespace Game
}  // namespace Saiga
