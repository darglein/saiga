/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"
#include "saiga/vulkan/svulkan.h"

#include <array>
#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * The uniform buffer shared by all world passes (set 0, binding 0). The layout matches WorldUBO in
 * shader/vulkan/world/common.glsl (std140: only mat4 and vec4 members).
 */
struct WorldUniforms
{
    // Camera relative: projection * view without the translation (the shaders transform position - cameraPos, so the
    // float precision does not depend on the distance to the origin)
    mat4 viewProjRel;
    mat4 invViewProjRel;
    // xyz: camera position (world), w: time in seconds
    vec4 cameraPos;
    // xyz: the camera position used for the level of detail and culling (differs if culling is frozen)
    vec4 lodCameraPos;
    // xyz: direction towards the sun, w: sun intensity
    vec4 sunDir;
    // x: albedo, y: ambient, z: flat shading (0/1), w: 0 display values (clamped, sRGB encoded in the shader), > 0
    // linear values multiplied by w (the pre-exposure, WorldRenderer::setOutputLinear)
    vec4 shading;
    // x: density, y: height falloff, z: enabled (0/1), w: water level
    vec4 fog;
    // rgb: fog and horizon color
    vec4 fogColor;
    // rgb: zenith color
    vec4 skyColor;
    // Culling frustum planes (xyz: normal, w: d), dot(n, p) + d >= 0 inside. Left, right, bottom, top, near.
    vec4 frustum[6];
    // x: height map size (texels), y: texel spacing (m), z: range of terrain level 0 (m), w: morph start ratio
    vec4 terrain;
    // x: highest terrain level, y: debug mode (0 none, 1 level of detail tint), z: wind strength, w: alpha cutoff
    vec4 terrain2;
    // x: tree level 1 distance, y: tree level 2 distance, z: max tree distance, w: tree count
    vec4 trees;
    // x: grass radius, y: grass cell size, z: grass density, w: MSAA sample count
    vec4 grass;
    // w, h, 1/w, 1/h
    vec4 viewport;
};

// Frustum planes of a (reversed, infinite) projection: left, right, bottom, top, near (normalized). The 6th plane is
// a dummy that is always passed (0,0,0,1).
inline std::array<vec4, 6> frustumPlanes(const mat4& viewProj)
{
    auto row = [&](int r) { return vec4(viewProj(r, 0), viewProj(r, 1), viewProj(r, 2), viewProj(r, 3)); };
    vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    std::array<vec4, 6> p = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2, vec4(0, 0, 0, 1)};
    for (int i = 0; i < 5; ++i)
    {
        float l = make_vec3(p[i]).norm();
        p[i] /= l;
    }
    return p;
}

// True if the axis aligned box is (partly) inside the frustum planes.
inline bool boxInFrustum(const std::array<vec4, 6>& planes, const vec3& bmin, const vec3& bmax)
{
    for (auto& p : planes)
    {
        // The corner furthest in the direction of the plane normal
        vec3 v(p(0) >= 0 ? bmax(0) : bmin(0), p(1) >= 0 ? bmax(1) : bmin(1), p(2) >= 0 ? bmax(2) : bmin(2));
        if (p(0) * v(0) + p(1) * v(1) + p(2) * v(2) + p(3) < 0) return false;
    }
    return true;
}

// Squared distance from the point to the axis aligned box (0 inside).
inline float boxDistanceSquared(const vec3& p, const vec3& bmin, const vec3& bmax)
{
    float d = 0;
    for (int i = 0; i < 3; ++i)
    {
        float v = std::max({bmin(i) - p(i), 0.0f, p(i) - bmax(i)});
        d += v * v;
    }
    return d;
}

/**
 * Creates a graphics pipeline of the world: descriptor set 0 with the given layout, a push constant range of
 * pushConstantSize bytes for the vertex and fragment stage, the shaders (file names relative to the shader
 * directory) and the pipeline state. The depth test is reversed (eGreaterOrEqual).
 */
inline void createWorldPipeline(Pipeline& pipeline, VulkanBase& base, const DescriptorSetLayout& layout,
                                uint32_t pushConstantSize, const std::vector<std::string>& shaders,
                                const RenderingFormats& formats, PipelineInfo info)
{
    pipeline.init(base, 1);
    pipeline.addDescriptorSetLayout(layout);
    if (pushConstantSize > 0)
    {
        pipeline.addPushConstantRange(
            {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstantSize});
    }
    pipeline.shaderPipeline.load(base.device, shaders);
    pipeline.create(formats, info);
}

// The default pipeline state of the world passes: reversed depth test, back face culling.
inline PipelineInfo worldPipelineInfo()
{
    PipelineInfo info;
    info.depthStencilState.depthCompareOp = vk::CompareOp::eGreaterOrEqual;
    return info;
}

// Line rasterization (PolygonMode::eLine) needs the fillModeNonSolid feature, which Base disables if the device
// lacks it. The wireframe pipelines are only created (and the wireframe setting only used) if it is enabled.
inline bool wireframeSupported(const VulkanBase& base)
{
    return base.enabledFeatures.fillModeNonSolid == VK_TRUE;
}

// The wireframe overlay variant of a pipeline state: lines, drawn on top of the filled surfaces with a depth bias.
inline PipelineInfo wireframeVariant(PipelineInfo info)
{
    info.rasterizationState.polygonMode     = vk::PolygonMode::eLine;
    info.rasterizationState.depthBiasEnable = VK_TRUE;
    // Reversed depth: a positive bias moves the lines towards the camera.
    info.rasterizationState.depthBiasConstantFactor = 2.0f;
    info.rasterizationState.depthBiasSlopeFactor    = 1.0f;
    info.depthStencilState.depthWriteEnable         = VK_FALSE;
    info.multisampleState.alphaToCoverageEnable     = VK_FALSE;
    return info;
}

}  // namespace Vulkan
}  // namespace Saiga
