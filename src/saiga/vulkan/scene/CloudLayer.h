/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <optional>

namespace Saiga
{
namespace Vulkan
{
class SceneLightingSet;

/**
 * The 2D cloud layer of RenderScene (S36a1c, A4 of docs/stages/S36_lighting.plan.md, D3; docs/engine_design.md §10.11):
 * a tiling coverage texture (kSize^2 R8 with mips, period kTile metres, generated on the GPU from CloudSettings::seed:
 * tiling fBm, shader/vulkan/scene/lighting/cloud_noise.comp) and a camera-centred polar grid mesh at the altitude
 * (cloud_layer.vert/.frag) drawn after the sky in the scene pass ("scene.clouds"): depth tested against the scene, no
 * depth write, premultiplied alpha, with the earth's curvature and an alpha fade over the outer 30% of the radius. The
 * coverage is binding 3 of the lighting set: the cloud shadow of every surface (cloudShadow in shadows.glsl, S42's
 * hook) and the environment capture read it. The noise's histogram (read back once per seed) gives the mean coverage at
 * a cloudiness (the cloud shadow beyond its fade distance, the statistics). Volumetric clouds (S42) replace the mesh.
 */
class SAIGA_VULKAN_API CloudLayer
{
   public:
    static constexpr uint32_t kSize = 1024;
    static constexpr double kTile   = 16384.0;
    // The mesh: rings (quadratic spacing) x segments
    static constexpr int kRings = 64, kSegments = 128;
    // The cloud shadow's opacity of full coverage (shader: kCloudShadowOpacity)
    static constexpr float kShadowOpacity = 0.85f;

    void init(VulkanBase& base, const RenderingFormats& formats, uint32_t seed);
    void destroy();
    ~CloudLayer() { destroy(); }

    // In the environment section: a new coverage texture when the seed (or the test blob) changed
    void record(vk::CommandBuffer cmd, uint32_t seed);
    // Inside the scene pass: the layer (set 1: the frame's lighting set)
    void draw(vk::CommandBuffer cmd, vk::DescriptorSet lightingSet, const mat4& viewProj, float preExposure);
    vk::DescriptorImageInfo coverageInfo() { return coverage.getDescriptorInfo(linearRepeat); }
    Texture2D& coverageTexture() { return coverage; }
    // The previews (recorded on demand in the frame slot; RGBA16F, the value in rgb): mode 1 the coverage at the
    // cloudiness over the tile (512^2), mode 2 the cloud shadow on the plane of the camera height within +- the shadow
    // fade around the camera (256^2). block: the frame's lighting block (the clouds, the origin, the light)
    struct DebugInput
    {
        vec4 clouds  = vec4(2000, 40000, 0.3f, 8000);
        vec2 origin  = vec2(0, 0);
        float mean   = 0;
        float height = 0;
        vec3 light   = vec3(0, 1, 0);
    };
    Texture2D* recordDebug(vk::CommandBuffer cmd, uint32_t frameIndex, int mode, const DebugInput& input);

    // The layer's cloudiness for the weather's (the shaders' cloudiness): 1 - the noise value below which a fraction 1
    // - cloudiness of the texels lies (from the noise's histogram), so the mean coverage follows the weather's
    // cloudiness; the weather's value before the histogram was read back
    float effectiveCloudiness(float cloudiness) const;
    // The mean coverage of the texture at a layer cloudiness (from the noise's histogram; 0 before it was read back)
    float meanCoverage(float layerCloudiness) const;
    // Reads the histogram once the frame that generated the texture has completed
    void harvest();
    // The coverage at a noise value and the edge alpha at a fraction of the radius (cloud_noise.comp,
    // atmosphere_common.glsl)
    static float coverageFromNoise(float noise, float cloudiness);
    static float edgeAlpha(float r01);
    // Tests: a disc of coverage 1 (centre in texels of the tile, radius in texels) instead of the noise; nullopt: the
    // noise of the seed. Applied by the next record
    void setTestBlob(std::optional<vec3> blob) { testBlob = blob; }
    uint64_t generations() const { return generationCount; }
    uint64_t memoryBytes() const;

   private:
    void generate(vk::CommandBuffer cmd, uint32_t seed);
    VulkanBase* base = nullptr;
    Texture2D coverage;
    Sampler linearRepeat;
    Buffer noise;
    ComputePipeline noisePipeline;
    // The generation's, then per frame slot one per preview mode
    std::array<vk::DescriptorSet, 1 + 2 * VulkanRenderer::framesInFlight> noiseSets{};
    std::array<Texture2D, 2> debugImages;
    // The generation's set binds 1x1 dummies to the bindings it does not use (the shader declares them)
    Texture2D dummySampled, dummyStorage;
    Pipeline layerPipeline;
    std::optional<uint32_t> generatedSeed;
    std::optional<vec3> testBlob, generatedBlob;
    std::array<uint32_t, 256> histogram{};
    uint32_t histogramTotal  = 0;
    bool histogramPending    = false;
    uint64_t pendingValue    = 0;
    uint64_t generationCount = 0;
};

}  // namespace Vulkan
}  // namespace Saiga
