/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <optional>

namespace Saiga
{
namespace Vulkan
{
struct EnvironmentState;

/**
 * The environment map of the sky ambient (S36, A4 of docs/stages/S36_lighting.plan.md): a capture cube (the sky seen
 * from the camera height without the sun disc, its clouds from step a1c.3; box mips by blits) and two prefiltered cubes
 * of environmentResolution with 5 GGX mips plus two SH9 buffers, **front and back**: a re-capture writes the capture,
 * prefilters the back cube one mip per frame and then computes the back SH from the capture's 32 x 32 box mip; the
 * front cube and SH are swapped in only when all of it is complete, so a frame never mixes two captures. Due when the
 * sun or the moon moved by more than environmentUpdateDegrees, the weather changed by more than 1%, the camera height
 * moved by 200 m or the settings changed; forceFullUpdates (and the first capture) does all of it in one frame. The
 * front cube is binding 4 of the lighting set, the front SH binding 2 (through SkyAmbient). Recorded in the
 * "environment" section.
 */
class SAIGA_VULKAN_API EnvironmentMap
{
   public:
    static constexpr int kMips = 5;
    // The SH source: the capture's mip of this face size
    static constexpr int kShSize = 32;

    void init(VulkanBase& base, int resolution);
    void destroy();
    ~EnvironmentMap() { destroy(); }

    // The inputs of a capture: the atmosphere's sky-view LUT and transmittance, the cloud layer's coverage (the same
    // textures for the scene's life), the frame's cloud values and the shadow light
    struct CaptureInputs
    {
        vk::DescriptorImageInfo skyView, transmittance, coverage;
        vec2 cloudOrigin       = vec2(0, 0);
        float cloudiness       = 0;  // the layer's (CloudLayer::effectiveCloudiness)
        bool clouds            = false;
        vec3 light             = vec3(0, 1, 0);
        float lightIlluminance = 0;
    };
    // Records this frame's work (after the atmosphere's sky-view LUT): a new capture when due, the next step of one in
    // progress, the swap when complete. frame: RenderScene's frame number
    void record(vk::CommandBuffer cmd, const EnvironmentState& environment, float cameraHeight,
                const LightingParameters& parameters, const CaptureInputs& inputs, uint64_t frame);
    // A different resolution recreates the cubes (from the next record; deferDelete)
    int resolution() const { return size; }

    bool valid() const { return frontValid; }
    vk::DescriptorImageInfo frontCubeInfo();
    vk::Buffer frontSh() const;
    static constexpr size_t kShBytes = 9 * 16;
    // The front SH (lux) as read back after its frame completed (zero before)
    const std::array<vec3, 9>& sh() const { return frontShValues; }
    const EnvironmentMapStatistics& statistics() const { return stats; }
    // Reads the front SH once the frame that computed it has completed
    void harvest();
    // The debug images of the previews (LightingDebug's sources; recorded on demand in the frame slot): mode 0 the
    // front cube's mip as a cross (radiance), mode 1 the front SH's irradiance as an equirectangular image (lux);
    // RGBA16F in eShaderReadOnlyOptimal afterwards
    Texture2D* recordDebug(vk::CommandBuffer cmd, uint32_t frameIndex, int mode, int mip);
    // Tests: a uniform radiance (cd/m^2) instead of the sky in the next captures (nullopt: the sky)
    void setCaptureOverride(std::optional<vec3> radiance) { captureOverride = radiance; }
    // The capture cube and the front cube (previews, tests)
    TextureCube& captureCube() { return capture; }
    TextureCube& frontCube() { return cubes[front]; }
    uint64_t memoryBytes() const;

   private:
    struct Key
    {
        vec3 sunDir = vec3(0, 0, 0), moonDir = vec3(0, 0, 0);
        float sunE = 0, moonE = 0, cloudiness = -1, cameraHeight = 0, cloudOffsetX = 0, cloudOffsetY = 0;
        AtmosphereSettings atmosphere;
        CloudSettings clouds;
        bool enclosed = false, cloudsOn = true;
    };
    bool due(const Key& k, const LightingParameters& p) const;
    void createImages(int resolution);
    void destroyImages();
    void recordCapture(vk::CommandBuffer cmd, const EnvironmentState& e, const CaptureInputs& in, float cameraHeight);
    void recordPrefilter(vk::CommandBuffer cmd, int mip);
    void recordSh(vk::CommandBuffer cmd);

    VulkanBase* base = nullptr;
    int size         = 0;
    TextureCube capture;
    std::array<TextureCube, 2> cubes;
    std::array<Buffer, 2> shBuffers;
    // Per cube and mip: a 2D array view for the storage writes; the capture's mip 0 view
    std::array<std::array<vk::ImageView, kMips>, 2> mipViews{};
    vk::ImageView captureMip0;
    Sampler linear;
    ComputePipeline capturePipeline, prefilterPipeline, shPipeline, debugPipeline;
    std::array<std::array<vk::DescriptorSet, 2>, 2> debugSets{};
    std::array<Texture2D, 2> debugImages;
    std::optional<vec3> captureOverride;
    vk::DescriptorSet captureSet;
    std::array<std::array<vk::DescriptorSet, kMips>, 2> prefilterSets{};
    std::array<vk::DescriptorSet, 2> shSets{};
    int front       = 0;
    bool frontValid = false;
    // The capture in progress: the next step (0 none, 1..kMips the prefilter of mip step - 1, kMips + 1 the SH)
    int step = 0;
    Key current, pending;
    bool haveKey        = false;
    bool skyViewWritten = false;
    std::array<vec3, 9> frontShValues{};
    uint64_t shReadyValue = 0;
    bool shPending        = false;
    EnvironmentMapStatistics stats;
};

}  // namespace Vulkan
}  // namespace Saiga
