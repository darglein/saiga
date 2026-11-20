/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/managedImage.h"
#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;
class SceneFrameRenderer;
class MeshPool;
class InstanceBuffer;
class SceneCulling;
struct SceneCamera;

/**
 * The frame of the engine app (docs/engine_design.md §8.3, §10.1, §10.5; S13: the frame shell, S17: the HDR frame):
 * the swap chain of the window (or offscreen images for a HeadlessWindow), two frames in flight on the frame timeline,
 * GPU timestamps, screenshots. Per frame:
 *   1. transfer (uploads), outside of any pass: the game's callback, then the scene content's uploads (S18);
 *   2. cull (S18): the GPU culling of the scene's instances (count, scan, write; docs/engine_design.md §10.2);
 *   3. the scene pass: HDR color (R16G16B16A16 float, msaaSamples) + reversed-Z depth (D32, infinite far plane),
 *      cleared to the clear color and depth 0; the render callback draws here (pipelines: formats()), then the
 *      scene's instances (indexed indirect draws per pipeline class);
 *   4. the MSAA resolve: Identity (clay mode) with the hardware average of the scene pass, the other tone maps with a
 *      tone-map weighted custom resolve (bright edges do not alias); none with 1 sample;
 *   5. post (exposure, tone map, output transfer) and ImGui on top, single-sampled, into the swap chain image.
 *
 * Pre-exposure contract: while sceneWritesLinear() every scene shader writes linear radiance multiplied by
 * preExposure() (negatives clamped, no clamp to 1); otherwise display values clamped to [0, 1] (clay mode).
 *
 * Scene content (S18, §10.1): materials, meshes and the palette by explicit IDs; static instance sets per region and
 * dynamic instances (prev and curr pose relative to a chunk, interpolated on the GPU with setInterpolation's alpha) in
 * one instance buffer each; every frame the offset table holds origin - camera (double, split into float hi + lo) per
 * set and per chunk of a dynamic, the instances are culled on the GPU and drawn camera-relative (the view matrix holds
 * only the camera rotation). Nothing is culled or drawn before the first setCamera. Main thread only; the calls take
 * effect in the next renderFrame (also from the render callback: the frame draws its transfer's snapshot). The storage
 * buffers grow at most to maxStorageBufferRange (RenderSceneParameters' capacities; more is a fatal error).
 */
class SAIGA_VULKAN_API RenderScene
{
   public:
    RenderScene(VulkanWindow& window, const VulkanParameters& vulkanParameters,
                const RenderSceneParameters& parameters);
    ~RenderScene();
    RenderScene(const RenderScene&)            = delete;
    RenderScene& operator=(const RenderScene&) = delete;

    using Callback = std::function<void(vk::CommandBuffer)>;
    // Records and submits one frame: transfer (uploads), render (inside the scene pass), gui (ImGui calls). False if no
    // frame was submitted (a minimized window, an out-of-date swap chain that is recreated). The tone map, exposure and
    // pre-exposure are latched when the frame's recording starts, before any callback (not without a frame).
    bool renderFrame(const Callback& transfer, const Callback& render, const std::function<void()>& gui);

    // The scene pass (the HDR color format, D32 depth, the sample count): the game's pipelines use it
    RenderingFormats formats() const;
    // The output format (swap chain or offscreen image)
    vk::Format colorFormat() const;
    VulkanBase& base();
    // CPU time of the last frame's command recording and submit (milliseconds)
    double lastRecordingMs() const;
    bool headless() const;
    // The size of the rendered images (0 before the first frame)
    ivec2 size() const;
    uint64_t frames() const { return frameCount; }

    // The next rendered frame (tone mapped, with ImGui) as RGBA, alpha 255
    void requestScreenshot(VulkanRenderer::ScreenshotCallback callback);
    void setImguiVisible(bool visible);
    bool imguiVisible() const;
    // Draws ImGui (true) or not (false) in the rendered frames until cleared, without changing imguiVisible (the
    // user's choice, e.g. a key that toggles it); for screenshots with or without the GUI
    void setImguiOverride(std::optional<bool> visible) { imguiOverride = visible; }
    std::optional<bool> imguiOverridden() const { return imguiOverride; }

    // From the next frame on (R-MECH-1). The exposure is clamped to [1/1024, 1024]
    void setToneMap(ToneMap toneMap);
    void setExposure(float exposure);
    // The requested values (the next frame's)
    const RenderSceneParameters& parameters() const { return params; }
    // The latched values of the frame being recorded (the last frame between frames)
    ToneMap toneMap() const { return frameToneMap; }
    // The effective exposure (1 with Identity)
    float exposure() const { return frameExposure; }
    // toneMap() != Identity: the scene shaders write linear values multiplied by preExposure()
    bool sceneWritesLinear() const { return frameToneMap != ToneMap::Identity; }
    // The exposure the scene pass of this frame writes with: the previous frame's effective exposure (1 after
    // Identity), or this frame's if it changed by more than a factor of kMaxPreExposureChange (fp16 range)
    float preExposure() const { return framePreExposure; }
    static constexpr float kMaxPreExposureChange = 16;

    // The GPU times of the oldest completed frame not yet taken (2-3 frames after it was rendered); false if none
    bool takeGpuFrameTimes(FrameTimes& out);
    // The sections of FrameTimes::gpuSectionMs in frame order: transfer, cull, scene, resolve, post, imgui
    static const std::vector<const char*>& gpuSectionNames();

    //==================================================================================================
    // Scene content (S18)

    // Explicit IDs (asserted free, names unique; ID 0 is none)
    void addMaterial(MaterialId id, const std::string& name, const MaterialData& data);
    // A change of alphaMask, twoSided or pipelineClass rebuilds the bucket layout
    void setMaterial(MaterialId id, const MaterialData& data);
    // The material must be registered; 1..4 LODs with ascending end distances
    void addMesh(MeshId id, const std::string& name, const MeshLods& lods, MaterialId material);
    // The game's palette (at most 4096 entries), read by the materials with usePalette
    void setPalette(const std::vector<PaletteEntry>& palette);

    // A static instance set of the region (region-relative positions; its meshes must be registered); capacity >=
    // instances.size() (0: the size)
    InstanceSetHandle addInstanceSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                     uint32_t capacity = 0);
    // Writes [first, first + size) (first + size <= capacity; appends up to it) and recomputes the bounds
    void updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& instances);
    // A tombstone (scale 0)
    void removeInstance(InstanceSetHandle set, uint32_t index);
    void removeInstanceSet(InstanceSetHandle set);

    // A dynamic instance (drawn after its first setDynamic)
    InstanceHandle addDynamic(MeshId mesh);
    // The previous and current pose relative to the chunk's origin; boneOffset unused before S34
    void setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                    uint32_t boneOffset = 0);
    void removeDynamic(InstanceHandle h);

    // The camera (double; fovy: vertical field of view in radians); the aspect from size(), the near plane from the
    // parameters (reversed-Z, infinite far plane). The camera looks along -z of its rotation, y is up.
    void setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy);
    bool hasCamera() const { return cameraSet; }
    // The interpolation of the dynamic instances between prev (0) and curr (1)
    void setInterpolation(float alpha);
    // The camera-relative matrices of the frame being recorded (the last frame between frames): the view matrix holds
    // only the camera rotation (positions relative to the camera, R-MATH-1)
    const mat4& viewMatrix() const { return frameView; }
    const mat4& projectionMatrix() const { return frameProjection; }
    mat4 viewProjection() const { return frameProjection * frameView; }

    // The statistics of the newest completed frame (tagged with its frame number)
    const RenderStatistics& statistics();
    // The meshes whose drawn instances per LOD the statistics carry (RenderStatistics::trackedMeshes), at most
    // maxTrackedMeshes (more is a fatal error); from the next recorded frame (each frame's statistics carry its list)
    void trackMeshStatistics(const std::vector<MeshId>& meshes);
    // The indirect draws use multiDrawIndirect (device feature and RenderSceneParameters::allowMultiDrawIndirect)
    bool multiDrawIndirect() const;

    // Waits until the GPU has finished every frame (before resources used by frames are destroyed)
    void waitIdle();

   private:
    friend class SceneFrameRenderer;
    void latchFrame();
    RenderSceneParameters params;
    ToneMap frameToneMap   = ToneMap::Identity;
    float frameExposure    = 1;
    float framePreExposure = 1;
    // The effective exposure of the last submitted frame (the next frame's pre-exposure)
    float lastExposure = 1;

    std::unique_ptr<SceneFrameRenderer> renderer;
    std::unique_ptr<MeshPool> meshPool;
    std::unique_ptr<InstanceBuffer> instanceBuffer;
    std::unique_ptr<SceneCulling> culling;

    // The camera (setCamera)
    bool cameraSet       = false;
    SpaceId cameraSpace  = 0;
    Vec3 cameraPosition  = Vec3::Zero();
    Quat cameraRotation  = Quat::Identity();
    double cameraFovy    = 1;
    float alpha          = 1;
    mat4 frameView       = mat4::Identity();
    mat4 frameProjection = mat4::Identity();

    // The statistics of the frame slots, read when the frame has completed
    struct SlotStatistics
    {
        bool pending           = false;
        uint64_t frame         = 0;
        uint64_t timelineValue = 0;
        uint32_t drawCommands  = 0;
    };
    std::array<SlotStatistics, 2> slotStatistics;
    RenderStatistics stats;
    void harvestStatistics();
    // The camera of the frame being recorded (SceneFrameRenderer::render)
    void beginSceneFrame(uint32_t frameIndex, int width, int height, SceneCamera& camera);
    const Callback* transferFn         = nullptr;
    const Callback* renderFn           = nullptr;
    const std::function<void()>* guiFn = nullptr;
    uint64_t frameCount                = 0;
    std::optional<bool> imguiOverride;
    // The frame numbers of the submitted frames by timeline value (GPU times arrive by timeline value)
    std::deque<std::pair<uint64_t, uint64_t>> submittedFrames;
};

}  // namespace Vulkan
}  // namespace Saiga
