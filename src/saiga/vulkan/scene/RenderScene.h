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
#include "saiga/vulkan/scene/TerrainAtlas.h"
#include "saiga/vulkan/scene/TerrainPass.h"

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
class GrassPass;
struct Texture2DArray;
struct SceneCamera;
struct SceneLighting;
struct PrototypeSettings;

/**
 * The frame of the engine app (docs/engine_design.md §8.3, §10.1, §10.5; S13: the frame shell, S17: the HDR frame):
 * the swap chain of the window (or offscreen images for a HeadlessWindow), two frames in flight on the frame timeline,
 * GPU timestamps, screenshots. Per frame:
 *   1. transfer (uploads), outside of any pass: the game's callback, then the terrain uploads, node selection, grass
 *      blocks and TerrainFrame (S20) and the scene content's uploads (S18);
 *   2. cull (S18): the GPU culling of the scene's instances (count, scan, write; docs/engine_design.md §10.2);
 *   3. depth (S19a, §10.5 passes 2-3): the depth pre-pass of the opaque content (the terrain and water, the
 *      opaque instance classes) into the scene pass's depth (PrototypeSettings::depthPrePass; empty without it);
 *   4. the scene pass: HDR color (R16G16B16A16 float, msaaSamples) + reversed-Z depth (D32, infinite far plane), the
 *      color cleared to the clear color, the depth cleared to 0 or loaded from the pre-pass; in order: the terrain and
 *      water (EQUAL after the pre-pass), the render callback (pipelines: formats()), the scene's instances (indexed
 *      indirect draws per pipeline class; the opaque classes EQUAL after the pre-pass), the grass and the sky; the
 *      depth is resolved with SAMPLE_ZERO into a single-sampled depth image (with 1 sample the depth image
 *      itself), left in eShaderReadOnlyOptimal for later consumers (S37);
 *   5. the MSAA resolve: Identity (clay mode) with the hardware average of the scene pass, the other tone maps with a
 *      tone-map weighted custom resolve (bright edges do not alias); none with 1 sample;
 *   6. post (exposure, tone map, output transfer) and ImGui on top, single-sampled, into the swap chain image.
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
 *
 * Terrain (S20): the GPU terrain set of one space (TerrainAtlas) filled through the terrain calls (the render sink's),
 * drawn as CDLOD (TerrainPass) with the water plane and the grass (GrassPass). The prototype environment
 * (setPrototypeEnvironment, until S36) adds the sky and the foliage texture of the PrototypeFoliage pipeline class (the
 * trees are instance sets the game creates, PrototypeForest); its settings set the instances' lighting block (sun,
 * hemisphere ambient, height fog, wind; without it S18's fixed light, fog off). The time of the wind phases comes from
 * setTime (no wall clock).
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
    // CPU time the last frame blocked for the frame slot, the swap chain image and the present (VulkanRenderer::
    // lastWaitMs): idle time, not work
    double lastWaitMs() const;
    // Its parts (VulkanRenderer::lastSlotWaitMs, lastAcquireWaitMs, lastPresentWaitMs): {slot, acquire, present}
    std::array<double, 3> lastWaitParts() const;
    // The last frame was presented (headless: rendered); renderFrame's result only says it was submitted
    bool lastPresented() const;
    // True if presented with vsync (the frame rate follows the display)
    bool vsync() const;
    bool headless() const;
    // The size of the rendered images (0 before the first frame)
    ivec2 size() const;
    uint64_t frames() const { return frameCount; }

    // The next rendered frame (tone mapped, with ImGui) as RGBA, alpha 255
    void requestScreenshot(VulkanRenderer::ScreenshotCallback callback);
    // The screenshot requests not yet taken by a frame, in request order (requestScreenshot gives them back)
    std::vector<VulkanRenderer::ScreenshotCallback> takeScreenshotRequests();
    void setImguiVisible(bool visible);
    bool imguiVisible() const;
    // True if the last recorded frame drew ImGui
    bool lastDrewImgui() const { return frameDrewImgui; }
    // Draws ImGui (true) or not (false) in the rendered frames that are read back (requestScreenshot) until cleared,
    // without changing imguiVisible (the user's choice, e.g. a key that toggles it); for screenshots with or without
    // the GUI (the other frames keep the user's choice)
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
    // The newest frame takeGpuFrameTimes returned (frame 0: none yet)
    const FrameTimes& lastGpuFrameTimes() const { return lastTaken; }
    // The sections of FrameTimes::gpuSectionMs in frame order: transfer, cull, depth, depth.terrain, scene,
    // scene.terrain, scene.instances, scene.grass, scene.sky, resolve, post, imgui. The dotted names are sub-sections
    // of the pass before the dot (consecutive ALL_COMMANDS timestamps inside the pass; S19a); every section is written
    // in every frame, also when its content is off
    static const std::vector<const char*>& gpuSectionNames();

    //==================================================================================================
    // Scene content (S18)

    // Explicit IDs (asserted free, names unique; ID 0 is none). PipelineClass::PrototypeFoliage needs the foliage
    // texture of setPrototypeEnvironment (asserted)
    void addMaterial(MaterialId id, const std::string& name, const MaterialData& data);
    // A change of alphaMask, twoSided or pipelineClass rebuilds the bucket layout
    void setMaterial(MaterialId id, const MaterialData& data);
    // The material must be registered; 1..4 LODs with non-decreasing end distances
    void addMesh(MeshId id, const std::string& name, const MeshLods& lods, MaterialId material);
    // Rewrites only the mesh record (S19a): the end distances (the same LOD count, non-decreasing, >= 0) and the bounds
    // padding; no geometry upload, no bucket rebuild. The sets with the mesh get their bounds from their unpadded
    // bounds and largest scale. From the next frame
    void setMeshLodDistances(MeshId id, const std::vector<float>& endDistances, float boundsPadding);
    // Reserves count consecutive IDs of the game range [1, kGameReservedIds) for content the game registers directly
    // on the scene (S19a: the prototype's tree meshes and material); returns the first (the range is asserted)
    uint16_t reserveIds(SceneIdKind kind, uint32_t count);
    // The ID of a registered mesh or material name (0: none); e.g. a game re-attaching its content to the scene
    MeshId findMesh(const std::string& name) const;
    MaterialId findMaterial(const std::string& name) const;
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

    // The render settings of a space (S21): the camera's space decides what the frame draws (an enclosed space: no
    // terrain, grass, water or sky, the background as the clear color, kEnclosedLighting); a space without settings
    // is drawn as open world
    void setSpaceSettings(SpaceId space, const SpaceRenderSettings& settings);
    const SpaceRenderSettings& spaceSettings(SpaceId space) const;

    // The camera (double; fovy: vertical field of view in radians); the aspect from size(), the near plane from the
    // parameters (reversed-Z, infinite far plane). The camera looks along -z of its rotation, y is up. Also from the
    // transfer callback: the frame being recorded uses it (viewMatrix() / projectionMatrix() follow at once)
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
    // The statistics of the oldest completed frame not yet taken, in frame order (at most 16 are kept); false if none.
    // Every frame's statistics, also when several frames complete between two calls (statistics() only the newest).
    bool takeStatistics(RenderStatistics& out);
    // The meshes whose drawn instances per LOD the statistics carry (RenderStatistics::trackedMeshes), at most
    // maxTrackedMeshes (more is a fatal error); from the next recorded frame (each frame's statistics carry its list)
    void trackMeshStatistics(const std::vector<MeshId>& meshes);
    // The indirect draws use multiDrawIndirect (device feature and RenderSceneParameters::allowMultiDrawIndirect)
    bool multiDrawIndirect() const;

    // Waits until the GPU has finished every frame (before resources used by frames are destroyed)
    void waitIdle();

    //==================================================================================================
    // Terrain (S20, docs/engine_design.md §10.1, §12.2): the GPU terrain set of one space (TerrainAtlas), filled
    // through these calls (the RenderSink's terrain calls; main thread, effective in the next renderFrame).

    // Another space or layout than the current set's: a new GPU set (the old set's tiles and masks are dropped); the
    // same: the pyramid data is replaced in place. Its seaLevel is the water plane
    void setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid);
    // New: a free layer of its level (asserted); already set: replaced in the same layer
    void setTerrainTile(const RegionId& region, const TerrainTileData& tile);
    void removeTerrainTile(const RegionId& region);
    // A set mask is replaced in its layer
    void setGrassMask(const ChunkId& chunk, const GrassMaskData& mask);
    void removeGrassMask(const ChunkId& chunk);
    // E1: drops the set; no terrain, grass or water is drawn until the next pyramid
    void clearTerrain();
    // E2: absolute heights of a sub-rectangle of a tile's (the pyramid's) sample grid, w values per row, copied at the
    // call and queued behind a pending full copy; without a layer or for another space ignored and counted
    // (TerrainAtlas::Stats::patchesIgnored)
    void patchTerrainTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights);
    void patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights);
    // The bytes of terrain uploads not yet recorded
    size_t terrainUploadsPending() const;
    // The set, its residency, bytes and patches
    const TerrainAtlas::Stats& terrainAtlasStats() const;
    // The GPU set itself (tests: readLayerForTests, node bounds)
    TerrainAtlas& terrainAtlas() { return *atlas; }
    // The last recorded frame's terrain selection: nodes per level, refusals for residency, the water node
    const TerrainPass::Stats& terrainPassStats() const;
    // The terrain pass (tests: selectedNodes, probeVertices)
    TerrainPass& terrainPass() { return *terrain; }
    // The last recorded frame's grass blocks (GrassPass, S20)
    const GrassPass& grassPass() const { return *grass; }

    // The prototype environment of the new terrain path (S20, until S36): the sun, fog, clay shading and wind of the
    // settings for the terrain, water, grass and instances, and the foliage texture of the grass and of the
    // PrototypeFoliage class (foliageSeed). Outside of renderFrame (it waits for the device and uploads); a second
    // call replaces the texture
    void setPrototypeEnvironment(const PrototypeSettings& settings, uint32_t foliageSeed);
    bool hasPrototypeEnvironment() const { return foliage != nullptr; }
    // Tests: the last frame's single-sampled depth (the SAMPLE_ZERO resolve target; waits for the device), row 0 at the
    // top; reversed-Z (1 at the near plane, 0 at infinity)
    std::vector<float> readDepthForTests(int& width, int& height);

    //==================================================================================================
    // Prototype settings (S19a, docs/engine_design.md §10.1)

    // From the next frame (game_set_settings), or from this frame in its transfer callback: a frame takes one snapshot
    // after that callback (a change in the render callback is the next frame's); depthPrePass also applies without
    // terrain
    void setPrototypeSettings(const PrototypeSettings& settings);
    const PrototypeSettings& prototypeSettings() const;
    // The time of the wind phases (seconds, default 0): the shaders get fmod(time * f, 2 pi) per wind frequency,
    // computed in double (§10.1: no absolute time in a shader). Games pass world.time() + alpha * dt; goldens and
    // benchmarks keep 0. Transitional exception to the camera-relative rule with it: the foliage rebuilds its float
    // world xz for the per-tree wind phase (valid within a few km of the origin; S23 replaces the prototype trees)
    void setTime(double seconds) { sceneTime = seconds; }
    double time() const { return sceneTime; }

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
    std::unique_ptr<PrototypeSettings> protoSettings;
    std::unique_ptr<TerrainAtlas> atlas;
    std::unique_ptr<TerrainPass> terrain;
    std::unique_ptr<GrassPass> grass;
    // The foliage texture of setPrototypeEnvironment
    std::unique_ptr<Texture2DArray> foliage;
    double sceneTime = 0;
    // The next IDs of reserveIds (mesh, material)
    std::array<uint32_t, 2> nextReservedId = {1, 1};
    FrameTimes lastTaken;

    // The render settings per SpaceId (setSpaceSettings; missing: open world)
    std::vector<SpaceRenderSettings> spaceRenderSettings;
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
        uint32_t terrainNodes  = 0;
    };
    std::array<SlotStatistics, 2> slotStatistics;
    RenderStatistics stats;
    std::deque<RenderStatistics> completedStatistics;
    void harvestStatistics();
    // The statistics slot of the frame being recorded (SceneFrameRenderer::render)
    void beginSceneFrame(uint32_t frameIndex);
    // The camera of the frame being recorded (again after the transfer callback, which may set it)
    void frameCamera(int width, int height, SceneCamera& camera);
    // The matrices of the camera for the size
    void updateMatrices(int width, int height);
    // The lighting block of the frame's instance draws
    void frameLighting(const SceneCamera& camera, const PrototypeSettings& proto, bool enclosed, int samples,
                       SceneLighting& out) const;
    const Callback* transferFn         = nullptr;
    const Callback* renderFn           = nullptr;
    const std::function<void()>* guiFn = nullptr;
    uint64_t frameCount                = 0;
    std::optional<bool> imguiOverride;
    bool frameDrewImgui = false;
    // The frame numbers of the submitted frames by timeline value (GPU times arrive by timeline value)
    std::deque<std::pair<uint64_t, uint64_t>> submittedFrames;
};

}  // namespace Vulkan
}  // namespace Saiga
