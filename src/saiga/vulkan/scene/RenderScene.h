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

#include <algorithm>
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
class SceneLightingSet;
class SkyAmbient;
class BrdfLut;
class Atmosphere;
class EnvironmentMap;
class CloudLayer;
class LightingDebug;
class DebugLines;
class ImGuiVulkanRenderer;
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
 * drawn as CDLOD (TerrainPass) with the water plane and the grass (GrassPass). setFoliageTexture adds the foliage
 * texture of the grass and of the PrototypeFoliage pipeline class (the trees are instance sets the game creates,
 * PrototypeForest) and the S20 sky. The light (S36, docs/engine_design.md §10.7): the environment of setEnvironment
 * (sun, moon, sky, clouds, exposure; the wind phases), clay mode's studio environment, the atmosphere's sky in the
 * physical path.
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
    // The ImGui renderer of the frame (nullptr without ImGui): ImGui textures (S36: the Lighting tab's previews)
    ImGuiVulkanRenderer* imguiRenderer();
    void setImguiVisible(bool visible);
    bool imguiVisible() const;
    // True if the last recorded frame drew ImGui
    bool lastDrewImgui() const { return frameDrewImgui; }
    // Draws ImGui (true) or not (false) in the rendered frames that are read back (requestScreenshot) until cleared,
    // without changing imguiVisible (the user's choice, e.g. a key that toggles it); for screenshots with or without
    // the GUI (the other frames keep the user's choice)
    void setImguiOverride(std::optional<bool> visible) { imguiOverride = visible; }
    std::optional<bool> imguiOverridden() const { return imguiOverride; }

    // From the next frame on (R-MECH-1; the tone map after a running probe). The exposure is clamped to [1/1024, 1024]
    void setToneMap(ToneMap toneMap);
    void setExposure(float exposure);
    // The requested values (the next frame's)
    const RenderSceneParameters& parameters() const { return params; }
    // The latched values of the frame being recorded (the last frame between frames)
    ToneMap toneMap() const { return frameToneMap; }
    // The frame's exposure multiplier: params.exposure as latched (1 with Identity). Without an environment it is the
    // effective exposure; with one (S36) the multiplier on top of the EV100
    float exposure() const { return frameMultiplier; }
    // The effective exposure the post pass uses: the multiplier, or with an environment exposureFromEv100(EV100,
    // multiplier) (S36, D18; 1 with Identity)
    float frameExposure() const { return frameEffectiveExposure; }
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
    // The sections of FrameTimes::gpuSectionMs in frame order: transfer, environment (S36: the atmosphere and the
    // environment map), cull, depth, depth.terrain, scene,
    // scene.terrain, scene.instances, scene.grass, scene.sky, scene.clouds (S36: the 2D cloud layer), scene.debug (S36:
    // the debug lines), resolve,
    // lighting.preview (S36: the preview conversions), post, imgui. The dotted names are sub-sections
    // of the pass before the dot (consecutive ALL_COMMANDS timestamps inside the pass; S19a); every section is written
    // in every frame, also when its content is off
    static const std::vector<const char*>& gpuSectionNames();

    //==================================================================================================
    // Scene content (S18)

    // Explicit IDs (asserted free, names unique; ID 0 is none). A PipelineClass::PrototypeFoliage material can be
    // added and set at any time (S23c, F4); a frame with instances of that class to draw asserts the foliage texture
    // of setFoliageTexture
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
    // terrain, grass, water or sky, the background as the clear color, the template's light); a space without settings
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

    // The foliage texture of the grass and of the PrototypeFoliage class from the seed (S36: replaces S20's prototype
    // environment call, whose sun and fog are the environment's; the passes and LOD settings are
    // setPrototypeSettings'). Outside of renderFrame (it waits for the device and uploads); a second call replaces the
    // texture. With it the S20 sky is drawn also without an environment
    void setFoliageTexture(uint32_t foliageSeed);
    bool hasFoliageTexture() const { return foliage != nullptr; }
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

    // ---- environment and lighting (S36, docs/engine_design.md §10.7) -----------------------------------------------
    // The environment of the next frames (main thread, from the next frame; also from the transfer callback like
    // setCamera). Until the first call the scene lights with the studio environment and uses params.exposure unchanged
    void setEnvironment(const EnvironmentState& environment);
    const EnvironmentState& environment() const { return env; }
    bool hasEnvironment() const { return envSet; }
    // Applied at the next frame start (resources are recreated through deferDelete when a size changes)
    void setLightingParameters(const LightingParameters& parameters);
    const LightingParameters& lightingParameters() const { return params.lighting; }
    // The studio environment of clay mode (A3, D7): by default sceneStudio(), with the foliage texture (the prototype)
    // prototypeStudio() with PrototypeSettings::albedo; an enclosed space uses prototypeStudio() without fog.
    // setStudioEnvironment overrides both (tests, a game that wants another clay light)
    void setStudioEnvironment(std::optional<StudioEnvironment> studio) { studioOverride = studio; }
    // The studio environment of the next frame for an open (or an enclosed) camera space
    StudioEnvironment studioEnvironment(bool enclosed = false) const;
    // The frame being recorded uses the physical path: an environment and a tone map other
    // than Identity (latched with the tone map)
    bool physicalLighting() const { return framePhysical; }
    // The lighting's resources (tests, the debug tools)
    SkyAmbient& skyAmbient() { return *ambient; }
    BrdfLut& brdfLut() { return *brdf; }
    Atmosphere& atmosphereLuts() { return *atmosphere; }
    EnvironmentMap& environmentMap() { return *envMap; }
    CloudLayer& cloudLayer() { return *clouds; }
    // The scene pipelines created since the start (SceneCulling's, the terrain's and the grass's; a debug view switch
    // creates none)
    uint64_t scenePipelineCreations() const;

    // ---- the lighting's debug core (S36a1b, A8) ------------------------------------------------------------------
    // The debug view and overlays of the next frames. While a view other than None is active a frame latches
    // ToneMap::Identity (no pipeline changes) and does not update the pre-exposure history
    void setLightingDebug(const LightingDebugState& state) { userDebug = state; }
    const LightingDebugState& lightingDebug() const { return userDebug; }
    // A one-frame override for the frames that are read back (requestScreenshot), like setImguiOverride
    void setLightingDebugOverride(std::optional<LightingDebugState> state) { debugOverride = state; }
    // The debug state the frame being recorded latched
    const LightingDebugState& frameLightingDebug() const { return frameDebug; }
    // The Exposure view's heat map: centre (EV100; nullopt: the frame's EV100 in the physical path, the display
    // mid-gray log2(0.18 * 8) in clay) and range (EV)
    void setExposureViewRange(std::optional<float> centre, float range)
    {
        exposureViewCentre = centre;
        exposureViewRange  = std::clamp(range, 0.5f, 64.0f);
    }
    static constexpr float kDefaultExposureViewRange = 8;
    std::optional<float> exposureViewCentreSetting() const { return exposureViewCentre; }
    float exposureViewRangeSetting() const { return exposureViewRange; }
    // The centre of the frame being recorded (the automatic one without a setting)
    float frameExposureViewCentre() const;
    // The previews and pixel readbacks (LightingDebug::request, requestPixels) and the debug lines
    // (DebugLines::addLine)
    LightingDebug& lightingDebugImages() { return *debugImages; }
    // The next frame updates every incremental effect fully (the environment map; capture_lighting full: true), like a
    // frame with LightingParameters::forceFullUpdates
    void requestFullUpdate() { fullUpdateRequested = true; }
    // The mip of the Environment preview (the front cube's, 0..EnvironmentMap::kMips - 1)
    void setEnvironmentPreviewMip(int mip) { environmentPreviewMip = mip; }
    int previewMip() const { return environmentPreviewMip; }
    DebugLines& debugLines() { return *lines; }
    // The statistics of the newest recorded frame (the luminance reduction when it was requested)
    const LightingStatistics& lightingStatistics() const { return lightStats; }
    // probe_lighting's sequence at a pixel of the next frames (8 frames; requests made before the sequence starts share
    // it). The camera and the environment (with it the wind phases) are held during the sequence (setCamera /
    // setEnvironment take effect after it); the callback runs after the last frame. While it runs, screenshots and
    // preview conversions wait for the first frame after it (they would read the debug frames), and the output shows
    // frame 1's image (the window does not flash the debug frames; ImGui stays live)
    void requestProbe(ivec2 pixel, std::function<void(const LightingProbe&)> callback);
    // The same at the pixel of a world point, projected with the camera of the sequence's frame 1 (outside the image:
    // inImage false)
    void requestProbe(const Vec3& world, std::function<void(const LightingProbe&)> callback);
    bool probeRunning() const { return probeRun != nullptr; }
    // A probe sequence runs in the next recorded frame (running or requested)
    bool probePending() const { return probeRun != nullptr || !probeRequests.empty(); }
    static constexpr int kProbeFrames = 8;

   private:
    friend class SceneFrameRenderer;
    void latchFrame();
    RenderSceneParameters params;
    ToneMap frameToneMap         = ToneMap::Identity;
    float frameEffectiveExposure = 1;
    float frameMultiplier        = 1;
    // setEnvironment (S36)
    EnvironmentState env;
    bool envSet = false;
    // S36: the physical path of the frame, the studio override
    bool framePhysical = false;
    std::optional<StudioEnvironment> studioOverride;
    std::unique_ptr<SceneLightingSet> lightingSet;
    std::unique_ptr<SkyAmbient> ambient;
    std::unique_ptr<BrdfLut> brdf;
    std::unique_ptr<Atmosphere> atmosphere;
    std::unique_ptr<EnvironmentMap> envMap;
    std::unique_ptr<CloudLayer> clouds;
    // The debug core (S36a1b)
    std::unique_ptr<LightingDebug> debugImages;
    std::unique_ptr<DebugLines> lines;
    LightingDebugState userDebug, frameDebug;
    std::optional<LightingDebugState> debugOverride;
    // The radiance views write raw values (the probe frames)
    bool frameDebugRaw        = false;
    int environmentPreviewMip = 0;
    bool fullUpdateRequested  = false;
    std::optional<float> exposureViewCentre;
    float exposureViewRange = kDefaultExposureViewRange;
    LightingStatistics lightStats;
    struct ProbeRun;
    struct ProbeRequest
    {
        ivec2 pixel = ivec2(0, 0);
        std::optional<Vec3> world;
        std::function<void(const LightingProbe&)> callback;
    };
    std::unique_ptr<ProbeRun> probeRun;
    std::vector<ProbeRequest> probeRequests;
    // While a probe runs: the environment, tone map and camera given meanwhile (applied after it)
    std::optional<EnvironmentState> heldEnvironment;
    std::optional<ToneMap> heldToneMap;
    struct HeldCamera
    {
        SpaceId space;
        Vec3 position;
        Quat rotation;
        double fovy;
    };
    std::optional<HeldCamera> heldCamera;
    void finishProbe();
    // The view of the frame being recorded (the probe's step, the override of a read-back frame, the user's)
    void latchDebug();
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
    // The foliage texture of setFoliageTexture
    std::unique_ptr<Texture2DArray> foliage;
    // The frame's check without the foliage texture (S23c, F4; only with SAIGA_ASSERTS, SceneCulling never draws the
    // class without the texture): asserts that no instance set or dynamic instance draws a mesh of the PrototypeFoliage
    // class
    void checkFoliageInstances() const;
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
    // The lighting block of the frame's draws (the legacy fields and S36's)
    void frameLighting(const SceneCamera& camera, const PrototypeSettings& proto, bool enclosed, int samples,
                       SceneLighting& out) const;
    // The frame slot's lighting set (block, ambient, foliage, LUTs): in the transfer section
    void writeLightingSet(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneLighting& block);
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
