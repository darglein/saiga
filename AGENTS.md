# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project Overview

**Saiga** is a C++ utility, rendering and computer-vision framework (OpenGL and Vulkan
renderers, a CUDA module, and a vision module for SLAM, reconstruction, registration and
optimization).

- Language: C++17 (`cxx_std_17` on `saiga_core`), CUDA code is compiled as C++17 and needs CUDA 12.0 or newer
- Build system: CMake (>= 3.16), version 1.3.2 (root `CMakeLists.txt`)
- Namespace: `Saiga`
- Libraries: one shared library per module — `saiga_core`, `saiga_opengl`, `saiga_vulkan`, `saiga_engine`,
  `saiga_vision`, `saiga_cuda` (exported as `Saiga::<name>`); games are samples in `samples/games/<name>/` (`game_<name>`)
- Tests: googletest (`submodules/googletest`), one executable per test file, run with `ctest`
- License: MIT

## Repository Layout

```
CMakeLists.txt          Root build: all SAIGA_* options, dependency includes, configured headers, subdirs
cmake/
  Dependencies_<module>.cmake   Finds each module's dependencies; sets MODULE_<NAME> if the module can be built
  Module_CMakeLists.txt         Unused old module recipe; each module's own CMakeLists.txt globs its sources
  Saiga_compiler_flags.cmake    Warning / optimization / sanitizer flags
  select_compute_arch.cmake     CUDA arch selection (SAIGA_CUDA_ARCH, else autodetect)
  Find*.cmake                   Find modules for the optional dependencies
src/
  saiga/
    core/               saiga_core: math, image (load/store/decode), util (threads, ini, file io), geometry,
                        camera, imgui, window, sound, model, time, sophus, ...
                        util/mcp/: McpServer (MCP over HTTP JSON-RPC on loopback; tools run in poll())
                        image/stb_image.h and image/stb_image_write.h export stb (vendored in image/stb/)
    vision/             saiga_vision: ba, pgo, imu, features, reconstruction, slam, recursive solvers, ceres/g2o glue
    cuda/               saiga_cuda: device helpers, reduce/scan/shfl, memory, streams, imageProcessing, thrust helpers
    opengl/             saiga_opengl: buffers, shaders, textures, rendering, window, imgui, ffmpeg, vr
                        window/StandaloneWindow: the base of OpenGL apps; its window backend is chosen at runtime
                        (WindowParameters::backend, `window_backend` in config.ini): glfw_Window or, headless, the
                        OffscreenWindow (egl/, an EGL pbuffer with ImGui and no window system), or without EGL (Windows) the
                        HiddenGLFWWindow (a GLFW window that is never shown, with the same ImGui)
                        pipeline.h: applyPipelineState sets the complete fixed-function state of a PipelineState
                        (core/rendering/PipelineState.h, API independent; Vulkan: PipelineInfo::setState),
                        GraphicsPipeline pairs it with a shader. Every pass of the renderers applies its state before
                        drawing or calling the application, and clears with clearFramebuffer; code that draws inside
                        someone else's pass changes a copy of currentPipelineState(). PipelineStateCache (off by
                        default) skips unchanged values; its validation mode finds code that bypasses it.
                        setClipSpace(ClipSpace::Vulkan) (glClipControl) renders in Vulkan's clip space with the
                        projections of a Camera with vulkanTransform: images have Vulkan's rows in memory
                        tests/test_opengl_renderers.cpp renders every renderer configuration; with
                        SAIGA_RENDERER_TEST_OUTPUT=<dir> it writes the images to compare them before and after a change
    vulkan/             saiga_vulkan: instance, swapchain, pipelines, memory, renderer (VulkanForwardRenderer with
                        ForwardRenderingParameters: MSAA, reversed-Z with an infinite far plane). Vulkan 1.3 core only
                        (instance/device API version 1.3 also on 1.4 drivers; extensions: only surface/swapchain and
                        debug_utils for validation) and uses its core
                        features: dynamic rendering (pipelines take RenderingFormats, no VkRenderPass),
                        synchronization2 (pipelineBarrier2/submit2), timeline semaphores
                        (VulkanBase::frameTimeline; VulkanBase::deferDelete for objects in use by frames in flight) and
                        shaderDemoteToHelperInvocation (shaders are compiled for SPIR-V 1.6: discard is a demote).
                        Renderer (VulkanRenderer): the base of VulkanForwardRenderer and RenderScene's frame (swap chain
                        or offscreen images in VulkanParameters::offscreenFormat, command buffers and submit, ImGui
                        frame, screenshots, selectSampleCount, the GPU sections).
                        Resources: buffer/ (vertex, index, uniform, staging, indirect), texture/ (Texture2D, 2DArray,
                        Cube, 3D with mipmaps and download; Sampler; RenderTarget for offscreen passes incl. MSAA and
                        multiple color attachments)
                        memory/: VulkanMemory allocates every buffer/image with VMA (submodules/VulkanMemoryAllocator,
                        own vk::Buffer per allocation, offset 0); mark_dynamic() moves a resource into VMA pools that
                        VMA's defragmentation compacts (full_defrag(), or incremental passes in update() per frame);
                        moved handles reach descriptors through DynamicDescriptorSet, images keep the layout tracked in
                        ImageData::layout (Texture::transitionImageLayout / setImageLayout)
                        world/: the game side of the world prototype (S19a; used by game_world, flight, racing): the
                        prototype height hook PrototypeTerrain (prototypeHeight, prototypeSurfaceHeight; S20) and the
                        generators, WorldSettings (GUI, game_* MCP tools), PrototypeForest (the trees as RenderScene
                        instance sets), PrototypeScene (attach to the scene, settings widgets, statistics; no GPU
                        resources), benchmark paths and samples WorldBenchmark
                        scene/RenderScene: the engine app's HDR frame (S17; shader/vulkan/scene/): fp16 MSAA scene pass
                        with reversed-Z D32, clay mode (identity tone map, hardware average resolve) or ACES (tone-map
                        weighted resolve), pre-exposure (sceneWritesLinear/preExposure), post + ImGui in one pass;
                        scene content (S18): MeshPool (meshes, materials, palette; vertex/index buffers sub-allocated
                        with VMA virtual blocks, SubAllocatedBuffer), InstanceBuffer (static instance sets per region,
                        dynamic instances), SceneCulling (offset table, GPU culling count/scan/write passes, indexed
                        indirect draws per pipeline class, flat shading), MeshImport (glTF via assimp, flat colors);
                        terrain (S20): TerrainAtlas (the GPU terrain set: int32 storage buffers of tiles, pyramid, page
                        table and grass masks, the upload queue with a per-frame byte cap), TerrainPass (CDLOD in
                        double with a 64 m frame anchor, skirts, the water plane, the sky, TerrainFrame), GrassPass
                        (chunk-relative grass from masks), TerrainTiles (pure tile / pyramid / mask generation);
                        prototype passes (S19a): setFoliageTexture (S36; the foliage texture and the S20 sky), the
                        PrototypeFoliage pipeline class of the trees, the depth pre-pass (off for the prototype) and the
                        SAMPLE_ZERO depth resolve; GPU sections with sub-sections (scene.terrain, scene.instances, ...);
                        lighting (S36, shader/vulkan/scene/lighting/): SceneLighting (the block) and its owner
                        SceneLightingSet (the set-1 layout of every scene pipeline incl. terrain, grass and sky, 1x1
                        dummies), lighting.glsl (shadeSurface, the one light model: clay's studio environment, which
                        reproduces S19a, or GGX in lux with an environment), BrdfLut, AmbientProvider / SkyAmbient
                        (the environment map's SH), Atmosphere (Hillaire LUTs, the sky with sun, moon and stars),
                        EnvironmentMap (capture, prefilter, SH; front / back), CloudLayer (the 2D clouds and their
                        shadows); the environment GPU section (atmosphere, environment map), scene.clouds; the debug
                        core: LightingDebug (debug views, previews with their reduction, pixel readbacks, the probe
                        frames), DebugLines (sections scene.debug, lighting.preview); S36a2 the sun and moon shadows:
                        CascadedShadows (fitting, staggering, the shared depth range, the D32 array), ShadowPipelineTable
                        (the classes' shadow pipelines, lighting/shadow_*.vert/.frag; a class without an entry casts
                        nothing), the shadow views of SceneCulling (view masks per work chunk), TerrainPass::
                        selectForShadow (terrain casters), HorizonShadows (the far terrain horizon map, horizon_*.comp);
                        receivers in lighting/shadows.glsl (the grass per vertex); sections shadow, shadow.cascades,
                        shadow.horizon; S36b the local lights: LocalLights (the CPU table, the frame's selection and
                        light buffer; addLight / addLights), LightClusters (light_cluster_bounds.comp, light_cluster.comp:
                        clustered light culling, section lights; clusters.glsl), ShadowAtlas (the cached two-layer D16
                        atlas of the shadowed local lights, faces as SceneCulling views, the shadow pipelines' and the
                        terrain's ATLAS_DEPTH variants, atlas_depth.frag, atlas_shadow.glsl; section shadow.atlas)
                        VulkanDevice/createVulkanDevice: instance, surface and device (VulkanRenderer and the app;
                        setHeadlessDeviceReuse: the tests' pool of headless devices)
                        window/StandaloneWindow: the base of Vulkan apps (VulkanExampleBase); the window backend is
                        chosen at runtime as in opengl/: GLFWWindow or HeadlessWindow (offscreen images, ImGui through
                        ImGuiHeadlessRenderer; core/imgui/imgui_headless_platform.h is shared with OpenGL)
    engine/             saiga_engine (namespace Saiga::Engine, needs the vulkan module): the game engine of
                        docs/engine_design.md; core/: EngineVersion, JobPool (parallelFor, the caller
                        participates), Rng (PCG32), Hash (FNV-1a 64), LogRing and SAIGA_LOG (Log.h), Profiler, StreamingPool (S20:
                        background generation jobs, drained on the main thread); terrain/ (S20): TerrainSystem (the
                        World's terrain: terrainHeight, visual streaming of the view camera's space, completeView,
                        readiness, the single send point to the sink), TerrainStore, TerrainGeneration; S22
                        terraforming: TerrainBrush and the integer BrushKernel, applyTerrainEdit (World::editTerrain),
                        TerrainEditStore (delta blocks, per-chunk sequence, World::terrainEdits / terrainSample),
                        TerrainDeltaCodec, TerrainMessages (TerrainModification), the visual path
                        (TerrainVisualPatch: merge at the send point and edited grass masks, TerrainDirtyQueue: the
                        patch step); tests/test_engine_terrain_edit.cpp, _net.cpp, _benchmark.cpp; S23a: HookGenerator
                        (an S20 height hook as a Generator: the games' prototype fixture, the tests' hooks);
                        content/ (S23a, docs/engine_design.md §12.1): GenerationPipeline (phase A: generators, space
                        list, plans of every space instance, sitesByChunk, plan hashes; phase B: prefab entities
                        against classes, components, meshes; heights with the composed site patches, chunkBase /
                        chunkPlacement, ContentSource for pool jobs, stampList / staticList), GeneratorList, Generator.h
                        / WorldPlan.h / ChunkContent.h (the interface), Prefab / PrefabStore (versioned prefab files),
                        ContentCache (WorldSetup::sharedContent), ContentHash (plan hashes), ContentHashFile
                        (checkContentHashes), ContentConvert (exact conversions), ChunkRecords (CREC state: placement
                        versions and, S23b, tombstones), chunkStampSystem (phase 7, host); S23b: Tombstones
                        (World::tombstone's checks and apply, tombstone-on-edit inside applyTerrainEdit,
                        tierOneSurface), ContentMessages (TombstoneModification),
                        ContentRenderSystem (the content render sets: one instance set per chunk, centres None / View
                        (EngineApp) / Players (ScenarioWorld), the re-snap queue); content/gen/ (integer only, linted): FixedMath.h, TrigTable,
                        IntNoise, SitePatch, ChunkGeneration, SectionHash, FlatGenerator (GeneratorId 0),
                        AuthoredSpaceGenerator; api/ComponentJson (the component-JSON path of spawn and stamping);
                        net/GenerationInfo (S26: the generation info of JoinAccept and Resync); tests/test_engine_content.cpp (+ content_lint.h, the fixture in
                        tests/content_lint_fixture/, content_test_generator.h, content_test_setup.h),
                        _content_benchmark.cpp; streaming/ (S24, §12.3; worlds with WorldConfig::streaming):
                        ChunkResidency (the resident chunk set around the streaming centres: pinned level-0 base tiles,
                        held placements, residency jobs on the streaming pool applied in phase 7, the step-start
                        refresh, synchronous mode for ScenarioWorld, completeResidency, dropAll), SimulationSet (the
                        Simulated tag by 32 m cells), Islands (IslandLimitEvent, IslandState), StreamingStats /
                        StreamingSettings, the systems streaming and simulation (phase 7), Arrivals.cpp
                        (requestArrival, checkArrival, arrivalProgress, the held state) and WorldStreaming.cpp (the
                        World's streaming members); tests/test_engine_streaming.cpp, _streaming_scenarios.cpp,
                        _streaming_benchmark.cpp (SAIGA_STREAMING_BENCHMARK=full: the local form), streaming_test_setup.h;
                        ecs/: Registry,
                        View, ComponentPool, visit rules and binary visitors (Visit.h, VisitBinary.h, FieldList.h),
                        VisitJson.h (JSON form of visit), SAIGA_LOG_E; components/: EngineIds.h, Transform, Parent,
                        LocalTransform, Player, Health, PlayerInput, CharacterIntent, Persistent, RenderInstance
                        (S19b: the mesh name an entity renders), SpaceLink (S21: an entrance to another space),
                        PendingTransition (S21: a requested transition; S24: an arrival, Replicate::Host),
                        StreamingAnchor / IslandState (S24), LightSource (S36b: an entity's local light,
                        sent by the extract: ExtractSystem::updateLights), WorldTime / Weather (S36: the time of day
                        (global) and the weather (space state) with WorldSetup::environment.enabled, else plain
                        components); lighting/ (S36): EnvironmentSetup, EnvironmentFile (environment.json), the
                        celestial model (Celestial), worldTimeSystem (addEnvironmentSystems), the validated writes
                        setWorldTime / setWeather (EnvironmentWrites), the environment extract (EnvironmentExtract),
                        set_lighting's key table and validateLightingValues (LightingKeys; tool, tab and file);
                        world/: World (also the mesh and material name
                        tables, WorldSetup::renderAssets, setRenderSink; S20: SpaceTemplate and
                        WorldConfig::spaceTemplates, terrainHeight, spaceSeed; S21: several spaces incl.
                        SpaceTemplate::enclosed, moveToSpace and the space primitive setSubtreeSpace/teleportTo, a
                        space-state entity per space, requestTransition, spaceReadiness; SpaceTransitionSystem: phase 1,
                        completes the PendingTransitions; tests/test_engine_spaces.cpp)
                        (step loop, the setup's context WorldSetup::setContext / World::context, phases, SystemList,
                        commands, StructuralLog, EventQueues, transformSystem, SpatialGrid and spatialGridSystem for
                        World::queryRadius, Events.h); gameplay/: GameObject,
                        Behaviour (+ World::spawn), ClassList, TickPolicy, the gameplay systems (Systems.h; S21: the
                        optional spaceLinkSystem, SpaceLinkSystem.cpp); api/:
                        WorldApi (the world as JSON by names, for MCP, scenarios, editors; snapshots); tools/:
                        registerWorldTools (MCP world tools), ScenarioWorld/ScenarioList/ScenarioRunner
                        (scenario tests, §17.2); save/: World::save/tryLoad (SaveFormat.h, SchemaTable,
                        MigrationList, SaveDecoder, diffSaves; S21: the optional sections of kSaveSections, SPCE with the
                        space table and its prefix rule; S25: the world save directory: SaveFileSystem (disk) and
                        MemorySaveFileSystem (the crash model of the tests), SaveDirectoryFormat.h / ChunkRecordCodec
                        (records RTER / RREC / RGRP / RSPC, the index CIDX, header, world-state file), ChunkStore (stored
                        entity groups, unloaded spaces, the index of the newest commit), SaveDirectory (open, commits on
                        the SaveIoQueue lane, compaction, retention), EntityGroup (the group form) and GroupLoader on the
                        StagingCore shared with the world loader, EntityUnloading (the step-start hook and the phase-7
                        systems of Systems.h: space_load, chunk_entities, space_unload), EGRP; tests/test_engine_unload.cpp,
                        _save_directory.cpp, _save_crash.cpp, _save_benchmark.cpp, tests/unload_test_setup.h); data/: data files (DataTypeList, DataStore, DataRef,
                        readDataFile); net/: Transport, LoopbackTransport, Fragmenter, MessageList (network messages,
                        World::send), the replication (S16: Replication with the host's NetId table and the client
                        handlers, NetVisitors.h, EngineMessages, NetOwner.h, HostClock, InterpolationSystem;
                        tests/test_engine_replication.cpp; the host + client scenario variants in ScenarioWorld;
                        S26 the world over the network: the handshake JoinRequest / JoinReject / JoinAccept and Resync
                        v2 (JoinMessages, JoinCheck, kProtocolVersion; admission gates the peer lists), NetSchema (the
                        wire's registration, section hashes), GenerationInfo (the adoption of the host's pins,
                        World::replaceGeneration), ChunkMessages (ChunkInterest, ChunkSnapshot, ChunkIndex),
                        ClientChunkSync (per chunk Pending / Synced, the queue, re-requests, unload drops, the `all`
                        subscription of worlds without streaming), HostChunkSubscriptions (subscriptions, the budgeted
                        snapshot queue, per-peer filtering of the live modifications), WireFormat.h;
                        tests/test_engine_world_net.cpp, tests/world_net_test_setup.h);
                        sync/ (S19b, depends on saiga_vulkan): RenderSink, VulkanRenderSink (over RenderScene),
                        RecordingRenderSink (tests, every scenario world), ExtractSystem (render instances from the
                        structural log, pose pairs from Transform or RenderPose, the catch-up between steps,
                        interpolatedPose; tests/test_engine_extract.cpp); app/: EngineApp (the engine's frame loop:
                        window or headless, RenderScene and the World's VulkanRenderSink, the view camera
                        setViewCamera, the render hooks init/shutdown/beforeRender/gui, MCP,
                        input map, frame/GPU profiles; S19c: the debug UI DebugUi with its tabs F1-F5 (S36: F5 the
                        LightingTab; game tabs F6-F10), DebugConsole whose
                        commands are the MCP tools, McpCallLog of every tool call, set_ui_tab; S21: LoadingScreen, the
                        load of a view space switch, EngineApp::spaceLoading), engineMain and the command line of engine
                        games, InputMap, the app tools (quit, app_status, set_input; screenshot, set_camera (S21: space)
                        /move_camera with the camera hold, list_scenarios/run_scenario); S36: the app's
                        EnvironmentSettings, LightingTools (get_lighting, set_time_of_day, set_weather, set_lighting;
                        S36a1b: set_lighting_debug, capture_lighting, probe_lighting, screenshot lighting_view /
                        overlays)
    export.h            SAIGA_<MODULE>_API dll export/import macros
    *.h.in              Configured into build/include/saiga/ (saiga_buildconfig.h, saiga_modules.h, git sha)
  internal/             stb_image wrappers, precompiled-header lists
docs/                   engine_design.md: design and implementation stages of the game engine;
                        stages/Sxx_<name>.plan.md: one plan per stage (template: S05_math_additions.plan.md)
shader/                 GLSL shaders (OpenGL / Vulkan)
data/                   Sample assets (textures, models); test goldens in data/tests/golden/, benchmark baselines per
                        GPU in data/tests/benchmarks.json; the glTF files of the mesh import tests in data/tests/scene/
                        (generated by its make_gltf.py); engine test data in data/tests/engine/ (data files, the
                        golden save save_v1.bin/.json and the engine schema schema.json, R-SAVE-1): after a deliberate
                        version change regenerate only the schema (SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_save
                        --gtest_filter='SaveFormat.EngineSchemaIsCommitted'); the golden save is rewritten only for a new
                        save format version (it must keep loading otherwise); S23a: the test prefabs in prefabs/, the
                        committed content hashes content_hashes.json (see Tests) and the golden save save_s23a.bin
                        (GENR, CREC; rewritten with SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_content
                        --gtest_filter='ContentSave.CommittedSaveLoads' only for a new GENR / CREC format version);
                        S23b: save_s23b.bin (CREC format 2 with tombstones; the same way with
                        --gtest_filter='ChunkRecords.CommittedSaveLoads'); S25: the save directory save_dir_s25/ (a stored
                        group, edited chunks, an unloaded space; rewritten only for a new directory format version with
                        SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_save_directory
                        --gtest_filter='SaveDirectory.CommittedDirectoryOpens', in a dedicated commit)
samples/<module>/       One sample executable per sample_<module>_*.cpp/.cu
samples/games/          Games on saiga_engine, one executable per samples/games/<name>/ (game_<name>, built with samples,
                        the engine and GLFW): the world game world/ (the world prototype on
                        EngineApp: game_world, --benchmark, --lamps (S36b: Lamps.*), --scenarios (the lamp policy), MCP of .mcp.json), the tech demos flight/, platformer/, racing/
                        (on EngineApp: simulation in a World, <Game>World.* and <Game>Render.*; data in data/<game>/;
                        S23c: flight's integer OverworldGenerator in flight/content/ (linted), its prefab
                        data/flight/prefabs/ruin.json and hash file data/flight/content_hashes.json) and
                        their shared common/, compiled into each game (docs/engine_design.md §18.4). Not tested by tests/
                        (the tests test the engine; the games' scenarios run with --scenarios)
tests/                  test_<module>_*.cpp/.cu + CMakeLists.txt (explicit saiga_test(...) list); test_process_setup.cpp
                        (linked into every test: no core dumps, Vulkan device reuse; see Tests, "Test speed")
submodules/             Vendored git submodules: eigen, glog, googletest, glfw, assimp, zlib, zstd, libpng,
                        libtiff, libzip, libavif, aom, SVT-AV1, TinyTIFF, glslang (built from submodules/CMakeLists.txt),
                        VulkanMemoryAllocator (header only, included by saiga_vulkan), plus tiny-eigen and tiny-glog, which are plain files maintained in this repository
.mcp.json               MCP server "saiga-world" (http://127.0.0.1:8765/mcp) of the running game_world
scripts/setup_worktree.sh   Submodule init for a git worktree (borrows objects from the main checkout)
.github/workflows/tests.yml The CI: Ubuntu 24.04 with GCC and with Clang, CPU-only build (OpenGL on Mesa llvmpipe, Vulkan on
                        Mesa lavapipe) + ctest; Windows MSVC (Ninja + ccache, shared libraries, Core + Vulkan + Engine, no samples;
                        OpenGL and Vision are not built there; the Vulkan tests skip without a driver) + ctest. The
                        Ubuntu jobs fail on any skipped gtest case; the Clang job also builds the samples (incl. the
                        tech demo games, not run) and compiles the content code with SAIGA_CONTENT_UBSAN=ON. Changes of
                        only docs/ and Markdown files skip it
travis/, .travis.yml    Legacy Travis CI — unused
```

## Setup

```shell
git submodule update --init --recursive      # in a git worktree: scripts/setup_worktree.sh
```

System packages (what CI installs on Ubuntu 24.04):

```shell
sudo apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build ccache \
  xorg-dev libwayland-dev libxkbcommon-dev wayland-protocols extra-cmake-modules \
  libgl-dev libegl-dev libegl-mesa0 libgl1-mesa-dri \
  libvulkan-dev mesa-vulkan-drivers vulkan-validationlayers locales
sudo locale-gen de_DE.UTF-8   # Ini.IndependentOfCLocale, Json.NumbersIndependentOfCLocale and
                              # Prefab.IndependentOfCLocale skip without a decimal-comma locale (CI fails on skips);
                              # without sudo: localedef -i de_DE -f UTF-8 <dir>/de_DE.UTF-8 and LOCPATH=<dir>
```

The dependency list in `README.md` is partly outdated (e.g. GLM is no longer used — Eigen is the
math backend); most image/compression libraries come from `submodules/`.

## Build (Linux)

The canonical build is the CI recipe — no GPU, no samples (the OpenGL and Vulkan modules are built and tested on
Mesa's software renderers). Use it unless the change needs CUDA:

```shell
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSAIGA_BUILD_SAMPLES=OFF \
  -DSAIGA_ARCHNATIVE=OFF \
  -DSAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON
cmake --build build --parallel -- -k 0
```

- `SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON` writes the source dir into the generated headers so the
  tests find `shader/` and `data/` from the build dir.
- Without Ninja, drop `-G Ninja` and `-- -k 0` (Makefiles: `cmake --build build --parallel $(nproc)`).
- **CMake >= 4.0** refuses the `cmake_minimum_required(VERSION 3.0)` of `submodules/glog`; add
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the configure. (CI uses CMake 3.28 and does not need it.)
- **GCC 15** fails on warnings in `submodules/assimp`; add `-DASSIMP_WARNINGS_AS_ERRORS=OFF` to the configure. (CI uses
  GCC 13 and does not need it.)
- **Eigen backend check** (math code must also compile with real Eigen, R-MATH-2 of `docs/engine_design.md`; not in CI):
  a separate build directory with only the core module:
  ```shell
  cmake -S . -B build-eigen -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSAIGA_ARCHNATIVE=OFF \
    -DSAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON -DSAIGA_BUILD_SAMPLES=OFF \
    -DSAIGA_WITH_TINY_EIGEN=OFF -DSAIGA_WITH_TINY_GLOG=OFF \
    -DSAIGA_MODULE_VULKAN=OFF -DSAIGA_MODULE_OPENGL=OFF -DSAIGA_MODULE_CUDA=OFF -DSAIGA_MODULE_VISION=OFF
  cmake --build build-eigen --parallel --target test_core_math_functions
  (cd build-eigen && ctest -R test_core_math_functions --output-on-failure)
  ```
  (`test_core_vectorization` and `test_core_geometry` fail with Eigen on `master` too.)
- The CUDA module needs a GPU / CUDA toolkit and is **not covered by CI**. The Vulkan module is covered
  through its headless tests (no window, swap chain or samples); it needs the Vulkan headers. glslang comes from `submodules/glslang`, is linked
  statically into `saiga_vulkan` and needs CMake >= 3.22.1 (the module is skipped with a warning below that). The Vulkan Memory
  Allocator (header only) comes from `submodules/VulkanMemoryAllocator`; no saiga header includes it (the handles are forward declared). The OpenGL module is covered, but only through its
  headless tests (an EGL context or the headless OffscreenWindow with ImGui; no window system or samples). For a CUDA build leave `SAIGA_MODULE_CUDA=ON` and pin the archs with
  `-DSAIGA_CUDA_ARCH="86;89"` (dotless, e.g. `75;80;86;89+PTX`), otherwise the installed GPU is
  autodetected.
- The first build also compiles the submodules (their tests are off) and takes a while; later builds
  are incremental.

### Build options (CMake)

| Option | Default | Effect |
|--------|---------|--------|
| `SAIGA_MODULE_CORE` / `_VISION` / `_CUDA` / `_OPENGL` / `_VULKAN` / `_ENGINE` | ON | Build the module *if its dependencies are found* (silently skipped otherwise) |
| `SAIGA_BUILD_TESTS` | ON | Build the gtest executables + register them with ctest |
| `SAIGA_BUILD_SAMPLES` | ON | Build `samples/` |
| `SAIGA_BUILD_SHARED` | ON | Shared libraries (the export macros in `export.h` matter) |
| `SAIGA_ARCHNATIVE` | ON | `-march=native` (CI turns it OFF) |
| `SAIGA_ASSERTS` | ON | Enable `SAIGA_ASSERT` |
| `SAIGA_WARNINGS_AS_ERRORS` | OFF | Compiler and linker warnings in the saiga code (libraries, tests, samples; not the submodules) are errors (`-Werror` / `/WX`); all three CI jobs turn it ON, so a change must build without warnings on GCC, Clang and MSVC |
| `SAIGA_DEBUG_ASAN` / `_MSAN` / `_TSAN` | OFF | Sanitizers (TSAN excludes ASAN/MSAN) |
| `SAIGA_CUDA_ARCH` | "" | CUDA archs; empty = autodetect |
| `SAIGA_WITH_<LIB>` | mostly ON | Use an optional library if found (ASSIMP, LIBZIP, FFMPEG, FREETYPE, LIBTIFF, YAMLCPP, OPENAL, OPENMESH, G2O, CHOLMOD, ...) |
| `SAIGA_CONTENT_UBSAN` | OFF | Trapping UBSan (`-fsanitize=signed-integer-overflow,shift,integer-divide-by-zero,bounds,null,alignment,bool,enum,return,unreachable -fsanitize-trap=all`, no runtime) on the content generation sources (`src/saiga/engine/content/`) and `test_engine_content` (S23a); Clang and GCC (an error with MSVC); the Clang CI job turns it ON; undefined behaviour aborts the test (SIGILL) |
| `SAIGA_WITH_JOLT` | OFF | Download Jolt Physics v5.6.0 with FetchContent (`cmake/Dependencies_jolt.cmake`, not a submodule) and build the target `Jolt` (static, PIC) and `sample_vulkan_jolt`; no saiga library links it. Jolt's own options (e.g. `CROSS_PLATFORM_DETERMINISTIC`) can be set with `-D` |
| `SAIGA_WITH_TINY_EIGEN` / `_TINY_GLOG` | ON | Use the tiny replacements instead of Eigen / glog (disables vision + some tests) |

The configure prints `Module enabled: <Name>` and `Test enabled: <name>` lines — check them to see
what is actually built. With the CI options on a plain Ubuntu 24.04, that is **Core + Opengl + Vulkan + Engine** and
95 tests (incl. 10 compile-fail cases of the ECS field rules, `tests/compile_fail/`; a test binary whose tests all skip, e.g. the Vulkan tests without a driver, is reported by ctest as skipped). The Vision module and its tests need `-DSAIGA_WITH_TINY_EIGEN=OFF` (real Eigen); the Ceres-based vision tests (`test_vision_bundle_adjustment*`, `test_vision_pose_*`,
`test_vision_imu_decoupled_solver`) are only built when Ceres is installed.

## Tests

```shell
cd build && ctest --output-on-failure --parallel 4     # what CI runs
ctest -R test_core_ini --output-on-failure             # one test
./build/tests/test_core_ini --gtest_filter='Ini.*'    # the binary directly (binaries are in build/tests/)
```

- Each `tests/test_*.cpp/.cu` is its own executable, registered in `tests/CMakeLists.txt` with
  `saiga_test(<file> [extra libs])` (links `saiga_core` + gtest; vision tests pass `"saiga_vision"`,
  CUDA tests `"saiga_cuda"`, engine tests (`test_engine_<area>.cpp`) `"saiga_engine"` inside
  `if (MODULE_ENGINE)`; there are no game tests, the games are samples). Compile-time rules are tested with
  compile-fail cases (`tests/compile_fail/`, targets excluded from the build, built by ctest one at a time with
  `tests/compile_fail/expect_build_failure.cmake`: the build must fail and print the `static_assert` message). Build before running ctest (as CI does): on a stale build
  the first compile-fail case rebuilds the libraries while other tests run. **Tests are not globbed** — a new test file must be added there,
  inside the matching `MODULE_*` / dependency guard (e.g. the bundle-adjustment tests need
  `CERES_FOUND`).
- Test style: `using namespace Saiga;`, plain `TEST(Suite, Name)`, `EXPECT_*`/`ASSERT_*`; a
  regression test carries a short comment describing the bug it guards against.
- **Test speed**: a test binary should take about a second, at most a few (budget below). Every test executable links
  `tests/test_process_setup.cpp` (an object library in `tests/CMakeLists.txt`): no core dumps (a soft `RLIMIT_CORE` of
  1 byte; each `EXPECT_DEATH` child aborts, and its dump to systemd-coredump / apport took 0.1-0.3 s per death test;
  `SAIGA_TEST_CORE_DUMPS=1` keeps the shell's limit) and, in executables that link `saiga_vulkan`, the reuse of
  headless Vulkan devices (`Vulkan::setHeadlessDeviceReuse` in `VulkanDevice.h`: the instance and the logical device
  of a destroyed `HeadlessWindow` renderer stay in a pool, everything on the device is created per renderer; an
  instance and a device cost 0.25 s per test on the NVIDIA driver; leaked objects are reported by the validation layer
  when the pool is released after the last test; `SAIGA_TEST_NEW_DEVICES=1` creates a device per renderer). On a
  reused device the pipeline objects are reused as well (`VulkanBase::reusePipelines`: a destroyed pipeline returns to
  a pool on the device, keyed by the content of its create info with the shaders by their SPIR-V; a RenderScene's ~25
  pipelines cost ~55 ms on lavapipe, its construction now ~15 ms). A pooled pipeline creates no shader module (the key
  needs only the SPIR-V hashes; `vkCreateShaderModule` validates the SPIR-V with the validation layer, ~0.4 ms per
  lighting shader), and RenderScene creates its lighting subsystems and draw pipelines on first use (a clay scene
  without an environment, lights or shadows creates none of them; S36 review round 2). The process also caches the
  SPIR-V of compiled shaders (keyed by the file contents incl. includes) and glslang's tables.
  Every test still gets its own RenderScene / EngineApp / World (no suite-wide renderer with a reset between tests:
  the pipeline pool makes a new scene cheap, and test independence stays structural). What is shared between tests is
  pure, deterministic data, opt-in per test:
  - `TerrainTestHooks::fillTestTerrain(hook, seed, ...)` caches the pyramid and tiles of a height hook per process.
  - Content: a `ContentCache` shared by several worlds (`WorldSetup::sharedContent`) generates their plans, chunks and
    terrain pyramids once (the cache keeps a space's pyramid per plan; `ContentCache::grid`). The content tests share
    one per version (`sharedTestCache` in `test_engine_content.cpp`), the content scenarios one for all their variants
    (as `runScenarios` does), `test_engine_terrain_render` one for its hook worlds, and `checkContentHashes(..., cache)`
    one for the committed file and the small hash-file test. Only tests that do not count generations, cache hits or
    in-flight jobs opt in (their assertions must not depend on what an earlier test generated).
  Rules for new tests:
  - No fixed sleeps or wall-clock waits where a condition can be polled: loop until the condition, with a deadline
    only as the bound. MCP deadline tests use a short timeout with a large margin (`McpServer::answerMarginFraction`,
    `EngineAppSetup::mcpAnswerMarginFraction`; e.g. 0.6 s and 0.5: the tool answers after 0.3 s, 0.3 s before the
    timeout); keep the absolute margin >= 0.2 s (CI is loaded). A time constant that a test would wait out becomes a
    setting the test shortens (`EngineAppSetup::statisticsWarmupSeconds`, `ShaderModule::autoReloadInterval`).
  - Rendering on lavapipe costs CPU (its vertex stage is single-threaded, sub-pixel triangles are expensive; a CDLOD
    terrain frame is ~100k triangles, 50 ms): small targets (the fixtures' 64 x 48 / 96 x 64), few frames
    (`RenderScene::statistics()` is the newest frame's: one frame after a change, not three; a terrain tile is
    GPU-resident from the frame its copy is recorded, so two frames after `renderUntilUploaded`), small scenes
    (instances, LOD 0 distances, grass radius, terrain tiles: `fillTestTerrain(..., range0)`) unless size is what is
    tested. A test that reads the terrain selection, vertex probes, streaming or loading state but no terrain pixels
    switches the terrain draws off (`PrototypeSettings::terrain` / `water`; the selection, the node data and the
    uploads run anyway: `NoTerrainDraws` in `test_vulkan_terrain.cpp`, `noTerrainDraws` in `test_engine_app.cpp`).
    Goldens keep their setup (§17.6). Record the reason for a reduction in the test's comment.
  - Death tests are cheap (fork, no dump); `GTEST_FLAG_SET(death_test_style, "threadsafe")` re-runs the binary up
    to the death test (incl. its Vulkan device): only where the forked child would deadlock.
  - Measure with `./build/tests/<binary> --gtest_print_time=1` (or the gtest XML of `GTEST_OUTPUT=xml:<dir>/`) on
    lavapipe; a new test case above 0.5 s needs a reason in its comment. Budget per binary, alone on lavapipe (24
    threads, a RelWithDebInfo build with Makefiles; all binaries: 207 s before the first pass, 89 s after it, now 54 s):
    `test_vulkan_world` 9.5 s (prototype scenes with terrain and trees; S36: their frames shade with the lighting's
    `shadeSurface` and per-vertex light terms, ~1.5x lavapipe's frame cost of S19a's shaders), `test_engine_app` 8.5 s
    (75 tests; the MCP deadline tests wait ~3 s of real time; S24: the streaming app tests 0.6 s, an app whose complete
    point and arrival generate the residency of a terrain space; measured 8.1-8.3 s alone after S26, 6.9 s before S24), `test_engine_terrain_render` 6 s (goldens at 512 x 512 in
    the S36 golden mode: cascades at 1024, the horizon map, ACES), `test_engine_content` 6 s (the test generator's
    pyramids and the committed hashes), `test_vulkan_scene` 5.5 s, `test_vulkan_terrain` 5 s, the S36 lighting
    binaries `test_vulkan_shadows` 4 s (39 tests: cascade and horizon maps rendered per test), `test_engine_lighting_tools`
    4.7 s (34 tests: an EngineApp per test, the tab tests drive ImGui; measured 4.6-4.75 s after S36's UI rounds, the
    earlier 3.5 s line was stale), `test_engine_lighting_render` 3 s (goldens),
    `test_vulkan_lighting` 2.5 s (38 tests: the atmosphere and environment-map tests), S25's `test_engine_save_crash` 1 s
    (the crash enumeration: every operation of a commit in four image kinds), `test_engine_save_directory` 1 s and
    `test_engine_unload` 0.5 s (memory file system, flat worlds), S26's `test_engine_world_net` 1.5 s (a host and up
    to 8 clients per test over loopback; the join data of 50 edited 1 m chunks), every other binary <= 2 s. These
    are ceilings: a change that makes a binary slower than its budget makes it
    faster again or explains why. `ctest --parallel 4` on lavapipe: 14 s (before: 60 s, 30 s after the first pass); the
    10 compile-fail cases run one at a time (one build directory) and cost ~0.75 s each with Makefiles (most of it the
    dependency check; less with Ninja).
- Shared helpers: `tests/compare_numbers.h`, `tests/numeric_derivative.h` (numeric Jacobians for the
  vision derivative tests), `tests/opengl_test_context.h` (the `OpenGLTest` fixture),
  `tests/vulkan_test_context.h` (the `VulkanTest` fixture), `tests/render_scene_test_context.h` (`RenderSceneTest`: a
  `RenderScene` per test, also in the lighting binaries), `tests/golden_image.h` (`compareGolden`, golden images),
  `tests/mcp_test_helper.h` (`McpRequest`, the `Poller` that runs `McpServer::poll` in the background for
  `handleMessage` tests of MCP tools).
- **OpenGL tests** (`test_opengl_*`, built when OpenGL + EGL are found) need no window or X server: the
  `OpenGLTest` fixture creates a headless EGL context (`CreateEGLContext()`, from the first EGL device —
  e.g. the NVIDIA GPU — else Mesa) and **skips** every test if there is no EGL display. Derive the
  suite from `OpenGLTest` and use `TEST_F`; it fails a test that leaves a GL error behind. The core
  profile needs a bound VAO to draw. To run them on Mesa's software renderer as CI does:
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 ctest -R test_opengl`.
- **Window tests**: `test_opengl_window` and `test_vulkan_standalone_window` run a `StandaloneWindow` with the
  headless backend (ImGui drawn into the frame, clicks through ImGui's input queue, resizing). They create a window per
  test instead of using the `OpenGLTest` / `VulkanTest` fixtures; the OpenGL one checks for GL errors in `TearDown` as
  the fixture does. ImGui shows a new window
  from its second frame on, and spreads a mouse move and a button change of one frame over two frames, so they send one
  input event per frame.
- **Vulkan tests** (`test_vulkan_*`) need no window system either: the `VulkanTest` fixture creates a
  `Vulkan::HeadlessWindow` with the forward renderer (no surface or swap chain, the frames are rendered into offscreen
  images; `RenderFrame()` returns one) and **skips** every test without a Vulkan 1.3 driver. Derive the suite from
  `VulkanTest` and use `TEST_F`. `test_vulkan_scene` (RenderScene's HDR frame and, S18, its scene content and culling
  against a CPU reference) and the world goldens create a
  `HeadlessWindow` with a `RenderScene` per test instead (`tests/render_scene_test_context.h`: `RenderSceneTest`, golden
  mode `forceFullUpdates` so frames compare; `goldenSceneParameters`, the goldens' lighting: golden mode, the
  cascades at 1024 and the horizon map at 256 texels). The validation layer is used if installed (`vulkan-validationlayers`); a validation
  error fails an assertion and aborts the test. The best device is used (e.g. the NVIDIA GPU); to run them on Mesa's
  software driver lavapipe as CI does: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R test_vulkan`.
- **Golden images** (`tests/golden_image.h`, `compareGolden`; PNGs in `data/tests/golden/`): a render test compares its
  image with a golden; the comparison (and the committed counts of the world goldens) gates only on lavapipe (device
  name `llvmpipe`), other devices print the metrics. A gating failure writes `golden_failed/<name>_actual.png` and
  `_diff.png` (the diff only when a readable golden of the same size exists) into the working directory (`build/tests/` under ctest; CI uploads them as the artifact `golden-failed-<job>`, e.g. `golden-failed-ubuntu`, and
  prints the metrics of every run in the step "Golden image metrics"). The committed goldens are rendered by a local
  lavapipe (Mesa with LLVM 21.1.8); CI's lavapipe (Mesa 25.2.8, LLVM 20.1.2) differs slightly and passes within the
  tolerance (run 43754502053: `world_forest` mean 0.14/255, 0.19% of the pixels > 16/255; a Mesa update on the CI
  runner may need the recipe below). Create or update goldens only on lavapipe, in a dedicated commit:
  `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_terrain_render
  --gtest_filter='VulkanWorldGolden.*'` (the counts are not checked in this run; the goldens are written to the source tree, gray images as 8-bit grayscale
  PNGs); then copy the printed `trees per lod` counts into the `ExpectCounts` calls of `tests/test_engine_terrain_render.cpp`.
  The tone-map golden `hdr_tonemap` (S17) the same way with `./build/tests/test_vulkan_scene
  --gtest_filter='VulkanScene.AcesGolden'`, the imported-mesh golden `scene_flat_box` (S18) with
  `--gtest_filter='VulkanSceneContent.ImportedMeshGolden'`, the terraformed crater `terrain_edit_crater` (S22) with
  `./build/tests/test_engine_terrain_render --gtest_filter='TerrainRenderTest.EditCraterGolden'`, the generated content
  `generated_site` (S23b: the content test generator's site with tier-1 instances, static parts and a tombstone; its
  count delta and set check run on every device) with `--gtest_filter='TerrainRenderTest.GeneratedSite'`. These five
  clay goldens use clay's studio environment (S36), so the atmosphere, the clouds and the exposure do not move them
  (S36a2: they show the studio sun's cascade and horizon shadows; the tree counts are unchanged by shadows). The
  lighting goldens (S36a1c: `lighting_noon`, `lighting_sunset`, `lighting_night`, `lighting_clouds`; ACES, a fixed
  EV100, golden mode with the sun and moon shadows, `goldenSceneParameters`; S36b: `lighting_local_lights` and
  `lighting_enclosed_lights`, the local lights with the shadow atlas) with
  `./build/tests/test_engine_lighting_render --gtest_filter='LightingGolden.*'`.
  If CI fails after a Mesa update, take the actual images of the failed run (the artifact exists only for a run
  whose comparison failed; the actual images are saved like goldens):
  `gh run download <run-id> -n golden-failed-ubuntu -D /tmp/golden`, then
  `for f in /tmp/golden/*_actual.png; do cp "$f" data/tests/golden/$(basename "$f" _actual.png).png; done`, and copy
  the counts from the `trees per lod` lines of the step "Golden image metrics" into the `ExpectCounts` calls of
  `tests/test_engine_terrain_render.cpp`; then update the Mesa/LLVM versions named here and in `docs/engine_design.md` §17.6
  (the goldens now come from CI's lavapipe). See `docs/engine_design.md` §17.6.
- **Content hashes** (S23a, `docs/engine_design.md` §12.1): `ContentHash.CommittedHashesReproduce`
  (`test_engine_content`) regenerates the engine test generator's content (`tests/content_test_generator.h`, the setup
  of `tests/content_test_setup.h`) and compares it with `data/tests/engine/content_hashes.json` (plan hashes, chunk
  sections, a digest, the toolkit); every CI compiler must reproduce it (the Clang job with `SAIGA_CONTENT_UBSAN=ON`).
  A difference is a determinism bug, not a reason to update. Only for a new generator version (a new entry) or a
  deliberate format change: `SAIGA_UPDATE_CONTENT_HASHES=1 ./build/tests/test_engine_content
  --gtest_filter='ContentHash.*'` rewrites the file in the source tree and fails the run; commit it in a dedicated
  commit whose message gives the reason. An entry of a released version is never regenerated. Generator code (the
  files under `src/saiga/engine/content/gen/`, the interface headers, `tests/content_test_generator.h`, games'
  `content/` directories) is integer only: `ContentLint.*` refuses floating point, libm, `long`, `size_t`, unordered
  containers, sequential RNGs and the template header there.
  The games keep their own hash file (S23c): flight's `data/flight/content_hashes.json` (its `OverworldGenerator` of
  `samples/games/flight/content/`) is checked by the local game scenario `flight_content_hashes`
  (`./build/samples/games/game_flight --scenarios flight_content_hashes`; not run by CI). The same rule applies; the
  update (with `SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON`, it writes only the source tree's file) is
  `SAIGA_UPDATE_CONTENT_HASHES=1 ./build/samples/games/game_flight --scenarios flight_content_hashes`, again in a
  dedicated commit with the reason.
- **Benchmarks** (local, GPU and display; `docs/engine_design.md` §17.7). Baselines are measured in a separate release
  build without asserts (§17.10) and without `-march=native` (§17.7):
  ```shell
  cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DSAIGA_ASSERTS=OFF -DSAIGA_ARCHNATIVE=OFF \
    -DSAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON -DSAIGA_BUILD_TESTS=OFF -DSAIGA_BUILD_SAMPLES=ON
  # CMake >= 4: add -DCMAKE_POLICY_VERSION_MINIMUM=3.5; GCC 15: add -DASSIMP_WARNINGS_AS_ERRORS=OFF (see above)
  cmake --build build-bench --target game_world sample_vulkan_scene_benchmark --parallel
  (cd build-bench/samples/games && ./game_world --benchmark flyover)   # and --benchmark forest, --benchmark terraform
  (cd build-bench/samples/games && ./game_world --benchmark night)     # S36b: 400 lamps at 22:00, reported only
  (cd build-bench/samples/vulkan && ./sample_vulkan_scene_benchmark)     # culling benchmark (S18), headless
  # S24: the streaming benchmark (headless, no window; configure build-bench with -DSAIGA_BUILD_TESTS=ON for it):
  cmake --build build-bench --target test_engine_streaming_benchmark --parallel
  SAIGA_STREAMING_BENCHMARK=full ./build-bench/tests/test_engine_streaming_benchmark      # full-off: without streaming
  # S25: the save benchmark (memory file system; freeze, unload / reload, save(), open), reported only:
  cmake --build build-bench --target test_engine_save_benchmark --parallel
  SAIGA_SAVE_BENCHMARK=full ./build-bench/tests/test_engine_save_benchmark
  ```
  It flies a fixed path at 1920 x 1080 (validation, vsync, ImGui, defragmentation and MCP off; `SAIGA_SCREENSHOT` is
  ignored) and writes
  `benchmark_<name>.json` to the working directory (here `build-bench/samples/games/`); the exit code is non-zero if it
  did not finish, could not write the result, or the rendered size changed or a frame stalled (e.g. a minimized window)
  during the measured frames. The JSON
  records the build (`build: {type, asserts, archnative}`: the CMake configuration, `SAIGA_ASSERTS` and whether
  `-march=native` was really used) and the rendered size. Compare with the baseline of the GPU in
  `data/tests/benchmarks.json` (run each benchmark twice: the frame interval `cpu_ms` median varies below 1% between
  runs, the GPU pass medians about 1%, passes below 0.1 ms up to 5%); a regression of more than 5% must be explained
  (R-TEST-2), and a new baseline is committed when a change is accepted. Since S19a the `gpu_ms` keys (S03's names) come
  from RenderScene's sections: `trees` = `scene.instances`, `upload + cull` = `transfer` + `cull`, `terrain + water` =
  `depth.terrain` + `scene.terrain`, `grass`, `sky`; `cpu_work_ms` is the app's recording and submit (not comparable
  with the S03 value). Since S20 the benchmark world is `game_world`'s 32 km space at 1 m (the trees in the 8 km area
  at its centre, `kPrototypeAreaOrigin`; until S36 in its corner `[0, 8192]^2`: older baselines are not comparable);
  the terrain is completed in every frame (`completeView`), its time is excluded from the frame interval and reported
  as `terrain_stream_ms` with a `terrain` block; `--base-spacing 2` runs the same benchmark on a 2 m template (to
  separate code changes from the 1 m content). `terraform` (S22) edits the terrain once per frame from a static camera
  (one step per frame, no per-frame `completeView`; the game's `TerraformBenchmark`) and adds the terraform keys
  (`terrain_edit_ms`, `terrain_visual_ms`, `edit_bytes_per_second`, ...; it fails on a broken assertion, e.g. a hook read
  or a tile send in the measured frames); `tests/test_engine_terrain_benchmark.cpp` is its headless, non-gating
  counterpart. The culling benchmark (S18) renders the
  prototype's 200k trees as flat instances through RenderScene over the flyover path (headless, 1920 x 1080) and writes
  `benchmark_scene_cull.json` (the `cull` section and the instance draw time; not in `benchmarks.json`).

## Game Engine

The game engine (`src/saiga/engine/`, from stage S04; the world renderer in `src/saiga/vulkan/scene/` and `src/saiga/vulkan/world/` and the games built on it) is
designed in `docs/engine_design.md`. Read its Section 1 (Rules) before changing engine or game code.

Tech demo games (§18.4): `game_flight`, `game_platformer`, `game_racing` in `samples/games/` (with samples ON), on
`EngineApp` (the engine app's command line, no `config.ini`). `--demo` plays by itself (autopilot, demo runner, AI
driver) and simulates a few seconds before the first frame, so `SAIGA_SCREENSHOT=<file.png>` shows the action:
`SAIGA_SCREENSHOT=/tmp/racing.png ./build/samples/games/game_racing --demo`. Their scenarios run with
`./build/samples/games/game_<name> --scenarios` (headless, exit code 0 if all pass; also in the host + client
variants); they have no tests in `tests/` (the saiga tests test the engine). `--coop [delay]` (any engine game, S16b)
runs co-op in one process: a hidden host world and the rendered client world over loopback (default delay 2 steps), the
client's plane, car or character next to the host's demo one, the MCP tools on the client world:
`SAIGA_SCREENSHOT=/tmp/racing_coop.png ./build/samples/games/game_racing --coop --demo`. `--save <dir>` (any engine game,
S25; not with `--scenarios` or `--benchmark`) keeps the world in a save directory: opened after the game's start (an
existing one is loaded), committed every 300 s of world time, at the `quit` tool and at the window's close
(`./build/samples/games/game_world --headless --save /tmp/my_world`; `save_status` / `commit_save` over MCP). In
`game_flight` (the only streaming game) entities beyond the kept distance of every player unload into their chunk's
stored groups and reload when the chunk is ready again. The platformer's `--cellar`
(S21) runs its demo through the enclosed cellar (the screenshot shows the cellar, solo and with `--coop`):
`SAIGA_SCREENSHOT=/tmp/cellar.png ./build/samples/games/game_platformer --demo --cellar`. Stages that change the
renderer, physics, animation or camera keep the demos working, run their scenarios and show their feature in them.
The demos enable the environment (S36a1c, `samples/games/common/Environment.*`: `data/<game>/environment.json` if it
exists) and render **ACES by default** (`--tonemap identity`: the clay view with the studio light); flight's `--demo`
day passes in 10 minutes from 6:00, `game_world` is at 10:00 frozen. Every engine game takes `--time-of-day h` and
`--day-length s` (S36: the world's `WorldTime`, an error for a game without `EnvironmentSetup::enabled`), e.g.
`SAIGA_SCREENSHOT=/tmp/flight_sunset.png ./build/samples/games/game_flight --demo --time-of-day 19`, and `--exposure x`
(a multiplier on top of the EV100); `SAIGA_DEBUG_TAB=5` opens the debug UI's Lighting tab (F1 Game, F2 Performance, F3 Engine, F4 Console,
F5 Lighting; game tabs F6-F10). S36a2: the sun and moon cast shadows (cascades to `lighting.shadowDistance`, flight 2500
m in `data/flight/environment.json`; the far terrain horizon map) in every engine game; the environment file's `lighting`
key takes every `LightingParameters` field (e.g. `"sunShadows": false`), the Lighting tab's Cascades and Far terrain
shadows sections and `set_lighting` change them at run time. S36b: local point and spot lights (`LightSource` entities,
scene lights; clustered, the most important shadowed ones in the cached shadow atlas): `game_world --lamps N` (default
400 lamps in front of the start camera; on while the sun is below 4 degrees: `--time-of-day 22` shows them),
`--benchmark night`; the flight demo's navigation and landing lights below 5 degrees sun elevation
(`SAIGA_SCREENSHOT=/tmp/flight_night.png ./build/samples/games/game_flight --demo --time-of-day 23`).

## Code Conventions

- Formatting: `.clang-format` (Google base): 4-space indent, **Allman braces**, 120-column limit,
  left-aligned pointers (`int* x`), aligned consecutive assignments, `#  ` indented preprocessor
  directives. Include groups in order: `"saiga/..."`, `"internal/..."`, other `"..."`, `<...>`.
  Run `clang-format -i <file>` on the files you change (only those).
- Every source file starts with the MIT header block:
  ```cpp
  /**
   * Copyright (c) 2021 Darius Rückert
   * Licensed under the MIT License.
   * See LICENSE file for more information.
   */
  ```
- `#pragma once` in headers; everything in `namespace Saiga { ... }  // namespace Saiga`.
- Includes use the full path from `src/`: `#include "saiga/core/util/ini/ini.h"`.
- Exported (non-template, non-inline) symbols need the module's export macro from
  `src/saiga/export.h` — `SAIGA_CORE_API`, `SAIGA_VISION_API`, `SAIGA_CUDA_API`, `SAIGA_OPENGL_API`,
  `SAIGA_VULKAN_API` — or the Windows (DLL) build fails to link.
- Error checks: `SAIGA_ASSERT(cond)` / `SAIGA_ASSERT(cond, "msg")` (`saiga/core/util/assert.h`) —
  the norm across the library (glog `CHECK*` is used only in a few places). The message is only evaluated on failure
  and printed to stderr, so death tests can match it (`EXPECT_DEATH(stmt, "already has Health")`, inside
  `#ifdef SAIGA_ASSERTS`).
- Keep the code portable and free of warnings: it must build with GCC, Clang and MSVC (Linux and Windows); CI builds
  with `SAIGA_WARNINGS_AS_ERRORS=ON`. The MSVC CI job covers Core, Vulkan and Engine only (OpenGL, Vision and the
  samples are not built there).
  MSVC compiles the saiga targets at warning level 3 (`SAIGA_WARNING_CXX_FLAGS`), where the CRT deprecations (C4996)
  are warnings: use `Saiga::GetEnvIfSet` (`saiga/core/util/env.h`) instead of `std::getenv`, and the `_s` or `_r`
  variants of `localtime`/`gmtime`.

## Gotchas

- Module sources are **globbed** by the module's `CMakeLists.txt` files (per directory): after adding or removing a
  `.cpp`/`.cu` file, re-run the CMake configure.
- The `saiga-world` MCP server of `.mcp.json` is only reachable while `game_world` runs (it fails to connect
  otherwise; that is expected).
- A missing optional dependency silently disables a module, a feature or a test rather than failing
  the configure — a change can "build" while its code was never compiled. Check the
  `Module enabled` / `Test enabled` lines.
- Generated headers live in `build/include/saiga/` (`saiga_buildconfig.h`, `saiga_modules.h`) — do not
  edit them; change the `src/saiga/*.h.in` templates.
- Do not modify the git submodules under `submodules/` (third-party code pinned by commit). The exceptions are
  `submodules/tiny-eigen` and `submodules/tiny-glog`: they are regular files of this repository and are edited here
  (tests: `tests/test_core_tiny_eigen.cpp`). They keep their own formatting: do not clang-format them as a whole; new
  lines follow the house style.
- `build*/` and `cmake-build-*/` are git-ignored; build in-tree in `build/`.

## Verification

```shell
cmake -S . -B build <CI options above> && cmake --build build --parallel   # must compile
cd build && ctest --output-on-failure --parallel 4                         # all tests must pass
```

Changes to RenderScene, culling, terrain or streaming are performance relevant (R-TEST-2): run the benchmarks of the
recipe above before and after (`game_world --benchmark flyover|forest` and `sample_vulkan_scene_benchmark`; terrain
edits also `--benchmark terraform`).

For changes in the CUDA module, which CI does not build, also build that module locally (with a GPU) and say so
if it could not be built. CI covers the OpenGL and Vulkan modules only through the headless
`test_opengl_*` and `test_vulkan_*` tests; for rendering changes also run a sample on a GPU (Vulkan: `samples/vulkan`, they use GLFW). The Vulkan samples save a
screenshot of frame 60 and exit when the environment variable `SAIGA_SCREENSHOT=<file.png>` is set, which allows checking
the output without looking at the window (F12 saves `screenshot.png` interactively).

## Pull Requests

PRs are squash merged (one commit on `master` per PR; `gh pr merge <n> --squash`). For stacked PRs: after the lower PR
is squash merged, rebase the next branch onto `master` without the merged commits
(`git rebase --onto origin/master <last commit of the merged branch> <next branch>`), force-push it and retarget its PR
to `master`.
