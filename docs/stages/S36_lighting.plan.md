# S36: Lighting

Status: S36a1a implemented (2026-11-22, branch `stage-s36a1a`); a1b, a1c, a2 and b planned (plan written 2026-11-22 on
branch `plans-s36`, based on the S20-S23c stack; revised after review rounds 1 and 2 the same day).

Sections follow the stage plan template of S05; the design-doc amendments, the AGENTS.md changes, a decisions table,
the template's Design changes / Results placeholders, the review log and the questions left to the user are at the end.

**Split (D1, Q2 decided; a1 split again after review round 2, blocker 2).** The stage is implemented and merged as
five parts with one plan, each part its own PR (counted file lists in Files):

- **S36a1: World state, PBR, sky, 2D clouds**, in three PRs:
  - **S36a1a: World state and the lighting tools' first half (61 files, no image change).** `WorldTime` / `Weather`,
    the celestial model, `EnvironmentState`, `RenderSink::setEnvironment` (stored), the app's mutable
    `EnvironmentSettings`, the exposure rule (A3), ImGui textures, the `LightingTab` with Time and weather and Sun and
    moon, `get_lighting`, `set_time_of_day`, `set_weather`, `set_lighting` (settings only), `app_status.lighting`, the
    scenario `world_time_advances`.
  - **S36a1b: Lighting block, PBR, debug core (59 files, no image change).** `SceneLighting` v2 with its owner
    `SceneLightingSet` and the final set-1 layout, `lighting.glsl` / `pbr.glsl` / `ambient.glsl`, the studio
    environment (clay reproduces S19a exactly, A3), the prototype albedo colours (D21), all behind
    `LightingParameters::pbr = false`; the debug views, previews and `DebugLines`, `capture_lighting`, `probe_lighting`,
    `set_lighting_debug`, the `screenshot` arguments, the tab's Exposure, Ambient, GPU sections and toolbar.
  - **S36a1c: Sky, environment map, clouds, games, flip (86 files).** The Hillaire atmosphere with sun, moon and
    stars, the sky environment map and sky ambient, the 2D cloud layer with cloud shadows, the games with the
    environment enabled and **ACES by default** (D22), tuning, and the flip commit with the goldens; tech demo: the
    flight demo by day, at sunset and at night, without sun shadows.
- **S36a2: Sun and moon shadows.** Multi-view culling, cascaded shadow maps, far terrain horizon shadows; their tab
  sections, previews, overlays and tool keys (built with the cascades, so the cascades are tuned with their debug
  views); the measurement that fixes the cascade defaults (Q6); tech demo: the flight demo's shadows at noon, sunset
  and night.
- **S36b: Local lights.** `LightSource`, `LightData` and `addLight` / `setLight` / `removeLight` / `addLights`,
  clustered forward light culling, the cached shadow atlas; their tab section, overlays, previews and tools; tech demo:
  `game_world` at night with lamps, the flight demo's landing light.

Rules of the split: the parts are merged in order (a1a, a1b, a1c, a2, b); a part may be opened as a stacked PR on the
previous one (AGENTS.md "Pull Requests"); **every commit builds without warnings and passes CI** (a1a and a1b change
no image; the image changes of a1c, a2 and b are behind a default that flips only in the part's last commit, which
also commits the goldens); **each part stays under 100 changed files** (CodeRabbit's limit; the Files lists are
counted per file, not estimated: a1a 61, a1b 59, a1c 86, a2 63, b 61; if a1c grows past 95 at its last feature step,
the cloud layer (step a1c.3, 8 files) moves to a2, a seam with no dependency in a1c). Every section of Scope,
Implementation steps, Tests, Done when and Files is labelled with its part (**a1** covers a1a-a1c where the
sub-part does not matter).

## Goal

The engine lights its worlds physically: a sun and a moon that follow a replicated, saved time of day; a physically
based sky with stars and a 2D cloud layer that follow a replicated, saved weather; GGX shading in physical units with a
game-chosen exposure; sun/moon shadows near the camera through cascaded shadow maps and to the horizon through a
heightfield horizon map; sky ambient and sky reflections; and (part b) many local point and spot lights through
clustered forward shading, the most important of them with cached shadows.

Developers see and change every part of it in the debug UI's new **Lighting** tab (F5): the shadow map image of each
cascade, the horizon map, the shadow atlas, cascade coverage and light volume visualisations, cluster heat maps,
exposure, time and weather, scene debug views and the GPU times and memory of every lighting pass. Agents do the same
through MCP tools: they read the state and statistics, change time, weather and lighting parameters, switch debug
views, probe a pixel ("why is this dark"), and capture debug images (shadow maps, LUTs, debug views) with the numbers
behind them. Each part delivers the UI views, previews and tool keys of the features it adds.

The flight demo flies by day, at sunset and at night; `game_world` shows lamps at night.

## Depends on

- **S19** (S19a: the lighting block, `setTime`, the depth pre-pass, the GPU sub-sections; S19b: `RenderSink`,
  `VulkanRenderSink`, `RecordingRenderSink`, `ExtractSystem`, the view camera; S19c: `DebugUi`, `DebugConsole`,
  `McpCallLog`, `set_ui_tab`), **S20** (`TerrainAtlas`, `TerrainPass`, the pyramid, `TerrainFrameUniforms`, the
  anchor), **S21** (space-state entity per space, `SpaceRenderSettings`, `SpaceTemplate`), **S23b/S23c** (content
  render sets: the casters of the flight demo).
- Done stages used: S18 (`SceneCulling` with its view dimension, §10.2), S17 (HDR frame, pre-exposure, ACES), S16
  (replication of world-state entities, `HostClock`, the host + client scenario variants), S14 (the screenshot tool),
  S11 (saves), S10 (`global` / `spaceState`, `WorldApi`, world tools, the co-op hooks).
- Not done and therefore stubbed (§22: "the stage that is done later implements it"):
  - **S27 far content**: impostors cast no shadows in S36. S36 gives every pipeline class a shadow pipeline slot
    (`ShadowPipelineTable`, A5) and tests the slot with a test class (`Shadows.PipelineTableSlotForNewClass`); S27
    registers the impostor class's shadow variant and adds the far cascades' impostor casters. Beyond the last cascade
    only terrain shadows (horizon map, A6) and cloud shadows (A4) reach far content. The S27 row of §22 gets this item
    (amendment M12).
  - **S30 placed pieces**: piece lights are S30's (it calls `RenderScene::addLights(chunk, ...)`, B2, from its
    per-chunk light lists); the bulk path is tested in S36 (`Lights.BulkListPerChunk`).
  - **S37**: auto exposure, AO (the AO texture is white), quality tiers (S36 has `LightingParameters`, S37 maps tiers to
    them), the per-pass budget gate (§17.10: checked from S37; S36 reports).
  - **S38** (froxel fog) and **S42** (volumetric clouds) consume the cascades, the clusters and the cloud shadow hook
    (`cloudShadow()` in `lighting.glsl`); S36 keeps the analytic in-shader fog (§10.5 pass 10 deviation of S19a) with
    the atmosphere's fog colour.
  - **S43**: the full material set, wetness and snow on materials. S36 lights the existing flat-colour materials
    (palette roughness / metalness, `MaterialData`) and gives the prototype's terrain, water, grass and foliage, which
    are one gray albedo today (`PrototypeSettings::albedo` 0.55 in `terrain.frag`, `grass.frag`,
    `prototype_foliage.frag`), a few **constant albedo colours** for the ACES path (`PrototypeAlbedo`, A3, D21; S43
    replaces them); `SmoothTextured` keeps its base colour. `Weather::rain` and `snowCover` are state only in S36 (no shading).

## Rules that apply

- **R-MECH-1**: how time and weather change, day length, latitude, exposure curve, night darkness, minimum ambient,
  shadow distances, light budgets and the lamp placement are game parameters (`EnvironmentSetup`,
  `EnvironmentSettings`, `LightingParameters`, the games). The engine's `worldTimeSystem` is optional (a game registers
  it).
- **R-MATH-1**: `WorldTime`, `Weather`, `LightSource` are double components; float only in `EnvironmentState`,
  `LightData` and the shaders (render data, camera- or chunk-relative). Cascade centres, the horizon map anchor and
  light positions are kept in double and made camera-relative on the CPU.
- **R-MATH-2**: celestial and cascade math with the tiny-eigen subset (Eigen backend check for the new double helpers).
- **R-STATE-1/2**, **R-SYS-1/2**: time and weather in components of the state entities (§7.8); `worldTimeSystem` is a
  free system function; light changes are component writes.
- **R-DET-1**: `worldTimeSystem` advances by the step `dt`; render-only effects (flicker, cloud drift between steps)
  are presentation with phases computed in double on the CPU (§10.1: no absolute time in a shader).
- **R-NET-1/2**: `WorldTime` and `Weather` replicate from the host (`Replicate::Host`), `LightSource` with its entity
  (`Replicate::HostReliable`); the time scenario passes the host + client variants; clients never write them (in
  `--coop` the tools write the hidden host, A9).
- **R-REG-1/2**: the components are in `EngineIds.h` and **always** registered in `registerEngineComponents` (the
  prefix rule of `Registry::validateRegistration`); only the kind (plain component, or global / space state) depends on
  `WorldSetup::environment.enabled` (D4); MCP tools in an explicit registration function.
- **R-FILE-1**: one class per file (`Atmosphere`, `CascadedShadows`, `HorizonShadows`, `EnvironmentMap`,
  `LocalLights`, `ShadowAtlas`, `LightingDebug`, `LightingTab`, ...); value types in `RenderSceneTypes.h` and
  `LightingTypes.h`.
- **R-TEST-1**: every render feature has a headless Vulkan test and a golden where it changes the image; every engine
  feature a unit test; the time and light mechanisms engine scenarios; every tool a `handleMessage` test.
- **R-TEST-2**: `RenderScene`, culling and terrain change: `game_world --benchmark flyover|forest` and
  `sample_vulkan_scene_benchmark` before and after each part, numbers reported (with the new lighting off and on
  separately, Benchmarks below); the new GPU sections reported against §17.10's shadow budget (gate from S37).
- **R-MCP-1**: generic names (`get_lighting`, `set_time_of_day`, ...), registered before `start`, descriptions state
  what they read and change, a `handleMessage` test each.
- **R-DATA-1**: the environment can come from a data file (`data/<game>/environment.json`, optional; read by
  `readEnvironmentFile` (A1) through `readDataFile`): missing keys keep defaults, unknown keys are fatal.
- **R-SAVE-1**: new components only (no migration); the committed engine schema (`data/tests/engine/schema.json`) is
  regenerated once in a1 and once in b (AGENTS.md recipe); the golden save keeps loading (it has no `WorldTime`; a world
  with the environment enabled creates the world-state entity with its defaults on load, as S21 does for a new space).
- AGENTS.md: Vulkan 1.3 core only (no optional device feature is required: depth clamping is done in the shadow vertex
  shaders, A5; the comparison sampler uses `D32_SFLOAT` / `D16_UNORM` with linear filtering only where the format
  feature reports it, nearest-with-manual-PCF otherwise), export macros, warnings as errors on GCC/Clang/MSVC (no
  identifiers `near` / `far`: `minwindef.h` macros), no submodule changes, clang-format.

## Scope

### A1. World time and weather (`saiga_engine`) [a1a]

- **Components** (`engine/components/WorldTime.h`, `Weather.h`; IDs `EngineComponent::WorldTime`, `Weather` appended
  after `PendingTransition`):

  ```cpp
  struct WorldTime            // global (§7.8) when the environment is enabled
  {
      double timeOfDay = 12;  // hours, [0, 24)
      int64_t day      = 0;
      double rate      = 0;   // game hours per simulated second (0: frozen); D5: added to §10.7's fields, so a game
                              // and agents change the speed through the component (replicated, saved)
      double validAt   = 0;   // s, World::time() of the step that wrote timeOfDay (review 2, finding 18: the
                              // client's extrapolation reference; replicated and saved with the value it belongs to)
  };
  struct Weather              // space state (§7.8) when the environment is enabled
  {
      double cloudiness = 0.3, rain = 0, fogDensity = 0.00015, snowCover = 0, waveAmplitude = 1, waveSteepness = 0.5;
      Vec2 wind        = Vec2(3, 1);    // m/s in x, z
      Vec2 cloudOffset = Vec2::Zero();  // m, accumulated from the wind by the host, wrapped to kCloudTile (§10.7)
      double validAt   = 0;             // s, World::time() of the step that wrote cloudOffset (as WorldTime)
  };
  ```

  `validAt` is written by `worldTimeSystem` and by `setWorldTime` / `setWeather` together with the value; it is
  `World::time()` (seconds, continuous across saves through `timeBase`, S11), not a step index, so a save loaded with
  another step rate keeps it meaningful.

  Ranges: `timeOfDay` [0, 24), `cloudiness`, `rain`, `snowCover` [0, 1], `fogDensity` [0, 0.05], `rate` [-3600, 3600],
  every value finite. `Visit.h` has no value limits, so a raw `set_global` can write anything: `worldTimeSystem`
  **normalises every step regardless of `rate`** (wrap `timeOfDay` with the day count, clamp the ranges, a non-finite
  value -> the default), `environmentState` sanitises what it reads the same way (a world without the system), and the
  tools and the tab write through the validated functions `setWorldTime(World&, ...)` / `setWeather(World&, SpaceId,
  ...)` (`engine/lighting/EnvironmentWrites.{h,cpp}`: range errors name the field, then a `WorldApi` write).
- **Registration (D4, blocker 2)**: both types are **always registered** (the engine IDs stay a registered prefix,
  `Registry.cpp`'s `validateRegistration`), and the schema is the same for every world.
  `registerEngineComponents(Registry&, const WorldSetup&)` (signature change; `World.cpp` passes its setup) registers
  `WorldTime` with `registerGlobal(Replicate::Host)` and `Weather` with `registerSpaceState(Replicate::Host)` when
  `setup.environment.enabled`, otherwise both with `registerComponent` (no world-state entity: S10 creates it only for
  a registered global; entity counts, hashes and the golden save of plain worlds stay). Later engine components
  (`LightSource`, b) are appended after them and registered unconditionally as usual. Host and client must agree on
  `enabled` (it changes the kind of two registrations): they are built from the same `WorldSetup` in every engine
  game and scenario variant; there is no join check of the setup until S26's handshake (`Replication.cpp`: "until
  S26's handshake"), so the S26 row of §22 gets "the environment flag in the join check" (M12; review 2, finding 22).
- **`EnvironmentSetup`** (`engine/lighting/EnvironmentSetup.h`, inside `WorldSetup`, constant after the world exists;
  only what the simulation reads): `bool enabled = false; CelestialConfig celestial; double cloudWindScale = 1;`.
  Everything presentational lives in the app's mutable `EnvironmentSettings` (A2).
- **Data file**: `EnvironmentFile readEnvironmentFile(const std::string& path)` (`engine/lighting/EnvironmentFile.{h,cpp}`,
  through `readDataFile`): `{EnvironmentSetup setup; EnvironmentSettings settings; LightingParameters lighting;}` from
  one JSON object with the keys `celestial`, `exposure`, `ambient`, `clouds`, `atmosphere`, `lighting`
  (R-DATA-1: missing keys keep defaults, unknown keys are fatal). Used by flight and `game_world`; racing and the
  platformer use code defaults.
- **`worldTimeSystem`** (`engine/lighting/WorldTimeSystem.cpp`, phase `Gameplay`, `RunOn::Host`, registered by
  `addEnvironmentSystems(SystemList&)` which games call; R-MECH-1): `timeOfDay += rate * dt / 3600`, wrapped into
  [0, 24) with `day` incremented or decremented; `cloudOffset += wind * dt * cloudWindScale`, wrapped component-wise
  into [0, `kCloudTile`) (`kCloudTile` = 16384 m, a multiple of the cloud texture period so the wrap never jumps); the
  normalisation above. Weather blending (target values, rates) is game policy: the demos write `Weather` from their own
  systems or tools.
- **Celestial model** (`engine/lighting/Celestial.{h,cpp}`, pure, double): `CelestialConfig { double latitudeDeg = 45,
  axialTiltDeg = 23.44, dayOfYearAtDay0 = 80 /* equinox */, yearLengthDays = 365, sunIlluminance = 120000 /* lux, top
  of the atmosphere */, moonIlluminance = 0.27 /* full moon */, moonPhase = 0.5 /* 0 new, 0.5 full; a parameter, no
  synodic model */, starIntensity = 1; }`; `CelestialState celestialState(const CelestialConfig&, double timeOfDay,
  int64_t day)` -> `{Vec3 sunDir, moonDir /* towards the body, y up, north = -z */; double sunIlluminance,
  moonIlluminance, moonPhase, sunElevationDeg, hourAngle; }` (solar declination from the day of year, hour angle from
  the time; the moon's hour angle is the sun's plus `2 pi * moonPhase`, its illuminance scaled by the phase's lit
  fraction; `hourAngle` also rotates the stars about the celestial pole). **Intent** (review 2, finding 21): with the
  fixed default phase 0.5 the full moon is exactly opposite the sun, so it rises at sunset and sets at sunrise every
  game day, which is what the demo wants (a lit night all night); the phase parameter moves the moon's rise time
  (phase 0.25: rises about at noon); at phase 0 (new moon) the moon stands next to the sun, gives no light, and the
  night has only `minAmbient` and the stars (no shadow light, A5). Test `Celestial.NewMoonNightHasNoShadowLight`.
- **Environment extract** (`engine/lighting/EnvironmentExtract.{h,cpp}`): `EnvironmentState environmentState(const
  World&, SpaceId space, double alpha, const EnvironmentSettings&, const EnvironmentSetup&)`:
  - time (one rule for host and client; review 2, finding 18): the render time `tr = World::time() + (alpha - 1) *
    dt` (on a client `World::time()` is already the `HostClock`'s estimate of the host time, `World.h`:
    `hostStepOffset`), and `timeOfDay + rate * (tr - validAt) / 3600`, wrapped. On the host `validAt` is the current
    step's time, so this interpolates between the previous and the current step like the poses; on a client it
    extrapolates from the snapshot that wrote the value, so the sun and the cloud drift move smoothly between
    snapshots at any S32 rate. A rate change reaches the client with the snapshot that carries it (with its own
    `validAt`): no jump beyond the true change;
  - the space's `Weather` (sanitised; `cloudOffset + wind * (tr - validAt) * cloudWindScale`, wrapped), the exposure
    (A3), the ambient settings, the per-effect phases (cloud drift remainder, flicker base) as `fmod` in double;
  - an **enclosed space** (its `SpaceTemplate::kind == SpaceKind::Enclosed`, `SpaceTemplate.h`; nit 29): the
    template's `ambient` (lux-equivalent colour) and `ev100`, no sun, sky or sky ambient (the single authority for
    enclosed lighting in the ACES path; clay renders an enclosed space with the studio light without fog, A3, exactly
    as `kEnclosedLighting` does today; `kEnclosedLighting` goes with the a1c flip);
  - without `environment.enabled`: the defaults of `WorldTime` / `Weather` (noon, cloudiness 0.3) and the settings.
  The app calls it once per rendered frame for the view camera's space after the game's `beforeRender` and passes it to
  `RenderSink::setEnvironment` (`EngineApp::frame` step 6), in every app with a device (so enclosed spaces light
  correctly without a `WorldTime`).

### A2. `EnvironmentState`, `EnvironmentSettings` and the render interface [a1a; `pbr` a1b]

- `RenderSceneTypes.h` (render-only floats; `static_assert`ed layout where it reaches a shader):

  ```cpp
  struct EnvironmentState
  {
      vec3 sunDir = ...; float sunIlluminance = 120000;   // lux at the top of the atmosphere (the LUT attenuates)
      vec3 moonDir = ...; float moonIlluminance = 0.27f; float moonPhase = 0.5f;
      float starIntensity = 1; mat3 starRotation = identity;  // hour angle about the celestial pole
      // Weather (§10.7)
      float cloudiness = 0.3f, rain = 0, fogDensity = 0.00015f, snowCover = 0;
      vec2 wind = vec2(3, 1); vec2 cloudOffset = vec2(0, 0);   // cloudOffset already wrapped (float exact below 2^24)
      // Exposure (A3): the resolved EV100 of this frame (curve or fixed value, compensation applied) and where it came
      // from for the statistics (nit 31)
      float ev100 = 15; ExposureSource exposureSource = ExposureSource::Fixed;   // Curve, Fixed, Enclosed
      // Ambient (A3)
      float ambientScale = 1; vec3 minAmbient = vec3(0.002f);   // lux-equivalent floor (night policy, §10.7)
      // Enclosed space (A1)
      bool enclosed = false; vec3 enclosedAmbient = vec3(0);
      // Phases (radians, fmod in double): wind (trees 3, grass 2), cloud drift remainder, flicker base
      std::array<float, 8> phases = {};
  };
  ```

- **`EnvironmentSettings`** (`vulkan/scene/LightingTypes.h`; finding 13): the presentation settings, owned mutable by
  `EngineApp` (`EngineApp::environmentSettings()` / `setEnvironmentSettings`), initialised from
  `EngineAppSetup::environment` (filled by the game or `readEnvironmentFile`): `ExposureSettings exposure;
  AmbientSettings ambient; CloudSettings clouds {altitude 2000, radius 40000, seed, shadowFade 8000, fogFalloff};
  AtmosphereSettings atmosphere; std::optional<SunOverride> sunOverride;` (the clay `StudioEnvironment` is the
  scene's, A3, not a setting of the app).
  `environmentState` takes it; `set_lighting`, the tab and the game write it; `get_lighting` reports it. Only
  `EnvironmentSetup::enabled` and `celestial` (and `cloudWindScale`) stay in `WorldSetup`.
- `RenderScene::setEnvironment(const EnvironmentState&)` (main thread, from the next frame; also from the transfer
  callback like `setCamera`), `const EnvironmentState& environment() const`, `bool hasEnvironment() const` (false
  until the first call: the scene then lights with the studio environment in every tone map and uses
  `params.exposure` unchanged, A3). `setTime` stays until the a1c flip (the prototype passes read `phases` from then
  on; goldens pass `phases = {}`).
- `RenderSink::setEnvironment(const EnvironmentState&)` (pure virtual); `VulkanRenderSink` forwards,
  `RecordingRenderSink` stores the last one and counts calls (tests). The `World` does not forward an environment
  (it is per frame, sent by the app).
- `RenderSceneParameters::lighting` (`LightingParameters`, the S37 tier mapping's target; R-MECH-1; changed at run
  time through `RenderScene::setLightingParameters`, applied at the next frame start, resources recreated through
  `deferDelete` when a size changes). A game sets its values in `EngineAppSetup::render.lighting` (or the `lighting`
  key of its environment file); e.g. flight's `shadowDistance` 2500 m. Every field is run-time changeable and has a
  `set_lighting` key (A9):

  ```cpp
  struct LightingParameters
  {
      // a1
      bool pbr = false;                   // a1b-a1c: selects the pipeline variant (define LIGHTING_PBR; the legacy
                                          // S19a pipelines have no branch, review 2 finding 26); a change recreates
                                          // the scene pipelines (tests and the games' temporary setting only); the
                                          // field is removed in the a1c flip commit (no permanent second light model)
      int environmentResolution = 128;    // 32..512, power of two
      float environmentUpdateDegrees = 0.5f;
      bool clouds = true;                 // the cloud layer and its shadows
      bool forceFullUpdates = false;      // golden mode (§17.6): every incremental effect updates fully every frame
      // a2: cascades (A5)
      bool sunShadows = false;            // false until the a2 flip commit, then true (stays: tiers, comparison)
      int cascadeCount = 4;               // 1..4
      int cascadeResolution = 2048;       // 256..4096, power of two
      float shadowDistance = 1500;        // m, the far end of the last cascade (Q6: final value after the a2 measurement)
      float splitLambda = 0.8f;           // log/linear split blend
      std::array<int, 4> cascadeInterval = {1, 1, 1, 2};   // frames between updates per cascade (A5 staggering)
      float depthBiasTexels = 1.0f;       // receiver-side constant bias, in texels of the cascade
      float slopeBias = 1.5f;             // the rasterizer's depthBiasSlopeFactor
      float normalOffsetTexels = 1.0f;    // receiver normal offset, in texels of the cascade
      float cascadeBlend = 0.1f;          // blend band, fraction of a cascade's extent
      int shadowLodBias = 1;              // LOD levels added for cascades 2-3 (cascades 0-1: the main view's LOD)
      float smallCasterTexels = 1.5f;     // casters with a projected radius below this many texels are skipped
      float maxCasterHeight = 120;        // m above the receivers: the instances' near cap towards the light
      float minShadowElevationDeg = 5;    // the elevation the cascade volumes are built for at most (A5)
      float cascadeCasterDistance = 1371; // m towards the light: terrain casters in the cascades at every elevation
                                          // (default = the instance cap at minShadowElevationDeg); the horizon map's
                                          // channel G starts here (A5, A6; review 2 finding 4)
      bool terrainCastsInCascades = true;
      // a2: horizon map (A6)
      bool horizonShadows = false;        // false until the a2 flip commit
      int horizonResolution = 1024; float horizonExtent = 32768;  // m, square around the anchor
      int horizonRowsPerFrame = 64;       // base incremental budget (scaled with the light's angular speed, A6)
      // b
      bool localLights = false;           // false until the b flip commit
      bool clusteredLights = true;        // false: brute-force loop (the reference of the cluster tests)
      int maxLights = 1024; int clusterX = 16, clusterY = 9, clusterZ = 24; float clusterNear = 2, clusterFar = 1000;
      int maxLightsPerCluster = 128;
      int atlasResolution = 4096; int atlasTile = 512; int facesPerFrame = 8;
  };
  ```

### A3. Shading: PBR, units, exposure, ambient, clay [exposure rule a1a; shading, studio, colours a1b; migration a1c]

- **One lighting include** (`shader/vulkan/scene/lighting/lighting.glsl`) used by `flat.frag`,
  `prototype_foliage.frag`, `terrain.frag`, `grass.frag` and the cloud layer: the set-1 bindings (below), `pbr.glsl`
  (GGX NDF, height-correlated Smith visibility, Schlick Fresnel with `F0 = mix(0.04, albedo, metalness)`, Lambert
  diffuse weighted by `(1 - F) * (1 - metalness)` and the multi-scatter energy compensation of Fdez-Agüera with the BRDF
  LUT), `shadows.glsl` (a2; in a1 `shadowFactor` returns the cloud factor only), `ambient.glsl` (`AmbientProvider`),
  `atmosphere_common.glsl` (sun transmittance, fog colour). Function: `vec3 shadeSurface(SurfaceInput s)` with
  `SurfaceInput {vec3 albedo; float roughness, metalness; vec3 n; vec3 relPos; vec3 emissive; float ao; uint flags
  /* foliage: wrap */}` -> linear radiance (before pre-exposure); the four passes call it, so every surface lights the
  same way. The legacy S19a model (`prototypeLighting`, `protoClayLighting`, `protoApplyFog`) stays next to it as the
  other pipeline variant (`pbr`), until the a1c flip commit.
- **The lighting block** (`SceneLighting` v2, moved from `SceneCulling.h` into `vulkan/scene/SceneLighting.h`,
  std140, `static_assert`ed size): sun and moon direction and illuminance (the shader applies the transmittance LUT per
  pixel for the direct light; the CPU reads back nothing), the shadow light (A5), exposure (`exposure`,
  `preExposure`), the studio environment (clay, A3), the prototype albedo colours, `minAmbient`, the enclosed flag and ambient, fog (density, falloff, base, enabled),
  cloud layer parameters, the star rotation, the debug view and overlay flags (A8), the wind phases, the camera, the
  MSAA samples; the legacy fields (the S19a sun, ambient, fog colour) until the flip; a2: the cascade matrices (4 x
  mat4, camera-relative, rebuilt per frame from the cascades' double centres), the cascade boxes for the selection by
  projection, texel sizes, the horizon map transform; b: the cluster grid dimensions, depth slicing and the light
  count. Set 1 of every scene draw pipeline: binding 0 the block, 1 the foliage texture (as today), 2 the ambient SH
  buffer (its own binding, written by `env_sh.comp`; finding 38), 3 the cloud coverage, 4 the prefiltered environment
  cube, 5 the BRDF LUT, 6 the transmittance LUT, 7 the sky-view LUT; a2: 8 the cascade shadow map
  (`sampler2DArrayShadow`), 9 the horizon map; b: 10 lights, 11 cluster table, 12 light indices, 13 the atlas
  (`sampler2DArrayShadow`, static and dynamic layer), 14 the atlas tile table. Unused bindings of a part are bound to
  1x1 dummies (the layout is final from a1b, so a2 and b add no layout change to every pipeline). `TerrainFrameUniforms`
  keeps only terrain data (its `sunDir`, `shading`, `fog`, `fogColor`, `skyColor` move into the block at the flip; the
  terrain, grass and sky pipelines bind set 1 too).
- **Owner of set 1 (review 2, finding 11)**: today `SceneCulling` owns the lighting buffer and `lightingSet` per frame
  slot (`SceneCulling.h`), and the terrain, grass and sky pipelines have only set 0 (`grass.frag` has its foliage
  texture at set 0 binding 7). a1b adds **`SceneLightingSet`** (`vulkan/scene/SceneLightingSet.{h,cpp}`, owned by
  `RenderScene`): the set-1 layout, the block buffer and one descriptor set per frame slot, the 1x1 dummies, and
  `write(frameSlot, const SceneLightingBindings&)` once per frame (the bindings of A4-B4 as they exist); `SceneCulling`,
  `TerrainPass`, `GrassPass`, the sky and `CloudLayer` receive its layout at pipeline creation and its set per frame;
  `SceneCulling` loses its lighting buffer, and grass drops its own foliage binding (set 0 binding 7 -> set 1 binding
  1). The **shadow and atlas pipelines** (a2, b) get a second, small layout from the same class
  (`shadowLayout()`: binding 0 the block, 1 the foliage texture): a depth image that is the current attachment must
  not be statically used through a descriptor of the same draw (a validation error, which aborts the tests), so
  `shadow_depth.vert`, `shadow_alpha.frag`, `shadow_foliage.vert/.frag`, `terrain_shadow.vert` and `atlas_depth.frag`
  do not include `lighting.glsl` (they include `scene_block.glsl`, the block declaration alone).
- **The `SCENE_FOG` variant** (`SceneCulling.h:150`): removed at the a1c flip: the cloud shadow, the atmosphere fog and
  (a2) the shadows need `relPos` in every fragment; its cost is in the a1c benchmark (Benchmarks).
- **Units**: sun and moon illuminance in lux, local lights in lumen (b), radiance in cd/m^2 times `preExposure` (the
  S17 contract: half floats see values near 1).
- **Exposure in EV100 (blocker 1)**: `ExposureSettings {ExposureMode mode = TimeOfDay; float ev100 = 15;
  std::vector<vec2> curve = {{-18, -1}, {-6, 3}, {0, 9}, {10, 13}, {30, 14.5}, {90, 15}}; float compensation = 0;}`
  (two modes, nit 44: `TimeOfDay` reads the curve at the sun elevation in degrees, piecewise linear, clamped at the
  ends; `Fixed` uses `ev100`). The extract resolves the frame's EV100 (curve or fixed value minus `compensation`;
  enclosed: the template's) into `EnvironmentState::ev100`.
  - **One rule for the frame's exposure (review 2, finding 6)**, in `RenderScene::latchFrame` (today `frameExposure =
    params.exposure`, `RenderScene.cpp:689`): a scene with an environment (`hasEnvironment()`) uses
    `frameExposure = clamp(params.exposure / (1.2 * 2^ev100), 2^-24, 2^10)`; a scene on which `setEnvironment` was never
    called keeps today's rule `frameExposure = params.exposure` (unit-less) and lights with the studio environment
    (unit-less S18 / S19a values, below) also in ACES. So `RenderSceneParameters::exposure`, `setExposure(x)` (it sets
    `params.exposure`, clamp [1/1024, 1024] unchanged), `exposure()` and `--exposure x` (`EngineMain.cpp`) keep their
    meaning: a multiplier, now on top of EV100 in a scene with an environment (`--exposure 2` is one stop brighter);
    the Engine tab's exposure field likewise. **Unchanged tests**: `test_engine_app.cpp`'s `EngineAppRender` exposure
    case (`render.exposure = 2` gives `scene->exposure() == 2`), the six `setExposure` cases of `test_vulkan_scene.cpp`
    incl. the clamp at 1024 and 1/128, and `hdr_tonemap` (it draws raw values through the scene callback without an
    environment: not regenerated); the Vulkan samples that draw with ACES keep their brightness. New:
    `RenderScene::frameExposure()` for the statistics.
  - The post shader and the tone-map weighted resolve use `exposure / preExposure` (a ratio near 1 within
    `kMaxPreExposureChange`) and `preExposure` itself (scene shaders multiply radiance by it): both are checked for the
    new range (no `1 / exposure` that overflows fp32, the fp16 scene values stay within [2^-14, 65504] for radiance
    between 10^-3 and 10^9 cd/m^2 at the matching EV100). Test `Exposure.Ev100RangeReachesPost` (EV100 20, 15, -1, -8
    arrive unclamped in `frameExposure` and give the same post output for a radiance scaled to the matching exposure)
    and `Exposure.NoEnvironmentKeepsUnitlessExposure` (a scene without `setEnvironment`: `frameExposure ==
    params.exposure`).
  - Enclosed spaces use the template's `ev100` (A1).
- **Clay mode (finding 11, Q1; review 2, finding 8)** (`ToneMap::Identity`): renders with a fixed **studio
  environment** (`StudioEnvironment`, independent of `WorldTime` and `Weather`), defined so that **clay reproduces
  S19a exactly** (except the shadows of a2):
  - `StudioEnvironment {vec3 sunDir; float sunIntensity; float ambient; bool hemisphere /* 0.6 + 0.4 n.y */; float
    albedo; bool fog; float fogDensity, fogFalloff; vec3 fogColor;}`: two presets with today's values: `kSceneStudio`
    (S18's block defaults: sun (0.311, 0.778, 0.545) at 0.8, constant ambient 0.35, no fog) for a `RenderScene`
    without the prototype, and `kPrototypeStudio` (S19a's `PrototypeSettings` defaults: sun azimuth 35, elevation 38,
    intensity 1, hemisphere ambient 0.25, albedo 0.55, fog 0.00015 / 0.0012, the gray fog colour 0.42) set by
    `setPrototypeTerrain`; `RenderScene::setStudioEnvironment` overrides it (tests, a game that wants another clay
    light). An enclosed space in clay uses the preset without fog (today's `kEnclosedLighting`).
  - The clay branch of `shadeSurface` is S19a's model: Lambert without the Fresnel weight and without specular, `albedo
    * (sunIntensity * max(n.l, 0) * shadow + ambient * (hemisphere ? 0.6 + 0.4 n.y : 1)) * ao`, the studio fog through
    the same height-fog function, the S19a output encoding. The ambient is written as SH L0 + L1 (`ambient * (0.6 +
    0.4 n.y)` is exactly representable), so it runs through the same `ambientDiffuse` code as the ACES path.
  - **Flat-colour instances keep their material colour in clay** (`flat.frag` does today; `scene_flat_box` pins the
    baked base colours of an import); the prototype's surfaces keep the clay albedo 0.55 (the `PrototypeAlbedo` colours
    are the ACES path's).
  - No sky change (the S20 gradient sky), no clouds, no cloud shadow, no atmosphere, no exposure; (a2) the cascades and
    the horizon map multiply the sun term.
  So the five clay goldens (`world_overview`, `world_forest`, `generated_site`, `terrain_edit_crater`,
  `scene_flat_box`; review 2, finding 9) **pass unchanged through the a1c flip**: from a1b on, their existing tests
  also run with `pbr = true` against the same committed golden (a test parameter), which proves that the rewrite of
  every scene shader kept geometry and shading; they are regenerated once, in the a2 flip (shadows), and stay stable
  against atmosphere, cloud and exposure tuning.
- **Prototype albedo colours (blocker 1b, D21)**: `PrototypeSettings::colors` (`PrototypeAlbedo {vec3 ground =
  (0.26, 0.30, 0.16), rock = (0.36, 0.34, 0.31), shore = (0.50, 0.46, 0.36), water = (0.03, 0.06, 0.07), grass =
  (0.20, 0.30, 0.10), leaves = (0.11, 0.20, 0.07), bark = (0.25, 0.19, 0.14); float rockSlope = 0.75;}`, linear,
  values to be tuned in a1c.4): the terrain blends ground -> rock by slope (`smoothstep` on `n.y` around
  `rockSlope`) and ground -> shore in the 3 m above the water level, the water plane takes `water` (instead of the
  gray times 0.62), grass `grass`, the foliage cards `leaves` and the trunks `bark`. Seven constants in the block and
  one `smoothstep` per terrain fragment: no textures, no new pass; clay ignores them. Exposed in the tab's Ambient
  section as colour edits (not saved; S43 replaces them with materials), not as tool keys.
- **`AmbientProvider`** (`vulkan/scene/AmbientProvider.h`, §10.7): `class AmbientProvider { virtual void
  update(vk::CommandBuffer, const EnvironmentState&) = 0; virtual void bind(DescriptorWriter&) = 0; virtual
  AmbientStatistics statistics() const = 0; }`; `SkyAmbient` (the only implementation) gives the SH9 irradiance of the
  environment cube times `ambientScale`, at least `minAmbient`, and the prefiltered cube for specular. Start value
  (finding 38): until the first SH is computed the SH buffer holds `minAmbient` as band 0 (frame 1 is not black;
  `forceFullUpdates` computes it in frame 1). In GLSL `ambientDiffuse(n, relPos)` and `ambientSpecular(r, roughness,
  relPos)` (`ambient.glsl`), both multiplied by the AO input (white until S37); enclosed spaces: `enclosedAmbient` as a
  constant SH band 0, the specular environment a dim constant. Probes or occlusion methods replace `SkyAmbient` later
  without shader changes outside `ambient.glsl`.
- **Migration** of the prototype environment (`setPrototypeEnvironment` until S36, §10.1), **in the a1c flip commit**
  (blocker 3: until then the legacy path reads the old fields and the five existing clay goldens pass unchanged):
  `PrototypeSettings` loses `fog`, `fogDensity`, `fogFalloff`, `sunAzimuth`, `sunElevation`, `sunIntensity`, `ambient`
  (they come from `EnvironmentState`; fog density from `Weather::fogDensity`, falloff from `CloudSettings::fogFalloff`;
  their defaults become `kPrototypeStudio`); `setPrototypeEnvironment` becomes
  `setFoliageTexture(uint32_t seed)` (the passes and LOD settings stay in `PrototypeSettings`); the legacy shader
  functions, `kEnclosedLighting`, `setTime`, `LightingParameters::pbr` and the `SCENE_FOG` variant are removed.
  `WorldSettings` loses the sun and fog keys (`game_set_settings` reports them as unknown keys naming the replacement
  tools, D9); `albedo` stays (clay).

### A4. Atmosphere, stars, moon, 2D clouds, cloud shadows, environment map [a1c]

- **Atmosphere** (`vulkan/scene/Atmosphere.{h,cpp}`, shaders `lighting/atmosphere_*.comp`, `sky.frag` rewritten; Q7:
  Hillaire everywhere, D2): the precomputed LUTs of Hillaire 2020 (Earth defaults: Rayleigh, Mie, ozone; parameters in
  `EnvironmentSettings::atmosphere`): transmittance (256 x 64, RGBA16F, recomputed only when the atmosphere parameters
  change), multiple scattering (32 x 32, same), sky-view (192 x 108, every frame at the camera height). The sky pass
  samples the sky-view LUT, adds the sun disc (radiance clamped below the fp16 limit after pre-exposure, §10.5), the
  moon disc with its phase (lit by the sun direction, a fixed albedo, no texture) and stars (a hashed cube-face grid of
  points in the celestial frame, rotated by `starRotation`, at least 1.5 px wide so MSAA coverage does not change
  between Mesa versions, faded by the sky luminance; no twinkle); the fog colour of `applyFog` is the sky-view LUT at
  elevation 0 in the view direction (so fog and sky match at the horizon). Weather: `cloudiness` greys the sky (a mix
  towards the cloud layer's ambient), `rain` lowers the visibility (fog density multiplier) - parameters, not physics.
- **2D cloud layer** (`vulkan/scene/CloudLayer.{h,cpp}`, `lighting/cloud_layer.vert/.frag`, `cloud_noise.comp`): a
  tiling coverage texture (1024 x 1024 R8 with mips, period `kCloudTile` = 16384 m, generated once on the GPU from
  `CloudSettings::seed` with tiling fBm), drawn as a camera-centred polar grid mesh (radius `CloudSettings::radius` =
  40 km) at `CloudSettings::altitude` (default 2000 m above sea level) in the scene pass after the sky (depth test
  against the scene, no depth write, premultiplied alpha): coverage = `smoothstep(1 - cloudiness - 0.1, 1 - cloudiness +
  0.1, noise(xz + cloudOffset))`, lit by the sun transmittance at the layer and the sky ambient, fog applied at its
  distance. The mesh drops with the earth's curvature (`y -= r^2 / (2 * 6371 km)`, 126 m at 40 km) and its alpha fades
  over the outer 30% of the radius (no visible rim 3 degrees above the horizon). Mountains above the altitude occlude
  it; seen from above (flight) it covers the terrain below. Sub-section `scene.clouds`. Volumetric clouds (S42) replace
  this mesh per tier.
- **Cloud shadows**: `cloudShadow(relPos)` in `lighting.glsl` projects the receiver along the shadow light direction to
  the layer and samples the coverage (a blurred mip). It multiplies the direct sun/moon light of every surface. S42
  replaces the function body with its transmittance map (the hook). Two edge cases (review 2, finding 16):
  - a receiver **at or above the layer's altitude** (the aircraft, a mountain top above 2000 m) gets factor 1 (the
    projection towards the light goes away from the layer), with a 50 m blend below the altitude;
  - the fade to the mean coverage uses the **projected point's** horizontal distance from the camera (`|relPos.xz +
    (altitude - y) / tan(el) * d|`, d the horizontal light direction), not the receiver's: at 3 degrees the offset is
    38 km, far beyond the coverage the layer shows; beyond `CloudSettings::shadowFade` (8 km) of projected distance the
    factor is `1 - mean coverage`.
  Tests `Clouds.NoShadowAboveLayer`, `Clouds.LowSunFadesToMean`.
- **Environment map** (`vulkan/scene/EnvironmentMap.{h,cpp}`, `env_capture.comp`, `env_prefilter.comp`, `env_sh.comp`,
  `brdf_lut.comp`): two cubes of `environmentResolution` (RGBA16F with 5 mips), **front and back** (finding 17a):
  a re-capture writes the sky (sun-less: the sun's specular comes from the direct light), moon, stars and the cloud
  layer seen from the camera height into the back cube, prefilters its GGX mips (32 samples, a deterministic sample
  set) and computes the SH9 from a **box-downsampled** 32 x 32 mip (not a GGX mip) into the back SH buffer, spread
  over the next frames (capture, then one mip per frame, then SH); the front cube and SH are swapped in only when all
  of it is complete, so a frame never mixes two captures. Due when the shadow light direction moved by more than
  `environmentUpdateDegrees`, the weather changed by more than 1%, or the camera height moved by 200 m;
  `forceFullUpdates` does all of it in the frame. The **lower hemisphere** (finding 17b): below the horizon the capture
  writes `CloudSettings`-independent ground radiance `groundAlbedo / pi * skyIrradiance(up)` (`AtmosphereSettings::
  groundAlbedo` = 0.15), so downward normals and grazing reflections get a ground, not the sky. The BRDF LUT (64 x 64
  RG16F) is computed once at init. No terrain in the map (no local reflection probes, §10.7). Memory 2 x 0.6 MiB.

### A5. Sun/moon cascaded shadow maps (multi-view culling) [a2]

- **`CascadedShadows`** (`vulkan/scene/CascadedShadows.{h,cpp}`): one D32 `Texture2DArray` with `cascadeCount` layers of
  `cascadeResolution`. Splits: `lambda * log + (1 - lambda) * linear` between the camera's near plane and
  `shadowDistance` (`cascadeSplits(nearPlane, farPlane, count, lambda)`, pure, tested).
- **Light-space basis** (nit 49): `z` = towards the light; `x` = `normalize(cross(north, z))` with `north` = (0, 0, -1),
  falling back to (1, 0, 0) when `|dot(north, z)| > 0.999`; fixed for a given light direction, so texel snapping is
  stable.
- **Fitting** (finding 7): a cascade with interval 1 is fitted to the bounding sphere of its slice of the camera frustum
  (rotation-invariant: no shimmer when the camera turns). A cascade with interval > 1 is fitted around the **camera
  position** with radius = the slice's far distance + `interval * maxSpeed * dt` (`maxSpeed` = the camera speed of the
  last second, at least 50 m/s), so a stale cascade still covers every receiver of its slice after any turn and the
  move until its next update. Its centre is snapped to the texel grid in light space (no shimmer when the camera moves)
  and kept in double (world coordinates); the camera-relative matrix is rebuilt every frame from it (`centre - camera`
  in double). Default intervals `{1, 1, 1, 2}`; staggering further is S37's tiers; `cascadeInterval[0]` must be 1
  (cascade 0 is fitted to the frustum; `set_lighting` rejects another value). A receiver outside every cascade
  rectangle gets the horizon and cloud factor only.
- **One light-space depth range for all cascades (review 2, finding 3)**: a cascade is its **xy rectangle** in light
  space (the fitted sphere's square, snapped) times **the same depth range** for every cascade: from the receivers'
  far end of the last cascade (the light-space depth of the `shadowDistance` sphere's far side plus the blend band) to
  its near end. Casters nearer to the light are pancaked (below), so the depth range only has to hold receivers: about
  2 x `shadowDistance` (3 km, flight 5 km), which D32 resolves to well below a millimetre. Because an orthographic
  shadow never leaves its caster's xy footprint in light space, selection and de-duplication by the xy rectangle alone
  are exact (Receivers, Casters).
- **Depth and bias (findings 8, 9)**: orthographic, **not reversed** (D32 = 4 bytes per texel, clear 1.0, compare op
  `LESS_OR_EQUAL`, the `sampler2DArrayShadow` compares `ref <= stored`). The near plane is the receivers' near end of
  the shared depth range; casters nearer than it (up to the caster caps below) are **pancaked in the shadow vertex
  shaders** (`gl_Position.z = max(gl_Position.z, 0)`: exact for an orthographic projection except for triangles
  crossing the plane, which only occur for casters far outside the receiver volume); no `depthClamp` feature, no 4 km
  pull-back. Biases: the
  rasterizer's `depthBiasSlopeFactor` = `slopeBias` (dynamic state), constant factor 0; on the receiver a constant bias
  of `depthBiasTexels * texelSize` metres along the light direction plus the normal offset of `normalOffsetTexels *
  texelSize` metres along the normal, both in world metres of the cascade (so one setting works for every cascade size).
- **Shadow light (finding 29, D13)**: the sun while its **ground illuminance** (top-of-atmosphere illuminance times the
  transmittance at the camera, CPU from the same model as the LUT) is above the moon's, with a hysteresis of a factor
  2 (the switch does not alternate); otherwise the moon. Below -18 degrees with the moon under the horizon: no cascade
  is rendered (`shadow.cascades` empty) and the factor is 1. The instances' caster cap uses `max(elevation,
  minShadowElevationDeg)` (below 5 degrees the horizon map and the low light carry the image).
- **Casters (findings 5, 6; §10.2)**: `SceneCulling` gets views 1..cascadeCount, each a **light-space volume**: the 4
  side planes of the cascade's rectangle, a **far cap** (the shared range's receiver far end: nothing behind the
  receivers) and a **near cap** towards the light at `maxCasterHeight / tan(max(el, minShadowElevationDeg))` beyond
  the receivers' near end for instances (a tree 100-450 m towards the low sun is inside the volume of every cascade
  whose receivers it shadows, so its shadow does not vanish at a split); terrain uses its own cap (Terrain casters).
  **De-duplication (review 2, finding 3)**: a caster is rejected for cascade i when its bounding sphere's **xy
  footprint** (a circle in light space) lies inside cascade i-1's **rectangle** shrunk by the blend band: its whole
  shadow then falls into rectangle i-1, whose receivers sample cascade i-1 (selection by the same rectangles, below),
  and cascade i-1's volume contains the caster (same depth range). Example of review 2: a 20 m tree 30 m from the
  camera casts 230 m at 5 degrees, onto ground far outside cascade 0's frustum slice; those ground points project to
  the tree's xy footprint in light space, so they lie in rectangle 0, sample cascade 0 and get the shadow, and the
  tree is correctly rejected for cascade 1. The rule is exact, which the round-1 rule with 3D boxes was not (cascade 0's
  prism is long at low sun but narrow; the cost is a longer depth range, not more texels). The
  **small-caster rule** rejects a caster whose radius projects below `smallCasterTexels` texels of the cascade;
  cascades 2-3 draw `shadowLodBias` LOD levels coarser than the main view's LOD (D10 keeps cascades 0-1 at the main
  view's LOD: no self-shadowing from LOD mismatch where it is visible), clamped at the mesh's last LOD. The count pass's
  existing rejection by the mesh's last end distance **from the main camera** also applies to the shadow views (a
  caster the main view does not draw at any LOD casts no shadow; finding 20d).
  CPU: each instance set's bounds are tested against each active view and the set's work chunks get a **view mask**
  (finding 20b: `WorkChunk` stays 16 bytes, `instance_common.glsl` is also included by `instance.vert`: the mask goes
  into today's unused `pad` (16 bits: views 0..15), and `scratchBase` (the prefix sum of `256 * popcount(viewMask)`)
  into a parallel `uint` buffer read only by the count and write passes). The count pass loops over the mask's views
  for its instance (volume test, de-duplication, small-caster rule, LOD), one scratch entry per (invocation, view);
  the **write pass loops over the chunk's views the same way** (finding 20a: every invocation reads its scratch entry
  per view). Buckets, indirect commands and the single-workgroup scan grow by the view count (5 in a2, up to 13 in b:
  `view * bucketsPerView + ...` as S18 reserved); the scan's time is reported separately in the Q6 measurement (the
  flight demo has many meshes; finding 20c). `RenderStatistics::instancesPerLod` stays the main view's; new
  `cascadeInstances[4]` and `cascadeTriangles[4]`. The single-view case (shadows off) keeps its cost
  (`sample_vulkan_scene_benchmark` with shadows off within 5% of the baseline; finding 37).
- **Shadow pipelines** (`ShadowPipelineTable`, one per pipeline class; D32 depth only, `slopeBias` dynamic, cull none
  for two-sided classes, back faces otherwise): opaque and two-sided: `shadow_depth.vert` (the instance placement of
  `instance_common.glsl`, pancaking, no fragment shader); alpha-masked: `shadow_alpha.frag` (the palette / vertex alpha
  test); `PrototypeFoliage`: `shadow_foliage.vert/.frag` (the wind of `prototype_foliage.vert` with the frame's phases
  and the foliage texture's alpha, no alpha to coverage: the cutoff). A class without an entry casts nothing (S27's
  impostor class, S34's skinned class add theirs; tested with a test class). Grass casts no shadows (it receives).
- **Terrain casters (finding 30; review 2, finding 4)**: the height-based cap applies to instances only. Terrain is
  taller than `maxCasterHeight` in the flight demo's mountains (a 200 m cliff 150 m from a receiver at 45 degrees is
  beyond the 120 m cap), so terrain nodes are selected up to the fixed distance `cascadeCasterDistance` (default 1371
  m, the instance cap at `minShadowElevationDeg`) towards the light **at every elevation**, and the horizon map's
  channel `G` starts at that same constant distance (A6). `TerrainPass::selectForShadow(const ShadowView&)` selects
  CDLOD nodes inside the cascade's rectangle and that caster range with **the main view's LOD metric and morph** (the same level per node as the main view draws, so
  the shadow mesh equals the receiver mesh; no per-cascade minimum level) and draws them with `terrain_shadow.vert`
  (the terrain vertex function with the main camera's morph, pancaking). Off with `terrainCastsInCascades = false` (the
  horizon map then shadows the terrain alone). `LightingStatistics` reports terrain nodes and triangles per cascade.
- **Receivers (finding 6)**: `shadowFactor(relPos, n)` in `shadows.glsl`: **cascade by the xy rectangle** (the first
  cascade whose rectangle contains the offset receiver's light-space xy, with a blend into the next one over the band at
  the rectangle border; the depth range is shared), the biases above, a fixed PCF of **3 x 3 bilinear compares (4 x 4
  texels)** with the comparison sampler (deterministic goldens; nit 32; nearest compares with manual bilinear weights
  when the format lacks linear filtering), multiplied by the horizon factor (A6) and the cloud shadow (A4); outside
  every rectangle the horizon factor alone.
- GPU sub-section `shadow.cascades` (inside the new `shadow` section, after `cull`).
- **Defaults and the measurement (Q6)**: 4 x 2048 at 1500 m with the de-duplication, the small-caster rule and the LOD
  bias. Step a2.4 measures `shadow.cascades` (with the terrain casters up to `cascadeCasterDistance`), `cull` and the
  scan in `game_world --benchmark forest` and `flyover` on the RTX
  path (release build of the benchmark recipe); if the median exceeds 2 ms, the defaults become 3 cascades / 1000 m
  (the horizon map carries the rest) and the result is recorded in Design changes. Flight sets 2500 m (2.9 m texels in
  the last cascade: terrain and large casters only).

### A6. Far terrain horizon shadows [a2]

- **`HorizonShadows`** (`vulkan/scene/HorizonShadows.{h,cpp}`, `lighting/horizon_map.comp`): an RG32F map of
  `horizonResolution^2` over `horizonExtent` (32 m texels at the defaults: the pyramid spacing), **world-anchored**: the
  anchor is snapped to the texel grid in integer metres (the S20 anchor rule: no float world positions) and moves only
  when the camera leaves the inner half of the extent (finding 10a). Each texel stores two **shadow heights** relative
  to the anchor's reference height, with d the horizontal shadow-light direction and the march sampling `h(x + t d)`:
  `R = max over t >= 64 m (2 texels) of (h - t * tan(el))` and `G` = the same maximum over `t >=
  cascadeCasterDistance` (a constant, 1371 m by default, for every elevation: the terrain closer than that, measured
  horizontally towards the light, is drawn into the cascades, A5; review 2, finding 4).
- **The march (review 2, finding 5)**: point samples at geometric steps would miss ridges narrower than the step
  (130 m at 1 km, 1.3 km at 10 km) and make the far shadow edge jump with the light direction. So the march is
  **conservative**: a **max-height mip chain of the pyramid** (`horizon_maxmip.comp`: int32 storage buffers like the
  pyramid, each level the maximum of 2 x 2 of the level below, about a third of the pyramid's size; rebuilt for the
  changed region when the pyramid changes, i.e. on a terrain edit or a new space) is sampled at the level whose texel
  is at least the step length, at the maximum over the texels the step covers. Up to 4 km from the texel the march
  steps linearly at the texel size (32 m, on the streamed levels nearer than 1 km where resident: 64 m steps), beyond
  that geometrically up to 24 km (or the space edge) on the max mips: about 125 + 40 samples per texel. A crest is
  never missed; far shadows can only grow by at most one mip texel's height range (reported in the statistics as the
  level used at 10 km). Heights in the integer units of §12.2 converted to float relative to the anchor.
- **Receiver test (finding 4)**: one-sided below the threshold: lit by `smoothstep(H - s, H, y)` (a point at the
  threshold is lit), `s = max(4 m, 2 * texel * tan(el))` (soft long shadows, no band on flat ground); inside a
  cascade rectangle with `terrainCastsInCascades` the receiver uses `G` and multiplies with the cascade factor, outside
  every rectangle it uses `R` (near occluders are the cascades' job; coarse pyramid heights never override the fine cascade near the
  camera). Test `HorizonShadows.FlatGroundIsLitAtLowSun`.
- **Incremental update (finding 10)**: three maps (front, previous, back). A new map is computed into the back map in
  row bands when the light direction moved by more than 0.1 degrees, the anchor moved, or a terrain edit (S22 patches)
  marked rows dirty. Rows per frame = `max(horizonRowsPerFrame, rows needed to finish within the time the light needs
  for 0.1 degrees)` (a fast day computes faster instead of lagging). Receivers blend the previous and the front map by
  the back map's progress (`mix(prev, front, rowsDone / rows)`), so the shadows move continuously instead of jumping
  every swap; on completion previous <- front <- back. **Each map carries its own transform** (anchor and light
  direction; the block holds the previous and the front map's, review 2 finding 15), so the blend is correct also
  across an anchor change (the back map was computed for the new anchor; the previous one keeps the old). Small camera
  moves recompute nothing. `forceFullUpdates`
  computes the whole map in the frame (goldens, `capture_lighting(horizon, full: true)`). Memory 3 x 8 MiB plus the max
  chain.
- Applied to every receiver (terrain, instances, grass; the clouds' undersides are not shadowed). Below -18 degrees
  with no moon the map is not updated; when the moon is the shadow light its direction drives the map.
- GPU sub-section `shadow.horizon` (empty when nothing is updated).

### A7. Frame structure, GPU sections, memory, statistics [a1, a2, b]

- Pass order (§10.5 pass 1 now exists): transfer (uploads, the lighting block, the cloud coverage once) ->
  **environment** (sky-view LUT, environment capture / prefilter / SH as due) -> cull (main view; a2: + cascade views;
  b: + atlas faces) -> **lights** (b: cluster assignment) -> **shadow** (a2: `shadow.cascades`, `shadow.horizon`; b:
  `shadow.atlas`) -> depth -> scene (`scene.terrain`, `scene.instances`, `scene.grass`, `scene.sky`, `scene.clouds`,
  `scene.debug`: debug lines) -> resolve -> post -> imgui. `gpuSectionNames()` adds per part: a1 `environment`,
  `scene.clouds`, `scene.debug`, `lighting.preview` (the on-demand preview conversions and the luminance reduction);
  a2 `shadow`, `shadow.cascades`, `shadow.horizon`; b `lights`, `shadow.atlas`. Every section is written every frame
  (empty when off), as S19a requires.
- Barriers (sync2): shadow map writes (`LATE_FRAGMENT_TESTS` / depth write) -> fragment shader sampled reads; LUT,
  map and SH compute writes -> fragment and compute reads; preview compute writes -> the ImGui draw's fragment read of
  the same frame (A8). The shadow images are shared by the frames in flight like the HDR targets (frames serialize on
  the queue).
- Memory at the defaults: a1: environment cubes 2 x 0.6 MiB, LUTs < 1 MiB, cloud coverage 1.3 MiB; a2: cascades 4 x
  2048^2 x 4 B = 64 MiB, horizon 3 x 1024^2 x 8 B = 24 MiB; b: atlas 2 layers x 4096^2 x D16 = 64 MiB, light and
  cluster buffers ~2 MiB per frame slot. Reported by `lightingStatistics().memoryBytes` and the Lighting tab.
- `RenderScene::lightingStatistics()` (`LightingStatistics`, newest completed frame): a1: shadow light, sun/moon
  elevation, EV100 and exposure, environment map last capture frame and progress, SH9, cloud parameters, the
  luminance reduction (mean, max, 16-bin histogram of log2 luminance) when requested, preview dispatches, memory;
  a2: cascade splits, boxes, texel sizes, last update frames, instances and triangles per cascade, terrain shadow nodes
  and triangles, horizon map progress (rows done, light direction of the front map, anchor); b: lights (submitted,
  visible, culled by budget, without tiles), cluster occupancy (max, mean, overflowed clusters), atlas (tiles used,
  static / dynamic re-renders this frame, cache hits, queue length).
- **Debug views' cost (finding 27)**: the debug view and overlay branches in `lighting.glsl` are uniform. The step that
  adds them (a1b.2) runs the benchmarks with the views off; if the frame time rises by more than 1%, the views move into
  a debug variant of the scene pipelines created on first use (`LIGHTING_DEBUG` define) and the default pipelines have
  no branch.

### A8. Developer UI: the Lighting tab (user requirement) [a1a base, a1b-a1c, a2 and b add sections]

- **Tab and key (D6, Q3)**: a new **engine** tab `DebugTab::Lighting = 4` on **F5**, drawn by `LightingTab`
  (`engine/app/LightingTab.{h,cpp}`, set by `EngineApp` with `setTabContent` like Performance and Engine); game tabs
  move to index 5 on (F6..F10; no game uses `addTab` today; `tests/test_engine_debug_ui.cpp`'s `addTab` index changes
  from 4 to 5 and its key test gets F5; `DebugUi.h`'s comments on F5..F10 change: `tabKey` and the tab bar need no
  code change beyond the enum value and the comments, nit 34). Rule recorded in §17.1: engine tabs F1-F5, game tabs
  F6-F10. `set_ui_tab lighting` selects it; `SAIGA_DEBUG_TAB=5` (`EngineMain.cpp`) opens it at start (AGENTS.md
  names it). Shown in every app with a device; without
  `environment.enabled` the time and weather controls are replaced by a line "the world has no WorldTime
  (EnvironmentSetup::enabled)" and the render sections stay.
- **ImGui images** (prerequisite, `saiga_vulkan`, finding 39): the vendored ImGui is 1.92 (`IMGUI_HAS_TEXTURES`); the
  backend keeps the legacy font upload (it does not set `ImGuiBackendFlags_RendererHasTextures`; the font texture keeps
  its ID). `ImGuiVulkanRenderer` binds a descriptor set per draw command from `ImDrawCmd::GetTexID()` (today it binds the
  font set once): `ImTextureID addTexture(vk::ImageView, vk::Sampler, vk::ImageLayout)` / `removeTexture(ImTextureID)`
  (sets from a small pool; a removed texture's set and the view it references are released through `deferDelete`, as a
  frame in flight may still draw it). Also in `ImGuiHeadlessRenderer`'s Vulkan path (the tab test runs headless). The
  preview images are written by compute in the same frame and sampled by ImGui: a sync2 barrier
  (`COMPUTE_SHADER` write -> `FRAGMENT_SHADER` read, layout `GENERAL` -> `SHADER_READ_ONLY_OPTIMAL`) before the post +
  ImGui pass. Tested by `VulkanImGui.ImageDraws` (GLFW-less, headless: an `ImGui::Image` of a known texture appears in
  the frame).
- **Preview images** (`vulkan/scene/LightingDebug.{h,cpp}`, `lighting/debug_preview.comp`): depth and data images are
  not displayable, so on demand a compute pass converts them into RGBA8 preview textures (registered once as ImGui
  textures) and reduces their source values (min, max, mean, 16-bin histogram; for `capture_lighting`'s JSON). Images
  (`LightingImage`): a1 `CloudCoverage`, `CloudShadow` (top-down around the camera), `TransmittanceLut`, `SkyViewLut`,
  `Environment` (cube cross, mip), `Irradiance` and `BrdfLut` (capture only, not in the tab), `Luminance` (the resolved
  HDR luminance, log2); a2 `Cascade` (layer), `CascadeStrip` (all layers side by side), `Horizon` (R or G channel);
  b `Atlas` (static | dynamic layer, tile overlay), `AtlasTile` (light, face), `Clusters` (heat map, slice z or max over
  z). The value mapping `PreviewMapping {range: auto (the GPU min / max) or [min, max], log, color_map (gray, turbo),
  tile_owners}`. A preview is computed only in a frame in which the tab draws it or a capture or probe asks for it
  (`LightingDebug::request(LightingImage, mapping)`); cost otherwise zero (tested).
- **Scene debug views** (`LightingDebugView`, one at a time, in the lighting block, uniform branches in `lighting.glsl`,
  A7's cost rule): a1 `None`, `LightingOnly` (white albedo, roughness 0.5), `Albedo`, `Normals`, `Roughness`,
  `Metalness`, `DirectOnly`, `AmbientOnly`, `SpecularOnly`, `CloudShadow`, `ShadowFactor` (sun/moon visibility; in a1
  clouds only), `Exposure` (scene luminance in EV100 as a heat map with a legend); a2 `ShadowCascades` (the factor
  tinted by the cascade index), `HorizonShadow`; b `ClusterLights` (lights per cluster as a heat map),
  `LocalLightsOnly`, `AtlasShadow`. **Debug frames (review 2, finding 24)**: while a view other than `None` is active
  (`RenderScene::setLightingDebug`, or the one-frame override of `screenshot` / `probe_lighting`) the frame latches
  `ToneMap::Identity`. That changes no pipeline: the scene pipelines do not depend on the tone map (the output mode is
  the push constant `linear`, the resolve mode is chosen per frame in the rendering info, `RenderScene.cpp:265`;
  confirmed by a test that counts pipeline creations across a view switch, `LightingDebug.ViewSwitchCreatesNoPipeline`).
  Data views (`Normals`, `Albedo`, `Roughness`, `Metalness`, the factors, the probe views) write their **raw values**
  into the fp16 target (no transfer curve, no pre-exposure) and the post pass shows them unencoded (a `PostPush` flag),
  so a readback of the resolved fp16 image gives the data (`Normals` of a +y plane reads (0.5, 1, 0.5)); the
  `Exposure` heat map and the `*Only` radiance views write display colours as clay does. A debug frame does not update
  the pre-exposure history (`lastExposure`), so the next normal frame continues from the last ACES frame instead of
  restarting from 1.
- **Overlays** (flags, combinable, drawn on top in the scene pass by `DebugLines`: camera-relative line lists in double,
  `lighting/debug_lines.vert/.frag`, depth tested or `onTop`; a1 builds `DebugLines` with `SunDirection` (a line from
  the camera focus towards the sun and the moon), so the mechanism exists before the cascades): a2 `CascadeColors`
  (tint by cascade, also with other views), `CascadeBounds` (each cascade's box and caster volume as a wireframe in its
  colour; with **freeze cascades** the matrices and the camera frustum stay, so one flies around and sees what each
  cascade covers), `CameraFrustumSplits` (the split planes of the frozen camera), `HorizonAnchor` (the horizon map's
  extent and inner half); b `LightVolumes` (spheres of point lights, cones of spot lights, in the light colour, shadowed
  ones thick; the selected light highlighted), `ClusterGrid` (the cluster frustum slices), `AtlasOwners` (the light of
  each atlas tile labelled in the atlas preview). `DebugLines` is a general mechanism (games can add lines later; S36
  uses it only for lighting).
- **Tab content** (DebugUiStyle: property tables, collapsed sections with persisted state, a toolbar row):
  - Toolbar (a1): tone map (ACES / clay, `RenderScene::setToneMap`; with the note "clay: studio light, time of day
    has no effect" while clay is on), debug view combo, overlay toggles, "force full updates", "copy JSON" (the
    `get_lighting` answer to the clipboard); a2 adds "freeze cascades".
  - **Time and weather** (a1, open): time-of-day slider (0-24 h) with day and rate (game hours per second, buttons x0,
    x1, x60, x600), weather sliders (cloudiness, rain, fog density, snow cover, wind x/z), each a write through
    `EngineApp::queueBetweenSteps` with `setWorldTime` / `setWeather` (validated, then `WorldApi`), so it replicates
    and shows in the MCP call log; in `--coop` they write the hidden host (A9), on a pure client they are read-only.
  - **Sun and moon** (a1, open): elevation and azimuth of both, illuminance (lux) at the top and on the ground, colour
    swatches, the shadow light, moon phase; "override sun" (az/el sliders that freeze the direction for inspection,
    `EnvironmentSettings::sunOverride`).
  - **Exposure** (a1, open): mode (TimeOfDay, Fixed), EV100 now and from the curve, compensation, the effective
    exposure and pre-exposure, the luminance of the last frame (mean, max, histogram; the `Luminance` reduction on
    demand while the section is open), the curve as a small plot with the current point.
  - **Ambient** (a1): `ambientScale`, `minAmbient`, the SH9 coefficients and the dominant colour, the environment
    cube cross with a mip slider and its capture progress; the prototype albedo colours (colour edits, D21).
  - **Sky and clouds** (a1): transmittance and sky-view LUT previews, the cloud coverage and the cloud-shadow map
    around the camera, cloud altitude, radius and shadow fade, star intensity, atmosphere parameters.
  - **Cascades** (a2): per cascade a row (split near/far, texel size, update interval, last update frame, instances,
    triangles, terrain nodes) and its preview image (click: enlarge in the panel; mapping range), bias sliders (depth,
    slope, normal offset in texels), count, resolution, distance, lambda, LOD bias, small-caster texels
    (`setLightingParameters`); a warning line when `shadow.cascades` exceeds 2 ms.
  - **Far terrain shadows** (a2): the horizon map preview (R / G), its progress bar, the direction it was computed for,
    the anchor, rows per frame, "recompute now".
  - **Local lights** (b): counts (submitted, visible, over budget, shadowed, without tiles), a light table (sortable:
    index, entity, kind, distance, lumen, range, cluster count, atlas tiles, last shadow render), selecting a row
    highlights its volume (`LightVolumes`) and offers "move camera to light" (`setViewCameraByTool`) and, for an entity
    light, its `LightSource` fields as widgets (written through `set_component`); cluster heat-map toggle with its
    scale; the atlas preview (static / dynamic layer, owner overlay) with cache statistics (hits, re-renders static /
    dynamic this frame, queue length, faces per frame); a warning for a shadowed light with `range` > 100 m.
  - **GPU and memory** (a1, sections added per part): the lighting sections as means over 2 s (from the app's GPU
    profile) next to the §17.10 shadow budget (2.5 ms), the lighting memory.

### A9. MCP tools (user requirement) [a1a base, a1b-a1c, a2 and b add keys]

Registered by `registerLightingTools(McpServer&, EngineApp&)` (`engine/app/LightingTools.{h,cpp}`) after
`registerAppTools`; every tool is also a console command (`DebugConsole`), e.g. `set_time_of_day hours=18.5`. Argument
names are snake_case and the same across tools (finding 26: `cascade`, `light`, `face`, `slice`, `mip`, `layer`,
`range`, `color_map`, `log`; never an overloaded `index`); unknown arguments are errors, rejected in the handler as
the world tools do (`McpServer::ObjectSchema` emits no `additionalProperties`; `WorldApi.cpp`, `WorldTools.cpp`;
nit 30); numbers must be finite and in range (the error names the argument and the range); errors without a device
name "this app renders no frames (no device)". Edits of world state go through `setWorldTime` / `setWeather` (then
`WorldApi`, logged under `api`); in `--coop` they are applied to the hidden host (`WorldApi::coop().host`, as `step`
does; finding 22) and reach the client by replication; on a pure client they are refused with the S16 message.
Settings edits go to `EngineApp::environmentSettings()` or `RenderScene::setLightingParameters` and return the previous
values (an undo payload the tool accepts). **A settings call is validated as a whole and applied all or nothing**
(review 2, finding 23): besides the per-key ranges the cross-key rules `atlas_tile` divides `atlas_resolution`,
`cluster_near < cluster_far`, `cascade_interval[0] == 1`, `cascade_count <= 4`, and a memory check (the lighting
memory after the change, e.g. 256 MiB for 4 x 4096 cascades, is returned as `memory_bytes`; a change above 1 GiB of
lighting memory is refused with the number); the error names every violated key. Settings that recreate resources
(sizes, `pbr` in a1b-a1c) are applied at the next frame start.

| Tool | Reads / changes | Part |
|---|---|---|
| `get_lighting(sections?)` | Reads: `time` (time of day, day, rate; null without `WorldTime`), `weather` of the view space, `sun` / `moon` (direction, azimuth / elevation deg, illuminance top and ground, phase), `shadow_light`, `tone_map` (`aces` / `identity`; with `identity` a note that clay uses the studio light and ignores time and weather), `exposure` (mode, `source`: curve / fixed / enclosed, ev100, curve ev100, compensation, the effective exposure, pre-exposure; without an environment `null` and the unit-less multiplier), `ambient` (scale, min, SH9), `environment` (last capture frame, progress), `clouds` (settings), `settings` (the whole `EnvironmentSettings`), `parameters` (all `LightingParameters`), `debug` (view, overlays, frozen), `gpu` (lighting sections' last frame and 2 s mean ms), `memory`; answered at once (no GPU readback: the luminance statistics are `capture_lighting(image: luminance, stats_only: true)`, review 2 finding 13); a2: `cascades` (per cascade split near/far, box, texel m, interval, last update frame, instances, triangles, terrain nodes), `horizon` (progress, direction, anchor, extent); b: `lights` (counts), `clusters` (max, mean, overflow), `atlas` (tiles used / total, re-renders, hits, queue, lights without tiles). `sections` filters the top-level keys. Changes nothing | a1a (a1b, a1c, a2, b add keys) |
| `set_time_of_day(hours?, day?, rate?)` | Changes `WorldTime`; `hours` [0, 24), `rate` [-3600, 3600]; returns `values` and `previous`. Error without `WorldTime`; `--coop`: the host | a1a |
| `set_weather(space?, cloudiness?, rain?, fog_density?, snow_cover?, wind?: [x, z], cloud_offset?: [x, z], wave_amplitude?, wave_steepness?)` | Changes the space's `Weather`; ranges as A1; returns `values` and `previous` | a1a |
| `set_lighting(...)` | Changes the settings (not world state; not replicated, not saved), validated as a whole and applied all or nothing: `tone_map` (`aces`, `identity`; blocker 1c), `exposure_mode` (`time_of_day`, `fixed`), `ev100`, `exposure_compensation`, `ambient_scale`, `min_ambient` ([r, g, b] lux), `sun_override` ({azimuth_deg, elevation_deg} or null), `cloud_altitude`, `cloud_radius`, `cloud_shadow_fade`, `star_intensity`, `ground_albedo`, and every `LightingParameters` field by its snake_case name (a1: `clouds`, `environment_resolution`, `environment_update_degrees`, `force_full_updates`; a2: `sun_shadows`, `cascade_count`, `cascade_resolution`, `shadow_distance`, `split_lambda`, `cascade_interval` ([4]), `depth_bias_texels`, `slope_bias`, `normal_offset_texels`, `cascade_blend`, `shadow_lod_bias`, `small_caster_texels`, `max_caster_height`, `min_shadow_elevation_deg`, `terrain_casts_in_cascades`, `horizon_shadows`, `horizon_resolution`, `horizon_extent`, `horizon_rows_per_frame`; b: `local_lights`, `clustered_lights`, `max_lights`, `cluster_x`, `cluster_y`, `cluster_z`, `cluster_near`, `cluster_far`, `max_lights_per_cluster`, `atlas_resolution`, `atlas_tile`, `faces_per_frame`). Applied from the next frame; returns `values` and `previous` | a1a (a1b, a1c, a2, b add keys) |
| `set_lighting_debug(view?, overlays?, freeze_cascades?, highlight_light?, mapping?: {range, log, color_map})` | Changes the debug view (names of `LightingDebugView`, snake_case), the overlays (a list of names; replaces the set), the cascade freeze (a2), the highlighted light (b) and the tab's preview mapping; returns the state and the known names. Changes only the rendering | a1b |
| `capture_lighting(image, cascade?, light?, face?, slice?, channel?, mip?, layer?, range?, log?, color_map?, full?, max_width?, format?, stats_only?)` | Returns a data image (none with `stats_only: true`, the one path for the statistics of a GPU image) plus a JSON text block: what it shows (the mapping range, the cascade's world box and texel size, the horizon map's extent, anchor and direction, the atlas tiles' owners) and the **numbers** of the source values (`min`, `max`, `mean`, a 16-bin histogram). `format` png (default for data images) or jpg; downscaled to `max_width` (default 1280, 64..8192, as `screenshot`). `image`: a1 `cloud_coverage`, `cloud_shadow`, `transmittance_lut`, `sky_view_lut`, `environment` (`mip`), `irradiance`, `brdf_lut`, `luminance`; a2 `cascade` (`cascade`, or all side by side without it), `horizon` (`channel` r/g); b `atlas` (`layer` static/dynamic), `atlas_tile` (`light`, `face`), `clusters` (`slice` or max). `full: true` fully updates the horizon map and the environment map first. Deferred: answered from the frame that computed the preview (a frame callback, never rendered in the handler); a deadline error like `screenshot` | a1b (a1c, a2, b add images) |
| `screenshot(..., lighting_view?, overlays?)` (extended, finding 23) | The existing tool gets two optional arguments: the frame is rendered with that debug view and overlays (a one-frame override like `gui`; requests batch per (gui, view, overlays)), then restored. No second "render a frame" path | a1b |
| `probe_lighting(pixel?: [x, y], position?: [x, y, z])` (finding 25; review 2, finding 7) | Reads, for one pixel (or the screen position of a world point): depth and world position, albedo / roughness / metalness, normal, the three shadow factors (cascade, horizon, cloud; a2: the cascade index, the horizon `H` (R and G) and the receiver height), direct / ambient / specular radiance, luminance and its EV100; b: the cluster index and its light handles with each light's contribution. A **short frame sequence** (below); answers with `frames` and `consistent`. Changes nothing | a1b (a2, b add keys) |
| `list_lights(near?: {position, radius}, shadowed?, visible?, limit? (100, <= 1000), after?)` | Reads the scene's lights (entity lights and lights added on the scene): handle, entity (or null), kind, position, colour, lumen, range, cone angles, shadowed, visible this frame, distance to the camera, clusters touched, atlas tiles and their last render frame, cache state; paged (`next`) | b |

**`probe_lighting`'s sequence (review 2, finding 7)**: the renderer is forward, so after the scene pass only the
resolved colour and depth exist, and writing material values from fragment shaders into a buffer needs the optional
`fragmentStoresAndAtomics` (D17). So the probe renders **6 frames** (a1b): frame 1 normal (the luminance at the pixel
and the depth), then the data views `ProbeMaterialA` (albedo rgb, roughness), `ProbeMaterialB` (normal xyz encoded,
metalness), `DirectOnly`, `AmbientOnly`, `SpecularOnly` (radiance, linear); each frame copies the resolved fp16 pixel
(a 1 x 1 copy, the MSAA-averaged value, stated in the answer) into a readback buffer. In the last frame one compute
dispatch (`debug_probe.comp`) reconstructs the position from the depth and evaluates the depth-derived values at that
pixel: the cloud factor; a2 the cascade index and factor, the horizon heights and factor; b the cluster and each
light's contribution (with the normal and material read in the earlier frames, passed as push constants). During the
sequence the tool **holds the camera** (the camera hold of `set_camera`) and the scene **holds the environment state**
of frame 1 (`RenderScene::holdEnvironment`), so time of day and weather do not move between the frames; a dynamic
object may move: `consistent` is false when the pixel's depth changed by more than 1% between the frames. The deadline
and its error are `screenshot`'s; requests batch per frame like screenshots; the debug view the user set is restored.

`app_status` gets `lighting: {time_of_day, ev100, tone_map, debug_view}` (null without a device). `measure_performance(frames)`
reports the new GPU sections automatically (they are in `gpuSectionNames`).

### A10. Games and tech demo [a1, a2]

- **Clay is the default of every game today (review 2, blocker 1)**: `RenderSceneParameters::toneMap =
  ToneMap::Identity` (`RenderSceneTypes.h`), no game sets it, `--tonemap aces` is opt-in (`EngineMain.cpp`), and clay
  shows the studio light (A3). So without a change the demos would show the same studio image at noon and at 23:00.
  **Decision D22**: a game with the environment enabled renders **ACES by default**: `enableEnvironment` sets
  `appSetup.render.toneMap = ToneMap::Aces` (before `engineMain` parses the command line, so `--tonemap identity`
  still gives the clay view), and the time of day is visible in every demo. What follows, stated in the a1c PR:
  the demos' look changes (sky, exposure, colours, D21); the `resolve` GPU section is no longer empty (the tone-map
  weighted resolve of MSAA, S17); the post pass tone maps; the benchmark baselines of `game_world` move for that reason
  alone, which the ACES baseline before a1 separates from the lighting cost (Benchmarks); the Lighting tab's toolbar
  and `set_lighting(tone_map)` switch between both views at run time.
- `samples/games/common/Environment.{h,cpp}`: `enableEnvironment(WorldSetup&, EngineAppSetup&, const std::string&
  game)` reads `data/<game>/environment.json` if present (R-DATA-1) into the world setup, the app's
  `EnvironmentSettings` and `LightingParameters`, sets the ACES tone map and calls `addEnvironmentSystems`.
- **Engine options `--time-of-day h` and `--day-length s`** (real seconds per game day; sets `rate`; review 2, finding
  19): valued engine options of `parseCommandLine` (like `--exposure`, every engine game), stored in `AppOptions` and
  applied by the app after the world exists (`setWorldTime`); a game whose world has no `WorldTime` (environment not
  enabled) stops at start with "--time-of-day: this game has no world time (EnvironmentSetup::enabled)" and exit code
  1. Game flags stay value-less (`EngineMain.cpp`); `game_world` parses its valued `--lamps N` (b) itself, as it does
  with `--base-spacing` (`samples/games/world/main.cpp`), before calling `engineMain`.
- **Entities (review 2, finding 22)**: with the environment enabled a world gets a world-state entity (if it had none)
  and `Weather` on every space-state entity. a1c checks the demos' scenarios and `tests/` for committed entity counts
  or snapshot sizes of game worlds (the engine tests use plain worlds; the games have no tests in `tests/`) and runs
  every demo's `--scenarios` in all variants; the flight content hashes hash the generated content, not world
  entities, and must not change (checked by the scenario `flight_content_hashes` against
  `data/flight/content_hashes.json`).
- **Flight** (`data/flight/environment.json`: latitude 50, cloud altitude 1800 m, `lighting.shadowDistance` 2500 m in
  a2): with `--demo` a full day passes in **10 minutes** starting at 6:00 (sunrise, noon, sunset,
  night with the moon; Q5), so `SAIGA_SCREENSHOT` with `--time-of-day 12|19|23` shows each; `--day-length 240` for a
  quick review. a1: sky, clouds, cloud shadows, PBR, exposure by day and night; a2: its trees (content sets,
  `PrototypeFoliage`) and ruins cast cascade shadows, the mountains cast horizon shadows at sunrise and sunset.
- **game_world**: environment enabled (`data/world/environment.json`), time 10:00 frozen, cloudiness 0.3 by default
  (its benchmark keeps this fixed time and weather with `forceFullUpdates` off; the benchmark JSON records the time,
  cloudiness, EV100 and which lighting features are on); `--time-of-day`.
- **Racing, platformer**: environment enabled with code defaults (no data file): frozen noon (racing) and frozen noon
  for the platformer's flat space; enclosed spaces of the platformer use their template's ambient and EV100 (A1); they
  must look right in `--demo` screenshots under ACES (the cars, the character and the props keep their material
  colours).

### B1. `LightSource` and the extract (`saiga_engine`) [b]

- `struct LightSource` (`engine/components/LightSource.h`, ID `EngineComponent::LightSource` appended after `Weather`,
  always registered, `Replicate::HostReliable` (rarely changing, on/off toggles reliable, not in snapshots; S32 may
  move it to cosmetic interest), saved): `LightKind kind = Point; Vec3 color = Vec3(1, 0.9, 0.8) /* linear */; double
  lumen = 800; double range = 10; double innerAngleDeg = 25, outerAngleDeg = 35; Vec3 offset = Vec3::Zero(); Quat
  rotation = Quat::Identity() /* the spot's -z relative to the entity */; bool on = true; bool castShadows = false;
  double priority = 1; double flickerAmplitude = 0, flickerFrequency = 0;` (§10.7: presentation; its on/off state
  replicates with its owner). Ranges: `range` (0, 1000], angles 0 < inner <= outer < 90, `lumen` >= 0. Entity lights
  are for a moderate number of gameplay lights (lamps, torches, vehicle lights); hundreds of static lights of a base are
  S30's piece lights through `addLights` (one list per chunk, no entities; nit 52).
- **Extract** (`ExtractSystem`, S19b's pattern for `RenderInstance`): the structural log adds / removes the light
  (`RenderSink::addLight(chunk, LightData)` / `removeLight`); a changed `LightSource` (a write through the API or a
  replicated change: the component's change counter, as `RenderInstance` names) calls `setLight`; a moving entity's
  light follows the interpolated pose every frame (`setLight` with the new chunk-relative position; a chunk change is
  a new chunk in `setLight`). Flicker is a render effect: `LightData::flicker {amplitude, frequency, seed}`, the shader
  phase from `EnvironmentState::phases` (no wall clock).
- `RecordingRenderSink` keeps the live lights (`lights()`) for scenarios.

### B2. Lights in `RenderScene` [b]

- `LightData` (`RenderSceneTypes.h`): `LightKind kind; vec3 position /* chunk-relative */; vec3 direction; vec3 color;
  float lumen, range, cosInner, cosOuter; bool castShadows; float priority; LightFlicker flicker;`.
  `LightHandle addLight(const ChunkId&, const LightData&)`, `void setLight(LightHandle, const ChunkId&, const
  LightData&)`, `void removeLight(LightHandle)`, and the bulk path for S30's piece lights `LightListHandle
  addLights(const ChunkId&, const std::vector<LightData>&)` / `removeLights(LightListHandle)` (one CPU list per chunk,
  never entities; tested in S36: `Lights.BulkListPerChunk`).
- **`LocalLights`** (`vulkan/scene/LocalLights.{h,cpp}`): the CPU table (double-free positions: chunk + offset); per
  frame: lights of the camera's space within `clusterFar + range` of the camera and intersecting the frustum, sorted
  by importance (`priority * lumen / max(distance, range)^2`, ties by handle: deterministic), capped at `maxLights`
  (the rest counted as over budget); written camera-relative (double -> float) into the frame slot's light buffer
  (48 bytes each: position + range, direction + kind, colour * intensity in candela (point `lumen / 4 pi`, spot
  `lumen / pi`), cone cosines, shadow tile index, flicker).
- Light falloff: inverse square with Karis's window `saturate(1 - (d / range)^4)^2`; spot: smooth cone between the
  cosines. Evaluated by `shadeSurface` (same BRDF as the sun). Enclosed spaces get local lights like open ones (their
  main use: dungeons; tested and in the golden).

### B3. Clustered forward light culling [b]

- Cluster grid `clusterX x clusterY x clusterZ` (16 x 9 x 24) over the screen; depth slices: slice 0 from the near plane
  to `clusterNear` (2 m), then exponential from `clusterNear` to `clusterFar` (1000 m; beyond it no local light: the
  light selection drops them) (finding 32d); the cluster AABBs in view space are recomputed on a size or projection
  change (`light_cluster_bounds.comp`).
- `light_cluster.comp` (section `lights`, after `cull`): **one invocation per cluster** loops the frame's lights in
  importance order (the buffer is sorted) and tests them (sphere vs AABB; spots: sphere bound plus a cone test); each
  cluster has a **fixed slot** of `maxLightsPerCluster` `uint` indices (no `atomicAdd`, deterministic; `uint`, not
  `uint16`, which would need the optional `storageBuffer16BitAccess`), the count in the cluster table; the excess is
  counted as overflow and dropped (the least important ones, as the loop is in importance order). Memory at the
  defaults: 3456 clusters x 128 x 4 B = 1.7 MiB per frame slot. Fragment shaders find their cluster from
  `gl_FragCoord.xy` and the view depth and loop its lights (`clusters.glsl`).
- The forward passes need no depth pre-pass for this (the grid is depth-independent). S38's froxel fog reads the same
  table (the grid is bound as storage buffers, also for compute).

### B4. Cached shadow atlas [b]

- **`ShadowAtlas`** (`vulkan/scene/ShadowAtlas.{h,cpp}`): two D16 layers (`atlasResolution^2`, a `Texture2DArray` of 2
  layers): **static** (cached: static casters, terrain) and **dynamic** (dynamic instances only, re-rendered every
  frame for tiles whose light range contains a dynamic instance); receivers take the minimum visibility of both
  (§10.7). Fixed tiles of `atlasTile` (512: 64 tiles at the defaults); a spot light uses 1 tile, a point light 6 tiles
  (cube faces as 90-degree perspective views).
- **Depth (finding 14)**: the atlas passes write **linear depth** `gl_FragDepth = distance(light, p) / range` (depth-only
  passes, no early-z to lose): with D16 the step is `range / 65536` (1.5 mm at 100 m, 0.3 mm at 18 m), the same at
  every distance; clear 1.0, compare `LESS_OR_EQUAL`; the receiver compares its own `distance / range` minus a bias of
  one step plus a normal offset of one texel's footprint. Ranges up to 100 m are fine (the tab warns above).
- **Selection by tiles (finding 15)**: lights with `castShadows` that are visible, in importance order (as B2, with a
  hysteresis of 20% so lights near the cut do not alternate), take tiles (spot 1, point 6) until the atlas is full; the
  rest are unshadowed and counted (`atlas.lights_without_tiles`). At the defaults (64 tiles of 512) the atlas holds
  10 point lights or 64 spot lights; §10.7's "16 shadowed lights at High" becomes a **tile budget** (M5; review 2,
  finding 27). A light that loses its tiles frees them for the
  next; a light without rendered tiles yet is drawn unshadowed (flag in the light record) until its faces are done.
- **Cache (finding 16)**: each tile keeps `(light handle, the light's pose and cone hash, rendered frame)` and a valid
  flag. Content changes are **dirty bounds**: every `addInstanceSet`, `updateInstances`, `removeInstance`,
  `removeInstanceSet`, terrain patch and terrain tile change, and `setMaterial` of a caster class (its sets' bounds)
  pushes its world bounds (double, camera-independent) into a per-frame dirty list; a static tile whose light sphere
  intersects a dirty bound is invalidated. The static layer renders terrain at the **finest resident level** under the
  light sphere at render time (level 0 is resident only near players: a lamp 800 m away has only coarser tiles; review
  2, finding 25), not the camera's CDLOD selection (which changes after the tile was cached); when a finer level
  becomes resident inside a light sphere, the terrain tile change pushes its bounds as a dirty bound (the list above)
  and the tile is re-rendered once at the finer level. Foliage is drawn with **wind phase 0**. A face is re-rendered when invalid or its key changed; the re-render queue is ordered by importance then
  age and limited to `facesPerFrame` per frame (dynamic faces first; a skipped dynamic face keeps last frame's image:
  a one-frame lag for moving casters when over budget). A dynamic tile is cleared once when the last dynamic instance
  leaves its range.
- **Culling**: each re-rendered face is a view of `SceneCulling` (view mask bits beyond the cascades; at most
  `facesPerFrame` atlas views per frame; the 16-bit `viewMask` of A5 holds main + 4 cascades + 11 faces, so
  `facesPerFrame <= 11` is asserted and validated by `set_lighting`, nit 36); the static layer draws static sets and
  terrain nodes in range (`TerrainPass::selectForShadow` with a sphere and the fixed level), the dynamic layer only
  dynamics (a culling flag per view: static, dynamic, both).
- **Receivers**: `atlasShadow(light, relPos, n)`: the tile from the light record (point: the face by the major axis),
  2 x 2 bilinear PCF.
- GPU sub-section `shadow.atlas`.

### B5. Developer UI, part b

The Lighting tab's "Local lights" section, the `LightVolumes`, `ClusterGrid` and `AtlasOwners` overlays, the
`ClusterLights`, `LocalLightsOnly` and `AtlasShadow` views and the atlas and cluster previews of A8, built with the
feature steps (B steps 2-4).

### B6. MCP tools, part b

`list_lights`; the part-b keys of `get_lighting`, `set_lighting`, `probe_lighting`, `set_lighting_debug(highlight_light)`
and `capture_lighting(atlas | atlas_tile | clusters)` (A9's table), each with its feature step. Entity lights are edited
with the existing `set_component` / `add_component` / `remove_component` (`LightSource`), scene lights (pieces) are
read-only.

### B7. Tech demo, part b

- **game_world** at night: `--lamps N` (default 400; a valued option parsed by `game_world`'s `main` like
  `--base-spacing`, A10) spawns lamp entities (a pole mesh from `RenderAssets` with
  `RenderInstance` and a warm `LightSource`, 1200 lm, range 18 m) on a grid along the terrain around the start point
  (game policy, positions from the terrain height), every 8th with `castShadows` (the tile budget decides which get
  tiles: about 10 point lights at 64 tiles), a few spot lights (street lamps pointing down, 25/35 degrees) and 20
  flickering torches; `--time-of-day 22` shows them; `--benchmark night` (the flyover path at 22:00 with the lamps) is
  run and reported in Results only (not committed to `benchmarks.json`).
- **Flight**: a shadowed landing light (spot, 20000 lm, range 120 m) switched on below 5 degrees sun elevation by the
  game (policy); `--time-of-day 23 --demo` shows it.

## Out of scope

- Auto exposure, AO, quality tiers and the budget gate (S37); volumetric fog and light shafts (S38); volumetric clouds
  and their transmittance shadow map (S42); water reflections (S39); rain and snow particles (S41); the full material
  set, wetness and snow on materials (S43: `rain` and `snowCover` have no shading effect in S36; finding 35).
- Impostor shadows (S27), piece lights (S30), skinned shadow casters (S34 adds its class's shadow pipeline entry).
- Global illumination, contact shadows, foliage translucency, local reflection probes (§10.7 "not provided").
- Temporal filtering of shadows; screen-space shadows; variance or moment shadow maps.
- Light culling for transparents beyond what clusters give (S39 uses the same clusters).
- Cut after review round 1 (over-scoping): star twinkle, a synodic moon model (the phase is a parameter), the
  `HorizonProfile` overlay, the `Irradiance` and `BrdfLut` previews in the tab (they stay in `capture_lighting`), the
  flight demo's navigation lights and strobe, a committed `--benchmark night` baseline, environment files for racing
  and the platformer.

## Design notes

- **Why five parts** (D1): a1 changes every scene shader, the world state and the app but has no culling risk and
  gives the day/night demo early; it is too large for one PR (about 200 changed files with its tests and goldens), so
  it is cut where no flip is needed: a1a (state, extract, tools; no shader), a1b (the new light model as a second
  pipeline variant, proven by the unchanged clay goldens), a1c (sky, clouds, games, flip). a2 is the culling and shadow
  work with its tuning; b adds a self-contained subsystem on top. Each part is a PR of reviewable size under
  CodeRabbit's limits and keeps CI green per commit through a default that flips in its last commit together with the
  goldens.
- **World state registered always, global only when enabled** (D4): the registry's prefix rule needs every engine ID
  registered; registering `WorldTime` / `Weather` as plain components in plain worlds keeps their entity layout (no
  world-state entity), hashes and the golden save, and the schema is one for every world.
- **Presentation settings in the app, simulation settings in the world** (finding 13): `WorldSetup` is constant once
  the world exists; exposure, ambient, clouds, atmosphere and the sun override are changed by the tab and tools at run
  time, so the app owns them. The simulation reads only `enabled`, the celestial model and the wind scale.
- **Sun interpolation and clients**: the render uses `timeOfDay + (alpha - 1) * rate * dt` on the host and extrapolates
  from the last snapshot with `HostClock` on a client, so a fast demo day moves the sun smoothly at 60 fps with a 30 Hz
  step and any snapshot rate.
- **Clay with a studio environment that reproduces S19a** (D7): clay is the geometry look of §10.5; it keeps S19a's
  Lambert light and (a2) the shadows but not the time of day, the atmosphere or the clouds, so its goldens guard
  geometry and shadows only. Reproducing S19a exactly turns the five clay goldens into the regression guard of the
  shader rewrite (a1b-a1c) and regenerates them once (a2) instead of twice.
- **ACES for games with an environment, constant prototype colours** (D21, D22): the time of day, the sky and the
  exposure exist only in the ACES path; a gray world under a blue sky makes the sunset and the cloud shadows hard to
  judge, and seven constants cost nothing until S43's materials.
- **Stable cascades**: sphere fitting + texel snapping (no shimmer); camera-centred spheres for staggered cascades
  (coverage after turns); double centres (correct for cascades updated every N frames); pancaking in the vertex shader
  instead of an extended near plane (precision, no optional feature); the main view's LOD in cascades 0-1 (no
  self-shadowing from LOD mismatch); one shared light-space depth range, so a cascade is an xy rectangle and both the
  receiver selection and the caster de-duplication are exact by xy (an orthographic shadow stays in its caster's xy
  footprint; no long-shadow loss at splits, no repeated near casters); terrain casters to a fixed distance at every
  elevation (mountains taller than the instance cap). The fixed 3 x 3 bilinear PCF keeps goldens deterministic.
- **Horizon shadow height instead of a horizon angle map** (D11): a height threshold is valid for every receiver
  height (trees, aircraft over the valley), the angle only for receivers on the ground; both need one march per texel.
  The second channel `G` (occluders beyond the cascades' caster range) lets cascades and horizon combine without
  coarse terrain overriding fine shadows; the one-sided band keeps flat ground lit at low sun.
- **Three horizon maps and blending**: the shown shadows move continuously with the light; a world-anchored map does
  not recompute for camera moves; a transform per map keeps the blend right across an anchor move. The max-mip march
  never misses a crest (conservative far shadows instead of shadows that flicker with the light direction).
- **Clusters, not tiles**: forward shading with alpha to coverage and MSAA; the depth-independent grid needs no
  pre-pass and serves S38's froxels. Fixed per-cluster slots: deterministic and no atomics.
- **Two atlas layers**: a static layer that is rendered once per change and a dynamic layer of dynamics only keeps the
  per-frame cost to the faces near moving objects (the §10.7 requirement); minimum of both at the receiver. Linear
  depth makes D16 precise at every distance.
- **Preview textures for UI and MCP**: one compute conversion serves `ImGui::Image` and `capture_lighting`, so what an
  agent captures is exactly what a developer sees in the tab (same mapping), with the numbers alongside; nothing is
  computed when neither asks.
- **UI and tools grow with each feature** (finding 12): the cascade previews, bounds and freeze exist when the cascades
  are tuned; each step is reviewable with screenshots and tool output; the goldens are committed last, after tuning.
- **Debug views latch Identity and write raw data**: the views' colours are data (normals, factors, heat maps);
  exposure, ACES and the transfer curve would distort them. The same views feed `probe_lighting` (a frame sequence of
  1 x 1 readbacks), because a forward renderer has no G-buffer and fragment stores are an optional feature.
- **Physical units with pre-exposure**: the S17 contract already keeps fp16 values near 1; 120 000 lux and 0.1 lux
  both fit once the exposure clamp covers EV100 -10..20.

## Implementation steps

Each step builds without warnings (GCC, Clang, MSVC) and passes ctest (CI recipe, lavapipe). Images change only in the
last commit of each part (the flip commit with its goldens). Benchmarks are run before the first step and after the
last of each part (Benchmarks). Each feature step adds its tab section, statistics, debug views, previews and tool keys
with their tests (finding 12).

**Part a1a (S36a1a): world state and the first tools (no image change)**

1. **World time and weather.** `LightingTypes.h` (the value types only: `EnvironmentSettings`, `ExposureSettings`,
   `LightingParameters`, ...; review 2, finding 17: the types step 1 uses), `WorldTime`, `Weather` (with `validAt`;
   always registered; global / space state with `environment.enabled`), `registerEngineComponents(Registry&, const
   WorldSetup&)`, `EnvironmentSetup`, `EnvironmentFile` / `readEnvironmentFile`, `worldTimeSystem` (normalisation),
   `addEnvironmentSystems`, `EnvironmentWrites`, `Celestial`, `ExposureSettings::ev100At`; the test-local `Weather`
   structs of `test_engine_replication.cpp`, `test_engine_save.cpp` and `test_engine_world_api.cpp` renamed
   `TestWeather` (finding 33); `test_engine_lighting.cpp` (no device) and the scenario `world_time_advances` (engine
   scenarios, all variants, with the client extrapolation check); the schema regenerated (AGENTS.md recipe).
2. **Render interface.** `EnvironmentState`, `EnvironmentSettings` (owned by `EngineApp`),
   `RenderScene::setEnvironment` / `hasEnvironment` / `setLightingParameters` (stored, not used for shading yet), the
   exposure rule of A3 in `latchFrame` (`frameExposure`; a scene without an environment is unchanged),
   `RenderSink::setEnvironment` (+ both sinks), `environmentState` (incl. the client extrapolation with `validAt` and
   enclosed spaces from the template), the app's per-frame call, the engine options `--time-of-day` / `--day-length`;
   `SpaceTemplate::ambient` / `ev100`. Tests: extract and app (no device), `Exposure.Ev100RangeReachesPost`,
   `Exposure.NoEnvironmentKeepsUnitlessExposure`. (No game enables the environment yet: no image changes.)
3. **ImGui textures.** `ImGuiVulkanRenderer::addTexture` / `removeTexture`, per-command binding from `GetTexID()`,
   `deferDelete` of removed sets, the headless path; `VulkanImGui.ImageDraws`.
4. **Tab and tools, first half.** `DebugTab::Lighting` on F5 (game tabs F6-F10), `LightingTab` with Time and weather,
   Sun and moon and the exposure numbers (CPU values); `LightingTools` with `get_lighting` (the a1a keys),
   `set_time_of_day`, `set_weather`, `set_lighting` (the settings and `tone_map`; validated as a whole),
   `app_status.lighting`; the console commands. Tests: `DebugUi.*`, `EngineAppNoRender.SetUiTab`, the tool tests of
   these keys (one app per suite), `EngineAppRender.LightingTabRenders` (the a1a sections); `test_engine_app`'s ctest
   time before and after (finding 28). Benchmarks: none needed (no render change; `latchFrame` only).

**Part a1b (S36a1b): lighting block, PBR, debug core (no image change)**

1. **Lighting block and set 1.** `SceneLighting.h` v2, `SceneLightingSet` (A3: layout, block, per-frame sets,
   dummies; the shadow layout follows in a2), the final set-1 layout for every scene pipeline (terrain, grass and sky
   included; grass's foliage binding moves), `lighting.glsl`, `pbr.glsl`, `ambient.glsl` with a constant `SkyAmbient`
   (the SH of a uniform sky colour until a1c), the BRDF LUT (`BrdfLut`), the clay branch with `StudioEnvironment`
   (`kSceneStudio`, `kPrototypeStudio`), the prototype albedo colours, the `LIGHTING_PBR` pipeline variant selected by
   `pbr = false` (default). Tests: BRDF LUT vs CPU, white furnace, PBR unit cases, and the **five existing clay golden
   tests run a second time with `pbr = true` against the same committed goldens** (`world_overview`, `world_forest`,
   `generated_site`, `terrain_edit_crater` in `test_engine_terrain_render.cpp`, `scene_flat_box` in
   `test_vulkan_scene.cpp`; a test parameter; review 2, finding 9: no separate `Pbr.LegacyLightingMatchesGoldens`, the
   existing tests are the guard of the default path). `RenderSceneSuiteTest` in `render_scene_test_context.h`
   (finding 12).
2. **Debug core.** `LightingDebug` (the a1 debug views with raw data output, the preview conversion with its
   reduction, the debug-frame rules of A8), `DebugLines` with `SunDirection`; the tab's toolbar (tone map, view,
   overlays), Exposure, Ambient (with the albedo colours) and GPU and memory sections; `set_lighting_debug`,
   `capture_lighting(luminance, brdf_lut, stats_only)`, `probe_lighting` (the frame sequence), `screenshot
   (lighting_view, overlays)`. The debug views' cost measured (A7, on the `LIGHTING_PBR` variant). Tests: the debug and
   tool tests of these keys, `LightingDebug.ViewSwitchCreatesNoPipeline`; the new test binaries' ctest time measured on
   lavapipe (target below).

**Part a1c (S36a1c): sky, environment map, clouds, games, flip**

1. **Atmosphere and environment map.** `Atmosphere` (LUTs, sky pass, sun, moon, stars), the fog colour,
   `EnvironmentMap` (front / back capture, prefilter, SH, the lower hemisphere), `SkyAmbient` from the SH, the
   `environment` section; the tab's Sky section (LUT previews, environment cube); tool keys (`transmittance_lut`,
   `sky_view_lut`, `environment`, `irradiance`; `get_lighting.environment`). Tests: LUT reference values, SH of a
   uniform sky, incremental capture rules, no mixed captures.
2. **Games.** `enableEnvironment` (ACES by default, D22), `data/flight/environment.json`, `data/world/environment.json`,
   the flight day cycle with `--demo` (10 min), racing / platformer / `game_world` code settings, each game setting
   `pbr = true` in its setup until the flip; the entity check of A10; screenshots by day, sunset and night (now
   possible: ACES); the demos' scenarios (`--scenarios`, all variants).
3. **Clouds.** `CloudLayer` (coverage noise, mesh with curvature and edge fade, `scene.clouds`), `cloudShadow` (above
   the layer, low sun), clouds in the environment capture; the tab's cloud previews; tool keys (`cloud_coverage`,
   `cloud_shadow`; the cloud settings in `set_lighting`). Tests: coverage vs cloudiness, cloud shadow projection on a
   test plane, no shadow above the layer, the low-sun fade, the wrap of `cloudOffset` is seamless, no rim at the edge.
   (The seam that moves to a2 if a1c approaches 100 files: 8 files.)
4. **Tuning.** Defaults of exposure curve, atmosphere, clouds, ambient and the prototype albedo colours tuned with the
   tab and the tools on the flight demo and `game_world` (screenshots in the PR); benchmarks after (Benchmarks: the
   ACES baseline of the unchanged code, the legacy variant, the new path).
5. **Flip (last commit of a1c).** `pbr = true` and then the field removed with the legacy variant, the migration of A3
   (`PrototypeSettings`, `WorldSettings`, `setPrototypeEnvironment` -> `setFoliageTexture`, `setTime`,
   `kEnclosedLighting`, `SCENE_FOG`), the games' temporary `pbr = true` removed, the second (`pbr = true`) run of the
   clay golden tests removed (one path is left); the five clay goldens **pass unchanged** (if one needs a
   regeneration, the difference is explained in the PR or it is a bug of the rewrite); `hdr_tonemap` unchanged (no
   environment); the new goldens `lighting_noon`, `lighting_sunset`, `lighting_night`, `lighting_clouds` (ACES, without
   sun shadows; cloud shadows on). No tuning in this commit. Design-doc amendments of a1, AGENTS.md, Results (a1).

**Part a2 (S36a2)**

1. **Multi-view culling.** View volumes, view masks (in `WorkChunk::pad`, `scratchBase` in its parallel buffer), the
   per-view loops of the count and write passes, per-view scratch, buckets, de-duplication by xy rectangles, the
   small-caster rule, the LOD bias; CPU reference extended; `cascadeInstances` / `cascadeTriangles`. Tests: culling counts per view vs the
   CPU reference, main-view counts unchanged, single-view cost (benchmark).
2. **Cascades.** `ShadowPipelineTable` and the shadow shaders (pancaking; `SceneLightingSet::shadowLayout`, no
   `lighting.glsl`), `TerrainPass::selectForShadow` (to `cascadeCasterDistance`), `CascadedShadows` (basis, fitting,
   staggering, the shared depth range, bias units), `shadowFactor` (selection by the xy rectangle), the shadow
   light switch, `shadow` / `shadow.cascades` sections; `sunShadows = false` stays the default (tests set it). With it:
   the `ShadowCascades` view, the `Cascade` / `CascadeStrip` previews, the `CascadeColors`, `CascadeBounds`,
   `CameraFrustumSplits` overlays, freeze, the tab's Cascades section, `get_lighting.cascades`, `capture_lighting
   (cascade)`, the a2 `set_lighting` keys, `probe_lighting`'s cascade keys. Tests: splits, stability, coverage incl.
   stale cascades, box-on-plane, foliage alpha shadow, long shadow across a split in both directions, the cliff near
   the camera, the shadow light switch, the pipeline table slot.
3. **Horizon shadows.** `HorizonShadows` (world anchor, R/G, the max-height chain and the conservative march, three
   maps with a transform each and blending, adaptive rows, patch rows), the mountain test hook,
   `shadow.horizon`, the `HorizonShadow` view, the `Horizon` preview, `HorizonAnchor` overlay, the tab's Far terrain
   section, the tool keys. Tests: dense CPU reference, flat ground lit at low sun, incremental equals full, no recompute
   for small moves, anchor move, edit rows, long shadow at sunset.
4. **Measure and tune (Q6).** `game_world --benchmark forest|flyover|terraform` with `sunShadows` on (RTX path):
   `shadow.cascades` (incl. the terrain casters), `shadow.horizon`, `cull` and the scan; the 2 ms rule of A5 decides the defaults; bias and blend tuned with the previews and `probe_lighting`
   on the flight demo and `game_world`; flight's 2500 m in its environment file; screenshots at noon, sunset, night.
5. **Flip (last commit of a2).** `sunShadows = true`, `horizonShadows = true`; the five clay goldens regenerated
   (shadows; their only regeneration in S36, tree counts unchanged and still checked) and the four lighting goldens
   with shadows. Design-doc amendments of a2, AGENTS.md, Results (a2).

**Part b (S36b)**

1. **`LightSource`** + extract + `RecordingRenderSink::lights` + `RenderSink` light calls; the scenarios
   `light_follows_entity` and `light_toggle_replicates` (all variants); schema regenerated.
2. **`LocalLights`** (selection, buffer, `addLight` / `setLight` / `removeLight` / `addLights`), unshadowed lights
   evaluated with a brute-force loop (`clusteredLights = false`, the correctness reference); `localLights = false`
   stays the default; the `LocalLightsOnly` view, `LightVolumes` overlay, the tab's Local lights table, `list_lights`,
   `get_lighting.lights`.
3. **Clusters.** `light_cluster_bounds.comp`, `light_cluster.comp`, `clusters.glsl`, the `lights` section; the
   `ClusterLights` view, `ClusterGrid` overlay, `Clusters` preview, `get_lighting.clusters`, `probe_lighting`'s cluster
   keys.
4. **Shadow atlas.** `ShadowAtlas` (linear depth, tiles, selection by tiles with hysteresis, dirty bounds, fixed
   terrain level, the queue and budget, the two layers), atlas views in `SceneCulling` (16-bit masks, per-view static /
   dynamic flags), `atlasShadow`, `shadow.atlas`; the `AtlasShadow` view, `Atlas` / `AtlasTile` previews,
   `AtlasOwners`, `get_lighting.atlas`, `capture_lighting(atlas, atlas_tile)`.
5. **Demos and tuning.** Lamps in `game_world`, the flight landing light, `--benchmark night` (reported), screenshots.
6. **Flip (last commit of b).** `localLights = true`; the goldens `lighting_local_lights` and
   `lighting_enclosed_lights`. Amendments of b, AGENTS.md, Results (b).

## Tests

New test files are registered in `tests/CMakeLists.txt` inside `if (MODULE_ENGINE)` (engine) or the Vulkan guard.
**ctest time (finding 28; review 2, findings 12 and 28)**: the scene shaders are compiled at run time per
`RenderScene` (no SPIR-V cache), so the lighting test binaries create **one scene per test suite**. Today's
`RenderSceneTest` creates a window and scene per test (`render_scene_test_context.h`, `SetUp`, 96 x 64), so a1b adds
the fixture **`RenderSceneSuiteTest`** to that header (`SetUpTestSuite` / `TearDownTestSuite`: a static headless
window and scene at the suite's size and MSAA, 128 x 128 and 4 samples by default; `SetUp` resets the state through
the public calls: `setLightingParameters(defaults)`, `setEnvironment(defaults)`, `setLightingDebug(None)`, the content
cleared). Cases that need another size or sample count are their own suite (a derived fixture with its own
parameters); a change of `LightingParameters` recreates resources through `deferDelete` and is a normal case. A
validation error aborts the whole binary, so the first failure hides later cases: acceptable, and the binaries stay
per area (`test_vulkan_lighting` a1, `test_vulkan_shadows` a2, `test_vulkan_local_lights` b). The engine-app tests
(`test_engine_lighting_tools.cpp`, the Lighting tab cases in `test_engine_app.cpp`) use **one headless app per
suite** the same way. Each new binary stays below 60 s on CI's lavapipe, and `test_engine_app`'s time is reported
before and after each part (measured in a1a.4, a1b.2 and every later step that adds cases, reported in Results); if
a binary exceeds it, an in-process SPIR-V cache keyed by the source hash (`ShaderModule`) is added in that step.

**Unit (no device), `tests/test_engine_lighting.cpp`** (a1 unless marked):

- `Celestial.NoonEquinoxAtLatitude` (sun elevation = 90 - latitude at 12:00 on day 0 within 0.1 degrees, azimuth
  south), `.SunriseSunsetAtEquinox` (elevation 0 at 6:00 and 18:00 within 0.5 degrees), `.SeasonsMoveTheSun`
  (declination +-tilt at the solstices), `.MoonPhaseLitFraction` (full and new moon illuminance),
  `.StarsRotateWithHourAngle`, `.FullMoonRisesAtSunset` (phase 0.5: moon elevation 0 within 1 degree when the sun's is),
  `.NewMoonNightHasNoShadowLight` (phase 0 at 0:00: moon illuminance 0, the shadow light selection returns none).
- `Exposure.CurveInterpolatesAndClamps`, `.FixedMode`, `.CompensationIsLog2OfMultiplier` (`--exposure 2` = +1 EV).
- `WorldTime.SystemAdvancesWrapsAndCountsDays` (24 h wrap both directions, `rate` 0 frozen),
  `.SystemNormalisesInvalidValues` (`timeOfDay` 30, NaN, `cloudiness` 2 after `set_global`: normalised next step even
  with `rate` 0), `.CloudOffsetWrapsSeamless` (the offset modulo the tile equals the unwrapped one in the noise),
  `.RegisteredButNoEntityWithoutEnvironment` (a plain world registers both types, has no world-state entity: entity
  counts and the hash of a fixed scenario unchanged; `validateRegistration` passes with `LightSource` (b) after them -
  the regression guard of D4), `.SaveLoadRoundTrip`.
- `EnvironmentExtract.InterpolatesBetweenSteps` (alpha 0 gives the previous step, 1 the current),
  `.ClientExtrapolatesWithHostClock` (from `validAt` and the estimated host time), `.RateChangeDoesNotJump` (a snapshot
  with a new rate and its `validAt` continues the time without a jump beyond the true change), `.SanitisesOutOfRangeWeather`, `.EnclosedSpaceUsesTemplateAmbientAndEv100`.
- `EnvironmentFile.UnknownKeyIsFatal` (death test, R-DATA-1), `.MissingKeysKeepDefaults`.
- `EnvironmentWrites.RangeErrorsNameTheField`, `.CoopWritesTheHost`.
- Engine scenarios (`tests/test_engine_scenarios.cpp`, every variant incl. host + client): `world_time_advances` (the
  client's `WorldTime` equals the host's after the delay; the client's extracted sun direction is monotonic between
  snapshots (extrapolation); the client never writes it: write detector), b: `light_follows_entity` (a moving lamp's
  light in `RecordingRenderSink` follows the interpolated pose, changes chunk), `light_toggle_replicates` (`on = false`
  on the host removes the light's contribution on the client: the sink's light is off).
- `RecordingRenderSink.EnvironmentAndLightsRecorded` (a1: `setEnvironment` count and last value; b: lights).

**Vulkan scene, `tests/test_vulkan_lighting.cpp` (a1), `tests/test_vulkan_shadows.cpp` (a2),
`tests/test_vulkan_local_lights.cpp` (b)** (`RenderSceneSuiteTest` of `tests/render_scene_test_context.h`, one scene
per suite, lavapipe in CI, skip without a driver):

- a1a: `Exposure.Ev100RangeReachesPost` (round-1 blocker 1), `Exposure.NoEnvironmentKeepsUnitlessExposure`.
- a1b: `Pbr.BrdfLutMatchesCpuIntegration` (8 (NdotV, roughness) points within 0.01 of a CPU split-sum integration),
  `Pbr.WhiteFurnace` (a white dielectric and a white metal sphere under a uniform white environment with the sun off:
  mean radiance within 3% of the environment's for roughness 0.2, 0.5, 1: energy conservation),
  `Pbr.ClayUsesStudioEnvironment` (clay at noon and at midnight with clouds render the same image),
  `Pbr.ClayKeepsMaterialColours` (a red flat-colour box stays red in clay), `Pbr.PrototypeColoursOnlyInAces` (the
  terrain's mean colour is gray in clay and green-brown in ACES; a 45-degree slope takes the rock colour). The five clay
  golden tests with `pbr = true` (a1b-a1c; in their existing files) are the guard of the rewrite (review 2, finding 9).
- a1c: `Atmosphere.TransmittanceReference` (zenith and horizon transmittance vs reference values of Hillaire's model
  within 2%), `.SkyIsBlueAtNoonAndRedAtSunset` (hue relations of sky-view samples), `.NightSkyHasStarsAndMoon`.
- a1c: `EnvironmentMap.UniformSkyGivesConstantSh`, `.RecapturesOnlyWhenDue` (frames without a change capture nothing:
  statistics), `.SwapsOnlyCompleteCaptures` (during a re-capture the bound cube and SH are the previous complete ones),
  `.LowerHemisphereIsGround`, `.FirstFrameAmbientIsFloor`, `.ForceFullUpdatesCapturesEveryFrame`.
- a1c: `Clouds.CoverageFollowsCloudiness` (mean coverage monotonic in cloudiness, 0 at 0), `.ShadowProjectsAlongSun`
  (the shadow of a single test blob on a plane moves by `altitude * tan(zenith)`), `.OccludedByTerrainAboveAltitude`,
  `.EdgeFades` (alpha at 0.95 radius below 0.2), `.NoShadowAboveLayer` (a receiver at 2100 m under full coverage has
  factor 1), `.LowSunFadesToMean` (at 3 degrees the factor equals `1 - mean coverage`).
- a2: `Cascades.SplitsMatchFormula`, `.CoverCameraFrustum` (every point of the camera frustum up to `shadowDistance`
  lies inside some cascade box), `.StableUnderSubTexelMoves` (a camera move of 0.3 texel leaves the snapped matrices
  unchanged; a rotation leaves the sphere radius unchanged), `.IntervalKeepsMatrixCorrect` (a cascade not updated in a
  frame shadows the same world point after a camera move), `.StaleCascadeStillCoversFrustum` (turn 30 degrees and move
  20 m between updates of the interval-2 cascade: every receiver of its slice is inside its box), `.BasisStableNearZenith`.
- a2: `Cascades.CullingCountsPerViewMatchCpuReference` (S18's reference extended to views; §17.5 tolerance),
  `.MainViewCountsUnchanged` (instancesPerLod with and without shadows equal), `.NearCastersNotDuplicated` (a caster
  whose xy footprint is inside cascade 0's shrunk rectangle is drawn only into cascade 0),
  `.SmallCastersSkippedInFarCascades`, `.ShadowViewsKeepMainEndDistance` (a mesh beyond its last end distance from the
  main camera casts nothing).
- a2: `Shadows.BoxOnPlane` (the `ShadowFactor` view read back: under the box < 0.1, outside > 0.9, contact pixels
  shadowed: no peter-panning beyond 2 texels), `.FoliageAlphaShadow` (the PrototypeFoliage class casts a perforated
  shadow: mean factor in a band), `.LongShadowCrossesCascadeSplit` (at 5 degrees a 20 m pole 200 m towards the sun
  shadows receivers on both sides of the cascade 0/1 split), `.LongShadowLeavesNearCascade` (review 2, finding 3: a 20 m
  pole inside cascade 0, 30 m from the camera, at 5 degrees: its shadow on the ground 100 m and 200 m away, beyond
  cascade 0's frustum slice, is continuous: factor < 0.2 along the analytic shadow line), `.CliffNearCameraUsesCascade`
  (finding 4: a 200 m wall 150 m away at 45 degrees gives a shadow edge within 2 cascade texels of the analytic
  position), `.ShadowLightSwitchHasHysteresis` (the sun -> moon switch
  by ground illuminance, no alternation around the threshold), `.TerrainCastsInCascades`, `.PancakedCasterBehindNear`
  (a caster beyond the near cap still shadows), `.PipelineTableSlotForNewClass` (a test pipeline class registers its
  shadow entry and casts; a class without one casts nothing).
- a2: `HorizonShadows.MatchCpuReference` (the new `mountainRidgeHook` of `tests/terrain_test_hooks.h`: one ridge of
  about 300 m height and 600 m width across a 4 km space; `slopeRidgeHook`'s 4 m ridges are below the map's resolution,
  review 2 finding 14; the shadow heights (R and G) of 64 texels compared with a **dense** CPU march over every texel:
  never below it by more than 0.5 m (no crest missed) and above it by at most the reported mip height range, for this
  smooth ridge within 5 m; finding 5), `.FarEdgeStableUnderSmallRotation` (a light rotation of 0.1 degrees moves the
  far shadow edge by less than 2 texels), `.FlatGroundIsLitAtLowSun` (elevation 2, 5, 10 degrees: factor > 0.98 on a
  plane and on a 10% slope facing the sun), `.IncrementalEqualsFull` (after completion the map equals a full update),
  `.BlendsDuringRecompute` (the factor at a texel moves monotonically between the two complete maps),
  `.AnchorMoveKeepsShadows` (finding 15: during the recompute after an anchor move the factor at a fixed world point
  stays between the two complete maps' values), `.NoRecomputeForSmallCameraMoves`, `.EditMarksRowsDirty`,
  `.LongShadowAtSunset` (a valley point behind the ridge lit at noon, shadowed at elevation 3 degrees).
- a1b (a2, b add cases): `LightingDebug.EveryViewRenders` (each `LightingDebugView` and overlay of the part renders 2
  frames without a validation error; `Normals` of a plane facing +y reads (0.5, 1, 0.5) from the resolved fp16
  image), `.ViewSwitchCreatesNoPipeline` (finding 24), `.DebugFrameKeepsPreExposure` (the frame after a debug frame
  has the pre-exposure of the frame before it), `.PreviewsCompute` (each
  `LightingImage` preview of the part has the expected size and non-constant content where content exists, and its
  reduction's min / max match a CPU readback), `.NothingComputedWithoutRequest` (no preview dispatch in a frame without
  a request: statistics), `.ProbeMatchesViews` (`probe_lighting`'s values at a pixel equal the debug views' values
  there; the sequence takes 6 frames in a1b and holds camera and environment).
- a1a: `VulkanImGui.ImageDraws` (step a1a.3; in `tests/test_vulkan_headless_renderer.cpp`, the headless ImGui path).
- b: `Lights.ClusteredEqualsBruteForce` (the `LocalLightsOnly` image with clusters equals the brute-force loop within
  1/255; 300 random lights), `.ClusterListsMatchCpuReference` (per cluster the light sets of a CPU assignment, in
  importance order), `.BudgetKeepsMostImportant`, `.OverflowCounted`, `.SpotCone`, `.FlickerUsesPhases` (the same
  phases render the same image), `.BulkListPerChunk` (`addLights` / `removeLights` of two chunks), `.EnclosedSpaceLit`.
- b: `Atlas.StaticSceneCachesFaces` (frame 2 re-renders 0 faces; hits = faces), `.MovingLightRerendersItsFaces`,
  `.InstanceChangeInRangeRerenders` (an `updateInstances` / tombstone inside the range re-renders the light's faces,
  one outside does not; a set leaving the range and another changing in the same frame both invalidate correctly),
  `.CameraMoveKeepsStaticTiles` (the finest resident level: no re-render when the main view's CDLOD selection changes;
  one re-render when a finer terrain level becomes resident inside the light sphere, finding 25),
  `.DynamicLayerOnlyNearDynamics`, `.DynamicTileClearedWhenLastDynamicLeaves`, `.BudgetAndQueueOrder`,
  `.HysteresisAvoidsAlternation`, `.TileBudgetLimitsShadowedLights` (16 shadowed point lights: 10 get tiles, 6 are
  counted without tiles), `.NoAcneAtRange` (a floor at 0.9 * range under a point light is lit, a box on it shadows it),
  `.PointLightSixFaces` (a box inside a point light shadows the floor in all directions).

**Goldens** (`tests/test_engine_lighting_render.cpp`: a `World` with `environment.enabled`, the `mountainRidgeHook`
terrain of `tests/terrain_test_hooks.h` (a1c adds the hook for the sunset golden; the base spacing as coarse as the
ridge allows, recorded in the test, for CI cost), 40 test meshes of `content_test_setup.h`, 200
`PrototypeFoliage` trees, `VulkanRenderSink`, 512 x 512, 4 samples, ACES with a fixed EV100 per golden,
`forceFullUpdates`, `cascadeResolution` 1024 and `horizonResolution` 256 (recorded in the test), captured after 5
frames; written with `SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_lighting_render --gtest_filter='LightingGolden.*'`
on lavapipe; §17.6 tolerance; committed only in each part's flip commit, after tuning):

- a1 (a2 regenerates them with shadows): `lighting_noon` (12:00, cloudiness 0.2, EV100 15): sky, PBR, ambient, cloud
  shadows; `lighting_sunset` (18:40, sun ~3 degrees, EV100 10): red sky, (a2) long horizon shadows from the ridge;
  `lighting_night` (23:00, moon phase 0.5, EV100 -1): moon, stars, (a2) moonlight shadows; `lighting_clouds` (12:00,
  cloudiness 0.7, camera above the layer looking down at 2500 m): the cloud layer and cloud shadows.
- b: `lighting_local_lights` (22:00, 12 point and 3 spot lights, 4 shadowed, EV100 3), `lighting_enclosed_lights`
  (an enclosed space with 6 point lights, 2 shadowed; finding 34).
- The five clay goldens `world_overview`, `world_forest` (committed tree counts unchanged and still checked),
  `generated_site`, `terrain_edit_crater` (`test_engine_terrain_render.cpp`) and `scene_flat_box`
  (`test_vulkan_scene.cpp`): unchanged through a1 (the studio environment reproduces S19a; run with both paths in
  a1b-a1c), regenerated once in the a2 flip commit (shadows).
- The golden binary's runtime on CI's lavapipe is reported in Results (target < 60 s; if a single golden exceeds 15 s
  the golden setup drops to 2 cascades, recorded in the test).

**Debug UI** (`tests/test_engine_debug_ui.cpp`, `tests/test_engine_app.cpp`):

- `DebugUi.KeysSelectTabs` (extended: F5 selects Lighting, a game tab is on F6), `DebugUi.GameTabIndexAfterLighting`.
- `EngineAppNoRender.SetUiTab` (extended: `lighting`).
- `EngineAppRender.LightingTabRenders` (Vulkan, headless app with `environment.enabled`: the Lighting tab with every
  section of the part opened through the state file renders 3 frames without a validation error, the preview images
  are requested in those frames (statistics), a click on the F5 tab button through ImGui's input queue selects it),
  `.LightingTabControlsWriteWorldState` (a slider change through the tab's queued write changes `WorldTime` after the
  next poll and appears as an `api` line in the engine log), `.LightingTabSettingsChangeEnvironment` (the exposure
  compensation widget changes `environmentSettings()` and the next frame's EV100).

**MCP** (`tests/test_engine_lighting_tools.cpp`, `handleMessage` with `tests/mcp_test_helper.h`'s `Poller`; a headless
app with a device, skips without a driver; no-device cases with `--no-render`; each part's keys with their step):

- `get_lighting`: keys present, `sections` filter, without `WorldTime` `time: null`, `tone_map`, answered at once (no
  deferred section); without a device an error.
- `set_time_of_day`: sets and returns `previous`; out-of-range `hours` (24, -1, nan) and unknown keys are errors that
  name the argument; in a `--coop` app it changes the host's `WorldTime` and the client's after the delay; refused on a
  pure client world (the S16 message); the next frame's sun follows.
- `set_weather`: per space, range errors, unknown space, `--coop` writes the host.
- `set_lighting`: every key of the part round-trips through `previous` (`values` == `get_lighting.settings` /
  `parameters`); invalid values (cascade_count 5, resolution 1000) are errors; a cross-key error (`cluster_near` 50
  with `cluster_far` 10, or `atlas_tile` 300) names both keys and changes nothing of the call's other keys (all or
  nothing, finding 23); `tone_map identity` switches to clay and `get_lighting` reports it; applied from the next frame.
- `set_lighting_debug`: every view name, overlay list, unknown names listed in the error.
- `screenshot(lighting_view, overlays)`: the frame shows the view (the `Normals` of a plane), the next frame is back to
  `None`; two requests with different views in one batch are answered from two frames.
- `capture_lighting`: every image name of the part returns an image of the expected size (png by default) and the JSON
  block (mapping, bounds, `min` / `max` / `mean` / histogram); `stats_only` returns the block without an image;
  `cascade` / `light` / `face` / `slice` out of range errors; a deadline error when rendering is disabled
  (`setRenderingEnabled(false)`), answered once.
- `probe_lighting`: a1b: a pixel of a lit plane returns its albedo, normal and a direct radiance > 0, `frames` 6 and
  `consistent` true; a2: a pixel of the box-on-plane scene in the shadow returns a cascade factor < 0.1 and the cascade
  index; a pixel outside the image is an error; the user's debug view is restored after the sequence.
- b: `list_lights` paging, `near`, `shadowed` filter; `get_lighting.lights` counts.
- Console: `DebugConsole.RunsLightingCommands` (`set_time_of_day hours=6` through `callLocal` changes the time).

## Benchmarks (finding 37, R-TEST-2; review 2, finding 10)

- Each benchmark twice (AGENTS.md recipe), before the first step and after the last step of each part that changes
  rendering (a1b, a1c, a2, b; a1a changes only `latchFrame`'s arithmetic and is run once after, as a check).
- **The tone-map confound**: the committed baselines of `data/tests/benchmarks.json` are clay (hardware resolve, no
  tone map). From a1c `game_world` runs ACES (D22: custom resolve, post tone map, PBR). So **before a1b** the unchanged
  code is measured with `--tonemap aces` (`game_world --benchmark flyover|forest`, twice) and reported as the **ACES
  baseline**; the 5% rule of the features-off comparison of a1b and a1c compares against it. The clay number is
  reported once more after the a1c flip (`--tonemap identity`: the studio path) against the committed clay baseline
  (within 5%: the new clay path costs what the old one did).
- What they measure after S36: `game_world --benchmark flyover|forest` at 10:00, cloudiness 0.3, frozen time, ACES,
  all lighting features of the part on; recorded in the benchmark JSON (`lighting: {time, cloudiness, ev100, tone_map,
  features}`).
- Reported separately: (1) the part's features off (a1b: the default legacy variant against the ACES baseline; a1c:
  the legacy variant before the flip, and `clouds = false` with a uniform sky after it; a2: `sunShadows = false`,
  `horizonShadows = false`; b: `localLights = false`), compared with the baseline before the part: must be within the
  5% of R-TEST-2 (a culling or shader regression is not hidden in the new cost); the 5% comparison of a1b is made on
  the legacy pipeline variant, which the games use until the flip (finding 26); (2) the features on, the new sections
  reported against §17.10 (shadow budget 2.5 ms).
- **`--benchmark terraform`** (`TerraformBenchmark`) in a2 and b as well: terrain edits mark horizon rows, the max
  chain and atlas tiles dirty.
- `sample_vulkan_scene_benchmark` runs **clay** (its `RenderScene` has no environment and keeps `ToneMap::Identity`;
  unchanged by D22): with shadows off within 5% of its pre-a2 number (the multi-view count pass does not slow the single
  view).
- Mapping: the S03 `gpu_ms` keys get `shadows` = `shadow` (with its sub-sections in the JSON) and `sky` = `scene.sky` +
  `scene.clouds` + `environment`; the baselines of `data/tests/benchmarks.json` are re-committed after a part is
  accepted, from a1c on with ACES (the new work and the tone map are an expected, explained increase, not a
  regression; open question O2 at the end).

## Tech demo

- **a1a**: the Lighting tab (F5) with time and weather in any engine game (`SAIGA_DEBUG_TAB=5`); `set_time_of_day`
  through MCP changes `get_lighting`'s sun (no image change yet).
- **a1b**: `screenshot(lighting_view: normals | albedo | direct_only)` and `probe_lighting` on `game_world` with the
  game's temporary `pbr = true` (local review; the default images are unchanged).
- **a1c**: `SAIGA_SCREENSHOT=/tmp/flight_noon.png ./build/samples/games/game_flight --demo --time-of-day 12`, the same
  at `19` (sunset: red sky, cloud layer) and `23` (moon, stars) - ACES by default (D22), so the time of day is visible
  without `--tonemap`; the same with `--tonemap identity` shows the unchanged clay view; the demo's day (10 min, or
  `--day-length 240`) reviewed live; the Lighting tab (F5) screenshot (`screenshot gui=true` after `set_ui_tab
  lighting`); racing and platformer `--demo` screenshots in ACES with the prototype colours and their material colours
  (their look changes from clay: reviewed, not compared).
- **a2**: the same three flight screenshots (ACES) with cascade and horizon shadows (horizon shadows across the valleys at
  sunset); the Lighting tab's Cascades section with the previews; `screenshot(lighting_view: shadow_cascades,
  overlays: [cascade_bounds])` with frozen cascades.
- **b**: `SAIGA_SCREENSHOT=/tmp/world_night.png ./build/samples/games/game_world --time-of-day 22` (lamps, shadowed
  ones, flickering torches); the flight demo at 23:00 with the landing light; the Lighting tab with light volumes, the
  cluster heat map and the atlas.

## Done when

**Each part**: every commit passed CI (GCC, Clang, MSVC, warnings as errors); the part's tests pass; its goldens gate on
lavapipe; every new tool key has its `handleMessage` test; the Lighting tab renders headless with the part's sections;
the benchmarks were run before and after and reported as in Benchmarks; the new test binaries each run below 60 s on
CI's lavapipe; the demos' scenarios pass with `--scenarios` in all variants; the part's screenshots are reviewed; the
part's design-doc amendments and AGENTS.md changes are applied; the PR has fewer than 100 changed files.

**a1a and a1b also**: no committed image changed; the five clay golden tests pass with both pipeline variants (a1b).

**a1c also**: the games render ACES by default and the flight screenshots at 12, 19 and 23 differ as day, sunset
and night; the five clay goldens are unchanged; the ACES baseline and the clay re-check are reported.

**a2 also**: the Q6 measurement is recorded and the cascade defaults follow from it.

**b also**: `--benchmark night` reported (400 lamps: the `lights` and `shadow.atlas` sections, cache hit rate).

## Verification

```shell
# CI recipe (AGENTS.md "Build (Linux)"), then
ctest --test-dir build --output-on-failure --parallel 4   # from the repository root (the paths below are root-relative)
# lavapipe as CI; the pattern also matches test_engine_lighting_render and test_engine_lighting_tools (intended);
# build first (the compile-fail note of AGENTS.md: ctest must not rebuild the libraries while tests run; nit 35)
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build -R 'test_vulkan_lighting|test_vulkan_shadows|test_vulkan_local_lights|test_engine_lighting' --output-on-failure
# goldens (the flip commit of a1c, a2, b; lavapipe)
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_lighting_render \
  --gtest_filter='LightingGolden.*'
# the five clay goldens (a2 flip only): the world goldens are in test_engine_terrain_render (VulkanWorldGolden.*,
# TerrainRenderTest.EditCraterGolden, .GeneratedSite; AGENTS.md's recipe naming test_vulkan_world is corrected, see
# AGENTS.md changes), scene_flat_box in test_vulkan_scene (VulkanSceneContent.ImportedMeshGolden)
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_terrain_render \
  --gtest_filter='VulkanWorldGolden.*:TerrainRenderTest.EditCraterGolden:TerrainRenderTest.GeneratedSite'
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_vulkan_scene \
  --gtest_filter='VulkanSceneContent.ImportedMeshGolden'   # then the tree counts as in AGENTS.md (unchanged)
# benchmarks (AGENTS.md recipe, build-bench), before and after each part, each twice; before a1b also the ACES
# baseline of the unchanged code
(cd build-bench/samples/games && ./game_world --benchmark flyover && ./game_world --benchmark forest)
(cd build-bench/samples/games && ./game_world --tonemap aces --benchmark flyover && \
  ./game_world --tonemap aces --benchmark forest)                     # ACES baseline (before a1b)
(cd build-bench/samples/games && ./game_world --benchmark terraform)  # a2, b
(cd build-bench/samples/vulkan && ./sample_vulkan_scene_benchmark)
(cd build-bench/samples/games && ./game_world --benchmark night)        # part b, reported only
# demos
SAIGA_SCREENSHOT=/tmp/flight_sunset.png ./build/samples/games/game_flight --demo --time-of-day 19
./build/samples/games/game_flight --scenarios && ./build/samples/games/game_world --scenarios
# Eigen backend check for the celestial and cascade double helpers (AGENTS.md)
```

## Files

Counted per file (review 2, blocker 2: each list below was counted with a script, one path per entry, no estimate;
a file changed in several parts is counted in each). CodeRabbit limit 100: **a1a 61, a1b 59, a1c 86, a2 63, b 61**.
The counts are re-checked with `git diff --stat origin/master...HEAD | tail -1` before each PR is opened; a part
that grows past 95 files moves its last feature step to the next part (a1c: the cloud layer, 8 files).

**a1a (61 files)**: `src/saiga/engine/components/{WorldTime.h, Weather.h, EngineIds.h, EngineComponents.h,
EngineComponents.cpp}`; `src/saiga/engine/lighting/{EnvironmentSetup.h, EnvironmentFile.h, EnvironmentFile.cpp,
EnvironmentWrites.h, EnvironmentWrites.cpp, Celestial.h, Celestial.cpp, WorldTimeSystem.cpp, Systems.h,
EnvironmentExtract.h, EnvironmentExtract.cpp}`; `src/saiga/engine/world/{World.h, World.cpp, SpaceTemplate.h}`;
`src/saiga/engine/sync/{RenderSink.h, VulkanRenderSink.h, VulkanRenderSink.cpp, RecordingRenderSink.h,
RecordingRenderSink.cpp}`; `src/saiga/engine/app/{EngineApp.h, EngineApp.cpp, EngineAppSetup.h, EngineMain.h,
EngineMain.cpp, EngineAppDebugUi.cpp, DebugUi.h, DebugUi.cpp, LightingTab.h, LightingTab.cpp, LightingTools.h,
LightingTools.cpp, AppTools.cpp}`; `src/saiga/vulkan/imgui/{ImGuiVulkanRenderer.h, ImGuiVulkanRenderer.cpp,
ImGuiHeadlessRenderer.h, ImGuiHeadlessRenderer.cpp}`; `src/saiga/vulkan/scene/{RenderSceneTypes.h, LightingTypes.h,
RenderScene.h, RenderScene.cpp}`; `tests/{test_engine_lighting.cpp, test_engine_lighting_tools.cpp,
test_engine_debug_ui.cpp, test_engine_app.cpp, test_engine_scenarios.cpp, test_engine_replication.cpp,
test_engine_save.cpp, test_engine_world_api.cpp, test_engine_extract.cpp, test_vulkan_scene.cpp,
test_vulkan_headless_renderer.cpp, CMakeLists.txt}`; `data/tests/engine/schema.json`; `docs/engine_design.md`;
`AGENTS.md`; `docs/stages/S36_lighting.plan.md`.

**a1b (59 files)**: `src/saiga/vulkan/scene/{SceneLighting.h, SceneLightingSet.h, SceneLightingSet.cpp, SceneCulling.h,
SceneCulling.cpp, TerrainPass.h, TerrainPass.cpp, GrassPass.h, GrassPass.cpp, PrototypeWorld.h, PrototypeWorld.cpp,
RenderScene.h, RenderScene.cpp, RenderSceneTypes.h, LightingTypes.h, AmbientProvider.h, SkyAmbient.h, SkyAmbient.cpp,
BrdfLut.h, BrdfLut.cpp, LightingDebug.h, LightingDebug.cpp, DebugLines.h, DebugLines.cpp}`;
`shader/vulkan/scene/{flat.frag, instance.vert, prototype_foliage.frag, terrain.frag, terrain_common.glsl,
terrain_shading.glsl, grass.frag, sky.frag, post.frag, scene_lighting.glsl}`;
`shader/vulkan/world/prototype_shading.glsl`; `shader/vulkan/scene/lighting/{lighting.glsl, scene_block.glsl, pbr.glsl,
ambient.glsl, shadows.glsl, brdf_lut.comp, debug_preview.comp, debug_probe.comp, debug_lines.vert, debug_lines.frag}`;
`src/saiga/engine/app/{EngineApp.cpp, AppTools.cpp, LightingTab.h, LightingTab.cpp, LightingTools.cpp}`;
`tests/{test_vulkan_lighting.cpp, render_scene_test_context.h, test_engine_lighting_tools.cpp, test_engine_app.cpp,
test_engine_terrain_render.cpp, test_vulkan_scene.cpp, CMakeLists.txt}`; `AGENTS.md`;
`docs/stages/S36_lighting.plan.md`.

**a1c (86 files)**: `src/saiga/vulkan/scene/{Atmosphere.h, Atmosphere.cpp, EnvironmentMap.h, EnvironmentMap.cpp,
CloudLayer.h, CloudLayer.cpp, SkyAmbient.cpp, RenderScene.h, RenderScene.cpp, SceneLighting.h, SceneLightingSet.cpp,
LightingDebug.cpp, LightingTypes.h, RenderSceneTypes.h, PrototypeWorld.h, PrototypeWorld.cpp, SceneCulling.h,
SceneCulling.cpp, TerrainPass.cpp, GrassPass.cpp}`; `src/saiga/vulkan/world/{WorldSettings.h, WorldSettings.cpp,
PrototypeScene.h, PrototypeScene.cpp, WorldMcp.cpp, WorldBenchmark.h, WorldBenchmark.cpp}`;
`shader/vulkan/scene/lighting/{atmosphere_common.glsl, atmosphere_transmittance.comp, atmosphere_multiscatter.comp,
atmosphere_skyview.comp, stars.glsl, cloud_noise.comp, cloud_layer.vert, cloud_layer.frag, env_capture.comp,
env_prefilter.comp, env_sh.comp, lighting.glsl, ambient.glsl, debug_preview.comp}`; `shader/vulkan/scene/{sky.frag,
flat.frag, prototype_foliage.frag, terrain.frag, grass.frag, terrain_shading.glsl, scene_lighting.glsl}`;
`shader/vulkan/world/prototype_shading.glsl`; `src/saiga/engine/app/{LightingTab.cpp, LightingTools.cpp}`;
`samples/games/common/{GameEngine.h, Environment.h, Environment.cpp}`; `samples/games/flight/{main.cpp,
FlightRender.cpp, FlightWorld.cpp}`; `samples/games/racing/{main.cpp, RacingRender.cpp}`;
`samples/games/platformer/{main.cpp, PlatformerRender.cpp}`; `samples/games/world/main.cpp`;
`data/flight/environment.json`; `data/world/environment.json`; `tests/{test_vulkan_lighting.cpp,
test_engine_lighting_render.cpp, terrain_test_hooks.h, test_engine_lighting_tools.cpp, test_engine_terrain_render.cpp,
test_vulkan_scene.cpp, test_vulkan_world.cpp, test_engine_app.cpp, CMakeLists.txt}`;
`data/tests/golden/{lighting_noon.png, lighting_sunset.png, lighting_night.png, lighting_clouds.png, world_overview.png,
world_forest.png, generated_site.png, terrain_edit_crater.png, scene_flat_box.png}`; `data/tests/benchmarks.json`;
`docs/engine_design.md`; `AGENTS.md`; `docs/stages/S36_lighting.plan.md`.

**a2 (63 files)**: `src/saiga/vulkan/scene/{RenderScene.h, RenderScene.cpp, SceneCulling.h, SceneCulling.cpp,
TerrainPass.h, TerrainPass.cpp, CascadedShadows.h, CascadedShadows.cpp, ShadowPipelineTable.h, ShadowPipelineTable.cpp,
HorizonShadows.h, HorizonShadows.cpp, LightingDebug.h, LightingDebug.cpp, DebugLines.cpp, SceneLighting.h,
SceneLightingSet.h, SceneLightingSet.cpp, RenderSceneTypes.h, LightingTypes.h, TerrainAtlas.h, TerrainAtlas.cpp}`;
`shader/vulkan/scene/lighting/{shadows.glsl, lighting.glsl, shadow_depth.vert, shadow_alpha.frag, shadow_foliage.vert,
shadow_foliage.frag, terrain_shadow.vert, horizon_map.comp, horizon_maxmip.comp, debug_preview.comp, debug_probe.comp,
scene_block.glsl}`; `shader/vulkan/scene/{instance_common.glsl, instance_count.comp, instance_write.comp,
bucket_scan.comp}`; `src/saiga/engine/app/{LightingTab.cpp, LightingTools.cpp}`; `samples/games/world/main.cpp`;
`data/flight/environment.json`; `tests/{test_vulkan_shadows.cpp, test_vulkan_scene.cpp, test_vulkan_terrain.cpp,
test_engine_lighting_render.cpp, test_engine_lighting_tools.cpp, test_engine_terrain_render.cpp,
render_scene_test_context.h, CMakeLists.txt}`; `data/tests/golden/{lighting_noon.png, lighting_sunset.png,
lighting_night.png, lighting_clouds.png, world_overview.png, world_forest.png, generated_site.png,
terrain_edit_crater.png, scene_flat_box.png}`; `data/tests/benchmarks.json`; `docs/engine_design.md`; `AGENTS.md`;
`docs/stages/S36_lighting.plan.md`.

**b (61 files)**: `src/saiga/engine/components/{LightSource.h, EngineIds.h, EngineComponents.cpp}`;
`src/saiga/engine/sync/{ExtractSystem.h, ExtractSystem.cpp, RenderSink.h, VulkanRenderSink.h, VulkanRenderSink.cpp,
RecordingRenderSink.h, RecordingRenderSink.cpp}`; `src/saiga/vulkan/scene/{LocalLights.h, LocalLights.cpp,
ShadowAtlas.h, ShadowAtlas.cpp, RenderScene.h, RenderScene.cpp, SceneCulling.h, SceneCulling.cpp, TerrainPass.h,
TerrainPass.cpp, LightingDebug.h, LightingDebug.cpp, DebugLines.cpp, RenderSceneTypes.h, SceneLighting.h,
SceneLightingSet.cpp, LightingTypes.h, ShadowPipelineTable.cpp}`; `shader/vulkan/scene/lighting/{clusters.glsl,
light_cluster_bounds.comp, light_cluster.comp, atlas_shadow.glsl, atlas_depth.frag, lighting.glsl, debug_preview.comp,
debug_probe.comp}`; `shader/vulkan/scene/{instance_count.comp, instance_write.comp}`;
`src/saiga/engine/app/{LightingTab.cpp, LightingTools.cpp}`; `samples/games/world/{main.cpp, Lamps.h, Lamps.cpp}`;
`samples/games/flight/{FlightRender.h, FlightRender.cpp, FlightWorld.h, FlightWorld.cpp}`;
`tests/{test_engine_lighting.cpp, test_vulkan_local_lights.cpp, test_engine_lighting_render.cpp,
test_engine_lighting_tools.cpp, test_engine_scenarios.cpp, test_engine_extract.cpp, CMakeLists.txt}`;
`data/tests/golden/{lighting_local_lights.png, lighting_enclosed_lights.png}`; `data/tests/engine/schema.json`;
`data/tests/benchmarks.json`; `docs/engine_design.md`; `AGENTS.md`; `docs/stages/S36_lighting.plan.md`.

## Risks

- **CI time on lavapipe.** The atmosphere LUTs, the environment prefilter, the cascades and the horizon map in every
  golden frame (`forceFullUpdates`) are slow on a CPU rasterizer, and every `RenderScene` compiles its shaders.
  Mitigation: one scene per test suite, goldens at 512 x 512 with cascades at 1024 and the horizon map at 256, small
  scenes (§17.6), the prefilter at 64; measured per step against the 60 s target; the SPIR-V cache as the fallback.
- **Golden churn.** The clay goldens are regenerated once (a2: shadows; a1 keeps them by reproducing S19a) and the
  lighting goldens once more in a2; each regeneration is a reviewed flip commit and the tree counts stay checked. The
  studio environment keeps later atmosphere and cloud tuning out of the clay goldens. Risk: the clay branch does not
  reproduce S19a bit for bit (float order of operations in the new functions); the golden tolerance (§17.6) absorbs
  rounding, a larger difference is a bug found in a1b, not at the flip.
- **The games' new look (D22)**: ACES with constant colours is a large visible change of every demo in a1c, and
  `game_world`'s benchmark numbers move with the tone map; mitigated by the ACES baseline, the clay re-check and the
  `--tonemap identity` view.
- **Bias tuning** differs between lavapipe and GPUs (depth precision, rasterization rules): the goldens gate only on
  lavapipe; acne / peter-panning on the RTX path are checked in the demo screenshots and with `probe_lighting`, biases
  are parameters in texels.
- **Multi-view culling cost**: the count pass loops over up to 5 views (a2) or 13 (b) per instance; the CPU view masks
  keep most chunks at 1-2 views, the de-duplication and the small-caster rule cut the GPU work. Measured with
  `sample_vulkan_scene_benchmark` (200k trees) and the forest benchmark. Fallback: one dispatch per view group.
- **Terrain in the cascades**: a CDLOD selection per cascade with the main view's metric costs CPU
  (`selectForShadow`), and the fixed `cascadeCasterDistance` (1371 m towards the light at every elevation) draws more
  terrain at high sun than a height-based cap would; measured in a2.4 (`shadow.cascades` with and without terrain);
  the escapes are a smaller `cascadeCasterDistance` (the horizon map's `G` takes over from there) or
  `terrainCastsInCascades = false` (horizon map only).
- **Conservative horizon march**: the max-mip chain can grow far shadows by up to one mip texel's height range (at
  10 km a 1 km texel); acceptable for the far horizon, reported in the statistics; a finer far march is a parameter.
- **`probe_lighting` latency**: six frames (about 0.1 s at 60 fps, longer on lavapipe) with the camera and environment
  held; a moving object makes `consistent` false rather than a wrong answer.
- **ImGui renderer change** affects every Vulkan sample's ImGui: the font path keeps its set; `VulkanImGui.ImageDraws`
  and the existing window tests guard it.
- **Component name clashes**: test-local `Weather` structs are renamed in all three test files; engine component names
  must stay unique, so a test world with the environment enabled cannot also register a component named `Weather`.
- **The prototype foliage's world-xz exception** (§10.1) stays; the shadow foliage shader reuses it (same 8 km
  validity).
- **Flight's 3 km content radius vs a 2.5 km shadow distance**: trees between 2.5 and 3 km and all far content are
  unshadowed by instances (terrain shadows from the horizon map). S27 adds impostor casters (amendment).
- **fp16 and the sun disc**: the disc is clamped after pre-exposure (§10.5); bloom (S37) will want the real value.
- **File counts**: a1c is the largest part (86 of 100); the cloud layer is the documented seam to a2.
- **Five PRs and CodeRabbit's rate limit** (about 2 reviews per hour on the free plan): stacked PRs keep the work
  moving while a review waits (O1).

## Design-doc amendments

Applied in the last commit of each part (a1a: M1, M2, the a1a rows of M3, M10, M11; a1b: none (no behaviour change of
the default path; the plan and AGENTS.md only); a1c: the rest of M3, M6, M7, M9-M11, M13, M14 for a1; a2: M4, M8, the
a2 parts of M9-M14; b: M5, the b parts), with the status updates of §3, §10.1, §10.2, §10.5, §10.7, §10.11, §16.2, §17.1, §17.6,
§17.7, §18.4 and §22:

- **M1 (§7.8)**: `WorldTime` / `Weather` exist from S36a1, always registered, global / space state only with
  `WorldSetup::environment.enabled` (D4); `registerEngineComponents(Registry&, const WorldSetup&)`.
- **M2 (§10.7, fields)**: `WorldTime` gets `rate` (D5); the field list of `Weather` as A1; the celestial model,
  `EnvironmentSetup` (simulation) and `EnvironmentSettings` (app, presentation) named; `worldTimeSystem` is optional
  (R-MECH-1) and normalises; `validAt` and the client extrapolation from the estimated host time.
- **M3 (§10.7, exposure; §10.5, clay)**: `ExposureSettings` (curve by sun elevation, two modes), EV100, the frame
  exposure rule (`params.exposure` a multiplier; a scene without an environment unit-less) and the clamp of the
  effective exposure; clay mode's studio environment (S19a's light reproduced, material colours kept for flat
  classes); games with an environment default to ACES (D22); the prototype albedo colours until S43 (D21).
- **M4 (§10.7, shadows)**: the shadow light by ground illuminance with hysteresis; cascades: basis rule, sphere fit,
  camera-centred staggered cascades, texel snap, double centres, pancaking in the vertex shader, bias in texels,
  one shared light-space depth range, cascades as xy rectangles, caster volumes with caps (instances by height,
  terrain to `cascadeCasterDistance`), de-duplication and selection by the xy rectangle, the LOD rule; the horizon map
  as a world-anchored two-channel shadow height map with a conservative max-mip march, three buffers with a transform
  each and blending.
- **M5 (§10.7, local lights, b)**: the two-layer D16 atlas with linear depth, fixed tiles, the tile budget (the
  sentence "16 shadowed lights at High" becomes "a tile budget: 64 tiles of 512 at High = 10 point or 64 spot lights";
  review 2, finding 27), dirty-bound invalidation, the finest resident terrain level, hysteresis; clusters with fixed
  slots.
- **M6 (§10.11)**: the 2D cloud layer is a mesh in the scene pass after the sky (`scene.clouds`) with curvature and edge
  fade, the clouds are in the environment map; `cloudShadow()` is S42's hook.
- **M7 (§10.1)**: `setEnvironment`, `setLightingParameters`, `lightingStatistics`, `setLightingDebug`, `setFoliageTexture`
  (replaces `setPrototypeEnvironment`), `setTime` removed; `RenderSink::setEnvironment` and (b) the light calls exist;
  `SpaceTemplate::ambient` / `ev100` (enclosed lighting through the extract); `kEnclosedLighting` removed.
- **M8 (§10.2)**: the shadow views: light-space volumes, view masks per work chunk (in `WorkChunk::pad`, 16 bits),
  per-view scratch and loops in the count and write passes, the main camera's end distances for every view, the LOD
  rule, de-duplication and the small-caster rule; (b) atlas views (`facesPerFrame <= 11`) and static / dynamic view
  flags.
- **M9 (§10.5)**: pass 1 exists; the section list per part; pass 10 (the fog colour from the atmosphere's sky-view
  LUT, the `SCENE_FOG` variant gone); pass 11 (the 2D clouds move from the composite after the resolve into the MSAA
  scene pass).
- **M10 (§17.1)**: the Lighting tab on F5 (engine tabs F1-F5, game tabs F6-F10), its sections; `DebugLines`.
- **M11 (§16.2)**: the rows of `get_lighting`, `set_time_of_day`, `set_weather`, `set_lighting`, `set_lighting_debug`,
  `capture_lighting`, `probe_lighting`, the `screenshot` arguments `lighting_view` / `overlays`, (b) `list_lights`;
  `app_status.lighting`.
- **M12 (§22)**: S36 split into S36a1a / S36a1b / S36a1c / S36a2 / S36b rows; the dependencies become "S19 (S19a,
  S19b, S19c for the tab), S20, S21, S23b, S23c" as in "Depends on" (nit 37); the S27 row adds "impostor shadow casters
  in the far cascades (the shadow pipeline table's impostor entry)"; S30's row keeps "light descriptions" (through
  `addLights`); the S26 row adds "the environment flag in the join check" (finding 22).
- **M13 (§17.6)**: the lighting goldens, golden mode = fixed EV100 + `forceFullUpdates`, the golden resolutions
  (`cascadeResolution` 1024, `horizonResolution` 256, 512 x 512, 4 samples; nit 37), the clay goldens with the studio
  environment (D7: unchanged through a1, regenerated in a2), `setTime` gone from the golden setup (phases = {}).
- **M14 (§17.7)**: `gpu_ms` keys `shadows` and the new `sky`; the benchmark JSON's `lighting` block (with
  `tone_map`); the ACES baselines of `game_world` from a1c on; the lighting-off comparison; `terraform` in the shadow
  parts; `--benchmark night` (b, reported).

## AGENTS.md changes

- Layout: `engine/` gets `lighting/` (WorldTime / Weather writes, EnvironmentSetup / EnvironmentFile, celestial model,
  environment extract) and the components `WorldTime`, `Weather`, (b) `LightSource`; `engine/app/` the Lighting tab
  (F5) and `LightingTools`; `vulkan/scene/` the lighting classes (atmosphere, clouds, environment map, cascades, horizon
  shadows, debug previews and lines, (b) local lights and the shadow atlas) and `shader/vulkan/scene/lighting/`.
- Tests: the lighting golden recipe (`test_engine_lighting_render --gtest_filter='LightingGolden.*'`) next to the
  existing golden recipes; the world golden recipe corrected (the world goldens are in `test_engine_terrain_render`,
  `VulkanWorldGolden.*`, not in `test_vulkan_world`); `RenderSceneSuiteTest` next to the `VulkanTest` fixture; the
  test count in "Build options".
- Debug UI line: "F1 Game, F2 Performance, F3 Engine, F4 Console, F5 Lighting; game tabs F6-F10";
  `SAIGA_DEBUG_TAB=5` opens the Lighting tab.
- Tech demos: games with an environment render ACES by default (`--tonemap identity`: the clay view with the studio
  light); `--time-of-day` / `--day-length`, `--exposure` as a multiplier on EV100; `game_world --lamps`,
  `--benchmark night` (b).

## Decisions

| ID | Decision | Reason |
|---|---|---|
| D1 | Five parts: S36a1a (world state, tools; 61 files), S36a1b (lighting block, PBR variant, debug core; 59), S36a1c (sky, environment map, clouds, games, flip; 86), S36a2 (multi-view culling, cascades, horizon map; 63), S36b (local lights, clusters, atlas; 61); each its own PR (stacked allowed), CI green per commit, < 100 files counted per file (Q2; review 2, blocker 2) | Reviewable PRs under CodeRabbit's limits; a1 as one PR would be about 150-200 files; the a1 cuts need no flip of their own |
| D2 | Hillaire 2020 LUT atmosphere everywhere (Q7) | Physically based (§10.11), cheap per frame (192 x 108 sky-view LUT), gives transmittance for the sun colour and the fog colour; a tier choice only if S37's Steam Deck measurement asks for it |
| D3 | 2D cloud layer as a depth-tested mesh after the sky, with curvature and edge fade | Correct occlusion by mountains and from above without a depth read in the MSAA pass; no rim |
| D4 | `WorldTime` / `Weather` always registered; global / space state (world-state entity) only with `EnvironmentSetup::enabled` (Q4) | The registry's prefix rule; plain test worlds, hashes and the golden save keep their layout; one schema |
| D5 | `WorldTime::rate` in the component, client-side extrapolation with `HostClock` (Q5); flight `--demo` day 10 min | Replicated and saved speed; agents and the UI change it like other state; smooth sun on clients |
| D6 | Lighting is engine tab F5; game tabs F6-F10 (Q3) | The content is too large for a section of the Engine tab; no game uses game tabs yet |
| D7 | Clay mode keeps S19a's light model exactly (Lambert, studio sun and hemisphere ambient, studio fog, material colours of flat classes, studio light without fog in enclosed spaces) plus the a2 shadows; the five clay goldens pass unchanged through a1 and are regenerated once in a2 (Q1; review 2, finding 8) | §10.5: clay keeps geometry and lighting; independent of time, atmosphere and clouds; the unchanged goldens prove the shader rewrite |
| D8 | Preview textures (with a value reduction) shared by the UI and `capture_lighting` | Agents see exactly what developers see, with the numbers; zero cost when unused |
| D9 | `WorldSettings` / `game_set_settings` lose the sun and fog keys | One source of truth (`WorldTime`, `Weather`, `set_lighting`); the error names the replacement tools |
| D10 | Main-camera LOD for cascades 0-1, `shadowLodBias` for cascades 2-3 | No self-shadowing from LOD mismatch where visible; cheaper far cascades |
| D11 | Horizon map stores two shadow heights (all occluders from 64 m; occluders beyond the cascades' caster range) | Valid for receivers at any height; combines with the cascades without coarse overrides |
| D12 | Debug views latch the Identity tone map without a pipeline change; data views write raw values into the fp16 target; a debug frame keeps the pre-exposure history | Data values are exact and readable back; no stall or flash around a debug frame |
| D13 | Shadow light = the body with the higher ground illuminance, hysteresis factor 2 | One shadow caster; no dark gap between sunset and the moon taking over |
| D14 | Atlas: two D16 layers with linear depth, fixed 512 tiles, selection by tile budget, dirty-bound invalidation, hysteresis | Precise at every distance; cache-friendly; dynamic objects cost only the faces near them |
| D15 | Cascade defaults 4 x 2048 at 1500 m with de-duplication, small-caster rule and LOD bias; fallback 3 / 1000 m if `shadow.cascades` > 2 ms in the a2 measurement (Q6) | Fits the 2.5 ms shadow budget only with the caster reductions; measured, not guessed |
| D16 | Presentation settings (`EnvironmentSettings`) owned mutably by `EngineApp`; `WorldSetup` keeps `enabled`, `celestial`, `cloudWindScale` | `WorldSetup` is constant after the world exists; tab and tools need a mutable owner |
| D17 | No optional Vulkan feature: pancaking in the shadow vertex shaders, `uint` cluster indices, manual PCF fallback without linear depth filtering | Vulkan 1.3 core only (AGENTS.md) |
| D18 | `frameExposure = params.exposure / (1.2 * 2^ev100)` clamped to [2^-24, 2^10] once a scene has an environment; without `setEnvironment` the scene is unit-less (studio light, `params.exposure`) as today; `setExposure`, its clamp and `--exposure` keep their meaning (a multiplier) | Physical EV100 15 needs exposure 2.5e-5; S17's tests, `hdr_tonemap` and the samples stay valid (review 2, finding 6) |
| D19 | UI views, previews and tool keys are built with each feature; goldens committed last in each part (flip commit) | Cascades and the rest are tuned with their debug views; no second golden commit per part |
| D20 | `screenshot` gets `lighting_view` / `overlays`; `capture_lighting` for data images and (`stats_only`) the only path to GPU image statistics; `get_lighting` never waits for a frame; `probe_lighting` is a 6-frame debug-view sequence with 1 x 1 readbacks plus one compute evaluation (camera and environment held) | One frame-capture path; agents reason on numbers; a forward renderer has no G-buffer and fragment stores are optional (review 2, findings 7, 13) |
| D21 | The ACES path gets constant prototype albedo colours (ground, rock by slope, shore, water, grass, leaves, bark) until S43; clay keeps the gray 0.55 (blocker 1b) | Cheap (seven constants, one `smoothstep`); a gray world makes sunset and cloud shadows hard to judge |
| D22 | A game with the environment enabled renders ACES by default (`enableEnvironment` sets the tone map; `--tonemap identity` gives clay); `tone_map` in `get_lighting` / `set_lighting` and the tab toolbar (blocker 1) | Time of day exists only in ACES; every demo shows it; agents see why clay ignores it |
| D23 | One light-space depth range for all cascades; cascades are xy rectangles; selection and de-duplication by xy (review 2, finding 3) | Exact: an orthographic shadow stays in its caster's xy footprint; long shadows never end at a split |
| D24 | Terrain casters in the cascades to the fixed `cascadeCasterDistance` (1371 m) at every elevation, the horizon map's `G` from there; instances keep the height cap (finding 4) | Mountains taller than the instance cap shadow near receivers with cascade precision |
| D25 | Horizon march: linear at texel size to 4 km, then geometric on a max-height mip chain; dense CPU reference (finding 5) | No missed crests, no flicker of the far edge; conservative far shadows |
| D26 | `SceneLightingSet` owns set 1; shadow and atlas pipelines use a separate small layout and do not include `lighting.glsl` (finding 11) | One owner; no shadow map bound while it is the attachment (a validation error) |
| D27 | Benchmarks: an ACES baseline of the unchanged code before a1b; the clay path re-checked after the a1c flip; `terraform` in a2 and b (finding 10) | Separates the tone-map cost from the lighting cost |
| D28 | `WorldTime::validAt` / `Weather::validAt` (World::time() of the write) drive the render time on host and client (finding 18) | One formula; no jump at a rate change; no access to snapshot internals |

## Design changes

(filled during implementation: deviations from this plan and the amendments they cause)

**S36a1a** (2026-11-22):

- **The app sends the environment only for worlds with `EnvironmentSetup::enabled` until the a1c flip**
  (`EngineApp::sendsEnvironment`). A1 says every app with a device sends it (so enclosed spaces light correctly), but in
  a1a / a1b the scene shaders still write unit-less values: an app with `--tonemap aces` would get the EV100 exposure
  (about 2^-15 at noon) and render black, an image change of a part that must change none. No game enables the
  environment before a1c, whose flip commit makes the call unconditional.
- **`RenderScene::exposure()` is the latched multiplier** (`params.exposure`, 1 with Identity) and the new
  `frameExposure()` the effective exposure the post pass uses; without an environment both are equal, so S17's tests
  and `EngineAppRender`'s exposure case are unchanged (the plan's "keep their meaning" made explicit).
- **`WorldTime::rate` is game seconds per simulated second** (a time multiplier). A1's comment "game hours per
  simulated second" contradicted its own formula `rate * dt / 3600`, the range [-3600, 3600], the tab's rate buttons
  x0 / x1 / x60 / x600 and `--day-length s` (rate = 86400 / s); the formula is kept, the text corrected.
- **Celestial model details**: the moon's hour angle is the sun's **minus** `2 pi * moonPhase` (A1 says plus; minus
  gives A1's stated behaviour, phase 0.25 rises about at noon; phase 0.5 is the same either way) and its declination
  is the sun's times `cos(2 pi * phase)` (the full moon opposite the sun); the solar declination uses the day of the year
  at noon (`day + (timeOfDay - 12) / 24`), so day 0 at 12:00 is exactly the equinox.
- **`setWorldTime` / `setWeather` write the component directly** and log their `api` line themselves instead of going
  through a `WorldApi` call: the app's `WorldApi` is bound to the client world in `--coop` and a temporary one would
  attach listeners; validation, the S16 refusal on a client and the "between steps" rule are the same.
- **`EnvironmentSettings::starIntensity`** (a multiplier of `CelestialConfig::starIntensity`): `set_lighting
  star_intensity` changes a setting, and `CelestialConfig` stays constant in `WorldSetup`. In the environment file it
  is `atmosphere.starIntensity`, and `EnvironmentSetup::cloudWindScale` is `clouds.windScale` (the file's keys stay the
  six of A1).
- **`LightingParameters` holds only the a1 fields** (`environmentResolution`, `environmentUpdateDegrees`, `clouds`,
  `forceFullUpdates`); `pbr` comes with a1b, the a2 and b fields with their parts (with their `set_lighting` keys).
- **The shadow light of a1a** (`selectShadowLight`, D13): the body with the higher "ground illuminance" approximated as
  top illuminance times the sine of the elevation (no transmittance before a1c's atmosphere), hysteresis factor 2; the
  extract is stateless (no previous choice) until a2 keeps one.
- **`SpaceTemplate::ambient` = (50, 50, 50), `ev100` = 6** (a room of about 50 lux), validated finite by
  `validateSpaceTemplates`. The enclosed EV100 also subtracts the exposure compensation (the user's knob for every
  source).
- **`get_lighting` without an environment** answers `exposure: {environment: false, multiplier, effective, note}`
  instead of `null` plus a separate key. `get_lighting` and `set_lighting` need a device (the tone map and the
  parameters are the scene's); `set_time_of_day` / `set_weather` do not.
- **The scenario's extrapolation check** (`world_time_advances`): strictly monotonic in alpha inside a step, and
  non-decreasing at alpha 1 from step to step: in the async variant the client's estimated host time holds while
  snapshots are late, so the render time of alpha 0 of the next step can be earlier than alpha 0.99 of the previous one
  (the check of "monotonic between snapshots" as the extrapolation guarantees it).
- **Test placement**: `Exposure.Ev100RangeReachesPost` and `.NoEnvironmentKeepsUnitlessExposure` are
  `VulkanScene.*` cases in `test_vulkan_scene.cpp` (the a1a Files list; `test_vulkan_lighting.cpp` starts with a1b);
  `EnvironmentWrites.CoopWritesTheHost` and `DebugConsole.RunsLightingCommands` are in
  `test_engine_lighting_tools.cpp` (they need an app); `RecordingRenderSink.EnvironmentAndLightsRecorded` is in
  `test_engine_extract.cpp`.
- **ImGui textures**: `ImGuiHeadlessRenderer` needed no change (it inherits the per-command binding);
  `VulkanRenderer::imguiRenderer()` (`Renderer.h`, not in the Files list) gives the tab and the test the renderer. An
  unknown texture ID's command is skipped.
- **Review round 1 of a1a** (2026-11-22):
  - `normaliseWorldTime` resets a `timeOfDay` of 1e15 days or more (a raw `set_global` of e.g. 1e300) to the default
    and saturates `day` at the int64 limits (the float-to-int64 conversion and the day addition were undefined).
  - `setWorldTime` / `setWeather` first fold in the progress since `validAt` (the extract's extrapolation) for the
    fields they do not set, so a write in a world without `worldTimeSystem` does not jump the sun or the clouds back.
  - `worldTimeSystem` writes `WorldTime` / a space's `Weather` only when a value changes (rate != 0, the wind moves
    the clouds, or the normalisation repaired a value): with rate 0 and no wind `validAt` keeps its value, nothing is
    replicated or appears in a save diff, and the extract is the same.
  - The environment file is range-checked: the values with a `set_lighting` key through `validateLightingValues`
    (`LightingTools.h`: every key's own validation, so later parts' keys are covered; the error names the
    `set_lighting` key), the exposure curve (not empty, finite, strictly ascending elevations) and the values without a
    key (`celestial.*`, `clouds.windScale` / `fogFalloff`, the atmosphere scales; the error names the key path).
  - `get_lighting.exposure`: `ev100`, `source`, `multiplier` and `effective` are now the sample's (what the next frame
    uses, consistent with `curve_ev100` and `compensation` right after a `set_lighting`); the new `frame_ev100` /
    `frame_effective` are the last rendered frame's. `app_status.lighting.ev100` is the sample's as well. The
    `set_lighting` schema gives every number key its range.
  - The Lighting tab: a "fixed EV100" slider (enabled in the fixed mode); the time and weather widgets show the values
    of the world they write (with `--coop` the hidden host, plus a "client view" row), so incremental widgets do not
    fight the replication delay; the day is an int64 input; a refused write is logged (`api`). `DebugUi::markItem` /
    `itemCenter` record named controls (the rate buttons, "override sun") so a test clicks the real widgets.
  - `--time-of-day` / `--day-length` with `--scenarios` is an error (the scenarios build their own worlds).
  - Tests: `EngineAppRender.EnvironmentReachesTheScene` and `.LightingTabRenders` moved to the suite apps of
    `test_engine_lighting_tools.cpp` (`LightingTools.EnvironmentReachesTheScene`; `LightingTabTest` and
    `LightingTabPlainTest`: one app each with ImGui at 800 x 480, with and without the environment):
    `test_engine_app` constructs four apps fewer; the tab case clicks the x60 rate button and
    the sun override; the environment file tests use per-process temporary file names and remove them.

## Results

(filled per part: test counts and ctest times on lavapipe, golden runtime, benchmarks before / after with the features
off and on, the Q6 measurement, screenshots reviewed)

**S36a1a** (2026-11-22, local, lavapipe = Mesa with LLVM 21.1.8; build with samples and warnings as errors, GCC):

- Commits: step 1 world time and weather, step 2 render interface, step 3 ImGui textures, step 4 tab and tools (with
  the docs). Every step built without warnings and passed ctest.
- ctest on lavapipe (`VK_DRIVER_FILES=... ctest --parallel 8`): 82 of 82 passed locally (80 with CI's options, AGENTS.md;
  2 new binaries `test_engine_lighting`, 27 cases, 0.18 s, and `test_engine_lighting_tools`, 9 cases, 0.32 s). New
  cases elsewhere: the six `world_time_advances` scenario variants (`test_engine_scenarios`),
  `VulkanScene.Ev100RangeReachesPost`, `.NoEnvironmentKeepsUnitlessExposure`, `VulkanImGui.ImageDraws`,
  `RecordingRenderSink.EnvironmentAndLightsRecorded`, `DebugUi.GameTabIndexAfterLighting`, the extended
  `DebugUi.KeysSelectTabs` and `EngineAppNoRender.SetUiTab`, `AppCommandLine.TimeOfDayAndDayLength`,
  `EngineAppNoRender.EnvironmentSettingsAndTimeOption`, `EngineAppRender.EnvironmentReachesTheScene`,
  `.LightingTabRenders`.
- `test_engine_app` time (ctest, lavapipe, parallel 8; finding 28): 32.7 s before a1a, 33.8 s after (two new
  render cases).
- No committed image changed (no golden touched); the five clay goldens and `hdr_tonemap` pass unchanged.
- Game scenarios (`--scenarios`, all variants): flight 30 passed, platformer 48 passed, racing 18 passed.
- Screenshot reviewed: `SAIGA_DEBUG_TAB=5 SAIGA_SCREENSHOT=... game_racing --headless --demo`: the F5 Lighting tab with
  the line "the world has no WorldTime (EnvironmentSetup::enabled)" (no game enables the environment before a1c), Sun
  and moon (45 degrees at noon, 120000 lux top) and Exposure ("no environment", the curve's 14.62).
- Benchmarks: not run in a1a (the recipe needs a window; a1a changes no game's frame: the games send no environment, and
  `latchFrame` gained one branch). The ACES baseline is measured before a1b as planned.
- Engine schema regenerated (`WorldTime`, `Weather`); the golden save loads unchanged.

## Review log

Plan review rounds (local Opus) and later the PR reviews are recorded here: date, reviewer, findings, resolution.

### Review round 1 (Fable), 2026-11-22

3 blockers, 14 major, 23 minor, 12 nits, possible cuts, Q1-Q7. All adopted except where noted.

- **Blockers.** 1: exposure clamp to [2^-24, 2^10], EV100 formula with compensation, `--exposure` as `log2(x)`,
  post/resolve range check, `Exposure.Ev100RangeReachesPost` (A3, D18). 2: types always registered, kind chosen by
  `enabled`, `registerEngineComponents(Registry&, const WorldSetup&)`, test renamed
  `RegisteredButNoEntityWithoutEnvironment` (A1, D4). 3: the legacy path keeps its inputs until the flip commit, which
  also does the migration; the guard covers all four existing goldens (A3, a1 steps 4 and 9).
- **Major.** 4: one-sided band, `s` scaled with texel and elevation, march from 64 m, second channel G for occluders
  beyond the cascades' caster range, `FlatGroundIsLitAtLowSun` (A6). 5: light-space caster volumes with a near cap at
  `maxCasterHeight / tan(max(el, 5 deg))`, `LongShadowCrossesCascadeSplit` (A5). 6: far cap, cascade selection by
  projection, de-duplication against the previous cascade's shrunk box, small-caster rule, `shadowLodBias` for
  cascades 2-3, triangles per cascade (A5, A7). 7: staggered cascades fitted around the camera with a speed margin,
  default intervals {1, 1, 1, 2}, `StaleCascadeStillCoversFrustum` (A5). 8: no pull-back; bias in texels on the
  receiver, slope factor on the rasterizer, compare op, clear and not-reversed stated (A5). 9: no `depthClamp`
  (pancaking in the vertex shader), linear-filter fallback (Rules, A5, D17). 10: world-anchored map recomputed only
  outside the inner half, three maps with blending, adaptive rows per frame, 48 march steps,
  `NoRecomputeForSmallCameraMoves`, `BlendsDuringRecompute` (A6). 11: clay studio environment (A3, D7). 12: steps
  reordered: UI core and tools with the PBR step, each feature with its UI and tools, tuning, goldens last per part
  (Implementation steps, D19). 13: `EnvironmentSettings` owned by `EngineApp`; `WorldSetup` keeps `enabled`,
  `celestial`, `cloudWindScale`; games set `LightingParameters` in `EngineAppSetup::render.lighting` (A2, D16).
  14: linear depth in the atlas (D16 steps of range / 65536), `NoAcneAtRange` (B4). 15: selection by tile budget,
  `lights_without_tiles`, `TileBudgetLimitsShadowedLights` (B4); the optional second tile size is not adopted (one size
  keeps the allocator trivial; the budget rule already handles the demo). 16: dirty bounds, fixed terrain level, wind
  phase 0, dynamic tile clear (B4). 17: front/back cube and SH, box-downsampled SH source, ground below the horizon
  (A4).
- **Minor.** 18: three parts (D1). 19: `pbr` removed in the a1 flip commit. 20: client extrapolation with `HostClock`
  and the scenario check (A1). 21: normalisation every step, validated writes, the `visit` limits claim dropped (A1).
  22: `--coop` writes the host (A9, tests). 23: `screenshot` extended, `capture_lighting(frame)` dropped (A9, D20).
  24: png default for data images, statistics and histogram, `max_width` 1280 (A9). 25: `probe_lighting` and the
  `luminance` section (A9). 26: consistent argument names, every `LightingParameters` field run-time changeable with a
  key (A2, A9). 27: cost measured in a1.4 with a debug-variant fallback; `SCENE_FOG` removed at the flip (A3, A7). 28:
  one scene per suite, 60 s target per binary, SPIR-V cache fallback (Tests). 29: switch by ground illuminance with
  hysteresis, elevation clamp for the volumes (A5, D13). 30: shadow terrain at the main view's level and morph (A5).
  31: curvature and edge fade; M9 covers pass 11 (A4, M9). 32: one invocation per cluster, fixed `uint` slots, slices
  exponential from 2 m (B3). 33: all three test `Weather` structs renamed (a1 step 1, Risks). 34: the extract is the
  single authority from the template; `lighting_enclosed_lights` golden and `Lights.EnclosedSpaceLit` (A1, B2). 35:
  rain / snow shading dropped (Out of scope). 36: `Lights.BulkListPerChunk`, `Shadows.PipelineTableSlotForNewClass`.
  37: Benchmarks section (features off vs on, single-view check). 38: SH as its own binding, `minAmbient` start value
  (A3). 39: `GetTexID`, no `RendererHasTextures`, barrier, `deferDelete`, headless path (A8). 40: M12 dependencies, M9
  pass 10, M13 `setTime`.
- **Nits.** 41: `nearPlane` / `farPlane` and the rule. 42: 120000 lux in both. 43: `shadowLodBias`, `clusteredLights`
  in the struct. 44: two exposure modes plus `source`. 45: `AtmosphereSettings` in `EnvironmentSettings`. 46: R-DATA-1
  cites A1 and `readDataFile`. 47: `TerrainFrameUniforms`. 48: stars rotated by the hour angle, >= 1.5 px, no twinkle.
  49: basis rule with a zenith fallback, `BasisStableNearZenith`. 50: `(alpha - 1)`. 51: Design changes / Results
  placeholders added. 52: entity lights vs S30 piece lights noted (B1).
- **Possible cuts**: all adopted (Out of scope); `irradiance` / `brdf_lut` stay as `capture_lighting` images.
- **Q1-Q7**: decided as recommended (D7, D1, D6, D4, D5, D15, D2); demo day 10 min (Q5).

### Review round 2 (Fable), 2026-11-22

2 blockers, 9 major, 17 minor, 9 nits, plus the round-1 findings not resolved. All adopted except where noted.

- **Blockers.** 1: adopted. (a) D22: `enableEnvironment` sets ACES, `--tonemap identity` stays the clay view; the
  consequences (look, `resolve` section, benchmark baselines) in A10 and Benchmarks. (b) D21: seven constant prototype
  albedo colours (ground, rock by slope, shore, water, grass, leaves, bark) in the ACES path until S43, gray in clay.
  (c) `tone_map` in `get_lighting`, `set_lighting`, `app_status.lighting` and the tab toolbar. 2: adopted with the
  proposed seams: a1 is now a1a / a1b / a1c (Split, Implementation steps, D1); every Files list was counted with a
  script: a1a 61, a1b 59, a1c 86, a2 63, b 61, including the files the reviewer named as missing
  (`PrototypeWorld.{h,cpp}`, `EngineMain.h`, `render_scene_test_context.h`, `terrain_test_hooks.h`,
  `terrain_edit_crater.png`) and the new ones of this round (`SceneLightingSet.*`, `BrdfLut.*`, `scene_block.glsl`,
  `horizon_maxmip.comp`, `common/Environment.*`, `TerrainAtlas.*` for the max chain, `test_vulkan_shadows.cpp`,
  `test_vulkan_local_lights.cpp`). Differences to the reviewer's cut: `DebugLines` moved to a1b (it needs shaders),
  the tab's Exposure / Ambient sections to a1b (they need the block). The clay goldens now stay unchanged in a1c (finding
  8), so a1c commits only four goldens.
- **Major.** 3: one shared light-space depth range, cascades as xy rectangles, selection and de-duplication by xy
  (A5, D23); `Shadows.LongShadowLeavesNearCascade`. 4: terrain casters to `cascadeCasterDistance` at every elevation,
  `G` from the same constant (A5, A6, D24); `Shadows.CliffNearCameraUsesCascade`. 5: max-height mip chain, linear march
  to 4 km, dense CPU reference with a one-sided tolerance, `FarEdgeStableUnderSmallRotation` (A6, D25). 6: one exposure
  rule in `latchFrame`; a scene without `setEnvironment` stays unit-less with the studio light, so the existing
  exposure tests and `hdr_tonemap` are unchanged (A3, D18). Deviation: `--exposure` needs no conversion to a
  compensation any more (it stays the multiplier `params.exposure`). 7: `probe_lighting` is a 6-frame debug-view
  sequence with 1 x 1 readbacks plus one compute dispatch, camera and environment held, `consistent` flag (A9, D20);
  the full key set is kept for agents. 8: (a) studio fog with today's values, (b) flat classes keep their material
  colour in clay, (c) enclosed spaces in clay use the studio light without fog; and the suggestion to reproduce S19a
  exactly is adopted: the five clay goldens pass unchanged through a1 and are run with both pipeline variants in
  a1b-a1c (A3, D7). 9: `terrain_edit_crater` added everywhere; `Pbr.LegacyLightingMatchesGoldens` dropped, the existing
  golden tests are the guard (Tests). 10: ACES baseline before a1b, clay re-check after the flip, `terraform` in a2 and
  b, `sample_vulkan_scene_benchmark` runs clay (Benchmarks, D27). 11: `SceneLightingSet` and the separate shadow layout
  (A3, D26).
- **Minor.** 12: `RenderSceneSuiteTest`, suite rules, binaries per area (Tests). 13: `get_lighting.luminance` dropped,
  `capture_lighting(stats_only)` (A9). 14: `mountainRidgeHook` (Tests; added in a1c for the sunset golden). 15: a
  transform per horizon map, `AnchorMoveKeepsShadows` (A6). 16: no cloud shadow above the layer, fade by the projected
  distance, two tests (A4). 17: `LightingTypes.h` in step a1a.1. 18: `validAt` in `WorldTime` / `Weather` (the second
  option of the reviewer, as `World::time()` instead of a step: valid across saves with another step rate; A1, D28).
  19: valued engine options for `--time-of-day` / `--day-length` with an error without `WorldTime`; `--lamps` parsed
  by `game_world` like `--base-spacing` (A10, B7); `EngineMain.{h,cpp}` are in a1a. 20: (a) loops in the write pass,
  (b) `viewMask` in `pad`, `scratchBase` in a parallel buffer (16 bytes kept), (c) scan time reported, (d) main-camera
  end distances for shadow views, LOD bias clamped (A5). 21: the moon's intent stated, `NewMoonNightHasNoShadowLight`,
  `FullMoonRisesAtSunset` (A1). 22: entity check in a1c, content hashes unaffected (`flight_content_hashes`), no join
  check before S26 (M12 adds it to S26's row) (A1, A10). 23: all-or-nothing validation with cross-key rules and a 1
  GiB memory refusal (A9). 24: no pipeline depends on the tone map (`RenderScene.cpp:265`, push constant `linear`;
  tested), raw data views read from the fp16 image, pre-exposure history kept (A8, D12). 25: the finest resident level
  with re-render on residency (B4). 26: `pbr` is a pipeline variant (`LIGHTING_PBR`), the 5% comparison on the legacy
  variant (A2, Benchmarks). 27: tile budget in M5. 28: one app per suite for the tool tests, `test_engine_app`'s time
  before and after (Tests).
- **Nits.** 29: `SpaceKind::Enclosed`. 30: handler rejection. 31: `ev100` resolved + `exposureSource`. 32: "3 x 3
  bilinear compares (4 x 4 texels)". 33: D32 = 4 B stated in A5 (A7 was already right). 34: `SAIGA_DEBUG_TAB=5`, no code
  change beyond the enum and comments. 35: the ctest pattern and the build-first note (Verification). 36:
  `facesPerFrame <= 11` asserted (B4, M8). 37: M12 dependencies incl. S19c and S23c; M13 resolutions.
- **Round-1 findings not resolved**: R1-6 -> finding 3; R1-5 / R1-4 -> finding 4; R1-1 -> finding 6; R1-11 -> findings
  1 and 8; R1-18 -> blocker 2; R1-25 -> finding 7; R1-28 -> finding 12; R1-37 -> finding 10; R1-3 -> finding 9; R1-10
  -> finding 15. All resolved as above.
- **Not adopted**: none of the findings. Two details differ from the reviewer's proposal: finding 18 uses
  `World::time()` instead of a host step index (survives a save loaded with another step rate), and finding 7 keeps
  the full key set through the sequence instead of the reduced compute-only variant (the user requirement for agents).
  Also verified: the world goldens live in `test_engine_terrain_render.cpp` (`VulkanWorldGolden.*`), not in
  `test_vulkan_world` as AGENTS.md's recipe says; the plan's Verification uses the right binary and AGENTS.md changes
  correct the recipe.

### S36a1a code review round 1, 2026-11-22

2 major, 9 minor, 6 nits (`r1_a1a.md`, the diff `origin/master..stage-s36a1a`). Fixed in ba2a960c8: majors 1
(`normaliseWorldTime` saturating) and 2 (the tab's fixed EV100, the host's values in `--coop`), minors 3-11 (the
extrapolated writes, the range-checked environment file, `get_lighting.exposure` of one frame, the tool and tab tests,
the render cases on the suite apps, unique temporary files, the font check, writes only on change) and nits 12-17. Not
fixed: none. Recorded in round 2 (finding 21; the entry was missing). See Design changes "Review round 1 of a1a".

## Questions for the user

Decided by the plan (D1-D28; the user can still override any of them): the five-part split, ACES by default for games
with an environment (D22), the constant prototype colours (D21), clay reproducing S19a (D7), the probe sequence (D20).
Answered by the user (2026-11-22):

- **O1: Five PRs instead of three.** Accepted: S36a1a, S36a1b, S36a1c, S36a2 and S36b are five stacked PRs, each under
  CodeRabbit's 100 files.
- **O2: Re-baselining `game_world`'s benchmarks in ACES.** Accepted: from a1c the committed `data/tests/benchmarks.json`
  baselines of `game_world --benchmark flyover|forest` are ACES numbers (the game's new default); the clay numbers are
  reported once (the clay re-check after the a1c flip).
