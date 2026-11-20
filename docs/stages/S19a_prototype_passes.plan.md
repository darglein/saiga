# S19a: Prototype passes in RenderScene

Status: done.

Sections follow the stage plan template of S05.

## Goal

`RenderScene` draws the prototype world itself: terrain, grass, sky and water through `setPrototypeTerrain`, the trees
as S18 instance sets with a ported foliage pipeline class, an opaque depth pre-pass and a depth resolve, time from the
caller instead of the wall clock. `game_world`, flight and racing render their world through it; the S02 goldens and the
per-LOD tree counts hold, and the frame stays within the post-S18 baseline + 5 % (or an explained budget). Split from
S19 in planning: the engine side (render sink, extract, `RenderInstance`) is S19b.

## Depends on

S18 (instances, culling, explicit mesh IDs, the hand-over of the tree shading and the forest benchmark), S17 (HDR
frame, sections, output contract), S14 (`game_*` tools), S13b (the demos' world rendering), S02/S03 (goldens,
benchmarks).

## Rules that apply

R-DET-1 (the prototype's wall-clock time in `WorldRenderer::transfer` goes away: the §3 deviation is removed),
R-TEST-1, R-TEST-2 (`RenderScene`, culling, terrain: benchmarks before and after), R-MECH-1 (the toggles and distances
stay game settings), R-MATH-1, §5.2 (all in `saiga_vulkan`), §0 (other samples keep their renderers), AGENTS.md.

## Scope

### 1. Prototype content in `RenderScene` (`vulkan/scene/PrototypeWorld.{h,cpp}`, renderers moved from `vulkan/world/`)

```cpp
struct PrototypeSettings   // from WorldSettings: sun, fog, wind, terrain and grass LOD, toggles
{ ...; bool terrain = true, grass = true, sky = true, water = true; };   // racing turns grass off
void RenderScene::setPrototypeTerrain(std::shared_ptr<const HeightField> heights, const PrototypeSettings& settings,
                                      uint32_t foliageSeed);   // WorldParameters::seed (the foliage texture)
void RenderScene::setPrototypeSettings(const PrototypeSettings& settings);   // at run time (game_set_settings)
void RenderScene::setTime(double seconds);   // the caller's time (default 0); the phase uniforms fmod(time * f, 2π) are
                                             // computed in double for the five wind frequencies (trees 1.1, 0.9, 4.0;
                                             // grass 1.7, 1.3) (§10.1: no absolute time in a shader); games pass
                                             // world.time() + alpha * dt (no 60 Hz judder); benchmarks and goldens
                                             // call setTime(0) (replacing freezeTime)
```

- Terrain (CDLOD), grass, sky and the water plane are drawn by `RenderScene` until S20 replaces them. The prototype's
  in-shader fog stays (per sample, before the clamp and encoding; added to the instance shaders too) and the water
  plane stays a mode of the terrain draw in the opaque MSAA pass: moving fog to the pass-10 composite and water after
  the resolve would change foliage edges and the shoreline beyond the golden gate (CI already uses 0.19 % of the 0.5 %
  pixel budget on `world_forest`). Both deviations from §10.5's pass order are recorded there; the composites come with
  S36-S38 (fog) and S39 (water).
- The instance shading (S18's flat model) and the foliage class read the sun, ambient and fog from
  `PrototypeSettings` (a `RenderScene` lighting block until S36), so the demos' meshes keep the world's light; without
  `setPrototypeTerrain` the block equals S18's fixed sun and ambient with fog off (`scene_flat_box` unchanged).
- `freezeTime` and `freezeCulling` are removed (`setTime` covers goldens and benchmarks; the cull camera has no user).

### 2. Trees as instance sets

- `PrototypeFoliage` pipeline class (S18's `MaterialData::pipelineClass`): `tree.vert` (wind) and `foliage.frag` (the
  foliage texture array, alpha to coverage with the `alphaCutoff` uniform, LOD tint from the entry's LOD bits) ported
  onto `MeshVertex`; the color word packs the texture layer (8 bits, 0xFF = opaque parts), AO (8 bits) and the wind
  weight (8 bits; the conversion asserts AO and the weight in [0, 1]); the class's vertex shader passes AO and uv as
  smooth outputs (S18's color word is `flat`); the normal is S18's oct snorm16 (covered by the golden check). It
  asserts at `addMaterial` / `setMaterial` with `PrototypeFoliage` that `setPrototypeTerrain` provided the foliage
  texture (no rendered frame needed). The shading lives in an include (alpha sharpening,
  clay, tint, fog) shared with the grass, which keeps `grass.vert` + `foliage.frag` on `TreeVertex`.
- Wind phase: `tree.vert` derives the phase from the tree's world xz (`posYaw.x * 0.031 + posYaw.z * 0.027`, with the
  1.7 and 5.0 multiples); the foliage vertex shader rebuilds the prototype's float world xz as `instance.position +
  (hi + lo) + cameraPos.xz` (the camera position as a float uniform; valid within the prototype's 8 km), a transitional
  exception to the camera-relative rule recorded next to `setTime` (S20/S23 replace the prototype).
- `PrototypeForest` (saiga_vulkan) owns the tree meshes, their material and the set handles, with
  `apply(const WorldSettings&)` diffing `trees`, `treeLod*` and `wind`; `WorldSettings` (the JSON keys of
  `game_set_settings`, `fovy`) stays in `vulkan/world/` and maps to `PrototypeSettings` (`WorldLighting.h` reads it);
  used by the three games (`samples/games/common/`) and the tests, driven by the settings GUI and `game_set_settings`.
- `prototypeTreeSets(const HeightField&, int treeCount, uint32_t seed, species, meshIds) -> per-chunk StaticInstance
  vectors` (saiga_vulkan; from
  `ForestRenderer::placeTrees`, one set per 256 m chunk, region level 0, the chunk index clamped at the world's edge,
  the instances in `placeTrees` order: deterministic keys); the tree meshes use a reserved ID range (S18 reserves the
  top for S30; the prototype tree meshes and their material take IDs from the game range `1 .. kGameReservedIds - 1`
  (`RenderSceneTypes.h`, 256 IDs, names prefixed `prototype/`) through `RenderScene::reserveIds`; S19b's `World` name
  tables assign from `kGameReservedIds` upward; the tree meshes take `sphereCenter = (0, bounds.x, 0)` and
  `sphereRadius = bounds.y` from `TreeSpecies::bounds` and `boundsPadding = 0.025 * wind`, as `tree_cull.comp`): used by `game_world`, flight, racing and
  `test_vulkan_world`. Tree meshes per species and LOD with the prototype's distances as `endDistance` and the wind sway
  as `boundsPadding`. `ForestRenderer`'s culling and draw are removed; `placeTrees` and the mesh generation stay.
- LOD distances and wind (`treeLod1/2/MaxDistance`, `wind`): the settings are clamped independently (`treeLod1` 10-500,
  `treeLod2` 50-3000, `treeMaxDistance` 500-50000), so they map to `end = {min(l1, max), min(max(l1, l2), max), max}`
  (S18's assert is non-strict: equal distances give an empty LOD, as `tree_cull.comp` tolerates);
  `RenderScene::setMeshLodDistances(MeshId, endDistances, boundsPadding)` rewrites only the mesh record (same LOD count; no geometry upload, no bucket rebuild); `RenderScene`
  keeps per set the unpadded AABB and the maximum scale, so set bounds are recomputed with the new padding. The `trees`
  toggle (today max distance 0) removes and re-adds the tree sets. Settings decided: `flatShading` (terrain) and
  `debugMode` (LOD tint from the LOD bits) are kept; `alphaCutoff` is a foliage-class uniform; `wireframe` and
  `freezeTime` are dropped (`VulkanWorld.SettingsJson` and the benchmark's `wireframe` key follow). `game_get_state` and
  the statistics window keep the per-LOD tree counts and `tree_triangles` as sums over the tree meshes of S18's
  per-(tracked mesh, LOD) counts (S18's readback gains a short list of tracked mesh IDs, set by `PrototypeForest`, whose
  per-LOD counts the scan writes; no CPU bucket decoding), forest memory from the instance buffer; `game_set_settings` / `game_get_state` and their S14
  tests follow.

### 3. Depth pre-pass and resolve (§10.5 passes 2-3)

- Pre-pass: opaque content only. In S19a that is the terrain and water draw (a tree mesh is trunk and leaf cards in
  one `PrototypeFoliage` draw, which is never in the pre-pass; the demos' meshes are still `MeshRenderer`) and opaque
  S18 instances; then the opaque pass with `EQUAL` and no depth write; alpha-tested content (grass, foliage) keeps its
  depth-writing draw in the scene pass (recorded as a deviation of §10.5, revisited with S37's budgets). The terrain
  already draws first, so the pre-pass only removes terrain self-overdraw (a 0.05-0.09 ms pass) and costs a second CDLOD
  vertex pass: if the measured frame is worse with it, it is off for the prototype by
  `PrototypeSettings::depthPrePass` (the mechanism stays for S37), or kept with an explained budget. Each content uses one vertex
  shader source for both passes with `invariant gl_Position`, the same time and alpha, so the positions are
  bit-identical (CDLOD morph, wind).
- Depth resolve: the depth resolve attachment of the scene pass (`SAMPLE_ZERO`: lavapipe supports only it; reversed-Z
  "nearest" would be MAX, not needed before S37's consumers), so grass, foliage and the callback meshes are included;
  the scene pass loads the pre-pass depth (pre-pass `storeOp` store, scene `loadOp` load); at 1 sample there is no
  resolve and the depth image itself carries SAMPLED usage; the sampled depth's layout transitions follow S17's sync2
  rules; §10.5's wording is corrected.

### 4. Sections and benchmarks

- Scene-pass order: terrain → the game's render callback → instances → grass → sky (the sky's `EQUAL` at depth 0 last).
  Sub-sections inside one pass are consecutive `ALL_COMMANDS` timestamp marks (as the S03 pass times), not S17's
  `TOP_OF_PIPE`/`BOTTOM_OF_PIPE` pairs, which can overlap; all sections and sub-sections are written every frame, also
  when their content or the pre-pass is off (S17's rule).
- GPU sections (S17's flat list, extended): `transfer`, `cull`, `depth`, `depth.terrain`, `scene`, `scene.terrain`,
  `scene.instances`, `scene.grass`, `scene.sky`, `resolve`, `post`, `imgui` (the sub-sections as extra flat names, also
  in `gpuSectionNames` and `gpu_frame_ms`). `samples/games/world/main.cpp` writes the S03 `gpu_ms` keys from them:
  `trees` = `scene.instances`, `upload + cull` = `transfer` + `cull`, `terrain + water` = `depth.terrain` +
  `scene.terrain`, `grass`, `sky` (`wireframe` dropped). `WorldRenderer`'s own query pool, `FrameTimes` and
  `frameCallbacks` are removed. `WorldBenchmark` (saiga_vulkan, no engine `Profiler`) gets its own sample struct
  `{frame, intervalMs, workMs, sectionMs}`, fed by `samples/games/world/main.cpp`: frame numbers from
  `RenderScene::frames()`, the interval measured from `renderFrame` to `renderFrame` (the S03 `cpu_ms`),
  `cpu_work_ms` as the app's `frame/render/recording` (a baseline note: not comparable with the S03 value), the warm-up
  and measured frame numbering on these frames, `stalled_frames` from the interval.
- The bound: `transfer` + `cull` + `scene.instances` (measured in `game_world`, which has no other instances) <= S03
  `trees` + `upload + cull`; the whole frame (`cpu_ms` median and `gpu_frame_ms.gpu`) <= the baseline committed after
  S18 + 5 %, or an explained budget (R-TEST-2). A pre-pass
  prototype is measured first (the PR reports with and without).

### 5. Games

`game_world`, flight and racing call `setPrototypeTerrain` and create their tree sets in `renderHooks.init`; their
`WorldRenderer` keeps only the height field and generators (S20 replaces them). The demos' own meshes still draw with
`MeshRenderer` in the render callback, using `RenderScene::viewProjection()` (S19b moves them to entities).

## Out of scope

The render sink, extract, `RenderInstance` and the demos' entities (S19b), fog and water composites (S36-S39), the
alpha-tested pre-pass (S37), height tiles (S20).

## Design notes

- In-shader fog and water in the MSAA pass keep the goldens; the composites only pay off with volumetric fog and real
  water.
- An opaque-only pre-pass is the cheap part of the win (terrain overdraw); alpha-tested content is drawn once.

## Tests

`tests/test_vulkan_world.cpp` (lavapipe): `world_overview` and `world_forest` through `RenderScene` with
`setPrototypeTerrain` and `prototypeTreeSets` (`setTime(0)`, `nearPlane` 0.1, fovy 60 through the test's `setCamera`)
pass the S02 goldens; the tree counts per LOD keep the committed `EXPECT_EQ` values and the terrain node counts (45 / 38) (on a
mismatch, a CPU reference of the prototype rule over `placeTrees` with the §17.5 1e-3 band decides, and differences up
to the band population are accepted and the counts updated); wind 0 → 4 and crossed LOD settings (`tree_lod1 = 500, tree_lod2 = 50`) through
`PrototypeForest::apply`: counts equal the CPU reference; the
`trees` toggle; `SAMPLE_ZERO` resolve and the 1-sample sampled depth validation clean; `setTime` makes the image
independent of the wall clock; the foliage class asserts without the foliage texture (death test); racing's grass
toggle; `VulkanWorld.SettingsJson` without the dropped keys. The other world tests move to the `RenderScene`
fixture: `RendersTerrainTreesAndSky` (the sky toggle), `SharesAnExternalHeightField`,
`FlatShadingIsStableFarFromTheOrigin` (the 7 km regression); `lastFrameTimes` becomes a sections test (every completed
frame delivered, the sub-sections present); `WorldBenchmark.ResultJson` uses the new sample struct; the
forward-renderer fixture is then dropped (S17's note that it keeps `VulkanForwardRenderer` is updated). `tests/test_vulkan_scene.cpp`: the pre-pass gives zero
differing pixels against a render without it (`LESS`/depth write) with the terrain and an opaque S18 instance (static
and dynamic at alpha 0.5); `scene_flat_box` unchanged without `setPrototypeTerrain`.

Locally: `game_world --benchmark flyover` and `forest` before and after (the bound above; S18's culling benchmark as the
input), each twice; flight and racing `--demo` screenshots compared with S18's.

## Done when

- The S19a part of the S19 row of §22: S02 goldens within tolerance with equal drawn counts per LOD (§17.5) (CI); the
  performance bound (local).
- `docs/engine_design.md` updated: §10.1 (`setPrototypeTerrain` with a `shared_ptr<const HeightField>`, `setTime`,
  `PrototypeSettings`, `setMeshLodDistances`, `freezeTime` removed, the tree sets created by the game, not through
  `RenderSink::addInstanceSet`), §8.3 (the `passName` / `frameTimes` hooks), §17.6 / §17.7 (`freezeTime` and
  `gpuPassName` wording), §18.1 (files),
  §10.5 (pre-pass of opaque content, `SAMPLE_ZERO`, the fog and water deviations and their stages), §17.7 (the sections
  and the key mapping), §19 (the `ForestRenderer`, `WorldCommon.h` / `common.glsl` and `WorldBenchmark` rows), §3 (the
  wall-clock deviation removed), §22 (S19 split into S19a and S19b; the row's bound worded against the
  baseline committed after S18); `AGENTS.md` (the benchmark keys, the `vulkan/world/` and
  `vulkan/scene/` layout lines).

## Verification

The CI recipe of AGENTS.md with samples ON; `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R test_vulkan`;
the benchmark recipe of AGENTS.md; the demos' screenshots.

## Files

`src/saiga/vulkan/scene/` (`PrototypeWorld.*`, `RenderScene.*`), `src/saiga/vulkan/world/` (renderers moved or reduced,
`ForestRenderer` placement only, `WorldBenchmark`, `WorldStatistics.cpp`, `WorldMcp.*`), `tests/test_vulkan_scene.cpp`, `shader/vulkan/world/`, `shader/vulkan/scene/`,
`samples/games/{world,flight,racing,common}/`, `src/saiga/engine/app/` (`passNames` / `passTimes` removed for
`game_world`), `src/saiga/vulkan/scene/RenderSceneTypes.h` (`kGameReservedIds`), `tests/test_vulkan_world.cpp`, `tests/test_engine_app.cpp` (sections),
`data/tests/benchmarks.json`, `docs/engine_design.md`, `AGENTS.md`, this file.

## Design changes

- **Files**: `TerrainRenderer` and `GrassRenderer` moved to `vulkan/scene/` (used by `PrototypeWorld`); `WorldCommon.h`
  stays in `vulkan/world/` (the prototype's `WorldUniforms` and the frustum helpers of the tests); `WorldSettings` got
  its own `vulkan/world/WorldSettings.{h,cpp}` (`PrototypeForest` needs it, `WorldRenderer` owns a `PrototypeForest`);
  `PrototypeForest` (with `prototypeTreeSets`, `prototypeTreeMesh`, `prototypeTreeLodEnds`) is in
  `vulkan/world/PrototypeForest.{h,cpp}`. Shaders: `scene/prototype_foliage.vert/.frag`, `scene/scene_lighting.glsl`
  (the lighting block), `world/prototype_shading.glsl` (the shared include: clay light, fog, LOD tint, output, alpha
  sharpening; `shading.glsl` wraps it with the world uniforms); `world/tree.vert` and `tree_cull.comp` removed,
  `world/foliage.frag` stays for the grass.
- **`WorldRenderer` keeps more than the height field and generators**: the settings, the `PrototypeForest`, the GUI and
  the statistics window (from `RenderScene`'s statistics and the newest GPU frame), `attach(RenderScene&)` (the
  `setPrototypeTerrain` and the trees) and `update()` (per frame: `setPrototypeSettings`, `PrototypeForest::apply`). It
  has no GPU resources; `WorldMcp`, `WorldLighting.h` and the three games keep using it. `statsJson` (game_get_state)
  keeps the tree counts per LOD, `tree_triangles` and `terrain_nodes`; `gpu_world_ms` / `gpu_pass_ms` became
  `gpu_frame_ms` / `gpu_section_ms` (RenderScene's frame and sections), the CPU `transfer` / `render` times are gone.
- **The depth pre-pass is off for the prototype** (`PrototypeSettings::depthPrePass` and `WorldSettings::depthPrePass`
  default false; the setting `depth_pre_pass` in the GUI and `game_set_settings`): measured slower in both benchmarks
  (Results), as the plan anticipated. The mechanism stays (tests compare it pixel-exact) for S37.
- **A fog variant of the flat pipelines** (`SCENE_FOG`, injected define; `SceneCulling` picks the variant per frame from
  the lighting block's fog flag): the camera-relative position as a varying cost 7 % of the flat draws in the culling
  benchmark (1.375 vs 1.285 ms) also with the fog off. The foliage class always has the fog.
- **The camera from the transfer callback**: the frame's camera is computed again after the game's transfer callback,
  and `setCamera` updates `viewMatrix()` / `projectionMatrix()` at once (with the last surface size), so a game sets the
  camera in its transfer hook and its `MeshRenderer` reads `RenderScene`'s matrices in the same callback (the plan's
  "the demos' meshes ... using `RenderScene::viewProjection()`"; S19b adds `beforeRender`).
- **`reserveIds(SceneIdKind kind, uint32_t count) -> uint16_t`**: one function for the two ID spaces (mesh, material);
  `addMesh` / `addMaterial` do not check that a game-range ID was reserved.
- **Lighting block** (set 1 of the instance draws, `SceneLighting`, per frame slot, with the foliage texture in binding
  1): the flat model keeps S18's constant ambient by default (`scene_flat_box` bit-identical) and uses the prototype's
  hemisphere ambient `ambient * (0.6 + 0.4 n.y)` with `setPrototypeTerrain` (`mesh.frag`'s model, so S19b's demo meshes
  match the world).
- **Set bounds**: the padded set bounds are the unpadded AABB grown by the largest padding of the set's meshes times
  the largest scale, also at `addInstanceSet` (S18: exact per instance); the per-instance count pass still decides,
  the set test only gets more conservative.
- **Statistics**: `RenderStatistics::terrainNodes` is filled by the prototype (the goldens' terrain node count);
  `drawCommands` counts the scene pass's indirect draws (the pre-pass repeats those of the opaque classes).
- **Prototype draws need a camera**: `PrototypeWorld` draws nothing in a frame without a camera (its node lists and
  uniforms are written in the transfer only with one).
- **Foliage vertices**: the normals are normalized before the oct encoding; AO and the wind weight are unorm8 in the
  color word (rounded), the layer 0xFF for the opaque parts.
- **Benchmark**: the frame interval is measured at the game's transfer hook (each `renderFrame`'s recording);
  `WorldBenchmark` takes the GPU samples when `game_world` collects them from the app's profile (after the measured
  frames, as before); `--benchmark` still needs a window (the runs used `SAIGA_SCREENSHOT` with a timeout, which the
  benchmark ignores).
- **Engine app**: `GameRenderHooks::passNames` / `passTimes` stay (their test stays); `game_world` no longer uses them
  (its passes are `RenderScene`'s sections), S19b removes them with `transfer` / `render`.
- **Tests**: `test_vulkan_world` (23 tests; the forward-renderer fixture `VulkanWorldRender` dropped): new
  `VulkanWorld.PrototypeTreeSets`, `.TreeLodEnds`, `.FoliageColorWord`; the fixture `VulkanWorldScene` with
  `RendersTerrainTreesAndSky`, `SharesAnExternalHeightField`, `FlatShadingIsStableFarFromTheOrigin`,
  `ForestFollowsSettings` (wind 0 -> 4, crossed LOD distances, the trees toggle against the CPU reference),
  `GrassToggle`, `TimeIndependentOfWallClock`, `DepthResolveAndPrePass` (4 and 1 samples, validation clean, pixel-equal),
  `Sections`, `FoliageClassNeedsTexture` (death tests), `GameTools`; `VulkanWorldGolden` derives from it.
  `test_vulkan_scene` (28 tests): `DepthPrePassMatches` (terrain, a static and a dynamic opaque instance at alpha 0.5,
  4 and 1 samples), `GpuSections` with the 12 sections; `test_engine_app`: the gpu scopes with the new sections.

## Results

Tests (`VK_DRIVER_FILES=.../lvp_icd.json ctest --parallel 8`, RelWithDebInfo, GCC 15): 65 of 66 pass; only
`test_opengl_window` fails locally (no display, as before). `test_vulkan_*` and `test_engine_app` (14 tests) also pass
on the RTX 5080 Laptop GPU with the validation layer (no validation message). The demos' `--scenarios` pass (9/9 each);
the `--demo` screenshots of flight and racing and `game_world`'s start view (`SAIGA_SCREENSHOT`, RTX 5080, with
`--validation`: no message) look as S18's.

Goldens (lavapipe, Mesa with LLVM 21.1.8): unchanged and passing, with the committed counts: `world_overview` (0, 7,
3575) trees per LOD and 45 terrain nodes, mean 0.13/255, 0.0072% of the pixels > 16/255, max 38/255; `world_forest`
(43, 281, 677) and 38 terrain nodes, mean 0.11/255, 0.034%, max 28/255. The CPU reference of the culling rule over
`placeTrees` gives the same counts (14 and 34 trees in the 1e-3 band). `scene_flat_box` and `hdr_tonemap`: 0/255. On
the RTX 5080 (not gating): `world_overview` mean 0.13/255 (0.027%), `world_forest` mean 0.61/255 (0.60% > 16/255; the
NVIDIA alpha-to-coverage edges), the same counts.

Benchmarks (release build of AGENTS.md, RTX 5080 Laptop GPU, 1920 x 1080; before = the S18 tip (the plan commit, built
from `git archive` in a scratch directory), after = this stage with the pre-pass off; 5 runs each, before and after
interleaved (the GPU clocks drift between batches: the S18 report measured the forest frame at 1.47 ms); medians of the
run medians, ms):

| | flyover before | flyover after | forest before | forest after |
|---|---|---|---|---|
| `cpu_ms` (frame interval) | 2.412 (2.32-3.84) | 2.481 (2.39-2.50) (+2.9 %) | 2.878 (2.50-3.02) | 2.336 (2.27-2.37) |
| `recording` | 0.126 | 0.187 | 0.115 | 0.157 |
| `gpu_frame_ms.gpu` (whole frame) | 2.132 (2.03-2.32) | 2.120 (2.08-2.22) (-0.5 %) | 1.867 (1.80-1.88) | 1.837 (1.81-1.86) (-1.6 %) |
| trees (`scene.instances`) | 1.440 | 1.429 (-0.8 %) | 0.468 | 0.466 (-0.6 %) |
| upload + cull (`transfer` + `cull`) | 0.024 | 0.076 (0.005 + 0.070) | 0.024 | 0.038 (0.005 + 0.033) |
| terrain + water | 0.106 | 0.094 | 0.066 | 0.051 |
| grass | 0 | 0 | 0.675 | 0.687 (+1.8 %) |
| sky | 0.011 | 0.011 | 0.008 | 0.008 |

- The bound `transfer` + `cull` + `scene.instances` <= S03 `trees` + `upload + cull`: flyover 1.504 vs 1.464 ms
  (+0.040 ms, +2.7 %), forest 0.504 vs 0.492 ms (+0.012 ms, +2.4 %): exceeded, an explained budget (R-TEST-2): the tree
  draws are 0.6-0.8 % faster, the difference is the generic culling (S18's count, scan and write passes over 928 sets,
  0.070 ms in the flyover, the culling benchmark's 0.068 ms) against the prototype's single `tree_cull.comp` pass
  (inside 0.024 ms with the uploads). The whole frame is within the bound: the GPU frame -0.5 % / -1.6 %, the frame
  interval +2.9 % in the flyover (the forest's before runs are in the slow mode of the bimodal interval).
- `recording` (the app's recording and submit, `cpu_work_ms` from now on) grows by 0.04-0.06 ms: the instance path's
  per-frame CPU work (the work chunks and the offset table of 928 sets, the descriptor writes), as in S18's culling
  benchmark. `terrain + water` is 11-23 % lower (passes below 0.1 ms, timed now by marks inside the scene pass instead
  of the prototype's marks around its own draws; §17.7's timer noise for such passes is up to 5 %); the grass +1.8 %.
- With the depth pre-pass (2 runs each, not interleaved, same build otherwise): flyover `gpu_frame_ms.gpu` 2.38 / 2.29
  vs 2.16 / 2.13 without, `terrain + water` 0.154 (`depth.terrain` 0.068 + `scene.terrain` 0.086) vs 0.095; forest 1.96 /
  1.99 vs 1.93 / 1.92 (`depth.terrain` 0.032 + `scene.terrain` 0.046 vs 0.052). The second CDLOD vertex pass costs more
  than the removed terrain self-overdraw saves, so the pre-pass is off for the prototype.
- Culling benchmark (`sample_vulkan_scene_benchmark`, 3 runs each, interleaved with the S18 tip): `cull_ms` 0.068 (both),
  `draw_ms` 1.285 vs 1.284 (with the fog variant; 1.375 before it), `scene_empty_ms` 0.045 vs 0.035-0.047 (the
  `SAMPLE_ZERO` depth resolve, about 0.01 ms), `recording_ms` 0.14-0.16 (both).
- `data/tests/benchmarks.json` is not updated (S17's baseline is not accepted yet).

Left open: CI has not run on this branch (lavapipe on the runner: the world goldens' margin on CI's Mesa; the MSVC
job: the new code on MSVC); the bound on the culling and instance passes is exceeded by 2.4-2.7 % (the generic culling's
passes, explained above); the flyover interval stays bimodal (S17); no committed baseline after S18 exists yet (the
comparison is against the S18 tip measured here).
