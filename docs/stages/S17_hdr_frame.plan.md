# S17: HDR frame

Status: planned.

Sections follow the stage plan template of S05.

## Goal

`RenderScene` owns its frame instead of wrapping `VulkanForwardRenderer`: the scene is rendered into an HDR target
(R16G16B16A16 float, MSAA), resolved (a hardware average for the clay look, a tone-map-weighted custom resolve for tone
mapped output), tone mapped with pre-exposure into the swap chain (or offscreen) image, and ImGui draws on top
single-sampled (§10.5 passes 7, 14, 15). Shaders are compiled for Vulkan 1.3 / SPIR-V 1.6 (the §10.2 prerequisite of
S18). The prototype world and the tech demos render through the new frame unchanged in look (clay mode: identity tone
map), the S02 goldens still pass, the benchmarks stay within 5 %, and a new tone-map golden pins the HDR path.

## Depends on

S13 (`RenderScene` frame shell, `EngineApp`, frame and GPU profiles, screenshots), S14 (`screenshot` and
`measure_performance(frames)`; `WorldMcp` without the screenshot), S02/S03 (goldens, benchmarks).

## Rules that apply

R-TEST-1 (a headless Vulkan test per render feature, a golden when it changes the image), R-TEST-2 (`RenderScene` is
performance relevant: `--benchmark flyover` and `forest` before and after, within 5 % or explained), R-MECH-1 (tone map
and exposure are parameters), R-MATH-1, §0 (other samples keep `VulkanForwardRenderer`), R-DOC-1/2, R-FILE-1,
AGENTS.md (Vulkan 1.3 core only, dynamic rendering, synchronization2, validation clean).

## Scope

### 1. Shader target (`vulkan/Shader/GLSL.cpp`)

glslang compiles with `setEnvClient(EShClientVulkan, EShTargetVulkan_1_3)` and `setEnvTarget(EShTargetSpv,
EShTargetSpv_1_6)`. Every stage file under `shader/vulkan/` compiles unchanged (checked with `glslangValidator
--target-env vulkan1.3` in review). SPIR-V 1.6 turns `discard` into
`OpDemoteToHelperInvocation` (glslang `GlslangToSpv.cpp`; `foliage.frag` discards): the device enables
`shaderDemoteToHelperInvocation` (Vulkan 1.3 core, mandatory; `Base.cpp` next to `dynamicRendering` and
`synchronization2`, checked in `Instance.cpp`'s device suitability and named in its error text), and AGENTS.md's core feature list names it. A test
checks the SPIR-V version word (0x00010600) and compiles every stage file (`.vert`, `.frag`, `.comp`; `.glsl` includes
excluded; includes resolved through saiga's shader directory as the loader does); a pipeline with a `discard` fragment
shader built from SPIR-V 1.6 passes validation.

### 2. Shared renderer code (`vulkan/Renderer.{h,cpp}`, `VulkanForwardRenderer.{h,cpp}`)

`RenderScene` owns an internal `VulkanRenderer` subclass defined in `RenderScene.cpp` (instead of a
`VulkanForwardRenderer` and the `VulkanForwardRenderingInterface` base) and uses `VulkanRenderer::ScreenshotCallback`
(the forward renderer keeps an alias for the samples). `VulkanRenderer` today holds only the swap chain or offscreen
images, frame slots and timings. Moved from `VulkanForwardRenderer` into `VulkanRenderer` as
protected members, so both renderers share them: the per-frame command buffers and submit2, the ImGui frame begin/end,
`recordScreenshot` / `readScreenshot`, the screenshot request list and the `ScreenshotCallback` type,
`selectSampleCount`. `VulkanParameters::offscreenFormat` (headless color format: `R8G8B8A8Unorm` default, or
`R8G8B8A8Srgb`; today a constant in `Renderer.h`) lets tests run the SRGB output path. The GPU section names
become a constructor argument (today `TRANSFER`/`MAIN`/`IMGUI` are registered in the `VulkanRenderer` constructor
before `timings.create`); the forward renderer keeps its three.

### 3. The frame (`vulkan/scene/RenderScene.{h,cpp}`, `RenderSceneTypes.h`, `shader/vulkan/scene/`)

Per frame:

1. transfer callback (uploads), outside any pass;
2. scene pass (dynamic rendering): HDR color R16G16B16A16 float with `msaaSamples` + reversed-Z depth D32 (same
   samples), cleared to `clearColor` and depth 0; the render callback draws here;
3. resolve:
   - Identity (clay): the hardware resolve of the scene pass (`resolveMode eAverage` into the single-sampled HDR image,
     the MSAA color `storeOp DontCare`, as the forward renderer does today: no extra pass, no stored MSAA image);
   - other tone maps: the MSAA image is stored and a fullscreen pass reads every sample (`texelFetch` on a
     multisampled `sampler2DMS`), sanitizes it per channel (`isnan(c) ? 0 : clamp(c, 0, 65504)`), and computes
     `Σ wᵢ cᵢ / Σ wᵢ` with `wᵢ = 1 / (1 + luma(cᵢ · exposureScale))` (Karis; `exposureScale` = post's `exposure /
     preExposure` from the same latched values; the output stays pre-exposed, post applies the scale once), so bright
     edges do not alias after tone mapping;
   - 1 sample: the scene pass renders single-sampled, no resolve;
4. post and ImGui in one pass on the swap chain / offscreen image (formats `{colorFormat, no depth, 1 sample}`; ImGui
   is initialized with these formats): a fullscreen post draw from the resolved HDR image (`exposure` scaling, the tone
   map operator, the output transfer; post sanitizes again, which covers 1 sample), then ImGui. ACES is the Narkowicz
   fit; the output curve is the piecewise sRGB OETF (not `pow(1/2.2)`); the new golden pins both.

Output transfer: the scene shaders write linear values when the tone map is not Identity
(`RenderScene::sceneWritesLinear()`); post sRGB-encodes for UNORM targets and writes linear for SRGB targets (the
hardware encodes). Identity is the clay mode of §10.5: the scene shaders write display values (their own sRGB encoding,
as today), post copies them clamped to [0, 1] (no transfer on a UNORM target; on an SRGB target post decodes to linear
first so the hardware encode restores the display value).

Parameters (`RenderSceneParameters`, R-MECH-1; the first quality parameters, §17.9):

```cpp
enum class ToneMap : uint8_t { Identity, Aces };
struct RenderSceneParameters
{
    int msaaSamples = 4;                // clamped to the supported count (below)
    vec4 clearColor = ...;              // a display value; converted to linear when the tone map is not Identity
    bool imgui = true;
    ToneMap toneMap = ToneMap::Identity;
    float exposure = 1;                 // fixed until S37 (auto exposure)
};
void RenderScene::setToneMap(ToneMap); void RenderScene::setExposure(float);   // at run time
bool RenderScene::sceneWritesLinear() const;   // toneMap != Identity; the game's shaders read it per frame
float RenderScene::preExposure() const;        // the exposure the scene pass of this frame writes with
```

- **Run-time switches**: `renderFrame` latches `toneMap`, `exposure` and `preExposure` once at its start, before any
  callback (the GUI combo runs before `transfer`); the getters return the latched values; setters apply from the next
  frame.
- **Pre-exposure** (§10.5), a contract: whenever `sceneWritesLinear()` is true every scene shader multiplies its output
  by `preExposure()`, and the clear value is `linear(clearColor) * preExposure()` (latched values); the shaders' linear
  branch clamps only negatives (the clamp to [0, 1] stays in the display branch, so the sky's sun glow above 1 reaches
  the tone map) (the previous frame's effective exposure; 1 after Identity); post scales by `exposure /
  preExposure`, recorded in the same command buffer. The prototype (`finalColor` in `shading.glsl`) and the demos
  (`mesh.frag`) get one uniform and apply it in their linear branch. In Identity the exposure is ignored (forced to 1),
  so clay mode never changes.
- **Sample count**: `selectSampleCount` intersects the framebuffer color and depth counts with
  `sampledImageColorSampleCounts` and the format's `getImageFormatProperties(R16G16B16A16Sfloat, ColorAttachment |
  Sampled).sampleCounts`, taking the largest supported count not above the request; `formats().samples` reports it.
- **Synchronization** (sync2): the HDR MSAA image, the resolved image and the depth image are shared across frames in
  flight; at frame start the previous frame's sampled reads → color attachment write (Undefined →
  ColorAttachmentOptimal); after the scene pass (and the custom resolve) → fragment shader sampled read
  (ShaderReadOnlyOptimal); the tracked texture layouts (`setImageLayout`) follow; the resolve and post descriptor sets
  are rewritten in `createBuffers` after a resize.
- `RenderingFormats formats()` describes the HDR scene pass (`formats().color` is the fp16 format);
  `RenderScene::colorFormat()` is the output (swap chain / offscreen) format; the game's pipelines use it (unchanged call sites).
- GPU sections: `transfer`, `scene`, `resolve`, `post`, `imgui` (was `transfer`, `main`, `imgui`), registered in frame
  order and written every frame, because `FrameTimings` resets and waits for all queries of a frame (the value of an
  absent pass is not meaningful). `scene` is left after `endRendering` (Identity's hardware resolve counts as `scene`),
  `resolve` wraps only the custom pass, `post` and `imgui` sit inside the final pass; the app's `gpu/...` scopes follow, and `gpu/game/<pass>` now sits under `gpu/scene` (`EngineApp`'s scope tree and the S14
  `measure_performance` test expectations change with it).
- Screenshots read the final image after ImGui (as today), so they show the tone-mapped frame.
- `--tonemap identity|aces` and `--exposure x` (engine command line; no `config.ini`, §8.3) set the parameters; the
  "Engine statistics" window gets a tone map combo and an exposure slider.

### 4. Prototype renderers and their users

As §8.3 planned for S13: `WorldRenderer::init(VulkanBase&, const RenderingFormats&, const WorldParameters&, ...)` and
`MeshRenderer::init(VulkanBase&, const RenderingFormats&)` replace the `VulkanForwardRenderer&` overloads (their
sub-renderers already take these). What replaces each use of the forward renderer:

- `getColorFormat()` / "sRGB target" for the shaders' own encoding (`shading.w`, `WorldRenderer.cpp`, `MeshRenderer.cpp`):
  `WorldRenderer::setOutputLinear(bool)` / `MeshRenderer::setOutputLinear(bool)` with the pre-exposure value, called by
  the game in `transfer` from `RenderScene::sceneWritesLinear()` / `preExposure()`; default false (encode), which
  covers the remaining forward-renderer users (tests on the UNORM offscreen image);
- `sampleCount()` (also `WorldRenderer.cpp`'s pipeline and statistics uses): `formats.samples`;
- `parameters().reversedZ` (an assert): a `WorldParameters::reversedZ` the caller sets (RenderScene and the forward
  renderer are both reversed-Z);
- `WorldStatistics.cpp`'s uses, `WorldMcpContext::renderer` (S14 leaves only the ImGui toggle there: it moves to
  `RenderScene::setImguiVisible` through a callback in the context).

`RenderScene::forwardRenderer()` is removed. `tests/test_vulkan_world.cpp` (the `VulkanTest` fixture keeps the forward
renderer) passes its renderer's base and formats.

## Out of scope

Auto exposure, bloom, the full post chain, render scale and upscaling (S37), the depth pre-pass, depth resolve and
the prototype passes moved into `RenderScene` (S19), atmosphere, PBR and lights (S36), quality tiers as a whole (S37),
moving the other Vulkan samples off `VulkanForwardRenderer` (§0: they keep it).

## Design notes

- Clay mode keeps the hardware average: a luma-weighted resolve of display values moves a black/white edge pixel from
  0.5 to 0.33 (43/255), far outside the golden gate (mean < 1/255, at most 0.5 % of pixels > 16/255) on the foliage of
  `world_forest`. The weighted resolve is for tone-mapped HDR, where it is the standard fix.
- The clay path reproduces today's pixels because every scene shader clamps to [0, 1] before writing (`shading.glsl`,
  `mesh.frag`) and no scene pipeline blends (foliage uses alpha to coverage); the remaining difference is rounding after
  instead of before the average, at most 1/255 on edge pixels. On an SRGB swap chain today's shaders write linear and
  the hardware applies the piecewise sRGB curve; the new clay path (the shaders' `pow(1/2.2)`, post's decode, the
  hardware encode) changes every pixel there by up to a few /255 in the darks, so the SRGB look now equals the UNORM
  look: an accepted difference (the swap chain prefers `B8G8R8A8_UNORM`, the goldens are UNORM).
- Cost (R-TEST-2): in clay mode the frame adds only the post pass (one fullscreen read of the single-sampled fp16 image
  and a write of the swap chain image, about 0.05-0.1 ms at 1080p on the baseline GPU) and doubles the on-chip MSAA
  color size (fp16); the ACES path additionally stores and reads the 4x fp16 image (about 130 MB of traffic at 1080p,
  0.2-0.3 ms; not in the benchmark, which runs clay mode).
- §10.1's constructor `RenderScene(VulkanBase&, VulkanWindow*, ...)` stays as today's `RenderScene(VulkanWindow&,
  VulkanParameters, RenderSceneParameters)` (the base renderer creates the device); a design change recorded in §10.1.

## Tests

`tests/test_vulkan_scene.cpp` (new; skips without a device; lavapipe in CI):
- Fixtures: `VulkanWorldGolden` becomes its own fixture (`HeadlessWindow` + `RenderScene`, with `Capture`, `Finish`,
  `deviceName` and the feature skip on `scene.base()`); `VulkanWorldRender` keeps the forward renderer and calls
  `init(base, formats, params)` (one device per test as today).
- The frame: a test render callback (pre-exposure aware) draws an HDR gradient (0..16) and a bright-on-dark MSAA edge;
  with `Aces` the result matches a new golden `hdr_tonemap` (256 x 128; §17.6 recipe, a dedicated commit); with
  `Identity` the gradient clamps and the edge is the plain average; with `Aces` the weighted resolve keeps the edge
  pixel darker than the plain average. Exposure: `setExposure(2)` with `Aces`: a mid-gray pixel of the final image
  and a background pixel (the clear value) equal `aces(2 r)` in the next frame and stay equal in the frames after (no
  flash); `preExposure()` is 1 in the
  first frame and 2 in the second; in Identity `setExposure(2)` changes no pixel. A tone map toggle between two frames:
  the frame after is consistent (no double or missing encoding). With `Aces` at 1 and 4 samples, NaN, +inf and negative samples give
  black, white and black final pixels with unaffected neighbours (in Identity with MSAA the hardware average runs
  before sanitizing, so a NaN sample blackens its pixel: accepted, the clay shaders clamp). A request of 64 samples runs at the largest count of the intersection (computed in the test from
  the same queries; `formats().samples`); 1 and 4 samples validation clean; offscreen and the screenshot path; GPU
  sections present and in order also with Identity, 1 sample and hidden ImGui; Identity on the SRGB and the UNORM
  offscreen format (`VulkanParameters::offscreenFormat`) gives the same display values within 1/255; a clamped display-value
  callback through `RenderScene` (Identity) matches the forward renderer's output for the same draw (the demos'
  `MeshRenderer` has no golden; this covers the clay path for it).
- The prototype through the new frame: `test_vulkan_world`'s golden tests switch to `RenderScene` (one render per
  golden, no second copy; `imgui = false`, 4 samples, the `drawIndirectFirstInstance` skip kept) and pass the existing
  goldens and tree counts per LOD.
- `GLSL`: the SPIR-V version word, every stage file compiles, a SPIR-V 1.6 pipeline is validation clean.
- `tests/test_engine_app.cpp`: the GPU sections and the `gpu/scene/game/...` nesting, `--tonemap` / `--exposure`
  parsing.

Locally: `game_world --benchmark flyover` and `forest` before and after on the development GPU (Release build of
AGENTS.md; `samples/games/world/main.cpp` adds `gpu_frame_ms: {gpu, transfer, scene, resolve, post, imgui}` (each
median and p95) from the app's `gpu` profile over the measured frames, matched by frame number like the world passes;
the before run has only `gpu/main`, so the before/after comparison is `cpu_ms`, the world passes and the whole-frame
`gpu`; the `WorldBenchmark.h` JSON comment is updated; the baseline is updated in `data/tests/benchmarks.json` if
accepted); the three demos and `game_world` with `--demo` screenshots (unchanged look)
and `game_world --tonemap aces` (the scene linear, no double encoding).

## Done when

- The S17 row of §22: the headless frame test and a tone-map golden pass (CI).
- S02 goldens pass through `RenderScene`; the benchmarks are within 5 % of the baseline or explained (R-TEST-2).
- `docs/engine_design.md` updated: §10.1 (`RenderScene` on `VulkanRenderer`, the kept constructor, `formats`,
  pre-exposure, tone map API, `sceneWritesLinear`), §10.5 (S17 parts [exists]; pass 7 in clay mode is the hardware
  average), §10.2 prerequisite done, §8.3 (`forwardRenderer()` removed, `--tonemap`, `--exposure`), §17.1 (`gpu/scene/game/<pass>` scope names; the S14 plan's
  `gpu/game/<pass>` wording), §17.6 (the new golden), §17.7 (pass names, `gpu_frame_ms`), §3 (rows: shader target, RenderScene, `VulkanForwardRenderer` now only
  for the other samples and `test_vulkan_world`'s fixture), §22 (S17 done); `AGENTS.md` (`vulkan/scene/` description,
  the new test and the test count (one more than after S16b), `shaderDemoteToHelperInvocation` in the Vulkan 1.3 feature list).

## Verification

The CI recipe of AGENTS.md; `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R
"test_vulkan|test_engine_app"`; the benchmark recipe of AGENTS.md (flyover, forest, each twice); screenshots of the demos
with `SAIGA_SCREENSHOT=<file> timeout 60`.

## Files

`src/saiga/vulkan/{Renderer,VulkanForwardRenderer,Base,Instance,Parameters}.{h,cpp}`, `src/saiga/vulkan/scene/{RenderScene.{h,cpp},
RenderSceneTypes.h}`, `src/saiga/vulkan/Shader/GLSL.cpp`, `shader/vulkan/scene/{fullscreen.vert,resolve.frag,post.frag}`,
`src/saiga/vulkan/world/{WorldRenderer.{h,cpp},WorldStatistics.cpp,WorldMcp.{h,cpp}}`,
`samples/games/common/MeshRenderer.{h,cpp}`, `samples/games/*/` (init calls, `samples/games/world/main.cpp` incl. `gpu_frame_ms`),
`src/saiga/vulkan/world/WorldBenchmark.h` (the JSON comment), `shader/vulkan/{world/shading.glsl,games/mesh.frag}`
(pre-exposure),
`src/saiga/engine/app/{EngineApp,EngineMain}.cpp`, `EngineAppSetup.h` (`AppOptions`, the `gpu/scene/game/<name>`
comment), `EngineMain.h` (usage text), `tests/test_vulkan_scene.cpp`, `tests/test_vulkan_world.cpp`,
`tests/test_engine_app.cpp`, `tests/CMakeLists.txt`, `data/tests/golden/hdr_tonemap.png`, `data/tests/benchmarks.json` (if
updated), `docs/engine_design.md`, `AGENTS.md`, this file.
