# S13: Engine app

Status: implemented (in review).

Sections follow the stage plan template of S05.

## Goal

Engine games run in the engine's own app: `EngineApp` owns the window (a headless one for tests, agents and servers),
the Vulkan device, the `RenderScene` frame shell, the `World`, the MCP server and the input map, and runs the frame loop
of docs/engine_design.md §8.3 (poll input and MCP, advance the world, render). It measures benchmarks (§17.7) and runs
a game's scenarios (`--scenarios`, §17.2). The world prototype sample becomes `game_world` on it. Frame and GPU times
join the S10 profiler, shown in an ImGui "Engine statistics" window and readable over MCP.

The tech demos are ported in S13b (split from this stage in planning, see Design notes).

## Depends on

S12 (scenario runner), S11 (`readDataFile` for `input.json`, data directory), S10 (world tools, profiler), S02 (goldens
of the prototype renderer: they must not change), S03 (benchmark runner and baseline).

## Rules that apply

R-MECH-1 (the app takes policies from the game: render hooks, scenarios, tools, input bindings), R-DET-1 (headless
`runFrames` with a fixed frame time is deterministic; wall-clock only for frame timing), R-REG-1 (actions registered in
code), R-TEST-1 (engine features with unit tests), R-TEST-2 (`RenderScene` is performance relevant: `--benchmark
flyover` and `forest` before and after), R-MCP-1, R-FILE-1, R-DOC-1/2; AGENTS.md: the other modules and samples keep
working (§0).

## Scope

### 1. Device creation (`saiga/vulkan/VulkanDevice.h/.cpp`)

```cpp
struct VulkanDevice   // owns what VulkanRenderer's constructor creates today (VulkanBase is not movable)
{
    Instance instance; VulkanBase base; VkSurfaceKHR surface = VK_NULL_HANDLE;   // null: headless window
    ~VulkanDevice();   // waits idle, then destroys device, surface, instance
};
SAIGA_VULKAN_API std::unique_ptr<VulkanDevice> createVulkanDevice(VulkanWindow& window, const VulkanParameters& p,
                                                                  int framesInFlight);
```

- Factored out of `VulkanRenderer`'s constructor: instance, surface (none for the headless window), physical device, the
  forced features, logical device, `initMemory(framesInFlight)`. `VulkanRenderer` calls it; `window.setRenderer` stays
  in `VulkanRenderer` (the engine app does not use it).
- The device owns the surface; `VulkanSwapChain` no longer destroys it (changed for `VulkanRenderer` too). Destruction
  order in the app: `RenderScene` (swap chain, frames), the hooks' `shutdown`, the device (surface, instance), then the
  window (GLFW terminates last).
- `--no-render` (with `--headless`): no device at all; the app steps the world and serves MCP (servers, machines
  without a Vulkan driver; S33 dedicated hosts). Render hooks are not called.
- §8.3's `createVulkanDevice(VulkanWindow* /* nullptr: headless */) -> VulkanBase` becomes this signature: headless is
  a `HeadlessWindow` (it provides the headless ImGui), never a null window.

### 2. `RenderScene` frame shell (`saiga/vulkan/scene/RenderScene.h/.cpp`, `RenderSceneTypes.h`)

```cpp
struct RenderSceneParameters { int msaaSamples = 4; vk::ClearColorValue clearColor; bool vsync = true; bool imgui = true; };
class RenderScene
{
   public:
    RenderScene(VulkanDevice& device, VulkanWindow& window, const RenderSceneParameters& parameters);
    RenderingFormats formats() const;              // MSAA color + reversed-Z D32 (infinite far plane); samples incl.
    vk::Format colorFormat() const;                // the presented format (sRGB check of the prototype)
    bool renderFrame(const std::function<void(vk::CommandBuffer)>& transfer,
                     const std::function<void(vk::CommandBuffer)>& render, const std::function<void()>& gui);
                                                   // false: no frame (minimized, swap chain out of date)
    void requestScreenshot(std::function<void(const TemplatedImage<ucvec4>&)> callback);
    ivec2 size() const;
    bool gpuFrameTimes(FrameTimes& out);           // the oldest completed frame not yet reported (by frame number)
    void setImguiVisible(bool);
};
```

- Swap chain (window) or offscreen images (headless window), two frames in flight on the `frameTimeline`, one MSAA
  color + reversed-Z depth pass with resolve, ImGui on top, screenshots by copy after the frame, resize on a changed
  framebuffer size, the per-frame `finish_frame` (deferred deletes, memory update): the frame of
  `VulkanForwardRenderer::render`, without `SAIGA_SCREENSHOT` handling, `MainLoop` or the renderer class hierarchy.
- `FrameTimes { double cpuMs, cpuWorkMs; double gpuMs; std::vector<double> gpuPassMs; int frame; }` in
  `RenderSceneTypes.h`; the shell measures the whole GPU frame, `GameRenderHooks::frameTimes` adds a game's passes.
- The prototype's `WorldRenderer::init` becomes `(VulkanBase&, const RenderingFormats&, vk::Format colorFormat,
  const WorldParameters&)` (§8.3; every `renderer->` use is replaced: formats, sample count, color format, the
  reversed-Z assert). `VulkanForwardRenderer` exposes its `VulkanBase`, formats and color format, so the golden tests
  of `tests/test_vulkan_world.cpp` keep the forward renderer and stay unchanged.

### 3. `EngineApp` and `main` (`engine/app/EngineApp.h/.cpp`, `EngineAppSetup.h`, `CommandLine.h/.cpp`, `EngineMain.h/.cpp`)

The API of §8.3 (`EngineAppSetup`, `GameRenderHooks`, `run`, `runFrames`, `quit`, `frame`, `onFrameRendered`,
`setRenderingEnabled`, accessors), plus:

```cpp
struct AppOptions   // what the command line selects besides the setup
{
    std::string benchmark, screenshot, scenarioFilter, dataDirectory;
    int screenshotFrame = 60; bool runScenarios = false, noRender = false;
    std::vector<std::string> gameFlags;   // declared game flags that were given (e.g. --demo)
};
bool parseCommandLine(int argc, char** argv, const std::vector<std::string>& declaredGameFlags, EngineAppSetup& setup,
                      AppOptions& options, std::string& error);
int runGameScenarios(const EngineAppSetup& setup, const std::string& filter);   // no device, no window
int engineMain(EngineAppSetup setup, const AppOptions& options,
               const std::function<void(EngineApp&, const AppOptions&)>& start);   // scenarios, benchmark or run
void EngineApp::setLocalPlayer(Entity player);   // the entity the input writes (none: input moves the free camera)
```

- Options: `--headless`, `--benchmark <name>`, `--screenshot <file>` (after `--screenshot-frame N`, default 60, then
  quit), `--width/--height`, `--validation`, `--no-vsync`, `--scenarios [filter]`, `--no-render`, `--mcp-port N` (also
  `SAIGA_MCP_PORT`; 0: the server is not started, the tools still exist for `handleMessage`), `--data <dir>` (default:
  the source tree's `data/<EngineAppSetup::name>` when the build writes the project dir, else next to the executable);
  declared game flags (e.g. `--demo`) are passed on, anything else is an error (also `--benchmark` with `--headless`).
  The options are readable in the hooks (`EngineApp::options()`, e.g. `game_world` turns `freezeTime` on in benchmark
  mode before `renderHooks.init`). No `config.ini`. `SAIGA_SCREENSHOT=<file>` stays an alias of `--screenshot` (AGENTS.md uses it for every
  sample).
- `runGameScenarios`: the S12 runner over `EngineAppSetup::scenarios` with the game's `WorldSetup`; one line per
  scenario and variant, the failure reports, the count; exit code 0 if at least one ran and all passed, else 1 (`no
  scenarios matched`).
- The frame of §8.3: poll events and input; `mcp.poll()`; frame time (`steady_clock`, or the fixed `dt` of
  `runFrames`); camera look; input (section 4); `world.advance`; render (skipped when disabled or no frame can be
  presented; the loop then sleeps to the step rate and keeps polling MCP and advancing). Headless `run()` paces frames
  to the step rate.
- Window: `CreateVulkanWindow` with the GLFW or headless backend (PR #53); camera `Glfw_Camera<PerspectiveCamera>`
  until S35, moved by WASD only without a local player. ESC quits, F1 toggles the statistics window, F12 saves
  `screenshot.png`, G toggles ImGui. Without GLFW (`SAIGA_USE_GLFW` off) the app builds headless-only with a plain
  `PerspectiveCamera`.
- A load listener keeps the local player valid after `load_snapshot` (the handle survives a load; if it is dead, the
  first `Player` with peer 0 is used, else none).
- MCP: the server and the tools always exist (`start` only with a port): the S10 world tools, `EngineAppSetup::
  gameTools`, and the app tools of S13 (registered before `start`):
  - `quit` (moved from S14: agents must be able to stop a headless app; answers, then the loop ends after the frame);
  - `app_status` -> headless, rendering enabled, size, fps, frame and step counts, local player, benchmark, the input
    bindings (actions with ids and held bits, axes);
  - `set_input {axes?, held?, pressed?, frames?}`: a virtual input device merged into the `InputMap` for the next
    `frames` (default 1; 0: until changed), so agents drive the local player through the same path as a player.
  Tools that need app data are app tools; the world API stays world-only.

### 4. Input (`engine/app/InputMap.h/.cpp`, `engine/gameplay/ActionList.h`)

```cpp
// Actions are registered in code (R-REG-1): their IDs are what PlayerAction carries
void ActionList::add(uint32_t id, const char* name);          // WorldSetup::actions (World knows the names)
void ActionList::addHeld(int bit, const char* name);          // a held button: a bit of PlayerInput.heldButtons
struct InputState { std::array<bool, 512> keys; std::array<bool, 8> mouseButtons; Vec2 mouseDelta; double scroll; };
class InputMap
{
   public:
    static bool load(const std::string& file, const ActionList& actions, InputMap& out, std::string& error);
    void update(const InputState& state, bool keyboardCaptured /* ImGui */);
    bool held(const std::string& action) const; bool pressed(const std::string& action) const;   // edge
    double axis(const std::string& name) const;     // [-1, 1]
};
std::function<void(const InputMap&, World&, Entity localPlayer)> EngineAppSetup::applyInput;   // game controls
```

- `data/<game>/input.json` binds keys to the registered actions and held buttons and defines axes:
  `{"bindings": {"jump": ["Space"], "walk": ["LeftShift"]}, "axes": {"move_x": {"negative": ["A"], "positive": ["D"]},
  "look_x": {"mouse": "x", "scale": 0.002}}, "move": ["move_x", "move_z"]}`. Parsed by `InputMap` with S11's
  `readJsonFile` and the same error format (unknown keys, actions, axes and key names are fatal, naming file and key).
  Key names are the engine's own table (GLFW key codes, names like `Space`, `LeftShift`). An optional
  `input.user.json` in the working directory overrides bindings (no rebinding UI).
- `InputState` comes from the window: keys and buttons from saiga's `keyboard()`/`mouse` state (filled by the GLFW
  event handler), the mouse delta and scroll from the app's own GLFW listeners; the headless window gives an empty
  state. The virtual device of `set_input` is merged on top.
- With a local player the app writes, only when the input changed or is not neutral (a headless app with no
  `set_input` never writes, so components set by agents through `set_component` stay):
  `PlayerInput.move` (the `move` axes rotated by the camera yaw into world space, length clamped to 1),
  `PlayerInput.look` (the camera rotation), `PlayerInput.heldButtons` and `queueAction` for pressed actions; then
  `applyInput` (games write their own control components, e.g. flight sticks). Without a local player (e.g. a demo's
  autopilot drives) the app writes nothing.
- Agents: `describe_world` lists the actions and held buttons (the `World` knows them from `WorldSetup::actions`) and
  `queue_action` accepts `action` as a name or an id; `app_status` lists the bindings and axes.
- Action IDs are stable codes: saves store them as numbers (S11), so a game never renumbers an action (documented in
  §7.9; a name table in the save is not needed for that).

### 5. Profiling and the statistics window (`engine/core/Profiler.h/.cpp`, `engine/app/EngineStatistics.cpp`)

- The profiler gets sample domains, each with its own sample count, reset count and rings: `int domain(name)`,
  `scope(path, domain)`, `counter(name, domain)`, `endSample(domain)`, `sampleCount(domain)`, `resetCount(domain)`;
  the existing calls use the `step` domain. The app writes into `world.profiler()` (also with `WorldSetup::profiling`
  off it times frames; the flag keeps governing the step domain):
  - `frame` domain (one sample per frame loop): `frame`, `frame/poll`, `frame/advance` (contains that frame's steps),
    `frame/render`, `frame/present`; counters `fps`, `steps_per_frame`, `dropped_steps`;
  - `gpu` domain (one sample when a frame's GPU times arrive, by frame number, 2-3 frames later): `gpu/frame`,
    `gpu/<pass>`.
  Shares are computed within a domain.
- `get_profile` keeps the S10 shape for the step domain and adds `frame` and `gpu` keys (window `frames?`), so S10
  clients keep working; `measure_performance` also reports the frame and GPU samples of its window. The prototype's
  `measure_performance` is dropped (its numbers are now in the engine tool), so no `game_measure_frames` rename is
  needed; the frame-synchronous measurement mode stays S14.
- ImGui window "Engine statistics": frame and step time plots (last 600 samples), tables per phase and per system with
  avg / p50 / p95 / max and share, the GPU passes, the counters; like the prototype's "World Statistics".

### 6. Benchmark runner (`engine/app/AppBenchmark.h/.cpp`, from `vulkan/world/WorldBenchmark`)

`--benchmark <name>` in `engineMain`: the paths (`flyover`, `forest`) use `GameRenderHooks::groundHeight` and the
frame times of the shell and the hooks; same rules and result file as §17.7 (1920 x 1080, validation, vsync, ImGui, MCP
and defragmentation off), plus `step` and `recording` (`{median, p95}`). `WorldBenchmark`'s statistics move with it
(their unit tests move from `tests/test_vulkan_world.cpp` to `tests/test_engine_app.cpp`; the render and golden tests
stay).

### 7. `game_world` (`samples/games/world/`)

`sample_vulkan_world` becomes `game_world` on `EngineApp` (§19): the prototype world renderer through
`GameRenderHooks`, the prototype MCP tools (`WorldMcp`, ported from `VulkanForwardRenderer` to `EngineApp`/
`RenderScene`: screenshots and the ImGui toggle) as `gameTools` (without its `measure_performance`, see section 5; the
renames stay S14), `--headless`, `--benchmark`. `.mcp.json`, AGENTS.md and
§16.1 name `game_world`; `sample_vulkan_world` is removed. The benchmark baseline must hold (R-TEST-2).

## Out of scope

The tech demo ports (S13b), app MCP tools other than `quit` (`screenshot`, camera, `run_scenario`, frame-synchronous
`measure_performance`) and the `game_*` renames (S14), HDR frame (S17), instance rendering and extract (S18, S19),
gamepads and rebinding UI (S35).

## Design notes

- S13 is split: S13 (this plan, engine infrastructure with CI tests) and S13b (the three tech demos ported to game
  objects, samples only, verified locally). S14 depends on S13 only. The stage table (§22) gets the S13b row.
- `RenderScene` reimplements the frame of `VulkanForwardRenderer` rather than wrapping it: `EngineApp` must not depend
  on `MainLoop`/`StandaloneWindow` (§8.3), and S17 replaces the frame with the HDR frame anyway. The forward renderer
  stays for the other samples and the existing tests.
- Actions are registered in code and bound in `input.json` (keys are data, action IDs are code): game code and agents
  use the same names.
- `runGameScenarios` needs no device, so it runs where Vulkan is missing.

## Tests

- `tests/test_engine_app.cpp` (Vulkan parts skip without a driver; CI runs them on lavapipe; `mcpPort = 0` and
  `handleMessage`, no fixed port):
  - headless `EngineApp`: `runFrames(60, 1/60)` advances 60 steps and renders 60 frames; with rendering disabled it
    still advances; a `renderFrame` that returns false (injected) keeps `advance` and `mcp.poll` running; `--no-render`
    without a device; a screenshot has the size and clear color; `onFrameRendered` and `quit` from a callback; an MCP
    tool call (`world_status`, `quit`, `app_status`, `set_input`) answered while frames run; no local player; the local
    player after a `load_snapshot`;
  - `RenderScene` with the headless window: MSAA resolve and reversed-Z depth (a near quad covers a far one), ImGui
    drawn, resize; destruction order (validation clean);
  - `parseCommandLine` (every option, declared game flags, errors), `runGameScenarios` exit codes and output (passing,
    failing, non-matching filter; no device), `engineMain` dispatch;
  - input: `ActionList`, `input.json` load, keys and mouse to axes, held bits and actions, edges, keyboard capture, user
    override, errors; the app writing `PlayerInput` (camera-relative move, look) and queueing actions of the local
    player; neutral input not overwriting an agent's `set_component`; `set_input`; `applyInput`; action names in
    `describe_world` and `queue_action` by name;
  - profiler domains (independent sample and reset counts, GPU samples by frame number, `get_profile` S10 shape kept
    plus `frame` and `gpu`), the benchmark statistics (moved).
- `tests/test_vulkan_world.cpp` render and golden tests unchanged (with the new `WorldRenderer::init`);
  `tests/test_vulkan_*` unchanged after the `createVulkanDevice` factoring.
- Local: `game_world` runs with MCP and `--headless`; `--benchmark flyover` and `forest` within the noise of the
  baseline (§17.7); the swap-chain checklist with validation on: resize, minimize for 5 s while `world_status.step`
  advances, vsync off, F12, ESC.

## Done when

- The S13 row of §22 without the demos: headless `EngineApp` test (`runFrames` advances, also with rendering
  disabled); input map test (CI); `game_world` runs and `--benchmark flyover` works (local); `--scenarios` runner.
- `docs/engine_design.md` (§3, §7.9 `actions`, §8.3 (device, headless window, options, input, app tools), §10.1,
  §16.1, §16.2 (`quit`, `app_status`, `set_input`, `get_profile` keys), §17.1 (domains), §17.7, §18.2, §19, §22 with
  the S13 and S13b rows: S14 without `quit`, the S16 and S19 demo work depending on S13b), AGENTS.md and `.mcp.json`
  updated.

## Verification

- The CI recipe of AGENTS.md and the full `ctest`, also with samples ON (the Clang job builds the games).
- Local GPU: `SAIGA_SCREENSHOT=/tmp/x.png timeout 60 ./game_world`, `--headless`, the MCP tools; benchmarks in the
  release build of AGENTS.md (two runs each, compared with `data/tests/benchmarks.json`).

## Files

`src/saiga/vulkan/{VulkanDevice.h,VulkanDevice.cpp,Renderer.h,Renderer.cpp,SwapChain.h,SwapChain.cpp,
VulkanForwardRenderer.h,VulkanForwardRenderer.cpp}`, `src/saiga/vulkan/scene/{RenderScene.h,RenderScene.cpp,
RenderSceneTypes.h}`, `src/saiga/vulkan/world/{WorldRenderer.h,WorldRenderer.cpp,WorldMcp.h,WorldMcp.cpp}`,
`src/saiga/engine/app/{EngineApp.h,EngineApp.cpp,EngineAppSetup.h,CommandLine.h,CommandLine.cpp,EngineMain.h,
EngineMain.cpp,InputMap.h,InputMap.cpp,EngineStatistics.cpp,AppBenchmark.h,AppBenchmark.cpp}`,
`src/saiga/engine/gameplay/ActionList.h`, `src/saiga/engine/app/AppTools.{h,cpp}`, `src/saiga/engine/core/Profiler.{h,cpp}`, `src/saiga/engine/world/World.{h,cpp}`
(`actions`), `src/saiga/engine/api/WorldApi.{h,cpp}`, `src/saiga/engine/tools/WorldTools.cpp`, `samples/games/world/`,
`samples/vulkan/sample_vulkan_world.cpp` (removed), `data/world/input.json`, `tests/test_engine_app.cpp`,
`tests/test_vulkan_world.cpp`, `tests/CMakeLists.txt`, `.mcp.json`, `docs/engine_design.md`, `AGENTS.md`, this file.

## Design changes

- `RenderScene` wraps the forward renderer's frame (`VulkanForwardRenderer` driven by `renderFrame`, no `MainLoop`)
  instead of reimplementing it: S17 replaces the frame with the HDR frame, so a copy of 600 lines of frame code would
  be thrown away; the prototype renderers (`WorldRenderer`, the demos' `MeshRenderer`) keep their
  `init(VulkanForwardRenderer&)` and the goldens are untouched.
- `createVulkanDevice(VulkanWindow&, VulkanParameters, framesInFlight) -> std::unique_ptr<VulkanDevice>`: the device owns
  the surface (the swap chain no longer destroys it); `VulkanRenderer` creates its device with it.
- The frame and GPU numbers are two separate profilers of the app (`frame`, `gpu`), not sample domains of the world's
  profiler: each has its own sample and reset counts without changing `Profiler`; `get_profile` keeps the S10 shape and
  adds the keys, `measure_performance` reports them for its window. `Profiler::scopeHistory` for the plots.
- The benchmark paths and statistics stay in `vulkan/world/WorldBenchmark` (they measure `WorldRenderer`'s passes and
  move with it in S19); `--benchmark` is parsed by the engine's command line and run by `game_world`, whose result adds
  the engine's `step` and `recording`. The prototype's `measure_performance` is not registered in `game_world`.
- `GameRenderHooks::passTimes` (game pass names and times) replaces `frameTimes`/`passName`; `groundHeight` is not
  needed yet (the paths use the prototype's height field).
- `WindowBase::markRunning` (the app runs its own loop); `VulkanRenderer::submittedFrames`/`takeCompletedFrame` and
  `FrameTimings::popCompleted` (GPU times by timeline value).
- Not created (folded into existing files): `CommandLine.*` (in `EngineMain.*`), `EngineStatistics.cpp` (in
  `EngineApp.cpp`), `AppBenchmark.*` (the runner stays in `WorldBenchmark`, see above), `data/world/input.json`
  (`game_world` uses the default bindings). The default data directory without the project dir in the build is
  `data/<name>` in the working directory, not next to the executable.
- Review round 1: the release of the device input (the first neutral frame after an active one) is written to the
  local player once (a held button stayed down); `set_input` rejects unknown axes; without `input.json` the input map
  has default axes (`move_x` A/D, `move_z` S/W, mouse look) and `app_status` lists the actions, held buttons, axes and
  the benchmark; the loop sleeps to the step rate when no frame was presented (a minimized window spun); the camera
  moves with the frame's time and ignores the mouse over ImGui; `EngineAppSetup::cameraSpeed/cameraSpeedFast` (game_world
  keeps the prototype's 25/250 and sliders) and `EngineAppSetup::benchmarks` (`--benchmark` names are checked before
  the window opens); `--screenshot` with `--no-render` is an error (`SAIGA_SCREENSHOT` is ignored with a warning);
  `reset_profile` resets the frame and GPU profiles; `engineMain` reports exceptions with a hint and exits with 1;
  `game_world --benchmark` exits with 1 if the window closed before the end; action and held-button names share one
  namespace; the user input file is `input.<game>.user.json`.
- Review round 2: a screenshot that cannot be saved exits with 1; a throwing `renderHooks.init` runs `shutdown` before
  the exception leaves the constructor (the game's GPU state outlived the device); `set_input` lasts world steps
  (`steps`, was `frames`: a paused world or a frame shorter than a step released the input before a step read it);
  game GPU passes are declared in `GameRenderHooks::passNames` (scopes before the first sample) and frames without GPU
  times add no samples; `EngineAppSetup::mcpInstructions` (game_world passes the world renderer's instructions); a
  `--no-render` app does not need the shader directory; the benchmark's `step` statistic covers the steps of the
  measured frames; the usage text on a command line error, `catchSegFaults`, the benchmark's no-GPU-times warning.
- Review round 3: the benchmark's `recording` is the command recording and submit (`frame/render/recording`,
  `VulkanRenderer::lastRecordingMs`; `frame/render` also waits for the frame slot and the image and repeated
  `cpu_ms`), over the measured frames; `gpu/game` is the sum of the game's passes (it showed 0); an option is not
  taken as the value of the previous one (`--data --headless`); the hint about `--headless` is printed only when the
  app could not start; a `--no-render` app gets generic MCP instructions (the game's describe renderer tools); the
  test sets environment variables portably (MSVC has no `setenv`).
- CodeRabbit (PR #56): the benchmark's `step` and `recording` cover exactly the measured frames (taken at the frame
  boundaries that close them, no cap); `set_input` counts every world step since it was set (also steps between
  frames) and is released before the next step could read it; frame callbacks keep their state (called in place);
  changing the local player releases the previous player's active input; the input map example defines `move_z`;
  GPU durations are masked raw-tick differences (`timestampDifferenceMs`, counters with fewer than 64 valid bits).
  Follow-up: an expired `set_input` is released by a step listener right after its last step (also inside one
  `world().step(n)` or a multi-step `advance`); the benchmark's `step` statistic collects the step times of the
  measured frames frame by frame (the profile keeps only the last 600 steps).

Updated: §3, §7.9 (`actions`), §8.3, §16.1, §16.2, §17.1, §17.7, §18.2, §18.4, §19, R-TEST-1, §22 (S13, the new S13b row,
S14 without `quit`, S16 and S19 depend on S13b); AGENTS.md.

## Results

- `test_engine_app` (14 tests: command line, scenario exit codes, input map, the app without a device, the app on
  lavapipe with frames, steps, screenshots, profiles and tools); the Vulkan tests and goldens are unchanged; 63 ctest
  tests.
- `game_world` on the RTX 5080 Laptop GPU: renders like the prototype; `--benchmark flyover` (RelWithDebInfo with
  asserts): trees 1.49 ms (baseline 1.48 ms).
- Release benchmark (§17.7, no asserts, no `-march=native`), alternating with the prototype `sample_vulkan_world`
  (stage-s12) on the same machine, two runs each: GPU pass medians equal within 1-2% (flyover trees 1.45-1.46 ms vs
  1.46-1.49 ms; forest trees 0.47 ms, grass 0.63 ms in both); the frame interval `cpu_ms` median flyover 2.35/2.46 ms
  (prototype 3.22/2.34 ms), forest 2.14/2.12 ms (prototype 2.16/2.43 ms), `cpu_work_ms` 0.042-0.047 ms in both. The
  CPU numbers of both vary more between runs on this machine today than the committed baseline (2.36/1.97 ms) shows;
  the app is not slower than the prototype, the baseline is unchanged.
