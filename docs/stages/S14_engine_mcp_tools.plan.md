# S14: Engine MCP tools

Status: done.

Sections follow the stage plan template of S05.

## Goal

An agent can see and steer every engine game through generic app tools: `screenshot`, `set_camera`, `move_camera`, a
frame measurement in `measure_performance`, and `list_scenarios` / `run_scenario`, which run one of the game's
scenarios inside the live app and can load its end state into the live world. The prototype's renderer tools become
game tools of `game_world` (`game_get_state`, `game_set_settings`, `game_terrain_query`, `game_set_camera`). Every tool
has a `handleMessage` test (R-MCP-1), including the prototype's tools (the known deviation of §3 is removed).

## Depends on

S13 (`EngineApp`, `RenderScene::requestScreenshot`, app tools, frame and GPU profiles, the frame-tagged
`GameRenderHooks::passTimes` of the S13 review), S12 (`ScenarioWorld`, `runScenario`), S11 (snapshots,
`load_snapshot`, load listeners).

## Rules that apply

R-MCP-1 (registered before `McpServer::start`; engine tools generic, game tools `game_`; each description states what
the tool reads and changes; each tool has a `handleMessage` test), R-REG-1/2, R-DET-1 (a scenario run from a tool gives
the same result as `--scenarios`), §5.2 (`saiga_vulkan` never includes engine headers), R-TEST-1, R-DOC-1/2, R-FILE-1,
AGENTS.md conventions (export macros, MSVC).

## Scope

### 1. App tools (`engine/app/AppTools.{h,cpp}`, `EngineApp`)

Argument checks follow S10 (§16.2): unknown arguments, wrong types and out-of-range integers are errors, never clamped.
The deadline of a deferred or long tool is the request's `answerDeadline` (the tool timeout minus the answer margin),
exported from `WorldTools.h` (`SAIGA_ENGINE_API std::chrono::steady_clock::time_point answerDeadline(const
McpServer&)`; today a file-local function of `WorldTools.cpp`).

| Tool | Contract |
|---|---|
| `screenshot(gui?, format?, max_width?)` | Renders the next frame and returns it as an image (`jpg` default, `png`), downscaled to `max_width` when wider (default 1280, 64..8192; never upscaled), followed by a text part with the camera of that frame (position, forward, yaw/pitch), the step and the frame number. The readback callback (inside `renderFrame`) only stores the image; the `onFrameRendered` callback of the same frame builds the answer (camera, frame number) and takes the once flag. An F12 or `SAIGA_SCREENSHOT` request in the same frame shares the hidden GUI (accepted). GUI hidden unless `gui: true` (the previous ImGui visibility is restored after the frame). Error without a device (`noRender`). Error at the deadline when no frame was rendered (rendering disabled, minimized window): the deadline answer and the frame callback share a once flag; the deadline path restores the GUI flag and leaves the batch. The forward renderer's screenshot request cannot be cancelled: the next rendered frame is still read back and the abandoned callback is a no-op (accepted). Reads; changes nothing. |
| `set_camera(position?, look_at?, yaw_deg?, pitch_deg?, release?)` | Places the app camera (§2). `look_at` wins over yaw/pitch; missing values stay. Takes the camera from the game (camera hold) until `release: true` (no other argument allowed with it). Coordinates finite and within ±1e6 m (the camera is float: cm resolution there). Returns the camera. Changes the camera and, while held, the heading that relative moves of the input use (§2). The field of view stays a game setting (`game_set_settings` in `game_world`; the camera rig of S35 makes it an engine value). |
| `move_camera(forward?, right?, up?, yaw_deg?, pitch_deg?)` | Relative to the view: forward/right along the ground plane, up along +y, then turns (pitch clamped to ±89°: a clamp of the result, not an argument error). Takes the camera like `set_camera`. Returns the camera. |
| `measure_performance(steps? \| frames?)` | The S10/S13 tool (registered by `registerWorldTools`) with a new form `frames: n` (1..600) through a hook `WorldToolsOptions::measureFrames` that the app sets (below). `steps` and `frames` together: error. Without a device: error. |
| `list_scenarios(filter?)` | The game's scenario names in registration order with `local_only`. Reads; changes nothing. |
| `run_scenario(name, variants?, load?, keep?)` | Runs the scenario headless and synchronously in the tool handler (frames stop meanwhile) with the given variants (`["Plain"]` by default; `"Repeat"`, `"SaveLoad"`: the names of `variantName`) and returns per variant `passed`, `failures`, `steps`, `seconds` and, for a failed variant, `report` (truncated to 16 KB); variants not run because of the deadline are reported `skipped`. The scenario code is the same as `--scenarios`; the data store is the live world's (`sharedDataStore()`, so a data error cannot stop the live app; edited data files need `reload_data` first, and the tool-run worlds refuse `reload_data` and `World::reloadData`: `WorldSetup::sharedDataReadOnly`). With `keep: true` or `load: true` the Plain run's end state is stored as the snapshot `scenario_<name>` (overwritten): also for a run that finished with failed checks, not for one that ended with a `ScenarioAbort` or an exception (`finalSave` empty); `load` and `keep` need `"Plain"` in `variants`. With `load: true` the live world is saved as `before_scenario` before the run (the live world cannot change during the handler, so its bytes are checked against the limits up front; a failure is a tool error and nothing runs), unless the live world descends from a `run_scenario` load and `before_scenario` exists (an app flag set only by this tool's loads, cleared by any other committed load: a manual `load_snapshot scenario_<name>` counts as another load, so the next `load: true` saves that state as `before_scenario`; the description says so); after the run `scenario_<name>` is stored and loaded, paused, through the `load_snapshot` path, and the result carries its `hook_errors`, `listener_errors`, `removed` and the loaded hash. Before the run the slots are checked (one or two new names); a byte-limit failure of `scenario_<name>` after the run is a field `snapshot_error` (no load; the results are kept). Errors: unknown name or variant, no scenarios in the setup, `!world.canSave()` (inside a step or a game hook; checked before the run). At the deadline the running variant stops with the failure `aborted: stopped: tool time budget` (the runner's prefix); a Plain run that finished before still stores and loads. Scenario log lines go into the process log ring (`logs` shows them). An assertion in scenario code ends the app (as in `--scenarios`); the description says so. Changes the snapshots and, with `load`, the live world. |

- `measure_performance(frames: n)` (`WorldToolsOptions::measureFrames`: `std::function<void(int frames,
  std::chrono::steady_clock::time_point deadline, McpServer::Respond)>`, answered later):
  - Start: `EngineApp` keeps an in-frame flag; the first measured frame loop is S = `frameProfile().sampleCount()` + 1
    if the tool runs inside `frame()` (its sample is still open), else + 0. A poll listener records the world
    profiler's `sampleCount()` and `RenderScene::frames()` at the first poll where the frame sample count equals S
    (frame S's steps run after its poll), and again at the first poll where it equals S + n.
  - Frame and step parts: then the frame profile's last n samples and the world profile's samples since the recorded
    start (trailing windows at that moment; `Profiler` needs no change). More than 600 steps (the world profile's
    history) are reported as the last 600 with `steps.truncated: true` and `steps.ran`.
  - GPU part: the rendered frames of the range are (frames at the start, frames at the end]. The app logs its last
    600 GPU samples with their `RenderScene` frame numbers and the frame-matched game passes; the report feeds the
    logged samples of the range into a temporary `Profiler` with the gpu scopes, so its JSON equals `get_profile`'s,
    plus `frames: [first, last]`. The wait for frames in flight ends at the first logged sample above the range, when
    `RenderScene::frames()` reaches the last frame + 8 + frames in flight (the app's own drop bound for unmatched
    frames), after 8 frame loops without a rendered frame, at once for an empty range or a device without timestamps
    (these are normal results), or at the deadline (`stopped_early`).
  - A profile reset, a committed load or a `run_scenario` during the measurement answers with an error (as the steps
    form). The frames form needs no edit permission while paused (it runs no steps itself).
- `app_status` gains `camera` (the app camera) and `camera_held`.
- The generic MCP instructions (`EngineApp`) mention the new tools; `game_world`'s (`worldMcpInstructions()`, which loses
  its `measurePerformance` parameter) the renamed ones.
- `McpServer::addTool` stops on a duplicate tool name in every build (`SAIGA_EXIT_ERROR`; a second registration was
  silently listed twice).
- `EngineApp::restartFrameClock()`: `run_scenario` calls it after its handler, so the next frame's dt does not contain
  the blocked time (no burst of four steps and dropped steps). `runFrames` (fixed dt) is unaffected. The frame
  profile keeps the long `frame/poll` sample.
- The app's load listener also neutralizes the local player's `PlayerInput` (`move`, `heldButtons`) after any load, as
  the flight and racing demos do for their controls: device input is transient, a restored held state was never
  released by neutral device input. A scenario that ended with `move(...)` therefore does not keep the loaded hero
  running. This applies to every load, so the `load_snapshot` tool's description and the `WorldTools.h` comment say
  so, and the S13 load tests of `test_engine_app.cpp` are checked against it.

### 2. Camera hold (`EngineApp`)

```cpp
bool EngineApp::cameraHeldByTool() const;      // set by set_camera / move_camera, cleared by release
void EngineApp::releaseCamera();
```

The games decide the camera; the app only says that a tool holds it (R-MECH-1). A game that drives the camera itself
skips its camera update while `cameraHeldByTool()` and renders from `app.camera()` (view and eye; the eye is the float
camera position, widened to double where the game's renderer takes a double eye) instead of its own camera. On the
release edge (a previous-held flag in each renderer) it resets its camera (`ChaseCamera::reset()`: the chase camera
snaps behind the followed entity instead of gliding back). Per game: flight and racing skip `ChaseCamera::update`; the
platformer also skips its `cameraYaw` turn input. `game_world` needs nothing: the app camera is its render camera, so
the hold only marks that a tool placed it (its walk mode keeps adjusting the height). The free camera's keyboard and
mouse control runs only without a local player (as before); with a local player a held camera moves only through the
tools. Relative moves of the input map (keys and `set_input`) use the heading of `app.camera()`, so they follow a held
camera; this is stated in the `set_camera` description.

### 3. Scenario runs in the app (`ScenarioRunner`, `ScenarioWorld`, `WorldApi`)

```cpp
// ScenarioResult: the Plain run's final save, step index and hash; empty / 0 for other variants and for runs that
// ended with a ScenarioAbort (maxSteps, the deadline) or an exception; kept for runs with failed checks
std::vector<uint8_t> ScenarioResult::finalSave;   uint64_t ScenarioResult::finalStep, ScenarioResult::finalHash;
// ScenarioOptions: an optional wall-clock deadline; passing it aborts the run (ScenarioAbort, checked by the observer
// after each step without touching the world, its RNG or its profile); it never changes the result of a run that
// finishes (R-DET-1)
std::optional<std::chrono::steady_clock::time_point> ScenarioOptions::deadline;
// WorldSetup: the shared data store must not be reloaded by this world (World::reloadData and the reload_data operation
// fail with an error); set for the tool-run scenario worlds, which share the live world's store
bool WorldSetup::sharedDataReadOnly = false;
// WorldApi: store given save bytes as a snapshot (the limits and the revision handling of save_snapshot)
ApiResult WorldApi::putSnapshot(const std::string& name, std::vector<uint8_t> save, uint64_t step, uint64_t hash,
                                bool overwrite);
```

- The run builds its own `ScenarioWorld`s (as `--scenarios`); no scenario world, listener or API outlives the tool
  call. The live world is touched only through the snapshot path. The existing load listeners do the rest:
  `PoseHistory` (the chase camera snaps), the racing wheels and the platformer animation reset, the flight and racing
  control resets of the local player, the app's local player fallback (peer 0's player) and `PlayerInput` reset (§1),
  `changes_since` (`world_loaded`). The agent returns with `load_snapshot before_scenario` (paused unless `paused:
  false`).
- After the load the live world's hash can differ from the snapshot's: the load listeners neutralize the local
  player's input and controls (e.g. `accelerates_on_track` ends with `VehicleControls{1, 0}`); the tool reports the
  loaded hash.
- The loaded world continues with the live world's settings (its tick policy and profiling), not the scenario's (every
  object ticks every step): the state is the scenario's end state, the continuation is the game's.

### 4. Game tools of `game_world` (`vulkan/world/WorldMcp.{h,cpp}`, `samples/games/world`)

`WorldMcp` stays in `saiga_vulkan` and includes no engine header (§5.2); its context loses the renderer and the
`measurePerformance` flag and keeps `WorldRenderer*`, `Camera*`, `walkMode`, `eyeHeight`.

| Before | After |
|---|---|
| `screenshot` | removed (app tool) |
| `get_state` | `game_get_state`: world extent, render statistics, render settings, and of the camera only `ground_height`, `height_above_ground`, `walk_mode` (the rest is in `app_status`) |
| `set_settings` | `game_set_settings` (unchanged contract, incl. `fovy`) |
| `terrain_query` | `game_terrain_query` (unchanged contract) |
| `set_camera` with `height_above_ground`, `walk_mode` | `game_set_camera(position?, height_above_ground?, walk_mode?)`: `position` [x, y, z] or [x, z]; `height_above_ground` sets y to the terrain height + the value; `walk_mode` on or off. Returns the camera, `ground_height` and `height_above_ground`. No hold (the app camera is this game's render camera). |
| `move_camera` | removed (app tool) |
| `measure_performance` (frames + `gpu_world_ms`) | removed (the app's `measure_performance(frames)` reports `gpu/game/<pass>`; since S17 `gpu/scene/game/<pass>`) |

The screenshot batching moves to the app tool. Each description states what it reads and changes (R-MCP-1).

## Out of scope

Gamepad input and the camera rig (S35: the field of view becomes an engine camera value then; the camera hold stays),
rendering a scenario while it runs (a scenario is a straight C++ function; it runs to its end in one tool call), a
command-line option for the tool timeout, tools for clients (S16), UI tools (S46), `game_*` tools for the tech demos
(the world API and the app tools cover them).

## Design notes

- Rejected: keeping the scenario's `ScenarioWorld` as the live world (the scenario's own listeners capture the
  scenario lambda's locals and would run on in the live app; a second `WorldApi`; the app's, the games' and the API's
  per-world listeners would all need moving), and parking the game's world next to a shown scenario world (a world
  change listener for every game, listener IDs per world, a restore at shutdown). The snapshot path reuses what S11
  and S13b already tested; its costs are that the continuation uses the game's tick policy and that the load
  listeners neutralize input, both stated above.
- Snapshots only on request (`keep`, `load`): checking all of a game's scenarios must not fill the 16 snapshot slots.
- The camera hold is a flag, not a camera stack: one tool user at a time is the use case.

## Tests

`tests/test_engine_app.cpp`, with `handleMessage` from the test thread and the `Poller` running `app.mcp().poll()`
and `app.runFrames(1, 1.0 / 60)` on one thread (frames and polls never run concurrently; the test thread blocks in
`handleMessage`):
- `EngineAppRender` (headless lavapipe in CI; a 256 x 144 app; `render.imgui = true` where the GUI matters):
  `screenshot` returns a decodable jpg and png, 128 wide with `max_width: 128` and 256 wide by default, with the camera
  text; `gui: true` restores the previous ImGui visibility; argument errors (`max_width` 63 and 8193, unknown format);
  with rendering disabled the answer is an error at the deadline (a short tool timeout), and after re-enabling
  rendering no second answer arrives and the ImGui flag is unchanged (regression). `measure_performance(frames: n)`
  while paused returns n frame samples, zero steps and gpu samples whose `frames` lie in the range; running, the step
  profile has the steps of those frames; `steps` + `frames` is an error.
- `EngineAppNoRender`: `screenshot` and `measure_performance(frames)` are errors; `set_camera` (each argument,
  `look_at` precedence, non-finite and out-of-range errors, `release` alone and with another argument), `move_camera`
  (ground-plane motion, the pitch clamp), the hold in `app_status`, input moves relative to a held camera;
  `list_scenarios`; `run_scenario` with a test scenario list: pass and fail (the `report` of a failure), `variants`
  (names, an unknown variant, `load` without Plain), unknown name; no snapshot without `keep`/`load`;
  `keep` stores `scenario_<name>`; `load: true` with a test scenario that ends with neutral input (the live world has the scenario's end step and hash
  and is paused; a second `load` keeps the first `before_scenario`; deleting `before_scenario` makes the next `load`
  save again; `before_scenario` restores the game's hash), and one that ends with `move(...)` (the reported loaded
  hash differs from the snapshot's: the input reset); full snapshot slots are checked before the run; a throwing
  scenario with `keep` stores nothing; the deadline: a scenario with a sleeping step
  listener and a short tool timeout fails with `aborted: stopped: tool time budget`, no snapshot; `canSave()` false: no `Poller` and no frames; a thread loops `app.world().step()` and a test system calls
  `app.mcp().poll()` during the step, the only poll site, while `run_scenario` is pending, and gets the error; a
  tool-run scenario's `reload_data` operation and `World::reloadData` are refused. `measure_performance(frames)`
  started between frames and inside a frame (from the `Poller` and from a frame's poll) measures the same frame loops;
  a profile reset during it answers with an error. `restartFrameClock`: `dropped_steps` does not grow in the frame after a
  blocking `run_scenario` (measured `frame()`). The load listener neutralizes the local player's `PlayerInput`.
- `tests/test_engine_world_api.cpp`: `putSnapshot` (limits, overwrite, revision, `load_snapshot` of it).
- `tests/test_core_mcp_server.cpp`: a duplicate tool name stops (death test, every build).

`tests/test_vulkan_world.cpp` (lavapipe; links only `saiga_vulkan`): `handleMessage` tests of `game_get_state`,
`game_set_settings` (valid, clamped, invalid keys), `game_terrain_query` (known heights of the test field, limits,
errors), `game_set_camera` (height above ground, [x, z] positions, walk mode on and off, the returned ground height).

Locally: each tech demo and `game_world` through MCP: `screenshot`, `set_camera` with the chase camera held and
released (the screenshot shows the tool's view, then the chase view), `run_scenario` of one scenario per game with
`load` (flight `climbs_with_full_throttle`, platformer `jumps_onto_platform`, racing `ai_completes_a_lap`) and back
with `load_snapshot before_scenario`.

## Done when

- The S14 row of §22: every tool has a `handleMessage` test; a headless `screenshot` through `handleMessage` (CI).
- The tech demos honor the camera hold (local, screenshots in the PR).
- `docs/engine_design.md` updated: §16.2 (the tool table: the app tools [exists: S14], `measure_performance(frames)`,
  `list_scenarios`, `run_scenario` with `keep`/`load`, the prototype tools as `game_*`, "half the tool timeout"
  corrected to the answer deadline in the `step` row), §8.3 (the camera hold, `restartFrameClock`, the input reset on
  load; the sketch's "EngineApp holds `std::unique_ptr<ScenarioWorld>`" replaced by the snapshot path), §17.2
  (`run_scenario`, the deadline option, `finalSave`), §18.3 (the cookbook's `run_scenario` line: `load: true`,
  `load_snapshot before_scenario`), §19 (the WorldMcp migration row: the tools went to `engine/app/AppTools` and
  `game_world`), §3 (the WorldMcp deviation removed; rows), §22 (S14 done); `AGENTS.md` (tools in the layout); code
  comments (`WorldApi.h` at `rebind`, `AppTools.h`, `WorldTools.h`, the `EngineApp.h` class comment,
  `EngineAppSetup::scenarios`, `WorldMcp.h`'s tool list and its `VulkanForwardRenderer.h` include); `AGENTS.md`'s
  stale `sample_vulkan_world` / `saiga-world` lines (layout and Gotchas: the world prototype runs as `game_world`).

## Verification

The CI recipe of AGENTS.md; `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R
"test_engine_app|test_engine_world_api|test_vulkan_world|test_core_mcp_server"`; locally the games with samples ON as
listed in Tests, driven by a scripted JSON-RPC session (curl) against `--headless` runs under `timeout`.

## Files

`src/saiga/engine/app/{AppTools,EngineApp}.{h,cpp}`, `src/saiga/engine/tools/{WorldTools,ScenarioWorld,
ScenarioRunner}.{h,cpp}`, `src/saiga/engine/api/WorldApi.{h,cpp}`, `src/saiga/engine/world/World.{h,cpp}` (`sharedDataReadOnly`),
`src/saiga/core/util/mcp/McpServer.cpp`,
`src/saiga/vulkan/world/WorldMcp.{h,cpp}`, `samples/games/{world,flight,platformer,racing,common}/`,
`tests/test_engine_app.cpp`, `tests/test_engine_world_api.cpp`, `tests/test_vulkan_world.cpp`,
`tests/test_core_mcp_server.cpp`, `docs/engine_design.md`, `AGENTS.md`, this file.

## Design changes

- `measure_performance(frames)` returns `frame`, `steps` (with `ran`, `truncated`), `gpu` (with `frames`) and
  `requested_frames`; the app logs its last 600 GPU samples with their `RenderScene` frame and builds the gpu report from
  a temporary `Profiler` with the same scopes. The tools' helpers `answerDeadline`, `profileWindow` and `exactToolJson`
  are exported from `WorldTools.h`.
- `run_scenario` also reports `snapshot_hash` (the stored end state's hash) next to the loaded hash, so the effect of
  the input reset is visible; `load_error` when the load itself fails.
- The tech demos share `HeldCamera` (`samples/games/common/GameEngine.h`) for the release edge.
- `game_get_state` reports the camera's ground state under `camera` (ground height, height above ground, walk mode).

## Results

- Tests: `test_engine_app` 30 (12 new: frame tools without a device, set/move camera, input relative to a held camera,
  list/run scenarios, load of the end state with `before_scenario`, snapshot slots and the deadline, refusal inside a
  step, the frame clock restart, the input reset on load, screenshot, the screenshot deadline, frames while paused and
  running), `test_vulkan_world` `GameTools`, `test_engine_world_api` `PutSnapshot`, `test_core_mcp_server`
  `DuplicateToolName`. Full ctest on lavapipe passes except the known local `test_opengl_window` display check.
- A scripted session against `game_racing --headless`: the chase-camera screenshot, `set_camera` (the screenshot shows
  the tool's view), release, `run_scenario accelerates_on_track` with `load: true` (passed, 180 steps; the screenshot
  shows the chase camera behind the car at the scenario's end), `measure_performance(frames: 30)` (30 frames, 30 gpu
  samples of frames 39-68), `load_snapshot before_scenario`.
