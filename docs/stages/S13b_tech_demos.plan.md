# S13b: Tech demos on the engine

Status: done.

Sections follow the stage plan template of S05.

## Goal

The three tech demos (flight, platformer, racing; docs/engine_design.md §18.4) run on `EngineApp` with their simulation
as game objects, components and systems in a `World`, driven by the input map, with game scenarios that pass with
`--scenarios` in all three variants. They look and play as before. Split from S13 in planning.

## Depends on

S13 (`EngineApp`, `RenderScene`, input map, `engineMain`), S12 (scenarios), S11 (data files).

## Rules that apply

R-STATE-1 (all demo state in components; members only constant configuration and `cache`), R-OO-1, R-SYS-1, R-DET-1,
R-REG-1, R-TEST-1 (game scenarios through `--scenarios`), R-FILE-1, R-DOC-1/2.

## Scope

### 1. World context (`engine/world/World.h`, a §7.9 change)

```cpp
template <class T> void WorldSetup::setContext(std::shared_ptr<const T> context);   // stores the type too
template <class T> const T& World::context() const;   // asserts on a missing context or another type
```

Constant configuration that is not a data file (the platformer level, the race track, the ground height field) is
built once per process before the worlds and read in `configure()` and in systems (R-STATE-1 (a)). `--scenarios`
builds it once for all scenario worlds. Replaced for the ground by `world.terrainHeight` in S20.

### 2. The demos

Each demo becomes a `WorldSetup` (components, events, systems, classes, actions, data types, context) plus
`GameRenderHooks`, an `applyInput` and scenarios; `main.cpp` calls `engineMain`. `GameSampleHost` and `GameInput` are
removed.

| Demo | Components (state) | Systems (phase 4, host) | Game objects (decisions) |
|---|---|---|---|
| flight | `AircraftControls` (sticks, throttle lever), `AircraftState` (velocity, body rates, on ground, crashed, crash timer) | `flightModelSystem` (lift, drag, stall, ground contact) | `Aircraft` (respawn after a crash), `Autopilot` (demo: writes `AircraftControls`) |
| platformer | `CharacterMotion` (velocity, yaw, grounded, air time, jump buffer, landing speed) | `characterControllerSystem` (coyote time, jump buffer, variable jump height from the held jump bit, sub-stepped box collision) | `PlatformerHero`, `DemoRunner` (demo: writes `PlayerInput`), `Goal` (run time) |
| racing | `VehicleControls` (throttle, steer), `VehicleState` (velocity, angular velocity, per wheel compression, contact, spin; steer angle), `LapState` | `vehicleSystem` (springs, friction circle) | `Car`, `RaceDriver` (demo AI: writes `VehicleControls`), `LapTimer` |

- Position and orientation live in `Transform`; the player's vehicle or character is the `Player` entity (it ticks every
  step; scenarios spawn it with `spawnPlayer<T>`). Tunable parameters (aircraft, character, vehicle) become data files
  `data/<game>/*.json`; the state components hold a `DataRef` to their entry (`AircraftState.model`), so the systems
  read them per step and `reload_data` tunes them live over MCP.
- Input: `data/<game>/input.json` with the current key layouts; `applyInput` writes the game's control components from
  the axes (flight: pitch, roll, yaw, throttle lever; racing: throttle, steer). `--demo` sets no local player (the
  autopilot, demo runner or AI driver writes the controls) and steps the world a few seconds before the first frame, as
  today.
- Rendering stays `MeshRenderer` and `WorldLighting` (until S19; ported from `VulkanForwardRenderer&` to
  `VulkanBase&` + `RenderingFormats`) in the render hooks, interpolated by `world.alpha()` between the previous
  and current pose (a render-side cache filled by a step listener, cleared by a load listener); the character animation
  is render-side (until S34); the chase camera stays (until S35).
- Ground: flight and racing use the prototype height field at the game's size; the context generates it once on the
  CPU (no device; the time is measured and reported) and is shared by the game world, all scenario worlds and the
  render hooks (`WorldRenderer::init` gets an overload taking an external `HeightField`). The racing track is placed
  on that field as today, so scenarios test the real track.
- Behavior baseline: before the port, record numbers from the current code (flight demo position and altitude at steps
  600 and 1800, the racing AI's first lap time, the platformer demo's goal time); the scenarios assert them with a
  tolerance (the integration order may change slightly).
- Game scenarios (`--scenarios`, local, all three variants): flight `takes_off_and_climbs`, `crashes_into_terrain`;
  platformer `jumps_onto_platform`, `demo_reaches_goal`; racing `accelerates_on_track`, `ai_completes_a_lap`.

## Out of scope

Co-op variants (S16), `RenderScene` instances (S19), skinned characters (S34), the camera rig (S35), Jolt vehicles and
characters (S28, S29).

## Design notes

- Physics models are systems over components; decisions (autopilot, AI, respawn, lap logic) are game objects (§9.4).
- The context is constant data, not state: it is not saved and must not change after the world exists.

## Tests

No ctest tests (the saiga tests test the engine, R-TEST-1); the `World::context` accessor gets a unit test in
`tests/test_engine_world.cpp`. Locally: each demo's `--scenarios` passes; `--demo` runs without validation errors;
screenshots before and after the port are compared (PR description).

## Done when

- The S13b row of §22: each tech demo runs on `EngineApp` with its simulation in a `World` and has scenarios that pass
  with `--scenarios` (local); the games compile in the Clang CI job.
- `docs/engine_design.md` (§7.9 `context`, §18.4 (transitional parts; "from S13" becomes S13b), R-TEST-1 (game
  scenarios from S13b), §22) and AGENTS.md updated; screenshots in the PR.

## Verification

The CI recipe of AGENTS.md with samples ON; locally per demo: `./game_<name> --scenarios`,
`SAIGA_SCREENSHOT=/tmp/<name>.png timeout 60 ./game_<name> --demo`, a short interactive run.

## Files

`src/saiga/engine/world/World.{h,cpp}` (context), `src/saiga/vulkan/world/WorldRenderer.{h,cpp}` (external height
field), `samples/games/{common,flight,platformer,racing}/`,
`data/{flight,platformer,racing}/*.json`, `tests/test_engine_world.cpp`, `docs/engine_design.md`, `AGENTS.md`, this
file.

## Design changes

- Scenario names: flight `autopilot_flies_the_circuit` (the baseline positions), `climbs_with_full_throttle` and
  `crashes_into_terrain`; the plan's `takes_off_and_climbs` became `climbs_with_full_throttle` (the demo has no runway:
  the plane starts airborne on the circuit, a take-off on the uneven terrain is not part of the demo).
- Lap timing is a system (`lapTimerSystem`, after `vehicleSystem`; the gate crossing needs the position before the
  step, `VehicleState::previousPosition`), not a `LapTimer` game object: it is bookkeeping of the physics step, not a
  decision. The platformer's goal and run time live in the hero (`RunState`, checked in `HeroBase::update`), there is
  no separate `Goal` object (the flag is drawn from the level context).
- The physics models stay plain classes (`FlightModel`, `CharacterController`, `Vehicle`, `DriverModel` (was
  `RaceDriver`), `LapState` (was `LapTimer`, now the component)); the systems run them on a copy of the components.
  The models' internal state structs are renamed `AircraftKinematics`, `VehicleKinematics`; the controller's jump
  buffer moved into `CharacterState`.
- The demo runner holds the jump bit (full jump height) and presses through `CharacterMotion::jumpRequest`; a player
  presses through the bit's edge (`jumpWasHeld`). With only the bit the runner could not hold and press at once.
- The flight throttle lever: `AircraftControls::throttleChange` (the input axis), integrated by the flight model
  system per step (deterministic; `applyInput` has no step time).
- `MeshRenderer` keeps `init(VulkanForwardRenderer&)` (from `RenderScene::forwardRenderer()`): its port to
  `VulkanBase&` + `RenderingFormats` brings nothing before S19 replaces it.
- The demo objects override `tickPolicy()` with an every-step policy: a demo has no player near the autopilot or the AI
  driver, and the distance policy of §8.4 would tick them every 0.5 s.
- Render-side caches: `PoseHistory` (`samples/games/common/GameEngine.h`) records the followed entity's `Transform`
  after every step (a step listener; a load listener clears it; a teleport snaps the camera); the racing renderer
  caches the wheels the same way, the platformer's landings come from a step listener.
- The flight context holds only the height field (the trees are not needed for the flight); the racing context places
  the trees on the CPU (`ForestRenderer::placeTrees`) to find the same tree-poor site as before.
- Review round 1: the racing track's site uses a fixed tree count (`RacingContext::kSiteTreeCount`; only the
  renderer follows `SAIGA_WORLD_TREES`, which moved the track and failed the scenarios); `accelerates_on_track` checks
  the default track's length. Flight and platformer have no tree dependence. The GUI windows follow the entity
  themselves (they run before `transfer()`; the first frame's window had no fields and `imgui.ini` kept its size) and
  resize to their content. A load reseeds the racing wheel caches and resets the platformer's animation and pending
  landing. The render-side listeners are removed in `shutdown()` (`PoseHistory::detach`). `World::context` stops the
  program in every build on a missing context or another type (a death test outside `SAIGA_ASSERTS`), and
  `setContext` also takes a `shared_ptr<T>`. `WorldRenderer::init` takes a `shared_ptr<const HeightField>` and shares
  it (it copied 67 MB; the games pass an aliasing pointer into their context), tested on lavapipe
  (`VulkanWorldRender.SharesAnExternalHeightField`). The games build with Clang.

## Results

- Behavior baseline (recorded from the pre-port code with a throwaway program on the same height field): flight
  autopilot at step 600 (6506.433, 402.839, 4752.384), at step 1800 (5858.335, 362.351, 5875.344); platformer demo goal
  7.25 s; racing AI first lap 62.067 s (3724 steps). The ported games reproduce them (the scenarios assert them within
  1 m, 0.1 s and 0.5 s; the lap takes 3724 steps again).
- `--scenarios`: flight 9, platformer 6, racing 6 runs (3 scenarios x 3 variants, 2 x 3, 2 x 3) pass.
- `--demo` with `SAIGA_SCREENSHOT` on the RTX 5080 Laptop GPU: the plane in its turn over the world prototype, the
  character on the goal platform (the window ran at 20 fps, so frame 60 is past the 7.25 s goal; headless at 60 Hz it
  shows the jump between the platforms), the car on the track behind the chase camera; `--headless` and `--no-render`
  run for all three.
- `--scenarios` also pass with `SAIGA_WORLD_TREES=20000` and in the Clang build.
- `World::context` unit tests in `tests/test_engine_world.cpp`, the shared height field in `tests/test_vulkan_world.cpp`;
  63 ctest tests (all pass except the known local
  `test_opengl_window` display check).

