# S13b: Tech demos on the engine

Status: planned.

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
