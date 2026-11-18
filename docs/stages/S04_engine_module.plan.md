# S04: Engine module setup

Status: done (part 1: the engine module and the CI jobs; part 2: the tech-demo games).

Sections follow the stage plan template of S05.

## Goal

The game engine has its own module `saiga_engine` (namespace `Saiga::Engine`), built and tested in CI with GCC, Clang
and MSVC (shared libraries), so every later engine stage starts from a working module with test registration. On owner
request S04 also adds three small tech-demo games that later stages use to show their features. They are samples
(`samples/games/`, built with `SAIGA_BUILD_SAMPLES`); the saiga tests test the engine only.

## Depends on

Nothing (§22).

## Rules that apply

- §5.3 (module setup, exact list), §5.1/§5.2 (layers: the engine links `saiga_vulkan`, Jolt only from S28), §18.1
  (files, names, test registration), R-DOC-1/2, R-TEST-1.
- AGENTS.md: export macros on exported non-template symbols, portability (GCC, Clang, MSVC), tests not globbed.

## Scope

Part 1, the engine module (§5.3):

- Root `CMakeLists.txt`: `OptionsHelper(SAIGA_MODULE_ENGINE ...)` (ON), `unset(MODULE_ENGINE)`,
  `include(Dependencies_engine)` inside `if (MODULE_CORE)` after `Dependencies_vulkan` and `Dependencies_jolt`.
- `cmake/Dependencies_engine.cmake`: `MODULE_ENGINE` and `SAIGA_USE_ENGINE` if `MODULE_VULKAN`; `ENGINE_TARGETS`
  (`saiga_vulkan`), `ENGINE_INCLUDES`, `ENGINE_LIBS`.
- `src/saiga/engine/CMakeLists.txt` like the vulkan module (without VMA and PCH): `SAIGA_DLL_EXPORTS`,
  SHARED/STATIC, `saiga_add_version_resource`, install into `${LIB_EXPORT}`; `add_subdirectory(engine)` and
  `Module enabled: Engine` in `src/saiga/CMakeLists.txt`.
- `src/saiga/export.h`: `SAIGA_ENGINE_API`. `src/saiga/saiga_modules.h.in`: `#cmakedefine SAIGA_USE_ENGINE`.
  `cmake/SAIGAConfigNoInstall.cmake.in`: the `saiga_engine` entries.
- `src/saiga/engine/core/EngineVersion.h/.cpp`: `SAIGA_ENGINE_API const char* Engine::engineVersion()`, the first
  exported symbol (proves the DLL exports on MSVC).
- `tests/test_engine_smoke.cpp` in `if (MODULE_ENGINE)`.
- CI (`.github/workflows/tests.yml`): the GCC job checks `SAIGA_USE_ENGINE`; new jobs `ubuntu-clang` (same CPU-only
  setup with Clang, build and ctest) and `windows-msvc` (Visual Studio generator, shared libraries, Core + Vulkan +
  Engine and the tests; Vulkan SDK from `jakoch/install-vulkan-sdk-action`, pinned SDK version; without a Vulkan driver
  the Vulkan tests skip).

Part 2, tech-demo games (owner request; `docs/engine_design.md` §18.4):

- `samples/games/CMakeLists.txt` (added by `samples/CMakeLists.txt` when `MODULE_ENGINE` is set): `saiga_game(<name>)`
  (executable `game_<name>` from `samples/games/<name>/` and `samples/games/common/`, links `saiga_engine`, include
  directory `samples/`, headers included as `"games/<name>/<File>.h"`) for `flight`, `platformer`, `racing`.
- `samples/games/common/` (compiled into each game, transitional): `GameMath.h` (double helpers: axis-angle, yaw,
  transforms, smoothing, quaternion blending, `GroundHeight`), `FixedStep.h`, `Mesh.h/.cpp` (flat colored `MeshData`,
  builders: box, cylinder, cone, airplane, car body, wheel, traffic cone, flag), `MeshRenderer.h/.cpp` (one pipeline,
  camera relative model matrices from double, push constants of 128 bytes, the world's lighting and height fog;
  `shader/vulkan/games/mesh.{vert,frag}`, `mesh_common.glsl`), `ChaseCamera.h/.cpp` (`reset()` after a respawn),
  `GameInput.h`, `WorldLighting.h`, `GameSampleHost.h` (header only: the window, fixed-step and frame wiring shared by
  the three executables; each executable maps its own keys).
- `samples/games/flight/`: `FlightModel` (thrust, lift linear in the angle of attack with a post-stall drop, parasitic
  and induced drag, side force, body rates following the controls plus stability, gear and crash), `FlightAutopilot`
  (circuit, height and speed hold), `FlightGame` (world prototype + airplane + chase camera, interpolation, respawn
  after a crash); `main.cpp`.
- `samples/games/platformer/`: `Level` (boxes, start, goal, waypoints, `supportHeight`), `CharacterController`
  (kinematic box, sub-stepped axis moves, coyote time, jump buffer, variable jump height), `CharacterAnimation`
  (skeleton, poses, `blendPoses`, `jointTransforms`, `CharacterAnimator`), `PlatformerDemo` (waypoints, jumps at
  ledges), `PlatformerGame`; `main.cpp`.
- `samples/games/racing/`: `Vehicle` (rigid body, four spring-damper wheels sampling the ground, friction circle, rear
  drive, brakes, steering), `Track` (closed Catmull-Rom spline by arc length, checkpoints, loop generator, site search),
  `LapTimer`, `RaceDriver` (AI), `RacingGame` (track ribbon with curbs, cones, start banner, car, chase camera);
  `main.cpp`.
- No game tests: the saiga tests (`tests/`) test the engine, not the samples (owner decision, R-TEST-1).
- Docs: §18.4 (the demos, their transitional parts), §18.1 (games as samples, `samples/games/common`), §17.2, R-TEST-1,
  §20, §3, §5.3, the tech-demo work of later stages in §22 (S13, S16, S19, S20, S27, S28, S29, S34, S35, S36, S37, S38,
  S39, S42), `AGENTS.md`.

## Out of scope

- Any ECS, world or app code (S06+). Jolt linking (S28).
- Running the Vulkan tests on Windows (no driver on the hosted runners; they skip).

## Design notes

- The engine module is SHARED like the other modules; `engineVersion()` is a real exported function so the MSVC job
  catches a missing `SAIGA_ENGINE_API`.
- The games are samples: one executable each, no libraries and no export macros; the shared code of
  `samples/games/common` is compiled into each game.
- The simulations take the ground as a function (`GroundHeight`): the games bind it to the prototype's height field
  (flight: terrain or water, whichever is higher).
- The samples step the game in `MainLoop::update` (60 Hz) and render with the interpolation factor of
  `MainLoop::interpolate`; the platformer animation runs per rendered frame from the interpolated state.
- `--demo` simulates a few seconds before the first frame (and the platformer animation with it), so the frame-60
  screenshot of `SAIGA_SCREENSHOT` shows the plane in a turn, the character running over the bridge and the car racing.
- The racing track avoids the prototype's trees (they cannot be removed): the site search adds a penalty per tree near
  the track (trees counted per 16 m cell); grass is off in the racing demo (it would grow through the track).
- The Windows job writes all executables and DLLs to `build/bin/` (`CMAKE_RUNTIME_OUTPUT_DIRECTORY`), so the tests find
  the saiga DLLs; the repository has no common output directory otherwise.

## Tests

- `tests/test_engine_smoke.cpp`, `EngineSmoke.VersionAndModuleDefine`: the engine links, `engineVersion()` is not empty,
  `SAIGA_USE_ENGINE` is defined.
- `test_core_util`: `AssertDeathTest.AssertWithMessageFires` (the MSVC fix of `SAIGA_ASSERT(cond, msg)`).

## Done when

`Module enabled: Engine` is printed; the engine library and the smoke test build and run on all CI compilers (GCC,
Clang, MSVC) (CI); the games compile in the Clang job (CI); each demo runs with `--demo` without validation
errors and its screenshot shows the intended scene (local).

## Verification

- Local: the CI recipe of AGENTS.md (reconfigure `build/`), `ctest -R test_engine`, full ctest.
- Demos (samples ON, from a directory with a `config.ini` with the validation layer on):
  `SAIGA_SCREENSHOT=<dir>/<game>.png ./build/samples/games/game_<game> --demo` for flight, platformer and
  racing; look at the screenshots, check the log for validation messages.
- CI: `gh workflow run tests.yml --ref stage-s04`; all three jobs green. The Ubuntu jobs fail if any gtest test case
  is skipped (gtest XML reports; nothing may skip with lavapipe and llvmpipe); on Windows the Vulkan-based tests skip
  (no driver). The Clang job builds the samples, so the three demo executables are compiled in CI (not run).

## Files

`CMakeLists.txt`, `cmake/Dependencies_engine.cmake`, `cmake/SAIGAConfigNoInstall.cmake.in`, `src/saiga/CMakeLists.txt`,
`src/saiga/export.h`, `src/saiga/saiga_modules.h.in`, `src/saiga/engine/` (`CMakeLists.txt`, `core/EngineVersion.*`),
`tests/CMakeLists.txt`, `tests/test_engine_smoke.cpp`, `.github/workflows/tests.yml`, `AGENTS.md`,
`docs/engine_design.md`. Part 2: `samples/games/` (`CMakeLists.txt`, `common/`, `flight/`, `platformer/`, `racing/`),
`samples/CMakeLists.txt`, `shader/vulkan/games/`, `tests/test_core_util.cpp`.

## Design changes

- §18.4 is new (the tech demos, owner request); §18.1 and §5.3: games are samples (`samples/games/<name>/`, executable
  `game_<name>`, `saiga_game(<name>)`), headers included as `"games/<name>/<File>.h"`, shared game code in
  `samples/games/common/`.
- Owner decision after the reviews: the games moved from static libraries in `games/` (with game tests in `tests/`) to
  executables in `samples/games/`, and the game tests were removed; the saiga tests test the engine only (R-TEST-1).
  Engine scenarios use test-only game objects in `tests/` (S12, §17.2); games get their own scenarios, run with
  `game_<name> --scenarios` (S13, not in ctest).
- The demos are not built on the engine design yet (no `World`, own fixed-step loop, own mesh renderer, camera, physics
  and animation). §18.4 lists each transitional part with the stage that replaces it; §22 names the demo work of the
  later stages.

## Results

The results below were measured with the game tests, before the owner decision to move the games into the samples and
remove the game tests (Design changes).

- Local (GCC 15, RTX 5080 Laptop GPU): 44/44 test binaries pass, including the 4 game test binaries with 32 test
  cases (flight 7, platformer 10, racing 14, mesh renderer 1); on lavapipe the Vulkan and game tests pass (15/15).
- Demos with `--demo` and the validation layer: no validation messages. Frame 60: the airplane banking in the
  autopilot circuit over the forest; the character mid-stride on the bridge (blend weights move 0.97, run 0.92); the car
  at about 88 km/h on the track, few trees nearby.
- CI run 43767519357 (commit f0ffb060): GCC 43/43, Clang 43/43 tests pass; Windows MSVC (shared libraries): 26 tests
  pass and the 11 Vulkan tests skip (no driver on the hosted runner; ctest reported them as passed then, now as
  skipped); the three game tests pass on every job.
- After review round 2: CI run 43967612832 (commit fd19edbd) is green on all three jobs; the Ubuntu jobs now fail on any
  skipped test, so all 44 tests ran there (GCC and Clang); Windows MSVC runs the non-Vulkan tests and skips the
  Vulkan-based ones (no driver). Locally 44/44 on the GPU and 15/15 Vulkan and game tests on lavapipe.
