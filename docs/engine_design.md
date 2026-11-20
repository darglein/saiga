# Saiga Engine Design

Status: living design document; changes follow R-DOC-2. This document is the single reference for the game engine built
on saiga: rules, architecture, object model, ECS, simulation, rendering, terrain, physics, streaming, multiplayer,
persistence, tooling and verification. It is written for humans and coding agents.

## 0. How to use this document

- **Read Section 1 (Rules) before changing engine or game code.** The rules are self-contained; the other sections
  explain the design behind them.
- **Status markers.** APIs, tools and files are marked **[exists]** (implemented) or **[planned: Sxx]** (designed, to be
  implemented in stage Sxx, §22; "Sxx, extended in Syy" when two stages contribute). Code blocks are design sketches
  unless marked [exists]. A section heading marked [planned: Sxx] applies to its whole content unless an item says
  otherwise. §3 lists the current state and is updated with every stage.
- **The document and the code change together.** A change that deviates from this document updates it in the same
  change and says why. References use `§n.m`.
- **Existing saiga code is a starting point, not a constraint.** Classes such as `MainLoop`, `StandaloneWindow`,
  `VulkanForwardRenderer`, the window classes and the world prototype may be rewritten, replaced or removed when the
  engine needs it. Conditions: the other modules (core, OpenGL, vision, CUDA) and their tests keep working; existing
  samples and tests that depend on changed classes are updated in the same change (or removed if obsolete).
- All conventions of `AGENTS.md` apply (MIT header, `.clang-format`, `SAIGA_ASSERT`, `#pragma once`, includes from
  `src/`, export macros, GCC/Clang/MSVC portability, explicit test registration).

## 1. Rules

Each rule has a stable ID and the stage from which it applies. Rules are checked in review and, where noted,
automatically.

**Principle and documentation**

- **R-MECH-1** (now) The engine provides mechanisms; games decide policies. The engine does not decide game-level
  questions (which terrain kinds a game uses, what happens far from players, how fast time passes, how buildings
  behave). If engine code needs such a decision, it becomes a parameter, a hook or an optional module that the game
  configures (§2.3).
- **R-DOC-1** (now) Before using an API from this document, grep for it (`src/saiga/engine/` exists from S04). If it
  does not exist, it is planned: implement it as specified in its section, or stop and report. Never invent a
  signature that differs from this document.
- **R-DOC-2** (now) A design change updates this document in the same change.

**Math and precision**

- **R-MATH-1** (from S07) Simulation state is double. Components store `double`, `Vec2/3/4`, `Quat`, `Mat3/4`, integers,
  `bool`, enums, `Entity`, `PersistentId`, `ChunkId`, strings, never `float`, `vec*`, `quat`, `mat*`. Exception:
  components whose name starts with `Render` (render-only data) and that declare `static constexpr bool allowFloat =
  true` (both checked at compile time by `registerComponent`). Float exists only in the render path (extract, §10.3) and
  in render data relative to a chunk or region origin (§9.1). Simulation time and `dt` are `double`.
- **R-MATH-2** (from S05) Engine and game code compile with tiny-eigen (`SAIGA_WITH_TINY_EIGEN=ON`, the default and the
  CI configuration). Use the API subset of §7.10 and saiga's helpers in `core/math/EigenGLMInterface.h`; missing
  functions are added to tiny-eigen (part of this repository, tested in `tests/test_core_tiny_eigen.cpp`) and, where
  needed, as double overloads to the helpers (tested in `tests/test_core_math_functions.cpp`: run with tiny-eigen in
  CI; with Eigen built and run locally with the "Eigen backend check" of AGENTS.md; tiny-eigen accepts more than
  Eigen, so only that check proves Eigen compatibility). Every
  `Vec*`/`Quat`/`Mat*` field has an initializer (`= Vec3::Zero()`, `= Quat::Identity()`): default constructors leave
  them uninitialized.

**State**

- **R-STATE-1** (from S10) All game state lives in components. A `GameObject` subclass has no constructor parameters
  and may only have (a) members set in `configure()` and never changed afterwards and (b) one member named `cache`
  that `onLoad()` and `onReplicatedSpawn()` rebuild. Per-instance spawn parameters are initial components of `spawn`
  (§7.6). Checked automatically by the save/load variant of every scenario (§17.4): a game object whose state lives in
  members loses it on load and fails the scenario's checks.
- **R-STATE-2** (from S07) Entities refer to each other by `Entity` and check `world.alive(e)`; references that must
  survive chunk unloading or saving across sessions use `PersistentId` (§9.6). No pointers or references to components
  or game objects are kept across a structural change or across steps.
- **R-STATE-3** (from S08) `Transform` is written only through its setters (fields are private).

**Registration and structure**

- **R-REG-1** (from S07) Components, events, systems, game object classes, scenarios, data file types, network messages
  and MCP tools are registered in explicit lists, one list per kind and module (§18.2). Components, events and classes
  have IDs from an explicit enum per module (§7.2). No registration by static initializers or macros.
- **R-REG-2** (from S07) No string lookups of components, systems or classes in C++ code. Strings only at the data,
  save, network and MCP boundaries, through the registries.
- **R-FILE-1** (from S04) One class or one system per file. Exceptions: `Events.h` (all event structs of a module),
  `<area>/Systems.h` (system declarations of an area), `<Module>Ids.h` (the ID enums) and headers of small related value
  types (e.g. `Collision.h`). Each file starts with a comment: what it does and, for systems, which components it reads
  and writes.
- **R-OO-1** (from S10) At most three inheritance levels below `GameObject`; `override` on every override; leaf
  classes `final`. Capabilities shared across the hierarchy (burnable, lootable) are components.

**Systems and concurrency**

- **R-SYS-1** (from S08) A system is a free function `void <name>System(World& world, double dt)` registered with its
  phase, its order inside the phase and `RunOn` (§8.2).
- **R-SYS-2** (from S08) Structural changes during a step (spawn, destroy, add, remove), from systems and from
  `GameObject` code, are commands (§7.6). Events only for broadcasts (§7.7).
- **R-PAR-1** (from S06) Parallelism only through `parallelEach` inside a system, engine-internal per-frame jobs on the
  frame `JobPool` (`JobPool::parallelFor`: pose evaluation, extract; they write no components), Jolt, the streaming pool
  (generation, meshing, IO), the network thread and the audio thread (S47). No OpenMP, no raw threads in engine or game
  code. A `parallelEach` body writes only components of its own entity, reads components of other entities only if the
  view does not write that component type, and makes no structural changes, commands, events or RNG calls (asserted with
  `SAIGA_ASSERTS`); no reductions (do them in a serial pass).
- **R-DET-1** (from S08) Simulation code uses `world.random()` (§8.6) and the step `dt`. Never wall-clock time, never
  `rand()`. A scenario in `ScenarioWorld` gives the same result on every run (§8.6); the live game is not required to be
  reproducible.
- **R-CONTENT-1** (from S23) Content generators (world plan, terrain heights, placement, prefab stamping, collider
  shapes) compute in integer and fixed-point arithmetic. The content hash is computed over this integer output. The
  conversion to float/double uses only exact operations and integer lookup tables (e.g. sin/cos tables indexed by
  fixed-point angles), never libm functions, so every machine and compiler produces identical chunks (§12.1). This
  includes re-snapping tier-1 instances after terrain edits (the terrain surface rule of §12.2 on the edited tile), so
  host and clients agree on their colliders. Generators and stamping read only their code, the world plan and versioned
  prefabs (§15.3), never tunable data files (§14).

**Networking**

- **R-NET-1** (from S16) Every gameplay feature also passes its scenarios in the host + client variant (§17.3), unless
  it is registered as local-only.
- **R-NET-2** (from S16, convention, checked in review and by the host + client variant) Client-side code writes only
  (a) components registered `Replicate::Never`, (b) components of client-local entities, (c)
  `Replicate::OwnerAuthoritative` components of entities owned by this peer. On the host, systems do not write
  `OwnerAuthoritative` components of client-owned entities (except the engine's pose receiver in phase 5, §13.3).
  Systems that write such components iterate `world.ownedBy<...>(world.localPeer())`. Entities without `NetOwner` count
  as owned by the host (peer 0) (§13.3).

**Verification**

- **R-TEST-1** (from S12 for engine scenarios, S13 for game scenarios, now for the rest) The saiga tests (`tests/`) test
  the engine, not the games (samples). Every engine gameplay mechanism has an engine scenario with test-only game
  objects (§17.2), every game feature a game scenario (`--scenarios`, run locally); every engine feature a unit test;
  every render feature a headless Vulkan test, and a golden image if it changes the image (§17.6; golden images start
  with S02, the S01 prototype tests check pixel relations only).
- **R-TEST-2** (from S03) Changes under `engine/ecs/`, `engine/sync/`, `engine/net/`, `RenderScene`, culling, terrain,
  streaming or physics are performance relevant: run the relevant benchmarks before and after and report the numbers
  (ECS: `tests/test_engine_ecs_benchmark.cpp`; rendering, terrain, streaming: `--benchmark`, §17.7); they must stay
  within the budgets of §17.10.

**MCP, data, saves**

- **R-MCP-1** (from S14) Tools are registered before `McpServer::start`. Engine tools have generic names, game tools
  are prefixed `game_`. Each description states what the tool reads and changes. Each tool has a `handleMessage` test.
- **R-DATA-1** (from S11) Data files (JSON): missing keys keep the struct defaults, unknown keys and malformed values
  are fatal load errors in every build (`SAIGA_EXIT_ERROR` in the `World` constructor, not only with `SAIGA_ASSERTS`;
  tested with death tests) naming file and key. An unknown entry name at run time is a `DataError` (fatal unless a load
  or the world API reports it).
- **R-SAVE-1** (from S11) A change of a component's fields that existing saves contain (rename, type change, meaning
  change) increments the component's version and adds a migration (§15.2). Adding a field with a default needs no
  migration.

## 2. Goals, non-goals and principle

### 2.1 Goals

- Open-world low-poly games: tens of km, massive view distance, millions of vegetation instances, thousands of
  interactive objects, player-built bases.
- **Co-op multiplayer**: 2-8 players, one player hosts. Cheating is not a concern; clients are trusted (§13).
- Worlds that live for months, across game updates (§15).
- Desktop GPUs as the main target, scaling down to the Steam Deck (§17.10).
- High performance where the cost is: rendering, culling, streaming, physics and replication on flat arrays and the GPU.
- Readable game logic with object-oriented game objects (`class Npc : public Character`).
- Developed mostly by coding agents: features are local, explicit, compile-checked and verifiable without a human.
- Portable C++17, Vulkan 1.3 core, as the rest of saiga.

### 2.2 Non-goals

Cheat protection and server authority over player movement; competitive or large-scale multiplayer (dedicated servers,
MMO); modding; a visual editor (authored content comes as prefab data files, §14); a scripting language (§20); huge
agent simulations (RTS, city builders) as the main target.

### 2.3 Mechanism, not policy

The engine serves several games of the same genre. It offers capabilities; each game configures and combines them
(R-MECH-1). Examples:

| Capability (engine) | Game decides |
|---|---|
| Height-map terrain with runtime edits; separate spaces (caves, dungeons, interiors) entered through a loading screen (§12.4) | what can be dug, which spaces exist and how they connect |
| Placed pieces with per-piece state and support queries (§6, §12.5) | building rules, stability, materials |
| Simulation radius, per-entity `alwaysSimulate` (while the chunk is resident), catch-up hook on chunk reload (§9.6, §12.3) | what happens far from players |
| Time of day and weather as replicated state, lighting that supports any time (§10.7) | day length, weather patterns |
| Optional navigation module (§9.7) | whether NPCs path-find |
| Generation pipeline: world plan -> chunk generation -> prefab stamps (§12.1) | the generators and prefabs |
| Two material models (flat colors, smooth textured), post chain, analytic and volumetric fog, 2D or volumetric clouds, water with waves and underwater rendering (§10.5-§10.11) | the look, which effects are on per quality tier |

## 3. Current state

| Part | Where | Status |
|---|---|---|
| Forward renderer with MSAA, reversed-Z, infinite far plane | `src/saiga/vulkan/VulkanForwardRenderer.*` | [exists] |
| World prototype: CDLOD terrain (one R32 height map, `TerrainRenderer::init(..., const HeightField&, ...)`), GPU-culled trees (`ForestRenderer`, own placement, OpenMP), procedural grass, sky, fog, gray look, GPU timestamps and statistics window; the world shaders are camera relative (float camera position: positions = (large part - camera) + small local part, `WorldUniforms::viewProjRel` without the translation) | `src/saiga/vulkan/world/`, `shader/vulkan/world/` | [exists], prototype |
| MCP server (HTTP JSON-RPC, tools run in `poll()`, `handleMessage`, deferred `Respond`; binds 127.0.0.1 only and rejects non-local Origin/Host headers and non-JSON bodies) | `src/saiga/core/util/mcp/McpServer.*` | [exists] |
| World MCP tools: `screenshot`, `get_state`, `set_camera`, `move_camera`, `set_settings`, `terrain_query`, `measure_performance` | `src/saiga/vulkan/world/WorldMcp.*` | [exists], no `handleMessage` tests |
| Benchmark runner (`sample_vulkan_world --benchmark flyover\|forest`: paths, median/p95, result JSON; baselines in `data/tests/benchmarks.json`) | `src/saiga/vulkan/world/WorldBenchmark.*` | [exists], prototype (S03) |
| Sample (GLFW window, polls MCP in `update()`), `.mcp.json` (`saiga-world`, port 8765) | `samples/vulkan/sample_vulkan_world.cpp` | [exists] |
| Tests | `tests/test_vulkan_world.cpp`, `tests/test_core_mcp_server.cpp` (contains `Poller`) | [exists] |
| Golden images: `compareGolden` (lavapipe gating, `SAIGA_UPDATE_GOLDEN`), goldens `world_overview` and `world_forest`, CI upload of failed images | `tests/golden_image.h`, `data/tests/golden/`, `.github/workflows/tests.yml` | [exists] |
| Jolt (optional FetchContent, static, `SAIGA_WITH_JOLT` default OFF, not in CI, no saiga library links it) | `cmake/Dependencies_jolt.cmake` | [exists] |
| `MainLoop`: `update(float)` at `updatesPerSecond` (default 60), `interpolate(float dt, float alpha)` and `render()` once per frame, `framesPerSecond` (default 60, 0 = unlimited) | `src/saiga/core/window/MainLoop.*` | [exists] |
| `HeadlessWindow` (no surface; `renderFrames` calls `update` and `render`, not `interpolate`; no ImGui) | `src/saiga/vulkan/window/HeadlessWindow.*` | [exists] |
| Double math `Vec3`, `Mat4`, `Quat`, `AngleAxis` (tiny-eigen and Eigen) with the R-MATH-2 subset of §7.10; float helpers and the S05 double overloads in `EigenGLMInterface.h` | `src/saiga/core/math/`, `submodules/tiny-eigen/` | [exists] |
| Shader compiler (glslang, targets Vulkan 1.0 / SPIR-V 1.0) | `src/saiga/vulkan/Shader/GLSL.cpp` | [exists] |
| VMA (supports virtual blocks; submodule, private to `saiga_vulkan`) | `submodules/VulkanMemoryAllocator` | [exists] |
| Engine module `saiga_engine` (CMake module, `SAIGA_ENGINE_API`, `Engine::engineVersion()`, smoke test; CI jobs GCC, Clang, Windows MSVC with shared libraries) | `src/saiga/engine/`, `.github/workflows/tests.yml` | [exists] |
| Engine core utilities: `JobPool` (parallel loops, the caller participates), `Rng` (PCG32), FNV-1a 64 `Hash`/`Hasher`, log ring with `SAIGA_LOG` (§8.5, §7.10, §17.1) | `src/saiga/engine/core/`, `tests/test_engine_core.cpp` | [exists] |
| ECS core: `Registry` (entities with generations, FIFO slot reuse, pools with sparse pages, built-in `Transform`), deterministic views, `visit` with compile-time field checks, binary visitors, field lists, `ChunkId`/`RegionId`/`InstanceKey` (§7.1-§7.5) | `src/saiga/engine/ecs/`, `src/saiga/engine/components/`, `src/saiga/core/math/ChunkId.h`, `tests/test_engine_ecs.cpp`, `tests/test_engine_ecs_benchmark.cpp` | [exists] |
| World and step loop: `World(WorldSetup)`, phases and `SystemList`, commands with reserved entities, structural log, events, `Parent`/`LocalTransform` hierarchy (`transformSystem`), `advance`/`step`, `random()`, `parallelEach`, `Simulated` filter, `hash()` (§7.6-§7.10, §8.1-§8.3, §8.6) | `src/saiga/engine/world/`, `tests/test_engine_world.cpp` | [exists] |
| Scenarios: `ScenarioWorld` (helpers, step observation, save/load swap), `ScenarioList`, runner with the plain, repeat and save/load variants; engine scenarios with test-only game objects (`Wanderer`) (§17.2, §17.4) | `src/saiga/engine/tools/Scenario*`, `tests/test_engine_scenarios.cpp` | [exists] |
| Game objects (`GameObject`, `Behaviour`, `ClassList`, `spawn`, hooks), ticking by grid-cell rings (`tickSystem`), gameplay basics (`Player`, `Health`, `PlayerInput`, `CharacterIntent`, engine events, `queueAction`, player control, kinematic mover), world-state entities, `Persistent` (§7.8, §8.4, §9.2, §9.3, §9.6) | `src/saiga/engine/gameplay/`, `src/saiga/engine/components/`, `tests/test_engine_gameobject.cpp` | [exists] |
| World API (JSON by names, `WorldApi`), MCP world tools (`registerWorldTools`), JSON visitors, profiler (§16.2, §17.1) | `src/saiga/engine/api/`, `src/saiga/engine/tools/`, `src/saiga/engine/ecs/VisitJson.h`, `src/saiga/engine/core/Profiler.*`, `tests/test_engine_world_api.cpp` | [exists] |
| Saves: `World::save`/`tryLoad`/`load` (schema tables, migrations, removed types, validate-then-commit, game objects recreated with `configure` + `onLoad`), `diffSaves`/`World::diff`, `dumpJson`; data files (`DataTypeList`, `World::data`, `DataRef`, `reloadData`); world API snapshot and data tools (§14, §15.1, §15.2, §16.2) | `src/saiga/engine/save/`, `src/saiga/engine/data/`, `tests/test_engine_save.cpp`, `tests/test_engine_data.cpp`, `data/tests/engine/` | [exists] |
| Spatial grid: 32 m cells per space updated in phase 6 (`spatialGridSystem`), `World::queryRadius` sorted by entity index (§9.5) | `src/saiga/engine/world/SpatialGrid.*`, `SpatialGridSystem.*`, `tests/test_engine_spatial_grid.cpp` | [exists] |
| Tech demo games: flight simulator, platformer, racing (§18.4), sample executables `game_<name>` with `--demo` (not tested by the saiga tests); shared transitional code in `samples/games/common/` (mesh renderer, procedural meshes, chase camera, input helpers) | `samples/games/`, `shader/vulkan/games/` | [exists], transitional |

The rows above describe what exists today; none of it is fixed (§0). **Known deviations in the prototype** (deviation,
rule, stage that removes it):

- `#pragma omp` per-cell loops in `HeightField.cpp` and `ForestRenderer.cpp` (deterministic; the `placeTrees` sum is
  serial): R-PAR-1, accepted inside `vulkan/world/` until S23.
- Time advances with the wall clock in `WorldRenderer::transfer` unless `WorldSettings::freezeTime` holds it at 0
  (goldens, benchmarks): R-DET-1, S19.
- WorldMcp tools have no `handleMessage` tests: R-MCP-1, S14.
- Float noise in the generators: R-CONTENT-1, S23.

## 4. Terms

| Term | Meaning |
|---|---|
| step | one fixed-rate simulation update (default 60 Hz, `dt = 1/60` s, double) |
| frame | one rendered image; contains 0..4 steps |
| phase | a part of a step (§8.1); systems are registered to a phase |
| tick | a call of `GameObject::update`, at most once per step (§8.4) |
| tier | kind of world content: 1 bulk, 1b procedural detail, 1p placed pieces, 2 entities, 3 services (§6) |
| space | an independent world with its own chunk grid, terrain, physics system and spatial grid (overworld, a dungeon); `SpaceId` (`uint16_t`, also used in `ChunkId`/`RegionId`), 0 = overworld; instances defined by the world plan (§12.1); every entity is in exactly one space (`Transform::space`) |
| chunk | 256 m x 256 m square of a space, `ChunkId {uint16 space; int32 x, z}`, origin `(x * 256, 0, z * 256)` in its space; defined in `src/saiga/core/math/ChunkId.h` (namespace `Saiga`, usable by `saiga_vulkan`) |
| region | a square of 2^L x 2^L chunks for far content and terrain levels, `RegionId {uint16 space; uint8 level; int32 x, z}` (also in `ChunkId.h`, created in S07 with visitor support; `SpaceId` and `InstanceKey` are defined there too) |
| host / client / peer | the process that simulates the world / the other players' processes / either (§13) |
| local-only | a feature without network effect (debug views, UI) |

## 5. Architecture

### 5.1 Layers

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Games  (samples/games/<name>/: one executable per game)                       │
│   GameObject subclasses, game systems, generators, prefabs, data, look       │
├──────────────────────────────────────────────────────────────────────────────┤
│ saiga_engine  (src/saiga/engine/, namespace Saiga::Engine)                   │
│   core/       JobPool, Rng, logging, hashing                                 │
│   ecs/        Entity, Registry, pools, views, commands, events, visit        │
│   components/ engine components (Transform, Health, Player, ...)             │
│   world/      World, phases, GameObject layer, spatial grid, spaces, chunks, │
│               streaming, placed pieces, save/load                            │
│   terrain/    height tiles + edits (CPU side)                                │
│   sync/       extract (ECS -> RenderSink), physics sync                      │
│   physics/    Jolt integration (.cpp only, required from S28)                │
│   net/        ownership, replication, messages, transports                   │
│   anim/       skeletons, animation state, pose evaluation                    │
│   nav/        optional navigation module                                     │
│   app/        EngineApp (own main loop), benchmark runner, camera rig        │
│   tools/      engine MCP tools, scenario runner                              │
│   content/    generation pipeline, prefab stamping (integer math)            │
├────────────────────────────────────┬─────────────────────────────────────────┤
│ saiga_vulkan  (no engine types)    │ Jolt  (private to saiga_engine, S28)    │
│   RenderScene, passes, culling     │                                         │
├────────────────────────────────────┴─────────────────────────────────────────┤
│ saiga_core: math (tiny-eigen), ChunkId, McpServer, image, json11, zstd       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Dependency rules

1. `saiga_vulkan` renders arrays and handles; it never includes engine headers.
2. **No saiga header includes Jolt.** Jolt types appear only in `.cpp` files under `engine/physics/`; components store
   plain IDs (`uint32_t body`). Game code uses physics only through the `PhysicsWorld` API (§11). Reason: Jolt is a
   static library with global state; a second copy in the game binary breaks it.
3. The engine owns the mapping entity <-> render handle <-> body ID <-> network ID; only `sync/`, `physics/` and `net/`
   touch several sides, driven by the structural log (§7.6).
4. Game code goes through `World` for structural changes.

### 5.3 Module setup [exists]

Modeled on `saiga_vulkan` (`cmake/Module_CMakeLists.txt` is not used by the modules):

- Root `CMakeLists.txt`: `OptionsHelper(SAIGA_MODULE_ENGINE ...)`; `unset(MODULE_ENGINE)` next to the other unsets;
  `include(Dependencies_engine)` inside `if (MODULE_CORE)` after `Dependencies_vulkan` and `Dependencies_jolt`.
- Games are samples: `samples/CMakeLists.txt` adds `samples/games/` when `MODULE_ENGINE` is set (with
  `SAIGA_BUILD_SAMPLES` and GLFW, like the Vulkan samples). `samples/games/CMakeLists.txt` defines `saiga_game(<name>)`
  (an executable `game_<name>` from the `.cpp`/`.h` files of `samples/games/<name>/` with its subdirectories and
  `samples/games/common/`, linking `saiga_engine`; include directory `samples/`) and adds the games; S04 added the tech
  demos (§18.4). There are no game libraries, and the saiga tests (`tests/`) test the engine only (§17.2).
- `cmake/Dependencies_engine.cmake`: sets `MODULE_ENGINE` and `SAIGA_USE_ENGINE` if `MODULE_VULKAN`; `ENGINE_TARGETS`
  (`saiga_vulkan`). From S28 `saiga_engine` requires Jolt: `MODULE_ENGINE` is set only if `JOLT_FOUND`,
  `SAIGA_WITH_JOLT` defaults to ON when `SAIGA_MODULE_ENGINE` is ON, and every CI job that builds the engine builds
  Jolt. There is no physics-free engine build and no `SAIGA_ENGINE_PHYSICS` define.
- `src/saiga/engine/CMakeLists.txt` like `src/saiga/vulkan/CMakeLists.txt` (without VMA and PCH):
  `add_definitions(-DSAIGA_DLL_EXPORTS)`, SHARED/STATIC, `saiga_add_version_resource`,
  `install(TARGETS ... EXPORT ${LIB_EXPORT})`; `add_subdirectory(engine)` and the `Module enabled: Engine` line in
  `src/saiga/CMakeLists.txt`.
- `src/saiga/export.h`: `SAIGA_ENGINE_API` (`saiga_engine_EXPORTS` branch, empty in the static branch).
- `src/saiga/saiga_modules.h.in`: `#cmakedefine SAIGA_USE_ENGINE`.
- `cmake/SAIGAConfigNoInstall.cmake.in`: the `saiga_engine` entries.
- `tests/CMakeLists.txt`: engine tests (incl. physics tests from S28) in `if (MODULE_ENGINE)`.
- Jolt is not linked before S28. S28: `Dependencies_jolt.cmake` sets `DOUBLE_PRECISION ON` and its comment is updated;
  `engine/physics/*.cpp` has `#ifndef JPH_DOUBLE_PRECISION #error`; static saiga builds either add Jolt to the export
  set or require `SAIGA_BUILD_SHARED=ON` for physics, decided in S28.

## 6. Object model: tiers

| Tier | What | Count per world | Storage | Per-step CPU cost |
|---|---|---|---|---|
| 1 bulk content | terrain, trees, rocks, bushes (generated) | 10^6 - 10^8 | arrays per chunk, far content per region (§12), GPU buffers | none |
| 1b procedural detail | grass, ground clutter | unbounded | not stored; shaders generate it from the chunk's density mask | none |
| 1p placed pieces | player-built walls, floors, roofs, furniture | 10^4 - 10^6 | arrays per chunk with per-piece state (§12.5), GPU buffers, compound colliders | only edited pieces |
| 2 entities | players, NPCs, animals, items, projectiles, props, crafting stations with behavior | 10^2 - 10^5 resident | ECS (§7) | only what changes |
| 3 services | renderer, physics, streaming, network, audio, input, MCP | a few | classes owned by `World` | n/a |

- **Promotion** (tier 1/1p -> 2) [planned: S31]: `world.promote<FallingTree>(chunk, key, initial...)` writes a
  tombstone (stable key, §12.1) into the chunk's modifications and creates an entity of the given `GameObject` class
  with the same mesh (`RenderInstance`) and transform plus the initial components (e.g. `Collider`, `RigidBody`);
  signatures in §12.5. A promotion is replicated as a normal spawn message of the new entity. The
  promotion applies in the requesting step (§11). The tombstone is sent as a normal live modification to every client
  that has the chunk loaded; clients hide the key when the modification arrives (a gap of up to one snapshot before
  the entity appears is accepted).
- **Demotion** (2 -> 1p) [planned: S31]: `world.demote(entity, pieceType)` turns a resting object into a placed piece
  (game decides when).
- Rule of thumb: what is numerous and does not change by itself is never an entity.

## 7. ECS [exists: S07 (Registry, storage, views, visit), S08 (World, commands, structural log, events, hierarchy)]

Own implementation (no EnTT or flecs): ~1500 lines an agent can read completely, fitted to <= 10^5 resident entities
and <= 256 component types, DLL safe, with reflection and a deterministic iteration order built in.
S07 implements create/destroy/add/remove/get/read/has/view/mask on `Registry` (`engine/ecs/Registry.h`) with the
signatures of §7.10; `World` (S08) owns a `Registry`, forwards these calls and adds commands, reserved entities and the
`Simulated` filter.

Implemented in S07 (`src/saiga/engine/ecs/`, `src/saiga/engine/components/`) [exists]: `Entity`, `PersistentId`
(value type), `ComponentMask`, `Ids.h` (ID types, `Replicate`, `Saved`, `RunOn`), `Registry` (`registerComponent`,
`validateRegistration`, `componentInfo`, `create`, `destroy`, `alive`, `add`, `remove`, `has`, `get`, `tryGet`,
`read`, `tryRead`, `mask`, `entityMask`, `flags`/`setFlags`, `view`), `View<C...>::each`/`count`, `ComponentPool`,
`Visit.h` (field rules, `EnumNames`, `Quantize`, compile-time checks), `VisitBinary.h` (`BinaryWriter`/`BinaryReader`,
`toBytes`/`fromBytes`), `FieldList.h` (`describeFields<T>()`), `Transform` (§7.4 without the hierarchy),
`SAIGA_LOG_E` (`ecs/EntityLog.h`); `ChunkId`, `RegionId`, `InstanceKey`, `SpaceId` in `src/saiga/core/math/ChunkId.h`.
Structural changes while a view iterates assert; `SAIGA_ASSERT` messages name the entity (`entity i:g`, stale or
invalid) and the component.

### 7.1 Entities

```cpp
struct Entity { uint32_t index = 0; uint32_t generation = 0; };   // {0, 0} is invalid, slot 0 is never used
```

- A stale handle is detected by the generation (`world.alive(e)`). Free slots are reused in FIFO order.
- Per slot: generation, state (free / reserved / alive), flags (e.g. `Simulated`, §12.3; not component IDs), component
  mask `std::array<uint64_t, 4>`.
- `world.mask<A, B>()` builds a `ComponentMask`.

### 7.2 Components and IDs

```cpp
// engine/components/EngineIds.h
enum class EngineComponent : uint8_t { Transform = 0, Parent, LocalTransform, Behaviour, Health, Player, /* ... */ Count };

struct Health
{
    static constexpr const char* name = "Health";
    static constexpr ComponentId id = ComponentId(EngineComponent::Health);
    static constexpr uint32_t version = 1;   // R-SAVE-1
    double hp = 100, maxHp = 100;
    template <class V> void visit(V& v) { v("hp", hp); v("max_hp", maxHp); }
};
```

- **ID types**: `using ComponentId = uint8_t; using EventId = uint16_t; using ClassId = uint16_t;` (engine,
  `ecs/Ids.h`).
- **IDs** come from an explicit enum per module: engine components 0..63, game components 64..255 (the game enum starts
  at 64). Registration (`registry.registerComponent<Health>(Replicate::Host)`; modes §13.4; optionally `Saved::No` for
  derived components that are rebuilt after loading, e.g. `RenderPose`) asserts that the ID is unused;
  `validateRegistration()` (called by the world after the registration lists, S08) asserts that every enum value up to
  the last one registered is registered (the enums grow per stage). `Replicate` and `RunOn` are defined in S07 and
  stored, but only evaluated from S16. Templates use `T::id` directly: no type-to-ID map, no hashing on access. Events
  (`EngineEvent`, game from 64) use the same scheme; every `GameObject` subclass declares `static constexpr ClassId id =
  ClassId(<Game>Class::X); static constexpr const char* name = "...";` (`spawn<T>` and `Behaviour::classId` use
  `T::id`).
- Plain structs with noexcept move construction and assignment (checked by `registerComponent`, so removal cannot throw
  halfway); no pointers; entity references as `Entity` or `PersistentId`. Engine-internal exceptions (not in `visit`,
  rebuilt on load): `Behaviour::object` (`std::unique_ptr<GameObject>`) and `RigidBody::body` (the body is recreated
  from the components on load, §11).
- `visit` lists every field: `v(name, field)` or `v(name, field, options)`. Field types: those of R-MATH-1, plus (only
  in `Render*` components that declare `static constexpr bool allowFloat = true`) `float`, `vec*`, `quat`, `mat*`;
  integers only as the fixed-width types (`int8_t` .. `uint64_t`: `long` differs between platforms); `std::array` and
  `std::vector` of supported types (e.g. an inventory as `std::vector<ItemStack>` of a `visit`-able struct). Enums are
  visited with their names: `v("state", state, EnumNames{WandererStateNames})`; the enum ends with `Count` and the
  visitor checks `std::size(names) == size_t(E::Count)` at compile time; enums inside containers are not supported (wrap
  them in a nested struct). Network option: `Quantize{0.01}` (on double, `Vec`, `Mat` and `Quat` fields). Other types
  are compile errors (`registerComponent` instantiates `visit` with a checking visitor; `std::vector<bool>` is rejected
  too; compile-fail tests in `tests/compile_fail/`). The binary form (`VisitBinary.h`): little-endian integers, bit
  patterns of doubles, `bool` as one byte, enums as `int64`, strings and vectors with a u64 length, matrices
  column-major, quaternions w, x, y, z. The reader treats its input as untrusted (saves, packets): every check runs in
  every build, `fromBytes` returns false with a message (truncated input, an enum value out of range, a vector longer
  than the input can hold, bytes left over; vectors of zero-byte elements such as empty structs store only their count,
  capped at 2^16). `describeFields<T>()` lists name and type per field, enums with their value names.
- References to data entries: `DataRef<T>` (S11, `engine/data/DataRef.h`), a nested visit struct `{entry}` saved by
  name; `world.data(ref)` resolves it once per data store and then reads without a string lookup (§14). Component fields
  that hold a component, event or class ID are not remapped by a load (store names instead; `Behaviour::classId` is the
  one field the engine remaps, §15.2).
- At 200 registered types registration logs a warning (prefer fields on an existing component or a tag).

### 7.3 Storage

- Per type a pool: dense `std::vector<T>`, dense `std::vector<Entity>`, sparse pages (4096 entries) entity index ->
  dense index. Removal swaps with the last element; structural changes
  invalidate references (R-STATE-2).
- `Transform` has ID 0, is present on every entity, and is stored in an array indexed by the entity index.

### 7.4 Transform and hierarchy

```cpp
class Transform
{
   public:
    static constexpr const char* name = "Transform";
    static constexpr ComponentId id = 0;
    static constexpr uint32_t version = 1;      // schema version (R-SAVE-1)
    SpaceId space() const;                      // changed only by World::moveToSpace
    const Vec3& position() const; const Quat& rotation() const; const Vec3& scale() const;
    void setPosition(const Vec3&); void setRotation(const Quat&); void setScale(const Vec3&);
    void teleport(const Vec3&, const Quat&);   // no interpolation from the previous pose
    template <class V> void visit(V& v);        // space, position, rotation, scale
   private:
    SpaceId sp = 0; Vec3 p = Vec3::Zero(); Quat r = Quat::Identity(); Vec3 s = Vec3::Ones();
    uint8_t changedThisStep = 0, changedLastStep = 0, teleported = 0, teleportedLastStep = 0;
};
struct Parent { Entity entity; };                                         // engine component
struct LocalTransform { Vec3 position = Vec3::Zero(); Quat rotation = Quat::Identity(); Vec3 scale = Vec3::Ones(); };
// Parent and LocalTransform have name, id, version and visit like every component.
```

- A setter compares with the stored value and changes the flags only if the value differs (bitwise), so idle entities
  written every step cause no hierarchy, grid or extract work.
- Phase 6 scans the transforms once and builds the list of changed entity indices (sorted by index); hierarchy, grid and
  extract iterate this list (tick intervals and `Simulated` are re-evaluated round-robin, §9.2); changes made in phases
  7 and 8 enter the next step's list through `changedLastStep`.
- The flags are members of `Transform` (no back pointers; `parallelEach` writes only its own entity). At the end of
  every step `World::step` shifts them (`changedLastStep = changedThisStep`, `changedThisStep = teleported = 0`),
  independent of any `RenderSink`. Readers: hierarchy (phase 6), spatial grid, extract (phase 8); all of them process
  entities with either flag set, so changes made in phases 7 and 8 (chunk reloads, replication) are not missed. Physics
  compares a body's last written pose with `Transform` instead (§11). The flags are excluded from `visit`, the hash and
  saves; `load()` sets `changedThisStep` and `teleported` on every transform and rebuilds the depth list and the spatial
  grid (derived data, no hash change).
- `World::moveToSpace(e, space, position, rotation)` moves an entity and its descendants to another space (a teleport:
  the body moves to the other space's physics system, grid and interest are re-evaluated).
- Phase 6 updates children from parents in a depth-sorted list (rebuilt on structural changes); a changed parent marks
  its descendants changed. Destroying a parent destroys its children. `add<Parent>` rejects cycles and a dead parent
  (between steps an assert; during a flush the child is destroyed, like the child of a destroyed parent); `Parent` is
  read-only through the World (`get`, `tryGet` and non-const views; change it with remove + add). Teleports: phase 6
  teleports a child when its parent was teleported in this step; a teleport made in phase 7 reaches the children before
  phase 8 (`propagateTeleportsBeforeExtract`, also a second teleport in the step); after phase 8 the children whose
  parent was teleported after their last update are listed and teleported in the next phase 6.
  `Transform::teleportedLastStep` is shifted like `changedLastStep`, for readers of the next step (S08). Physics bodies
  only on roots; children with colliders are kinematic (§11).

### 7.5 Views

```cpp
world.ownedBy<Transform, PlayerInput>(world.localPeer()).each([](Entity e, Transform& t, PlayerInput& in) { ... });
world.view<const Health, Enemy>().each([](Entity e, const Health& h, Enemy& en) { ... });
world.view<Lifetime>().parallelEach([](Entity e, Lifetime& l) { ... }, 256);   // not owner-authoritative
const Health& h = world.read<Health>(e);   // const access
Health& w = world.get<Health>(e);          // write access
```

- A view iterates the dense array of its smallest pool (or the entity slots for a `Transform`-only view) and tests the
  masks; the order follows from the history of structural changes, so a `ScenarioWorld` run is reproducible. Reserved
  entities (§7.6) are skipped; in phases 2-5 entities without the `Simulated` flag are skipped (§12.3).
- `read`/`tryRead` return const references, `get`/`tryGet` mutable ones; the difference documents intent, nothing is
  tracked.
- `parallelEach(fn, chunk = 256)` runs on the frame `JobPool` (§8.5) with fixed chunks of the candidate list (the
  smallest pool's dense array or the slots), so its result does not depend on the thread count; while it runs,
  structural changes, commands, `events.emit` and `random()` assert. `world.view` passes the `Simulated` flag as a
  required flag in phases 2-5 (`Registry::view(requiredFlags)`); every new entity has the flag until S24.

### 7.6 Structural changes, commands, structural log [exists: S08]

- Game code calls `world.create/spawn/destroy/add/remove` (or the `GameObject` wrappers); the world decides whether they
  apply immediately or as commands. `commands()` is engine-internal.
- **Between steps** (setup code, tests, loading, scenario helpers, MCP tool handlers): changes apply immediately (not
  inside a view iteration, asserted). Events
  emitted between steps k and k+1 belong to step k and are readable in step k+1.
- **During a step** (systems and `GameObject` code): commands, applied at the end of each phase in recording order.
  - `spawn<T>(position, initial...)` reserves the entity and returns it immediately. When the command is applied:
    `Transform`, `Behaviour` and the initial components are added, then `configure()` and `onSpawn()` run. Reserved
    entities are skipped by views and `queryRadius` until then; `get` on them asserts.
  - `create()` during a step reserves and returns the entity like `spawn`; `add` on a reserved entity is recorded as a
    command; `get`/`read`/`has` on it assert until the phase flush.
  - `add<T>(args...)` on an entity that already has `T` asserts (use `get`/`write`). `add` and `remove` return `void`.
  - `destroy(e)` runs `GameObject::onDestroy()` when applied, then removes the entity and its children.
  - While a flush runs `configure`, `onSpawn` or `onDestroy`, structural changes made by them apply immediately (an
    added component is usable in the next statement).
- **Structural log** (engine-internal): every applied structural change appends `(entity, componentId,
  added/removed)` to the log (destroying an entity appends one record per removed component; changes between steps
  are appended too). The consumers `sync/`, `physics/` and `net/` each keep a read cursor and consume at fixed points
  (physics at the start of phase 5, extract and replication in phase 8) to create and free render instances, bodies
  and network IDs; records appended after a consumer ran are consumed in its next run. At the end of `World::step` the
  log is truncated up to the smallest cursor. Not an observer API; S08 tests it with a recording consumer.
- `load()` appends one add record per loaded component to the structural log; `sync/`, `physics/` and `net/` create
  render instances, bodies (from `Collider`, `RigidBody` and the saved velocities, §11) and network IDs from it.
  The replacement world of `ScenarioWorld` gets its `RecordingRenderSink` like any world (`setRenderSink`, at any time).

S08 implementation (`engine/world/World.h`, `Commands.h`, `StructuralLog.h`): `create()` during a step reserves a slot
(`Registry::reserve`; slots and `Transform`s live in stable pages, so this never moves a `Transform` a view callback
holds); `add` on a reserved entity is a command too; commands are type-erased closures (a component value is moved into
its command) applied in recording order after every phase; commands on an entity destroyed earlier in the same flush are
dropped (e.g. a second `destroy`, or one after the parent's); `add`/`remove` check the entity when recorded and the
component when applied (remove + add of the same component in one phase is valid). A system or command that throws
aborts the step: the pending commands are dropped, the reserved entities released, the step index does not advance (the
changes of the finished phases remain). Creating an entity logs a `Transform` record; destroying it logs one removed
record per component in ID order (`Transform` first). Consumers: `structuralLog().addConsumer()`, `consume(consumer,
fn)`, `unread(consumer)`, `removeConsumer(consumer)` (a consumer that stops reading must be removed, or the log grows).
A cursor moves past a record after the callback returned (a throwing consumer gets it again). Every change reserves its
record before it is made (`prepare`), so a record is never lost after a change; an aborted step leaves the step state
first and cleans up without allocating (the registry's free list always has room for releasing every reserved entity;
the events are restored by a swap: the readable events become pending again, the aborted step's own events are
dropped).

### 7.7 Events [exists: S08]

- An event is a struct with `name`, `id` (enum, §7.2) and `visit`; registered in `registerEngineEvents` /
  `WorldSetup::events`; all event structs of a module are in its `Events.h`.
- `world.events.emit(DamageEvent{...})`; `world.events.read<DamageEvent>()` returns the events of the previous step in
  emit order. Events have `name`, `id` (`EventId`), `version` and `visit` (the field rules of §7.2) and noexcept moves;
  `EventRegistry::add<E>(Replicate)` checks unique IDs and names, `validateRegistration` the contiguous ranges
  (`EngineEvent` 0..63, game events from 64). **Events emitted in step k are readable by all phases of step k+1, then
  cleared.** Pending events are part of the save and the hash.

### 7.8 World state

World-level state (time of day, weather, global game flags such as "boss killed") lives in components of special
entities, so saves, hashes, replication and MCP treat it like any other state: one **world-state entity** and one
**space-state entity** per space. The world-state entity is never unloaded and in every client's interest; a
space-state entity is resident while its space is loaded (stored with the space otherwise) and in the interest of the
clients whose player is in that space. Access:
`world.global<T>()` and `world.spaceState<T>(space)` (`T&`; const variants `readGlobal`, `readSpaceState`; a missing
component asserts: world-state components are added at world creation by their registration). Both kinds are flagged
in the slot state and excluded from the spatial grid, chunk membership, chunk unloading and radius interest; their
interest and residency follow only the rules above.
`WorldTime` (global) and `Weather` (per space) are added in S36. [exists: S10 for the mechanism (`global`,
`spaceState`, `readGlobal`, `readSpaceState`; only space 0 before S21)]

S10 implementation: the World creates the world-state entity when at least one component is registered with
`registerGlobal`, and the space-state entity of space 0 when one is registered with `registerSpaceState` (a world
without them has neither, so entity counts of plain worlds are unchanged); the slot flags `kWorldState` and
`kSpaceState` keep them out of the grid and ticking; `destroy` asserts on them (the world API returns an error). They
are visible in views and in the world API, so agents read and change them like other entities (`get_global`,
`set_global`). In every build they are not destroyed, get no `Parent` (a parent's cascade would take them along) and
keep their global and space-state components (the refusals assert in builds with asserts).

### 7.9 World setup

```cpp
struct WorldConfig                               // game policies (R-MECH-1)
{
    double stepsPerSecond = 60;                   // S08
    int maxStepsPerAdvance = 4;                   // S08
    TickPolicy ticking;                           // S10: §8.4
    std::vector<SpaceTemplate> spaceTemplates;    // S20 (one entry, space 0), S21 more: templates; space instances come from the world plan (§12.1)
    int maxPlayers = 8, maxSimulationIslands = 4; // S24 (§12.3)
    bool hardIslandLimit = false;                 // S24: true: requestArrival returns false beyond the limit
    bool clientPush = true;                       // S29: clients push host bodies with PushImpulse (§11)
    bool navigation = false;                      // in S45 (§9.7)
};

struct SpaceTemplate                              // S20
{
    std::string name;
    SpaceKind kind = SpaceKind::OpenWorld;        // OpenWorld (height map, sky) or Enclosed (authored scene)
    int baseSpacing = 2;                          // metres: 1 (257 x 257 tiles) or 2 (129 x 129 tiles); game_world: 1
    std::optional<int32_t> seaLevel = 0;          // S20: global water plane in height units, none if empty (§10.9)
    WaveSet waves;                                // S39: Gerstner waves of the sea (§10.9)
    int32_t sizeChunks = 128;                     // S20: the space covers chunks [0, sizeChunks)^2 (128 = 32 km);
                                                  // pyramid, tiles and the region quadtree are bounded by it
    double simulationRadius = 384, fullRadius = 1000;   // §12.3
    double clientPhysicsRadius = 384;             // S29: client proxy bodies (§11); default = simulationRadius
    double terrainEditMargin = 64;                // S28: metres of padding of a chunk's collision height range (§11)
    double prefetchMargin = 256;                  // S24: prefetch beyond fullRadius (§12.3)
    double unloadMargin = 128;                    // S24: hysteresis of chunk unloading beyond fullRadius (§12.3)
    double interestRadius = 512;                  // S32: replication interest (§13.4); asserts >= clientPhysicsRadius
    double interestMargin = 64;                   // S32: hysteresis of leaving the interest set (§13.4)
    int32_t (*prototypeHeight)(uint64_t seed, int64_t x, int64_t z) = nullptr;   // S20: x, z integer metres; result in
                                                  // integer height units (1/1024 m), point
                                                  // evaluable; transitional, replaced by `generator` in S23
    AmbientSettings ambient;                      // §10.7: Vec3 skyColor, groundColor; double intensity
    FogSettings fog;                              // §10.5: Vec3 color; double density, heightFalloff, baseHeight
                                                  // (both in saiga/vulkan/scene/RenderSceneTypes.h, prototype values)
    GeneratorId generator = 0;                    // §12.1 (S23)
};

struct WorldSetup
{
    uint64_t seed = 1;
    int workerThreads = 1;                        // threads of parallelEach incl. the caller (S08)
    WorldConfig config;
    NetRole role = NetRole::Host;                 // [planned: S16] Host or Client
    uint8_t localPeer = 0;                        // [planned: S16]
    Transport* transport = nullptr;               // [planned: S15] null: single player
    // The game's registration lists; the engine's lists are always registered first.
    void (*components)(Registry&) = nullptr;      // S07
    void (*events)(EventRegistry&) = nullptr;     // S08
    void (*systems)(SystemList&) = nullptr;       // S08
    void (*classes)(ClassList&) = nullptr;        // S10
    bool profiling = true;                        // S10: step, phase, system timings and counters (§17.1)
    void (*dataTypes)(DataTypeList&) = nullptr;   // S11
    std::string dataDirectory;                    // S11 e.g. SAIGA_PROJECT_SOURCE_DIR "/data/world"
    std::shared_ptr<DataStore> sharedData;        // S11: an already loaded store (worlds of one setup, scenario runs)
    void (*messages)(MessageList&) = nullptr;     // [planned: S15]
    void (*generators)(GeneratorList&) = nullptr; // [planned: S23] world plan, chunk generators, prefabs
    void (*pieceTypes)(PieceTypeList&) = nullptr; // [planned: S30]
    void (*migrations)(MigrationList&) = nullptr; // S11 §15.2
    void (*renderAssets)(World&) = nullptr;       // [planned: S19] registers meshes and materials by name into the
                                                  // World (world.registerMesh/registerMaterial); reads data files
    void (*collisionAssets)(CollisionAssetList&) = nullptr;     // [planned: S28] named collision meshes and shapes
};
explicit World::World(const WorldSetup& setup);   // registers engine then game lists, loads the data files

template <class T> void Registry::registerComponent(Replicate mode = Replicate::Never, Saved saved = Saved::Yes);  // S07
                                                  // Saved::No: not saved, rebuilt after loading (RenderPose)
template <class T> void Registry::registerGlobal(Replicate mode);       // S10: added to the world-state entity
template <class T> void Registry::registerSpaceState(Replicate mode);   // S10: added to each space-state entity
template <class T> void MigrationList::add(uint32_t fromVersion, void (*fn)(FieldMap& fields));   // [exists: S11]
void MigrationList::add(const char* typeName, uint32_t fromVersion, void (*fn)(FieldMap& fields));   // removed types
void MigrationList::addRemoved(const char* name, RemovedPolicy policy, const char* replacement = nullptr);
                                                  // DropEntity / DropComponent / Replace with `replacement`
                                                  // (components, classes, nested structs; FieldMap: the JSON fields)

template <class T> void ClassList::add();         // uses T::id and T::name; factory std::make_unique<T>()  [S10]
template <class E> void EventRegistry::add(Replicate mode = Replicate::Never);
void ScenarioList::add(const char* name, std::function<void(ScenarioWorld&)> run, bool localOnly = false);  // [S12]
template <class T> void DataTypeList::add();      // file <dataDirectory>/<snake_case(T::name)>.json  [exists: S11]
template <class M> void MessageList::add(Reliability r, void (*handler)(World&, uint8_t fromPeer, const M&));
                                                  // M: struct with name, id, visit; handlers run in phase 1  [S15]
```

Meshes and materials are registered by name into the `World` in `WorldSetup::renderAssets` (called once at world
creation): the `World` owns the mesh, material and piece type name tables, assigns the IDs and forwards every
registration to the render sink whenever one is set (also when it is set later), so headless worlds have the same name
tables and saving does not depend on a sink; saves store mesh and piece type names (§15.2). Collision geometry does not
come from the sink: named collision meshes and shapes (integer or fixed-point vertices) are registered in
`WorldSetup::collisionAssets` (`CollisionAssetList::add(const char* name, const CollisionMeshData&)`, S28), independent
of rendering, so headless hosts have the same shapes; `Collider::mesh` names them.

### 7.10 Core API

```cpp
class World
{
   public:
    Entity create();
    template <class T, class... C> Entity spawn(const Vec3& position, C&&... initial);   // space 0 [exists: S10]
    template <class T, class... C> Entity spawnIn(SpaceId space, const Vec3& position, C&&... initial);  // [S10]
                                                                    // (asserts space == 0 before S21)
    void moveToSpace(Entity e, SpaceId space, const Vec3& position, const Quat& rotation);              // [S21]
    void destroy(Entity e);
    bool alive(Entity e) const;
    template <class T, class... A> void add(Entity e, A&&... args);
    template <class T> void remove(Entity e);
    template <class T> T& get(Entity e);              template <class T> T* tryGet(Entity e);
    template <class T> const T& read(Entity e) const; template <class T> const T* tryRead(Entity e) const;
    template <class T> bool has(Entity e) const;
    template <class... C> View<C...> view();          // in phases 2-5: only `Simulated` entities (§12.3)
    template <class... C> View<C...> viewAll();       // all resident entities
    template <class... C> static ComponentMask mask();
    EventQueues events;
    Rng& random();                    // §8.6
    double time() const; uint64_t stepIndex() const;
    void advance(double frameDt);     // §8.3
    void step(int n = 1);
    void setPaused(bool); void setTimeScale(double); void setWorkerThreads(int n);
    uint64_t hash() const;            // debug and scenario repeat checks (§17.4)
    void stepNetworkOnly();           // phases 1 and 8 only (end-of-scenario sync, §17.3) [planned: S16]
    std::vector<uint8_t> save() const;                    // §15 [exists: S11]
    LoadResult tryLoad(const std::vector<uint8_t>& data); // rejected: unchanged; hook errors per entity [S11]
    void load(const std::vector<uint8_t>& data);          // tryLoad, fatal unless ok() [S11]
    int addLoadListener(std::function<void(const LoadInfo&)> fn);   // log range of the load, load count [S11]
    uint64_t loadCount() const;                           // [S11]
    std::string dumpJson(Entity e = {}) const;            // all entities if e is invalid [exists: S11]
    std::vector<Entity> queryRadius(SpaceId space, const Vec3& center, double radius,
                                    const ComponentMask& mask = {}) const;            // §9.5 [exists: S09]
    void queryRadius(SpaceId space, const Vec3& center, double radius, const ComponentMask& mask,
                     std::vector<Entity>& out) const;     // reuses out's capacity [S09]
    template <class T> T& global();  template <class T> T& spaceState(SpaceId space);  // §7.8 [exists: S10]
    template <class T> const T& readGlobal() const;
    template <class T> const T& readSpaceState(SpaceId space) const;
    PersistentId makePersistent(Entity e);   // §9.6 [exists: S10]
    void queueAction(const PlayerAction& a); // §9.3 [exists: S10; S15: a message]
    void setRenderSink(RenderSink* sink);    // §10.3; forwards the registered meshes and materials and creates the
                                             // render instances; null allowed; any time [planned: S19]
    MeshId registerMesh(const std::string& name, const MeshLods& lods, MaterialId material);   // [planned: S19]
    MaterialId registerMaterial(const std::string& name, const MaterialData& data);           // [planned: S19]
    MeshId findMesh(const std::string& name) const;                    // name tables owned by the World [S19]
    PhysicsWorld& physics();                 // §11 [planned: S28]
    Animation& anim();                       // §10.6 [planned: S34]
    Navigation& nav();                       // §9.7 [planned: S45]
    void requestTransition(Entity player, Entity link);   // §12.4 [planned: S21]; called by the game (policy)
    bool requestArrival(Entity e, SpaceId space, const Vec3& position);   // §12.3; false: rejected by a game limit
                                                                          // [planned: S24]
    template <class M> void send(uint8_t peer, const M& message);   // §13.6 [planned: S15]
    void setOwner(Entity root, uint8_t peer);   // host; before the entity is replicated (player characters at join);
                                                // no runtime ownership transfer (§13.3) [planned: S16]
    void ownerTeleport(Entity e, const Vec3& p, const Quat& r);    // host; direct or OwnerCommand [S16]
    void ownerImpulse(Entity e, const Vec3& impulse);               // host; direct or OwnerCommand [S28]
    static std::string diff(const World& a, const World& b, int maxLines = 20);  // diffSaves of both saves [S11]
    uint8_t localPeer() const; bool isHost() const;   // [exists: S10; before S16: 0 and true]
    template <class T> const T& data(const std::string& name) const;   // §14 [exists: S11]; DataError if missing
    template <class T> const T& data(const DataRef<T>& ref) const;     // resolved once per store [S11]
    bool reloadData(std::string& error);                               // entries updated in place [S11]
    double terrainHeight(SpaceId space, double x, double z) const;     // §12.2 [planned: S20]
    double waterHeight(SpaceId space, double x, double z) const;       // §10.9 [planned: S39]
    Entity resolve(PersistentId id) const;                             // §9.6 [exists: S10]
    template <class... C> View<C...> ownedBy(uint8_t peer);   // §13.3 [exists: S10: all entities; S16: filtered]
    Profiler& profiler();                    // §17.1 [exists: S10]
};

class EventQueues
{
   public:
    template <class E> void emit(const E& e);
    template <class E> const std::vector<E>& read() const;
};

class SystemList
{
   public:
    void add(Phase phase, const char* name, SystemFn fn, RunOn runOn = RunOn::Host);   // runs in registration order
};

class Rng   // PCG32 (pcg32 of pcg-random.org), engine/core/Rng.h [exists]
{
   public:
    explicit Rng(uint64_t seed = 0, uint64_t stream = 0);   // pcg32_srandom: Rng(42, 54) gives the reference values
    uint32_t nextU32(); uint64_t nextU64();
    double uniform();                                         // [0, 1), 53 bits from two outputs
    double uniform(double a, double b);                       // [a, b): fma(b - a, u, a); b -> largest double < b
    int64_t range(int64_t a, int64_t b);                      // [a, b] inclusive, no modulo bias
    struct State { uint64_t state, increment; };              // saved by the world
    State state() const; void setState(const State&);
};
// No std::*_distribution anywhere in simulation or content code: their results differ between standard libraries.
// Hashing: FNV-1a 64 (engine/core/Hash.h) for seeds, content hashes and world.hash(); never std::hash.
//   fnv1a64Bytes(data, size) and constexpr fnv1a64(string_view) (separate names: fnv1a64(str.c_str(), seed) must
//   not hash seed bytes); Hasher (addU8/U32/U64/I32/I64 as little-endian bytes,
//   addDouble as the bit pattern, addString with its length) [exists]; hashCombine(seed, value) for derived seeds.
```

**Math subset (R-MATH-2)** [exists]: `Vec2/3/4`, `Mat3/4`, `Quat`, `AngleAxis`; `+ - *`, `dot`, `cross`, `norm`,
`normalized`, `cast<T>()`, `transpose`, `inverse`, `block`, `Zero/Ones/Identity`, `UnitX/Y/Z/W`, `setIdentity`,
`isApprox` (Eigen semantics and default precision), `cwiseProduct/Min/Max`, `Matrix *=` (scalar and square matrix);
`Quat` `* *= inverse conjugate normalized slerp matrix() isApprox angularDistance`; `AngleAxis` (double, `angle()`,
`axis()`, `toRotationMatrix()`, `Quat(aa)`, `AngleAxis(q)`). Double helpers in `EigenGLMInterface.h` (templates
restricted to double scalars, so float arguments keep using the float functions and Eigen expressions are not
ambiguous): `createTRSmatrix(Vec3 or Vec4, Quat, Vec3 or Vec4)`, `lookAt(Vec3, Vec3, Vec3)` (asserts on
`eye == center` and on `up` parallel to the view direction), `make_quat(Mat3 or Mat4)`,
`quat_cast(Mat3 or Mat4)` (normalized). Added in S05: tiny-eigen tests in `tests/test_core_tiny_eigen.cpp`, tests of the
subset and the double helpers in `tests/test_core_math_functions.cpp` (run with tiny-eigen in CI; with Eigen built and
run locally with the "Eigen backend check" of AGENTS.md). For the S05 additions tiny-eigen rejects at compile time
what Eigen rejects: `isApprox`/`cwise*` need the same scalar type and dimensions and both matrices or both arrays, the
double helpers check the argument shapes (`RowsAtCompileTime`/`ColsAtCompileTime`, Eigen's names),
`AngleAxis` uses its axis as given (like Eigen, also in `toRotationMatrix()`); `UnitX/Y/Z/W`/`Unit(i)` are for column
vectors only (Eigen also allows row vectors; like Eigen the index of `Unit(i)` is not checked in release builds).
In general tiny-eigen accepts more than Eigen (e.g. mixed float/double arithmetic such as `Vec3 + vec3`, and
tiny-eigen-only API like `Quaternion::FromAngleAxis`), so a tiny-eigen build does not prove that code compiles with
Eigen; only the Eigen backend check does. Float overloads with Eigen expressions as arguments can be ambiguous (`vec3`/`vec4`,
`mat3`/`mat4` overloads); pass concrete types.

### 7.11 Not in the ECS

Archetypes, groups, automatic scheduling, observers, runtime-defined component types, relations beyond `Parent`,
generic change detection (only the `Transform` flags), access declarations per system.

## 8. Simulation

### 8.1 One step [exists: S08 (phases, systems, commands after every phase)]

| Phase | Work | Runs on |
|---|---|---|
| 1 Input | local input and received messages (client actions, owner commands, §13.3) in (peer, sequence) order -> components | all peers (each for what it owns) |
| 2 Events | `behaviourEventSystem` delivers the previous step's `DamageEvent`, `InteractEvent`, `ContactEvent` to `GameObject` handlers | host |
| 3 Behaviour | `behaviourSystem`: `GameObject::update` for objects whose tick is due (§8.4) | host |
| 4 Gameplay | engine and game systems in registration order (player control; the kinematic mover until S29) | per system (`RunOn`) |
| 5 Physics | owner poses in, Jolt step, active bodies -> `Transform`, sorted contacts -> events, character movers from S29 (§11) | host; clients for what they own |
| 6 Transform | hierarchy, spatial grid, chunk membership | all |
| 7 Streaming | apply chunk results that are ready, request new ones (§12.3) | all |
| 8 Extract | changed transforms and animation -> `RenderSink` (§10.3); host: replication (§13.4); owners: owner-authoritative state (§13.3); structural log consumed | all |

Commands are applied after every phase; events of this step become readable in the next step. Terrain and piece
edits and promotions apply at the flush of the phase that requested them (between steps: immediately) (§12.2, §11);
clients apply received live modifications in phase 1.

### 8.2 Registration [exists: S08]

```cpp
void registerEngineSystems(SystemList& s)
{
    s.add(Phase::Transform, "transform", transformSystem, RunOn::Both);   // S08: hierarchy, changed list
    s.add(Phase::Events, "behaviour_events", behaviourEventSystem);       // S10
    s.add(Phase::Behaviour, "behaviour", behaviourSystem);                // S10
    s.add(Phase::Gameplay, "player_control", playerControlSystem, RunOn::Both);   // S10: PlayerInput -> intent
    s.add(Phase::Gameplay, "character_move", characterMoveSystem, RunOn::Both);
        // S10: kinematic, phase 4 (position += desiredVelocity * dt; y = terrain height, 0 before S20).
        // From S29 the system is registered in Phase::Physics and runs after the Jolt step, in entity order, on the
        // owner (ownedBy); phase 4 only produces CharacterIntent.
    s.add(Phase::Transform, "spatial_grid", spatialGridSystem, RunOn::Both);   // S09: §9.5
    s.add(Phase::Transform, "tick", tickSystem, RunOn::Both);                  // S10: §8.4, after the grid
    // ...
}
```

- `RunOn::Host` (default), `RunOn::Client`, `RunOn::Both`. Systems that run `GameObject` code are serial.
- Systems that write `OwnerAuthoritative` components iterate `world.ownedBy<...>(localPeer)` (R-NET-2). Player control
  (input -> `CharacterIntent`) and the movers are such `RunOn::Both` systems, never `GameObject::update`, so they run
  on the owning peer.

### 8.3 Application and main loop [planned: S13; `World::advance` and `step`: exist (S08)]

The engine has its own main loop in `EngineApp` (`engine/app/EngineApp.h`); it does not use saiga's `MainLoop`,
`StandaloneWindow` or `VulkanForwardRenderer` (they stay for the other samples, §0). `EngineApp` owns:

- the window: saiga's GLFW `VulkanWindow` subclass, used only for the surface, the size and events (none when headless);
- the Vulkan instance and device: the creation code is factored out of `VulkanRenderer`'s constructor into a function
  (`createVulkanDevice(VulkanWindow* window /* nullptr: headless */, parameters) -> VulkanBase`) that `EngineApp` calls;
- the `RenderScene` (§10.1): **S13 creates its frame shell** (swap chain or offscreen images, frame synchronization,
  one MSAA color + reversed-Z depth pass, ImGui, screenshots); S17 adds the HDR frame, S18 and S19 the scene content;
- the `World`, the `McpServer` and the camera: a `Glfw_Camera<PerspectiveCamera>` (WASD and mouse) until S35 replaces
  it with the camera rig;
- the **input map** (`engine/app/InputMap.h`): devices (keyboard, mouse; gamepad from S35) are mapped to named actions
  (buttons) and axes by a per-game data file (`data/<game>/input.json`, rebinding writes a user copy). Game code and
  `PlayerInput`/`PlayerAction` see only actions and axes. S13 implements keyboard and mouse, S35 gamepads.

`main` fills `EngineAppSetup` from the command line (`--headless`, `--benchmark <name>`, `--screenshot <file>` after
N frames, `--width/--height`, `--validation`, `--no-vsync`, `--scenarios [filter]`: run the game's scenarios with
their variants headless, print the failures and their count, exit with 0 if at least one scenario ran and all pass,
else 1; a filter that matches no scenario prints `no scenarios matched` and fails, §17.2) and `SAIGA_MCP_PORT`; there
is no `config.ini`.

- **One frame** (`EngineApp::frame()`):
  1. poll window events and input (GLFW; nothing when headless);
  2. `mcp.poll()` (tool handlers run between steps);
  3. frame time: `steady_clock`, or the fixed `dt` given to `runFrames` (headless tests are deterministic);
  4. apply look input to the camera immediately (no step latency; the prototype camera before S35, the rig of §10.6
     after);
  5. `world.advance(frameDt)`;
  6. render: before S19: `RenderScene::renderFrame` calls the `GameRenderHooks` (the prototype world renderer); from S19
     the extract results and the interpolated camera go to the `RenderScene`, which records and submits the frame (ImGui
     and game UI included) and presents. Skipped when rendering is disabled or no frame can be presented; frame
     callbacks registered with `onFrameRendered` run after each rendered frame (screenshots, measurements).
- **The host never stalls the session on rendering**: when no frame can be presented (minimized window, swap chain
  unavailable, loading screen, shader compilation), the loop keeps calling `mcp.poll()` and `world.advance()` at least
  at the step rate; rendering work during a transition is budgeted per frame. The dropped-backlog count is a host
  statistic visible to clients. Test: a headless `EngineApp` with rendering disabled still advances.

```cpp
struct GameRenderHooks                                        // until S19 the game draws the prototype world renderer
{
    std::function<void(EngineApp&)> init;                     // after the device exists (create game GPU state)
    std::function<void(EngineApp&)> shutdown;                 // after the device is idle (destroy it)
    std::function<void(vk::CommandBuffer)> transfer, render;  // called by the frame recording
    std::function<void()> gui;
    std::function<bool(FrameTimes&)> frameTimes;              // GPU pass times of the last measured frame;
                                                              // FrameTimes { double cpuMs; std::vector<double>
                                                              // gpuPassMs; } in RenderSceneTypes.h from S13
    std::function<const char*(int)> passName;                 // pass names (from S19 RenderScene::gpuPassName)
    std::function<double(double x, double z)> groundHeight;   // benchmark paths; the only hook kept in S19, removed
                                                              // in S20 (world.terrainHeight)
};
struct EngineAppSetup
{
    WorldSetup world;
    bool headless = false;                                    // no window, offscreen images, no input
    int width = 1600, height = 900;
    bool vsync = true, validation = false;
    RenderSceneParameters render;                             // [planned: S13 (MSAA samples), extended in S17] quality, ...
    void (*scenarios)(ScenarioList&) = nullptr;               // for run_scenario
    void (*gameTools)(McpServer&, EngineApp&) = nullptr;      // game_* tools
    GameRenderHooks renderHooks;                              // transitional (S13 until S19; groundHeight until S20)
    int mcpPort = 8765;                                       // 0: no MCP
};
class EngineApp   // engine/app/EngineApp.h
{
   public:
    explicit EngineApp(const EngineAppSetup& setup);
    ~EngineApp();                                             // waits for the device, then destroys in order
    int run();                                                // until quit; returns the exit code
    void runFrames(int count, double dt);                     // fixed frame time (tests, benchmarks)
    void quit();
    World& world(); McpServer& mcp(); PerspectiveCamera& camera(); VulkanBase& device();
    RenderScene& scene();                                     // frame shell from S13
    void frame();                                             // one iteration of the loop
    void onFrameRendered(std::function<bool()> callback);     // return true to remove it
    void setRenderingEnabled(bool enabled);                   // false: steps and MCP continue, no frames rendered
    // run_scenario: EngineApp holds std::unique_ptr<ScenarioWorld>; world() returns its world while one is active
};
// main: parse the command line (above), fill EngineAppSetup, EngineApp app(setup); return app.run();
// RenderScene frame shell (S13), src/saiga/vulkan/scene/RenderScene.h:
//   RenderingFormats formats() const;   // for the prototype pipelines (MSAA, reversed-Z)
//   void requestScreenshot(std::function<void(const TemplatedImage<ucvec4>&)> callback);
//   void renderFrame(const std::function<void(vk::CommandBuffer)>& transfer,
//                    const std::function<void(vk::CommandBuffer)>& render, const std::function<void()>& gui);
// The prototype WorldRenderer::init changes to (VulkanBase&, const RenderingFormats&, const WorldParameters&) in S13.
```
- `World::advance(frameDt)`: accumulator += `frameDt * timeScale`; runs whole steps, at most 4; then drops the remaining
  backlog (`accumulator = min(accumulator, 0.999 * dt)`, counted in `droppedSteps()`), so `alpha = accumulator / dt` is
  in [0, 1). When paused: no steps. `world.step(n)` runs exactly n steps (also when paused). A NaN frame time is
  ignored, +inf counts as a long frame. [exists: S08]

### 8.4 Ticking game objects [exists: S10]

- `behaviourSystem` (phase 3) calls `update(dtSinceLastTick)` (seconds, double) for objects with `sinceTick >=
  tickInterval` (both seconds; a tolerance of 1e-9 s, so six steps of 1/60 s make 0.1 s), in the dense order of the
  `Behaviour` pool. Policy (R-MECH-1), by grid-cell rings around the players (decided in S10 instead of distances):
  `struct TickPolicy { double ringSize = 32; int nearRing = 1, midRing = 6; double midInterval = 0.1, farInterval =
  0.5; };` in `WorldConfig`. The x/z plane is divided into `ringSize` cells; the ring of an object is the Chebyshev
  distance in cells between its cell and the nearest player's cell in its space: `<= nearRing` every step (3 x 3 cells,
  96 m), `<= midRing` every `midInterval` (13 x 13 cells, about 200 m), else `farInterval` (also without a player in the
  space). A class overrides the rings and intervals with `virtual const TickPolicy* tickPolicy() const { return
  nullptr; }` (the cell size is always the world's); players and objects with `Behaviour::alwaysTickEveryStep` tick
  every step (the flag is read by `behaviourSystem`, not folded into `tickInterval`, so clearing it takes effect at
  once). The first tick of a spawned object is staggered by `Behaviour::tickOffset = (hashCombine(0, index) %
  1024) / 1024.0 * tickInterval` (it ticks when `sinceTick + tickOffset` reaches the interval; `update` gets the real
  `sinceTick`).
- `tickSystem` (phase 6, after the grid) keeps the sorted player cells: when they change (a player crossed a cell
  border, joined or left) every object is re-evaluated, otherwise only the objects of the changed list; `spawn`
  evaluates the new object at once. Exact and deterministic (no round-robin delay); cost per step is the moved objects
  (the S10 benchmark: 10^4 objects 0.09 ms per step with a standing player, 0.21 ms re-tiering all every step).
- `ScenarioWorld` ticks every object every step unless `w.enableDistanceTicking()`.

### 8.5 Threading [exists: S06 (JobPool); streaming pool: S20; network thread: S33]

| Thread(s) | Work |
|---|---|
| main | steps (systems serially), extract, render recording, MCP |
| frame `JobPool` (`setWorkerThreads`, the caller participates) | `parallelEach`; per-frame `JobPool::parallelFor` jobs (pose evaluation, extract); Jolt jobs through a job system adapter (S28 adds a non-blocking submit/barrier API to `JobPool` for it, or uses Jolt's `JobSystemThreadPool` with the same worker count; `parallelFor` alone cannot serve Jolt) |
| streaming pool (2-3 low-priority workers) | chunk and region generation, terrain meshing, file IO, compression |
| network thread | socket IO only, queues to and from the main thread |
| audio (S47) | device callback / mixing; reads a lock-free command queue filled by the main thread, writes no components |

`JobPool` (`engine/core/JobPool.h`) [exists]: `JobPool(int workerThreads)`, `setWorkerThreads`, `parallelFor(int64_t
begin, int64_t end, int64_t chunk, fn(chunkBegin, chunkEnd))`. The chunk boundaries are fixed (they never depend on the
thread count, only the assignment of chunks to threads does); the caller takes chunks too; with 0 workers or nested
inside a chunk of the same pool the chunks run serially in order (a chunk of another pool calls the pool normally);
calls from different threads are serialized, except nested calls (two chunks of one job that call the pool again run
their nested chunks at the same time); pools that call each other in a cycle from different threads deadlock and are not
allowed; an exception of a chunk is rethrown in the caller (the remaining chunks are skipped from the moment a thread
sees the failure, best effort); more than INT64_MAX chunks throw `std::length_error`. saiga's `ThreadPool` is not used.

### 8.6 Determinism [exists: S08 (world.random(), fixed dt, deterministic commands, events and parallelEach)]

**Only scenarios are reproducible**: the same build and the same scenario (seed, scripted inputs) give the same
result on every run in `ScenarioWorld`, so tests never flake. `ScenarioWorld` runs headless with a fixed `dt`, a fixed
worker thread count (default 1) and synchronous chunk loading (§12.3). Ensured by: fixed `dt`; one PCG32 stream
`world.random()` seeded from the world seed (its state is saved); deterministic iteration (§7.5), commands and events;
`parallelEach` without cross-entity effects.

The live game is **not** required to be reproducible: the frame rate, streaming timing and the worker thread count
may change results, and nothing at run time depends on it (clients receive state from the host, §13). A loaded save
restores all saved state but does not continue bit-identically (§15.2). Content generation is a separate requirement:
it is identical on every machine and compiler (R-CONTENT-1), because clients generate chunks themselves and worlds
regenerate chunks after patches.

## 9. World and game objects

### 9.1 Coordinates

- Simulation in double world coordinates of its space (R-MATH-1); Jolt in double precision (positions `RVec3`; Jolt
  computes rotations and velocities in float internally, we store them as double).
- Render data of tier 1/1p and terrain is float relative to its chunk or region origin; rendering is camera relative
  (§10.1).

### 9.2 GameObject [exists: S10]

```cpp
class GameObject
{
   public:
    virtual ~GameObject() = default;
    virtual void configure() {}                  // const configuration (data files); on spawn, load, replication
    virtual void onSpawn() {}                    // host, on spawn only; initial components are already present
    virtual void onLoad() {}                     // host, after loading or chunk reload: rebuilds `cache`
    virtual void onCatchUp(double elapsedSeconds) {}   // host, after a chunk reload (§9.6); game policy
    virtual void onReplicatedSpawn() {}          // client: rebuilds `cache` from replicated components only
    virtual void onReplicatedDestroy() {}        // client: cosmetic reaction
    virtual void update(double dtSinceLastTick) {}
    virtual void onDamage(const DamageEvent&) {}
    virtual void onInteract(const InteractEvent&) {}
    virtual void onContact(const ContactEvent&) {}
    virtual void onDestroy() {}                  // host; not called when the entity is unloaded (§9.6)
    virtual const TickPolicy* tickPolicy() const { return nullptr; }   // §8.4

    Entity entity() const;
    World& world();
    template <class T> const T& read() const;
    template <class T> T& write();
    template <class T, class... A> void add(A&&...);
    template <class T> void remove();
    template <class T, class... C> Entity spawn(const Vec3& position, C&&... initial);   // in this object's space
    template <class T, class... C> Entity spawnIn(SpaceId space, const Vec3& position, C&&... initial);
    void destroy(Entity e);   // destroy() without argument destroys this object's entity
    void destroy();
    const Transform& transform() const;
    Transform& transformWrite();
};
```

- Optional per entity, owned by the `Behaviour` component (`std::unique_ptr<GameObject>`, `classId`, `tickInterval`,
  `sinceTick`, `tickOffset`, `alwaysSimulate`, `alwaysTickEveryStep`; `visit` covers all but the pointer; the object is recreated
  through its factory).
  `tickInterval` is recomputed in phase 6 from the grid-cell rings around the players (§8.4; `ScenarioWorld`: all
  objects tick every step).
- S10 implementation (`engine/gameplay/`): `GameObject.h` (the class; `entity()`, `world()`, `read/write/has/add/
  remove<T>`, `spawn`, `spawnIn`, `destroy`, `transform()`, `transformWrite()`), `Behaviour.h` (the component and
  `World::spawn`/`spawnIn`), `ClassList.h` (`add<T>`, IDs contiguous like components: `EngineClass` 0..63, game classes
  from `kFirstGameClass` = 64; names unique; `find(name)` for the world API), `TickPolicy.h`, `Systems.h`.
  `spawn<T>(position, initial...)`: between steps at once; during a step the entity is reserved and returned, the rest
  is a command (dropped if the entity is destroyed earlier in the same flush): `Transform` position, `Behaviour` with
  the object from the factory, the initial components, the tick interval, `configure()`, `onSpawn()` (skipped if
  `configure` destroyed the entity). `Behaviour` and `Persistent` cannot be added with `add` (compile error).
  `destroy` calls `onDestroy()` of the entity and its descendants (children first, while all components are present)
  before any component is removed; a hook may destroy other entities, itself (no second `onDestroy`) or spawn new ones
  (loot). Destroys requested by `onDestroy` hooks are queued and applied after the current destroy, iteratively (no
  recursion per chained destroy). After the hooks the descendants are collected again: entities a hook attached to a
  doomed parent are destroyed with it, without their own `onDestroy` (a hook that keeps respawning children cannot
  loop); an entity a hook released from the cascade (e.g. removed its `Parent`) lives on and gets `onDestroy` again on
  its real destroy. A game object stays valid while one of its hooks runs, also if the hook destroys its entity or
  removes its `Behaviour` (it is released when no hook runs). An exception in `onDestroy` aborts the destroy (nothing
  is destroyed; a later destroy calls the hooks again); an exception in `configure`, `onSpawn` or an initial add
  destroys the half-spawned entity (without `onDestroy`) before it propagates. Two deferred adds of a component, or
  removes of a missing one, are refused when applied in every build (hooks see the state before the flush). During a flush, operations on an entity reserved later in the
  same phase follow its activation (appended to the commands); so does `add<Parent>` with such a parent. `remove<Behaviour>` drops the object without
  `onDestroy`. C++ `spawn<T>` adds the initial components before `configure`/`onSpawn` (a hook that adds one of them
  again asserts: check `has<T>()`); the world API's `spawn` applies them after `onSpawn`, merged into components the
  object added itself. The first tick of a spawned object is staggered by
  `Behaviour::tickOffset`; `update` always gets the real time since the last update. `behaviourEventSystem` skips dead targets,
  entities without a game object and entities without `Simulated`.
- The player's `GameObject` holds only host-side logic (damage, inventory, quests); movement is a system (§8.2).
- Game-specific events reach game objects through the game's own systems, which call methods of the game's own
  subclasses; the engine `GameObject` is not changed for game events.

### 9.3 Engine gameplay basics [exists: S10]

| Type | Fields | Replicate |
|---|---|---|
| `Player` | `uint8_t peer; uint64_t characterId; double moveSpeed = 5` (player characters; used by ticking, streaming, interest, ownership). `peer` is a session slot; `characterId` != 0 is random, created with the character file and stored in it (before S33 set by the game; in `ScenarioWorld::spawnPlayer` numbered 1, 2, ...; from S33 created by the client with its character file and sent on join); everything that names a player across sessions (piece owner, game references) stores `characterId`, never a peer index | Host |
| `Health` | `double hp, maxHp` | Host |
| `PlayerInput` | `Vec3 move; Quat look; uint32_t heldButtons` (latest wins) | OwnerAuthoritative |
| `PlayerAction` (message) | `Entity player; uint32_t action; Vec3 aim`: one-shot actions, reliable, delivered in phase 1 in (peer, sequence) order as `ActionEvent` on the host, exactly once; before S15 it is queued locally and delivered the same way | – |
| `ActionEvent` | `Entity player; uint32_t action; Vec3 aim` | Never |
| `CharacterIntent` | `Vec3 desiredVelocity; Entity target` (computed on each peer) | Never |
| `Transform` | §7.4 | OwnerAuthoritative |
| `DamageEvent` | `Entity target, source; double amount` | Never |
| `InteractEvent` | `Entity target, actor; uint32_t action` | Never |
| `ContactEvent` | `Entity a, b; Vec3 point, normal` | Never |

Flat terrain (before S20) is the plane y = 0.

Engine system `playerControlSystem` (phase 4, `RunOn::Both`, over `ownedBy<const Player, const PlayerInput,
CharacterIntent>`): `desiredVelocity = move * Player::moveSpeed` (a `move` longer than 1 is shortened to 1; NaN gives 0);
games add their own control systems for actions.

S10 implementation: the components in `engine/components/` (`Player.h`, `Health.h` (default 100/100), `PlayerInput.h`,
`CharacterIntent.h`), the events in `engine/world/Events.h` (`EngineEvent` 0..3), `PlayerAction` in
`engine/gameplay/PlayerAction.h`. `world.queueAction(action)` (the player must be alive with a `Player`) queues with
the player's peer and a sequence number; at the start of the next step the queue becomes pending `ActionEvent`s in
(peer, sequence) order, readable in that step (an aborted step keeps them like other events); the queue is part of
the hash. `characterMoveSystem` (phase 4) moves entities with a non-zero `desiredVelocity` (`y = 0`) and does not
write idle ones.

### 9.4 Classes decide, systems execute

`update` writes intent; movement, animation, physics, damage, replication and rendering are systems over all entities.
Reusable mechanics (telegraphed attacks, projectiles, summons, status effects) are components plus systems.

| Situation | Use |
|---|---|
| Unique or state-machine-heavy behavior (boss, quest NPC, player logic) | `GameObject` subclass, state in `<Name>State` |
| Many similar simple objects (projectiles, particles, props) | components + system |
| Reusable mechanic | component + system |
| Numerous static content | tier 1; player-built: tier 1p |

### 9.5 Spatial grid [exists: S09]

Uniform 32 m grid per space, updated in phase 6 for changed transforms. `queryRadius` returns entities sorted by entity
index (deterministic). Used for ticking, gameplay queries and replication interest.

S09 implementation (`engine/world/SpatialGrid.h`, `SpatialGridSystem.h`): square cells of 32 m in x and z (columns, y
is not divided), per space a hash map from the cell to its items `{Entity, position}`; per entity index the cell and
the slot in it (swap-and-pop removal). `spatialGridSystem` (phase 6, after `transformSystem`) reads its own structural
log consumer (created entities enter, destroyed ones leave, in log order) and the changed list (an update without a
cell change only rewrites the stored position); idle entities cost nothing. `queryRadius` tests the Euclidean distance
in 3D (boundary included) against the position stored at the last phase 6: an entity moved since is found at its old
position, one created since is not found until then (reserved entities never); destroyed entities are never returned
(stale handles are filtered by `alive`); it visits the cells of the radius' square, or every occupied cell of the space
if those are fewer; a negative or NaN radius finds nothing, non-finite positions are never found. `mask` defaults to no
requirement; `Simulated` is not required. Queries only read and are allowed
in `parallelEach`. Engine-internal `world.forEachWithin(space, center, radius, mask, fn(Entity, gridPosition))` is the
unsorted form for order-insensitive callers (S16 interest, the world API): it reports the stored position the decision
was made with, so distances (hysteresis, nearest player) use the same state. The grid is derived data (not saved, not hashed); `rebuildSpatialGrid(world)` (engine-internal)
rebuilds it from all entities (S11 `load`). World-state entities (S10) are excluded.

### 9.6 Far from players: persistence and simulation policy [planned: S25]

Stages: the `PersistentId` type in S07 (`engine/ecs/PersistentId.h`); `Persistent`, `makePersistent` and `resolve` in
S10; unloading and the rest of this section in S25.

S10 implementation: `Persistent {PersistentId persistentId}` (engine component, `Replicate::Host`); `makePersistent`
returns the existing ID or assigns the next counter value (between steps the component is added at once, during a step
at the flush; the counter is part of the hash). The World keeps the ID -> entity index in `add`, `remove` and `destroy`
of `Persistent` (not from the structural log: resolve is then current between steps); `resolve` returns `Entity{}`
for a destroyed entity (also after its slot is reused). `add<Persistent>` does not compile (use `makePersistent`);
`Persistent` is read-only through `get`, `tryGet` and views. During a step `makePersistent` is idempotent (a second call
in the same phase returns the pending ID) and accepts entities reserved in the step (a fresh `spawn`).

- Player entities of connected players are never unloaded by chunk unloading (leaving players: §13.7). Entities with
  `Behaviour::alwaysSimulate` (game decides: world events, important NPCs) keep ticking while their chunk is resident,
  even outside the simulation radius; they do not keep chunks loaded. When their chunk unloads they unload like every
  other entity; a game that needs traveling NPCs far from players moves them abstractly (policy, R-MECH-1).
- When a chunk is unloaded (rule in §12.3: `fullRadius + unloadMargin`), the root entities (no `Parent`) in it
  are serialized with their descendants into the chunk's state and removed with the
  engine-internal `unload` command (no `onDestroy`; structural log records flagged `unloaded`; replication sends
  `Despawn`, never destroy) (commands
  in phase 7). Inside such a group `Entity` fields are stored as group-local indices and remapped on reload; fields
  pointing outside the group are stored as null (use `PersistentId`).
- In phase 6, a root entity whose chunk is not resident and not requested (spawned with `spawnIn` far from players,
  flung or teleported there) is unloaded with the same group rules into an in-memory pending list of that chunk (part
  of the world-state snapshot of each commit); the list is merged into the chunk record when the chunk reloads or at
  the next commit on the streaming pool. Phase 6 does no IO.
- Player entities are never written into chunk state.
- On reload (phase 7, commands, serialized order) the group is recreated with new handles; then `configure()`,
  `onLoad()` and `onCatchUp(elapsed)` run per game object (elapsed = simulation time since the unload; what catches up
  is game policy, R-MECH-1).
- `PersistentId {uint64_t}` for entities that must stay referable: engine component `Persistent {PersistentId id}`;
  `world.makePersistent(e)` assigns the next value of a world counter (saved) and is idempotent; the world
  keeps an id -> entity index, updated from the structural log; `world.resolve(pid)` returns `Entity{}` if not
  resident. Available from S10 (everything resident), unloading from S25.
- The states of unloaded chunks are part of the save.

### 9.7 Navigation (optional module) [planned: S45]

`engine/nav/`: per-chunk navigation data (grid or navmesh) generated from terrain, authored space geometry and placed
pieces, updated on edits, streamed with the full radius; `nav.findPath(space, from, to)`. Games enable it in
`WorldConfig` if their NPCs path-find; otherwise steering only.

## 10. Rendering

### 10.1 RenderScene [planned: S13 frame shell, S17 HDR frame, S18 instances]

The interface between the engine and `saiga_vulkan`; knows meshes, materials, regions and instances, not entities. Class
`RenderScene` in `src/saiga/vulkan/scene/RenderScene.h`, value types (`MeshLods`, `MaterialData`, `StaticInstance`,
`DynamicPose`, `RenderSceneParameters`, `QualityTier`, `EnvironmentState`, `LightData`, `WaterData`, `EmitterData`,
`FogVolumeData`, `TerrainTileData`, `TerrainPyramidData`, `GrassMaskData`, `MeshId`, `MaterialId` and the handle types)
in `src/saiga/vulkan/scene/RenderSceneTypes.h`; `ChunkId`/`RegionId` from `saiga_core`. Constructed with
`RenderScene(VulkanBase&, VulkanWindow* window, const RenderSceneParameters&)` (window null: headless, offscreen
images); it owns the swap chain or offscreen images, frame synchronization, the HDR targets, all passes and ImGui
(§10.5); it replaces `VulkanForwardRenderer` for engine apps.

```cpp
MaterialId addMaterial(const std::string& name, const MaterialData& data);              // §10.8
MeshId addMesh(const std::string& name, const MeshLods& lods, MaterialId material);     // LODs incl. impostor, skin
InstanceSetHandle addInstanceSet(RegionId region, const std::vector<StaticInstance>& instances,
                                 uint32_t capacity = 0);            // tier 1, 1p, far; capacity >= size (0: size)
void updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& instances);
                                                                    // first + size <= capacity (appends up to it)
void removeInstance(InstanceSetHandle set, uint32_t index);         // tombstone (scale 0)
void removeInstanceSet(InstanceSetHandle set);                      // chunk or region unloaded [S18]
// A set grows by being re-created with a larger capacity (piece placement, S30).
MeshId addGeneratedMesh(RegionId region, const MeshLods& lods, MaterialId material);   // [S30] merged far meshes
void removeGeneratedMesh(MeshId mesh);                                                 // [S30]; drawn like others
InstanceHandle addDynamic(MeshId mesh);
void setDynamic(InstanceHandle h, ChunkId chunk, const DynamicPose& prev, const DynamicPose& curr, uint32_t boneOffset);
void removeDynamic(InstanceHandle h);
void setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy);   // double
void setInterpolation(float alpha);
// [planned: S20] as in RenderSink: setTerrainTile, removeTerrainTile, setTerrainPyramid, setGrassMask
// [planned: S36 environment and lights, S38 fog volumes, S39 water, S41 emitters, S42 clouds]
void setEnvironment(const EnvironmentState& env);   // sun/moon direction and intensity, sky, weather, fog, clouds
// No absolute time reaches a shader: EnvironmentState carries per-effect phases (wind, foliage, flicker, particles)
// computed in double on the CPU modulo each effect's period, as for the waves (§10.9).
LightHandle addLight(ChunkId chunk, const LightData& light); void setLight(LightHandle, const LightData&);
void removeLight(LightHandle);
void setWater(SpaceId space, const WaterData& water);   // sea level and waves (§10.9)
FogVolumeHandle addFogVolume(ChunkId chunk, const FogVolumeData& volume); void removeFogVolume(FogVolumeHandle);
EmitterHandle addEmitter(InstanceHandle attachTo, ChunkId chunk, const EmitterData& emitter);   // invalid handle:
void removeEmitter(EmitterHandle);                                  // fixed position from EmitterData
```

- **Storage**: one GPU instance buffer, sub-allocated with a VMA virtual block (`VmaVirtualBlock` forward declared, no
  VMA include in headers; main thread). Capacity from `RenderSceneParameters`; on failure the buffer grows 2x with a
  GPU copy, the old one through `deferDelete`. `StaticInstance` (32 bytes): region-relative position (3 x float),
  rotation (4 x snorm16), uniform scale (float), mesh (uint16), flags (uint16), `InstanceKey` low bits (uint32).
  Dynamic instance: prev and curr pose, chunk, mesh, `boneOffset`.
- **Offset table**: each frame `origin - cameraPosition` is computed in double for every region and chunk referenced by
  an instance set or dynamic instance of the camera's space and uploaded as float. Shaders compute
  `p = offset[ref] + local`; the view matrix contains only the camera rotation.
- **Interpolation on the GPU** for dynamic instances with `alpha`; the camera is interpolated on the CPU in double.
- The host renders only the space and regions around its own camera; chunks resident only for remote players get no
  GPU uploads.
- Statistics (S18): `struct RenderStatistics { std::array<uint32_t, 4> instancesPerLod; uint32_t terrainNodes,
  drawCalls; }; const RenderStatistics& statistics() const; static const char* gpuPassName(int pass);
  const FrameTimes& lastFrameTimes() const;` (used by §17.5 and §17.7).
- Required device feature (asserted): `drawIndirectFirstInstance`; without `multiDrawIndirect` draws are issued one by
  one.
- **S19 (transitional, replaced in S20)**: `void setPrototypeTerrain(const HeightField& heights, const
  PrototypeSettings& settings);` hands the prototype terrain, grass, sky and water plane (renderers moved from
  `vulkan/world/`) to `RenderScene`, which draws them in the §10.5 pass order until S20 replaces them;
  `PrototypeSettings` takes the sun, fog, wind and LOD values of the prototype's `WorldSettings` (`shading` and
  `freezeTime` move to `RenderSceneParameters`). The game creates the tree instance sets once after `setRenderSink`
  through `RenderSink::addInstanceSet` from `placeTrees`; from S20 `placeTrees` samples `SpaceTemplate::prototypeHeight`
  (trees stay within the prototype's 8 km); from S23 the chunk pipeline does it.
- Engine side: `RenderInstance {MeshId mesh;}` (saved by mesh name) selects what an entity renders. `RenderSink`
  (`engine/sync/RenderSink.h`) is the engine's interface to rendering, implemented by `VulkanRenderSink`
  (`engine/sync/VulkanRenderSink.h`, over
  `RenderScene`) and `RecordingRenderSink` (headless tests; records calls and provides the mesh name table):

```cpp
class RenderSink   // engine/sync/RenderSink.h [planned: S19]; types from saiga/vulkan/scene/RenderSceneTypes.h
{
   public:
    virtual ~RenderSink() = default;
    // IDs are assigned by the World's name tables (§7.10); the World forwards every registration (also those made
    // before the sink was set) with its ID, the sink stores the GPU data under that ID
    virtual void registerMaterial(MaterialId id, const std::string& name, const MaterialData& data) = 0;
    virtual void registerMesh(MeshId id, const std::string& name, const MeshLods& lods, MaterialId material) = 0;
    virtual InstanceSetHandle addInstanceSet(RegionId region, const std::vector<StaticInstance>& instances,
                                             uint32_t capacity = 0) = 0;
    virtual void updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& v) = 0;
    virtual void removeInstance(InstanceSetHandle set, uint32_t index) = 0;
    virtual void removeInstanceSet(InstanceSetHandle set) = 0;
    virtual MeshId addGeneratedMesh(RegionId region, const MeshLods& lods, MaterialId material) = 0;   // [S30]
    virtual void removeGeneratedMesh(MeshId mesh) = 0;                                                 // [S30]
    virtual InstanceHandle addDynamic(MeshId mesh) = 0;
    virtual void setDynamic(InstanceHandle h, ChunkId chunk, const DynamicPose& prev, const DynamicPose& curr,
                            uint32_t boneOffset) = 0;
    virtual void removeDynamic(InstanceHandle h) = 0;
    virtual void setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy) = 0;
    virtual void setInterpolation(float alpha) = 0;
    // [planned: S20]
    virtual void setTerrainTile(RegionId region, const TerrainTileData& tile) = 0;   // integer heights, level
    virtual void removeTerrainTile(RegionId region) = 0;
    virtual void setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid) = 0;
    virtual void setGrassMask(ChunkId chunk, const GrassMaskData& mask) = 0;
    // [planned: S36 environment and lights, S38 fog volumes, S39 water, S41 emitters, S42 clouds]: setEnvironment,
    // add/set/removeLight, add/removeFogVolume, setWater, add/removeEmitter (as in RenderScene)
};
```

### 10.2 Culling [planned: S18]

Prerequisite (implemented in S17): `GLSL.cpp` targets Vulkan 1.3 / SPIR-V 1.6 (`setEnvClient`, `setEnvTarget`), with a
test.

1. CPU: instance set bounds against the views -> visible instance ranges.
2. GPU count pass over the visible ranges (indirect dispatch): frustum test, LOD (S18: by distance with the prototype's
   thresholds; from S27 also by projected size), count per (view, mesh LOD) bucket with `atomicAdd`.
3. Prefix sum -> bucket offsets; write pass -> compacted 32-bit instance indices; instance counts into the indirect
   commands.
4. Draw: `vkCmdDrawIndexedIndirect` per pipeline and view.

Buffers and shaders have a view dimension from the start; S18 implements one view, the shadow views (S36) add more.
The buffers do not preclude Hi-Z occlusion culling against the previous frame's depth (open option, §21).

### 10.3 Extract and interpolation [planned: S19]

- The extract keeps per rendered entity the previous and current pose in double.
- In phase 8, for entities with `RenderInstance` whose `Transform` changed in this or the previous step: prev = old
  curr, curr = new pose (`teleported`: prev = curr), both relative to the chunk of the current pose; render instances
  are created and freed from the structural log. Every peer uses `RenderPose` when present (clients for remote entities,
  the host for client-owned entities; written by the interpolation buffer, §13.3/§13.4) instead of `Transform`; then the
  selection uses the `RenderPose` changed flags set by the interpolation buffer (same this/previous-step rule), not the
  `Transform` flags (tested in S19). The GPU upload of all changes happens once per frame.
- `RenderSink` (engine interface in `engine/sync/`): the Vulkan implementation over `RenderScene` and a
  `RecordingRenderSink` for tests. `World` may run without a sink.

### 10.4 Far content [planned: S27]

- Far content is a quadtree of regions: level L covers 2^L x 2^L chunks; each node holds an instance set thinned to a
  fixed budget (impostors and coarse meshes) and is selected by distance.
- **Hierarchical placement**: candidate counters are ordered by importance (hierarchical jittered grid: coarse cells
  first), so level L keeps the candidates with `counter < keptCount(slot, L)` under the same `InstanceKey`. Far sets
  are therefore subsets of the chunk content: tombstones apply at every level and near/far match. Far sets are
  generated without generating the full chunk (the generator evaluates only counters below `keptCount`), on the
  streaming pool, and are cached on disk. A far set evaluates each contained chunk with that chunk's stored placement
  version (the newest version for chunks not generated yet); the region summary holds the run-length list of placement
  versions of its generated chunks; the disk cache key contains a hash of that list and the generator versions.
- Each region keeps a **modification summary** (tombstoned keys per level, coarse height patches, piece far-mesh input),
  updated when a modification applies, so far content, coarse tiles and clients far away reflect edits.
  Placed pieces (bases) contribute merged far meshes per region, rebuilt after edits.
- Every mesh has an impostor as its last LOD. **Impostors are relightable** (albedo + normal atlases, no baked
  lighting), because the time of day changes (§10.7). Octahedral vs billboard is decided in S27.

### 10.5 Frame structure and post-processing [planned: S17 (HDR target), S19 (passes), S37 (post chain)]

- **Frame** (S17): `RenderScene` renders the scene into an HDR target (R16G16B16A16 float, MSAA per quality tier),
  composites the tone-mapped result into the swap chain image (or offscreen image), then ImGui and the game UI draw on
  top (single-sampled). Frame synchronization (frames in flight, timeline semaphore, deferred deletion) reuses
  `VulkanBase`; the rest of the frame logic is new code in `RenderScene`.
- **Pre-exposure**: shaders write radiance multiplied by the previous frame's exposure (half floats end at 65504; the
  sun disc is clamped); exposure is applied before tone mapping.
- **Pass order** (stage in brackets; the pass exists from then on, the list is the target order):
  1. shadow maps (cascades, far terrain horizon map, local light atlas; cached, §10.7) [S36]
  2. depth pre-pass (opaque and alpha-tested) [S19]
  3. depth resolve (single sample, nearest; used by AO, fog, clouds, water, DOF) [S19]
  4. ambient occlusion (half resolution, normals reconstructed from depth) [S37]
  5. opaque passes (terrain, instances, skinned characters, grass); they sample the AO texture (white before S37) and
     apply it to the ambient term and, as specular occlusion, to the environment specular [S19]
  6. sky [S19: prototype sky; S36: atmosphere]
  7. MSAA resolve, tone-map weighted (custom resolve, not `resolveMode`, so bright edges do not alias) [S17]
  8. transparents on the far side of the water surface [S39]
  9. water (§10.9) [S19: prototype plane; S39: full water]; the refraction source is a copy of the HDR color taken
     after pass 8 [S39; before S39 directly after pass 7]; writes its depth into the resolved depth (a copy is kept for
     refraction and foam)
  10. analytic fog composite on opaque and water: analytic height fog to the horizon, beyond the froxel range when
      volumetric fog is on [S19]
  11. clouds composite (§10.11), applying analytic fog at their own (march) depth: 2D layer [S36]; ray march from the
      volumetric fog far plane with depth-aware upsample [S42]
  12. froxel volumetric fog composite inside its range (§10.5.1), nearest to the camera [S38]
  13. transparents and particles on the camera side (sorted back to front per object; they apply fog in their own
      shaders) [S39: transparents; S41: particles]
  14. post chain: depth of field, bloom, exposure, tone mapping, color grading, upscaling of a reduced render
      resolution (spatial: bilinear plus sharpening) [S17: simple tone map; S37: full chain]
  15. UI (ImGui, game UI) on the swap chain [S13]
- **Anti-aliasing**: MSAA with alpha to coverage for foliage on every tier (sample count per tier). The Low tier
  (Steam Deck) lowers the render scale and upscales spatially in the post chain. No temporal anti-aliasing, temporal
  upscaling or motion vectors (§20). Each effect is switchable per game and tier (R-MECH-1).
- **Clay mode** (`WorldSettings::shading = WorldShading::Clay` [exists], the prototype's gray look) is a debug shading
  mode, used for golden images; it bypasses materials and tone mapping (identity), keeps geometry and lighting.

#### 10.5.1 Volumetric fog [planned: S38]

Froxel grid aligned to the view frustum (default 160 x 90 x 64 at High, exponential depth slices up to 300 m; Medium
half resolution, Low off), filled in compute with density from `Weather` (fog density, height falloff), fog volumes
(`FogVolume` component -> `RenderScene::addFogVolume`, S38) and the cloud density field (§10.11); lit by sun/moon
(cascade shadows) and the clustered local lights; temporally filtered (reprojection of the froxel grid with the previous
camera); integrated front to back into a scattering/transmittance volume sampled by the froxel fog composite (opaque and
water, §10.5 pass 12) and the transparent shaders. Beyond its range the analytic height fog continues to the horizon
with matching density at the boundary. Files: `shader/vulkan/render/volumetric_fog_*.comp`, pass in `RenderScene`.

### 10.6 Characters, animation and camera [planned: S34 (characters, animation), S35 (camera)]

- Skinned meshes (glTF via assimp) with low bone counts; GPU skinning in the vertex shader from a bone palette buffer
  (`boneOffset` in the dynamic instance, reserved from S18). Skinned meshes have LODs (fewer bones and triangles) and an
  impostor as the far LOD; the animation update rate scales with distance.
- `AnimationState` (clips, times, blend weights) is simulation state, `OwnerAuthoritative` for owned characters (written
  by a `RunOn::Both` animation-control system on the owner), replicated otherwise. Render poses are evaluated per frame
  at the interpolated animation time for visible characters (`engine/anim/`, `JobPool::parallelFor`, writes no
  components), not replicated; step-rate evaluation exists only for `evaluateBone`. Visual attachments (weapons, carried
  items) are `RenderAttachment {Entity parent; uint16 bone; offset}`, resolved in the extract or on the GPU; the
  simulation `Transform` of an attached entity follows a fixed socket offset from the parent's root, so simulation never
  depends on pose LOD. Gameplay that needs bone positions (hit boxes) calls `anim.evaluateBone(e, bone)`, evaluated from
  `AnimationState` at full rate, independent of LOD.
- Third-person camera rig (`engine/app/`): orbit around the local player, collision against terrain, instances and
  pieces, look input applied per frame (§8.3); games configure it.

### 10.7 Lighting, time and weather [planned: S36]

- `WorldTime` (time of day, day count; global) and `Weather` (cloudiness, rain, fog density, snow cover; per space) are
  components of the world-state entities (§7.8); how they change is game policy (R-MECH-1).
- **Shading model**: physically based (GGX specular with energy-conserving diffuse, metal/roughness from the
  materials, §10.8), light intensities in physical units (sun illuminance in lux, local lights in lumen). S36 uses a
  fixed or per-time-of-day exposure from a game parameter (goldens use the fixed exposure of §17.6); auto exposure with
  per-game limits (S37) handles day and night. Both material models use it.
- `WorldTime` fields: `double timeOfDay` (hours), `int64 day`; `Weather` fields: `double cloudiness, rain, fogDensity,
  snowCover, waveAmplitude, waveSteepness; Vec2 wind; Vec2 cloudOffset` (accumulated and wrapped by the host,
  replicated, so clouds never jump), each blended slowly by game systems.
- **Sun and moon**: cascaded shadow maps (multi-view culling, §10.2; cascade count and resolution per quality tier);
  far content casts shadows from its impostors in the far cascades; far cascades update every N frames.
- **Far terrain shadows**: beyond the cascades, a heightfield shadow (horizon) map computed in compute from the coarse
  terrain levels for the current sun direction, updated incrementally as the sun moves; mountains cast long shadows
  over the far terrain at sunrise and sunset.
- **Local lights** (point and spot; color, intensity, range, optional flicker): clustered forward light culling with a
  light budget per quality tier. **Shadowed local lights**: the nearest/most important lights (budget per tier,
  default 0-1 on Low, 4 on Medium, 16 on High) render into a shadow atlas (point lights as cube faces); the rest are
  unshadowed. The atlas is **cached**: a face is re-rendered only when its light or a caster in its range changed, with
  static and dynamic casters in separate layers and a per-tier budget of re-rendered faces per frame.
  Lights come from entities (`LightSource` component, presentation, its on/off state replicated with its owner) and
  from placed pieces (per-chunk light lists in the 1p data, §12.5, so a base with hundreds of torches creates no
  entities).
- **Indirect light: ambient + ambient occlusion only.** Sky ambient (from the sky color, §10.11, per-space ambient
  settings for dungeons and caves) modulated by screen-space ambient occlusion. Accepted limitation: interiors of
  player-built houses do not darken by themselves (caves and dungeons are separate spaces with their own ambient,
  §12.4); games use per-space ambient, light placement and
  exposure to compensate. The lighting code keeps indirect light behind one interface (`AmbientProvider`), so probe-
  or occlusion-based methods can be added later without changing materials or passes.
- **Reflections**: a sky environment map (re-rendered as the time of day changes) for specular everywhere;
  screen-space reflections on water (§10.9). No local reflection probes.
- **Night**: moonlight, minimum ambient and exposure limits are parameters; how dark the night is is game policy.
- Weather parameters feed fog, sky, wetness and snow in the material shaders.
- Nothing far away bakes lighting (§10.4).
- Not provided: global illumination with bounce light, contact shadows, foliage translucency (can be added as options
  later).

### 10.8 Materials [planned: S18 (data model), S43 (full set)]

The engine supports two shading models; a game uses either or both (per material, R-MECH-1):

| Model | Data | Normals |
|---|---|---|
| **Flat colors** | per-vertex color or palette index (one palette texture per game), roughness/metalness per palette entry | flat (face normals; vertices duplicated per face at mesh import) |
| **Smooth textured** | low-resolution albedo texture (point or linear filtered, game choice), optional normal map, roughness | smooth vertex normals, optional normal map |

- Common: alpha-tested masks for foliage (alpha to coverage), emissive color (fires, glowing objects), wind
  parameters, two-sided flag; wetness and snow from weather applied by the shader.
- Material parameters are a data file type (`data/<game>/material.json`, S43); `WorldSetup::renderAssets` reads them and
  calls `registerMaterial` (before S43 materials are registered in code). Meshes reference materials by name; textures
  go into texture arrays per format to keep draws batched (one material buffer, indices per instance/mesh, no
  per-material descriptor sets).
- Impostor baking (§10.4) bakes albedo, normal and material parameters for both models, so far content relights
  correctly.

### 10.9 Water [planned: S39]

- **Sea level**: one global water plane per open-world space at `SpaceTemplate::seaLevel` (integer height units;
  enclosed spaces have none unless set). Everything below sea level is water, including pits dug inland (accepted,
  as in Valheim): there are no water extents, no lakes at other levels and no fill logic; digging below sea level
  simply exposes water. Rivers are channels carved below sea level by the world plan (§12.1): there is no sloped water.
  The terrain is a height field, so there is no water under solid ground.
- **Waves**: a fixed set of Gerstner waves per space (`SpaceTemplate::waves`: directions, wavelengths, speeds); weather
  scales only amplitude and steepness (a replicated value in `Weather`, blended slowly). The waves are a function of
  position and the simulation clock `world.time()` (never `WorldTime`, which games may skip). `world.waterHeight(space,
  x, z)` (double) inverts the horizontal displacement with a fixed number of iterations. The shader gets each wave's
  phase at the camera origin (computed in double on the CPU, modulo 2 pi) and adds only the camera-relative part; CPU
  and GPU agree within a tolerance. On clients `world.time()` is the estimated host time (§13.4).
- **Buoyancy** (needs S28 physics): bodies with `Buoyancy {sample points, volume}` sample `waterHeight` in phase 5 and
  get forces through `PhysicsWorld`.
- **Exclusion volumes**: meshes (e.g. a boat's interior) that mask the water surface and the underwater test, so hulls
  stay dry.
- **Rendering**: camera-centered clipmap grid displaced in the vertex shader (to the horizon at High), refraction from
  the resolved scene color and depth, sky and screen-space reflections (per tier), shore foam from the terrain depth.
- **Underwater**: when the camera is below the surface (and not in an exclusion volume): underwater
  fog and absorption, the surface seen from below, optional caustics, a post effect at the water line.

### 10.10 Particles and effects [planned: S41]

- GPU particle system: emitters (data files: spawn rate, lifetime, velocity, size and color curves, texture atlas,
  blend mode, lit or unlit) simulated in compute shaders, rendered sorted back to front per emitter, budget per
  quality tier.
- Effects are presentation: emitters are attached to entities (`ParticleEmitter`, `Replicate::Never`) or started by
  cosmetic events (§13.4); they never affect simulation state.
- Rain and snow are screen/camera-local particle fields driven by `Weather`.

### 10.11 Sky and clouds [planned: S36; volumetric clouds: S42]

- Physically based atmosphere (sun and moon, time of day from `WorldTime`), stars at night; the sky color also drives
  ambient light and fog color.
- **Clouds**, two mechanisms (game and quality tier choose, R-MECH-1):
  - **2D cloud layer**: a textured layer at a fixed altitude driven by `Weather` (cloudiness, `cloudOffset`); cheap,
    Steam Deck tier and fallback; its coverage texture is projected as the cloud shadow.
  - **Volumetric clouds (optional)**: a cloud band between two altitudes per space (e.g. 600-1500 m; mountains reach
    into and above it). Density from tiling 3D noise textures shaped by `Weather` (cloudiness, `cloudOffset`) and
    optional per-region coverage maps from the world plan. Ray-marched in a compute pass at reduced
    resolution with temporal reprojection, **depth aware** (the march stops at the scene depth, so clouds wrap around
    mountains and lie in valleys), lit by sun/moon with a short light march and ambient from the sky.
  - **Inside and above the clouds**: the same density field feeds the froxel volumetric fog near the camera (§10.5.1),
    so hiking up a mountain walks through the cloud as fog and comes out on top with the cloud layer below; seen from
    above, the clouds are rendered with the same ray march looking down, and terrain and far content below are
    attenuated by the clouds' transmittance.
  - **Cloud shadows**: a top-down transmittance map over the camera region (updated incrementally) darkens terrain,
    instances and water and is used by the volumetric fog; beyond the map, cloud shadows fade to the average
    transmittance.
- Clouds are presentation: they never affect simulation state; `Weather` (replicated) makes all peers see the same
  coverage, the exact shapes may differ slightly between peers (not replicated).

### 10.12 View distance

At the High tier terrain and forests are visible to the horizon (20+ km): terrain through the coarse levels and the
pyramid (§12.2), forests through the region quadtree with impostors at every level (§10.4), water to the horizon.
Lower tiers reduce the far-content budgets and distance and hide the edge with fog (§17.9).

## 11. Physics (Jolt) [planned: S28, extended in S29]

- API (no Jolt types), accessed with `world.physics()`:

```cpp
enum class MotionType { Static, Kinematic, Dynamic };
struct Collider { ShapeType shape = ShapeType::Box; Vec3 halfExtents = Vec3::Zero(); std::string mesh;
                  uint8_t layer = 0; };                             // box/sphere/capsule/mesh;
                                                                    // mesh: a name from WorldSetup::collisionAssets
struct RigidBody { uint32_t body = 0; MotionType type = MotionType::Dynamic; double mass = 1;   // body: set by engine,
                   Vec3 linearVelocity = Vec3::Zero(), angularVelocity = Vec3::Zero(); };  // not in visit; velocities
                                                                                           // written back in phase 5
class PhysicsWorld
{
   public:
    bool raycast(SpaceId space, const Vec3& from, const Vec3& to, RayHit& hit) const;
    std::vector<Entity> overlapSphere(SpaceId space, const Vec3& center, double radius) const;   // sorted by index
    void addImpulse(Entity e, const Vec3& impulse);
    void setVelocity(Entity e, const Vec3& v);
    Vec3 velocity(Entity e) const;
};
```

- **Frozen bodies**: bodies of non-`Simulated` entities are switched to `EMotionType::Static` (bodies are created with
  `mAllowDynamicOrKinematic = true`; velocities stay in `RigidBody`) and switched back when the entity becomes
  `Simulated` again, in entity order in phase 5. A frozen body still collides but is never moved by Jolt.
- `PhysicsWorld` (`engine/physics/`) owns one `JPH::PhysicsSystem` per space, layers (static, dynamic, character,
  trigger) and a job system adapter on the frame `JobPool`. Jolt's results may depend on the thread count (accepted,
  §8.6; `ScenarioWorld` uses a fixed worker count). Public API without Jolt types.
- **Terrain**: one `HeightFieldShape` per chunk from the integer heights (§12.2) with shared border samples
  (`mBitsPerSample = 8`).
  - Diagonal: the diagonal of §12.2 is fixed from S20 and never changes (the content hashes depend on it). S28 confirms
    that Jolt splits cells along it; if Jolt splits the other way, the engine rotates each chunk's height-field body by
    90 degrees about y and reorders the samples to match.
  - Accuracy: a test compares collision height and `terrainHeight` within Jolt's quantization error (at most the block
    height range / (2^bitsPerSample - 1), measured with the edit margin; the test uses that bound).
  - Edits: terrain edits patch the chunk's shape synchronously in the requesting step (§12.2) with
    `HeightFieldShape::SetHeights` on the edited block-aligned rectangle; a full rebuild is only the fallback. The shape
    is created with the chunk's height range padded by `SpaceTemplate::terrainEditMargin` (default +-64 m; an edit that
    leaves the range triggers the full rebuild; API to be confirmed in S28).
  - Borders: the sample count is padded to a multiple of Jolt's block size (padding samples `cNoCollisionValue`), so
    border samples are patchable; an edit on a shared border patches both chunks' shapes in the same step.
  - Enclosed spaces (§12.4) use static mesh colliders of their authored geometry.
- **Tier 1 and 1p colliders**: per chunk, static colliders are grouped into cells (e.g. 8 x 8 cells of 32 m); each cell
  is one static body with a `StaticCompoundShape` (it has a bounding tree, so queries stay fast with 10^4 sub-shapes).
  Piece edits and promotions apply in the requesting step (state, tombstone, render instance, replication); the
  affected cell's shape is rebuilt synchronously at the phase flush (cost measured in S30; a shape filter is the
  planned fallback if it is too slow, §20). S28 measures query cost against `MutableCompoundShape` and keeps the faster
  option.
- **Bodies** belong to entities with `Collider` + `RigidBody`, created and freed from the structural log. If
  `Transform` differs from the pose physics last wrote, game code moved it: the body is teleported. Kinematic children
  are moved in phase 5 from the parent's pose of this step.
- **Contacts** are collected per worker thread, sorted by (entity index pair, sub-shape IDs) after the step and emitted
  as `ContactEvent` in that order.
- **Characters**: `JPH::CharacterVirtual` for players and important characters (this is `characterMoveSystem`; with
  `RideFrame` it uses the platform pose written by this step's Jolt update; after the step, entity order,
  `CharacterVsCharacterCollisionSimple`). A cheaper mover for many creatures is added when a game needs it. Every
  character has `Collider` + `RigidBody{Kinematic}`: on its owner this body is the `CharacterVirtual` inner body
  (`mInnerBodyShape`); on the host for client-owned characters, and on clients for remote characters, it is the
  kinematic proxy. Raycasts and overlaps hit characters through this body on every peer.
- **Clients** create kinematic proxy bodies for replicated entities with `RigidBody` inside
  `SpaceTemplate::clientPhysicsRadius`. Proxies are moved every step to the interpolation buffer's pose for that step
  (the same timeline as `RenderPose`), with velocities derived from it.
- **Pushing** [S29]: when an owner's mover contacts a proxy of a host-owned dynamic body, it sends the contact impulse
  as an unreliable `PushImpulse {NetId body; Vec3 impulse, point}`; the host applies it in phase 5 with `addImpulse`.
  Whether a game enables it is policy (`WorldConfig::clientPush`, default true; `samples/games/world` uses it).
- **Save/load**: Jolt state is not saved. Loading recreates bodies from `Collider`, `RigidBody` (with the saved
  velocities) and `Transform` through the structural log, like a chunk reload (§7.6); characters are recreated from
  their components. Sleeping state and contact caches start fresh, so a loaded world continues equivalently, not
  bit-identically (§8.6). Jolt assigns body IDs itself.
- From S28 every CI job that builds the engine builds Jolt (§5.3).

## 12. Spaces, chunks, terrain and streaming

### 12.1 Generation pipeline and chunk content [planned: S23]

- Per space: **world plan** (global, low resolution: biomes, river channels carved below sea level (filled by the sea,
  §10.9), roads, sites of authored places; computed once from the seed) -> **chunk generation** (terrain heights, tier-1
  placement, grass masks) -> **prefab stamps** (authored places from prefab data files: terrain edits, placed pieces,
  entities, loot, entrances to other spaces). Generators and prefabs are game content (`samples/games/<name>/content/`);
  the pipeline and stamping are engine mechanisms (`engine/content/`). A space can also be fully authored (prefabs only,
  e.g. a dungeon).
- S23 scope (before streaming): chunks are generated on demand like the S20 tiles; the host keeps per space the set
  of chunks whose prefab entities are stamped (saved; from S24 a flag in the chunk record); in phase 7 it
  stamps, synchronously, **on the host only** (clients receive them by replication, §13.2) and in `ChunkId` order,
  every unstamped chunk within `fullRadius` of a player, independent of the visual tile cache. Stamped pieces are added
  in S30 (which regenerates the committed content hashes once).
- Deterministic and identical on every machine (R-CONTENT-1). Each tier-1 instance has a stable
  `InstanceKey {uint32 generatorSlot, uint32 counter}`; tombstones name keys.
- **Generator contract** (needed for far sets, §10.4): (a) terrain height is point-evaluable from the world plan
  (`height(x, z)` without neighboring tiles); (b) a tier-1 candidate is a pure function of (generator slot, chunk,
  counter): position and acceptance depend only on the world plan and point-evaluated terrain, never on other candidates
  (spacing from jittered grids); counters are ordered by importance (§10.4); (c) **sites** (authored places) are part of
  the world plan with footprint, height patch and exclusion mask: `height(x, z)` includes the site height patches,
  candidate acceptance tests the exclusion masks, and a site's pieces provide a far mesh generated per region from the
  plan's site list without chunk generation. Stamped terrain and stamped pieces are base content, not modifications;
  only later edits are. Piece modifications are: placed pieces (full records), a tombstone set of stamped piece keys,
  and state overrides for stamped pieces by key; the region summary lists the tombstoned stamped keys for the site far
  meshes. A CI test [S27] generates a far set directly and compares it with the filtered full chunk, including a chunk
  with a site.
- **Space instances**: the plan of space 0 lists all space instances; every other instance computes its own plan with
  `plan(version, instanceSeed, template)`, whose `spaces` list is empty. The list: `SpaceId` = index in the plan's space
  list (0 = overworld), template (`SpaceTemplate`), instance seed = hashCombine(world seed, site key); `SpaceLink::target` is
  filled at stamping from the plan; the list is part of the pinned plan version. Every space instance comes from the
  plan; spaces are not created or deleted at run time (§20).
- **Content hash** per generated chunk over the integer output; a CI test compares generated chunks with committed
  hashes (committed hashes are never regenerated for a shipped version; a hash change for an existing version is a
  bug); it runs on the Clang and Windows/MSVC (shared libraries) CI jobs that exist from S04.
- **Generator versions** (§15.3): a world pins the versions it was created with.

```cpp
class Generator   // engine/content/Generator.h [planned: S23]; implemented by games
{
   public:
    virtual ~Generator() = default;
    virtual uint16_t version() const = 0;
    // `version` is the pinned plan/terrain version for plan() and height(), the chunk's placement version for
    // candidate() and grassMask() (§15.3)
    virtual WorldPlan plan(uint16_t version, uint64_t seed, const SpaceTemplate& space) const = 0;
    virtual int32_t height(uint16_t version, const WorldPlan& plan, int64_t x, int64_t z) const = 0;
                                    // x, z: integer metres in the space; result: integer height units (1/1024 m)
    virtual bool candidate(uint16_t version, const WorldPlan& plan, ChunkId chunk, uint32_t slot, uint32_t counter,
                           Candidate& out) const = 0;   // false: rejected
    virtual uint32_t candidatesPerChunk(uint32_t slot) const = 0;
    virtual uint32_t keptCount(uint32_t slot, int level) const = 0;   // far level L keeps counter < keptCount
    virtual void grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const = 0;
};
struct WorldPlan { std::vector<Site> sites; std::vector<SpaceInstance> spaces;
                   PlanGrid biomes, rivers;                             // integer grids at plan resolution
                   PlanGrid cloudCoverage;  // [S42] presentation only, not in the content hash
                 };
struct Candidate { uint16_t meshSlot; int32_t x, z;   // 1/256 m, chunk relative; y = the terrain surface of
                                                      // §12.2 at (x, z) from the base-spacing height() samples
                                                      // (integer barycentric, 1/1024 m, rounded toward -inf)
                   uint16_t yaw;                         // index into the integer sin/cos table
                   uint16_t scale;                       // 1/1024
                   uint32_t flags; };                    // re-snap or tombstone on terrain edits, cosmetic
// The engine converts exactly to double and maps meshSlot to MeshId through the generator-slot name table (§15.2);
// the content hash covers these integers.
void GeneratorList::add(GeneratorId id, std::unique_ptr<Generator> generator);
// Prefabs: data/<game>/prefabs/<name>.json with terrain patches, pieces, entities (components as visit JSON),
// SpaceLinks and (enclosed spaces) static geometry mesh names; loaded by the generation pipeline (one file per
// prefab version, §15.3; not the one-file-per-type data files of §14).
```
- A chunk holds: terrain tile(s), tier-1 arrays per mesh, grass mask, placed pieces, static colliders,
  plus its **modifications** and the serialized entities of §9.6. **Modifications are state, not a log**: the current
  edited height tile (stored, saved and sent as the difference from the chunk's base, zero where unedited, zstd
  compressed), the tombstone set, the edited grass mask and the piece modifications. Brushes exist only on the
  host; applying one updates the state. Every chunk has a modification sequence number (uint32, saved),
  incremented by every applied modification. The modification state (plus the stored base of retired versions, §15.3)
  is what the save stores and what a client receives; entity groups stay on the host.

### 12.2 Height-map terrain [planned: S20; terraforming: S22]

- **Integer heights**: int32 with the fixed unit `kHeightUnit` (1/1024 m, `constexpr double` in
  `saiga/core/math/ChunkId.h`, no offset). Tiles, pyramid, CPU queries, physics, saves, content hashes, network
  rectangles and the generators use the same integer values. GPU tiles are R32F, relative to the tile's minimum height
  (exact for every tile whose height range is below 16 km); tile meshes have skirts (as the prototype CDLOD). The
  terrain is a pure height field: no holes, caves or overhangs (those are separate spaces, §12.4, or meshes placed on
  the terrain).
- **Terrain surface**: every consumer (render mesh, Jolt collision, `terrainHeight`, candidate y, re-snapping) uses the
  same triangulation: two triangles per cell with the fixed diagonal from (x, z) to (x+1, z+1) in sample coordinates,
  CPU consumers interpolate with integer barycentric weights, rounded toward negative infinity (double queries
  interpolate the same triangles exactly); Jolt and the GPU agree within their quantization and float error (§11).
- **Scope before S23** (S20, S22): tiles of all levels are generated on demand from `SpaceTemplate::prototypeHeight`
  (the prototype noise evaluated per point by `samples/games/world`, continuing beyond the prototype's 8 km; integer
  format; R-CONTENT-1 applies from S23) on the streaming pool, by camera distance, with an LRU budget per level; the
  chunk record of §12.1 is introduced with only the edited height tile; `World::save()` blobs contain the edited tiles.
  Simulation height queries (`terrainHeight`, mover, edits) evaluate the prototype height hook exactly if the base tile
  is not resident (the asynchronous path is visual only). Before S26 (S22) the host sends every live modification to
  every client, and clients keep all edited tiles in memory (never evicted); modification snapshots and the reload rules
  of §13.2 come with S26. Simulation radii and the chunk store follow in S24 and S25.
- The base spacing is a `SpaceTemplate` parameter (1 m with 257 x 257 tiles, or 2 m with 129 x 129 tiles; shared
  borders); `samples/games/world` uses 1 m (the engine default stays 2 m); coarser levels double the spacing (level L
  covers 2^L chunks) and are streamed by distance; a coarse whole-space pyramid (e.g. 32 m) is always resident on CPU
  and GPU. GPU: texture-array atlas with a page table.
- CDLOD refines a node only if its tiles are resident. The view distance does not depend on the load radius.
- **Terraforming** (engine mechanism; whether and how a game allows it is policy): `struct TerrainBrush { enum Op {Add,
  Subtract, Flatten, Smooth} op; Vec3 center; double radius, strength; };` `void World::editTerrain(SpaceId space, const
  TerrainBrush& brush);` [S22]. Edits are chunk modifications with an integer result. `editTerrain` computes the integer
  result and patches the collider (from S28; before S28 the resident height tile used by the kinematic mover)
  synchronously in the requesting step (between steps immediately; during a step as a command at the phase flush) and
  replicates the edit as a live modification (§13.2). The region modification summary [from S27] (tombstoned keys,
  coarse height patches, piece far-mesh input) is updated synchronously with the edit (§10.4); nav patch [S45], far
  meshes [S30] and impostor rebuilds [S27], coarser levels, pyramid GPU copy and GPU uploads follow asynchronously
  (visual and budgeted per frame). The edit reports the affected tier-1 instances and pieces: tier-1 instances are
  re-snapped or tombstoned according to a flag of their generator slot (by the later of S22 and S23, §22), pieces are
  handed to the game (policy).
- `world.terrainHeight(space, x, z)` (double) is exact everywhere: it uses the resident base-level tile (edits are
  visible from the requesting step on); otherwise it evaluates the three samples of the containing triangle exactly
  (`Generator::height` plus the chunk's modification state, loaded synchronously as for edits of non-resident chunks,
  §12.3; before S23 the prototype height hook). The pyramid is used only for rendering and far content (its CPU copy is
  updated synchronously with the edit, its GPU copy budgeted).
- Terrain, grass, water, sky/fog and culling shaders work region/camera relative.

### 12.3 Streaming [planned: S24]

| Radius around each player (defaults; `SpaceTemplate`) | Content |
|---|---|
| simulation (e.g. 384 m) | entities tick, physics active |
| full (e.g. 1 km) | base-level tiles, tier-1/1p instances, colliders, chunk entities |
| region levels | coarser tiles and far content by distance, to the horizon |
| always | coarse terrain pyramid, world plan |

- Requests are made in phase 7 from player positions (per space). Generation, loading and meshing run on the
  streaming pool. **Prefetch**: generation of a chunk starts when it enters `fullRadius + prefetchMargin`
  (`SpaceTemplate`, default 256 m), so its results are usually ready when it enters `fullRadius`.
- A chunk is unloaded when it leaves `fullRadius + unloadMargin` of all players.
- S24 scope: modification states of non-resident chunks stay in memory and in the `save()` blob; entities are not
  unloaded before S25.
- **Results apply when ready**: colliders, height tiles, modifications and chunk entities are applied in the first step
  after the streaming pool delivers them, on the host and on clients; nothing waits for them. While the collision chunk
  below an entity is not resident, its mover holds it (no gravity, no movement) on the peer that owns it, and host
  bodies there stay frozen. `ScenarioWorld` loads synchronously (results apply in the requesting step), so scenarios
  stay reproducible (§8.6).
- **Edits of non-resident or pending chunks**: `editTerrain`, piece edits, tombstones and promotions on a chunk that
  is not resident build the base rectangle they need synchronously on the main thread (heights are point-evaluable,
  §12.1; from S25 the stored base or modification state is loaded synchronously) and update the chunk's modification
  state and sequence number. A streaming result carries only base content and is applied together with the chunk's
  modification state as it is at that moment (re-patched if the sequence number changed since the request), so a
  pending result never undoes an edit. The synchronous load counts against the step budget (S25 measures it).
- The number of steps in which a player stood on a missing chunk is a statistic
  (`streamingStalls`, reported by `host_step` and the S27 flight). **Arrivals** (join, respawn, teleport outside the
  resident set, space transition): `World::requestArrival(entity, space, position)` preloads the target island; the
  entity stays held (frozen, invisible) until the island is complete and is then moved in phase 1 of the next step.
  **Clients** queue live modifications of a chunk until the host's modification snapshot is applied (§13.2); on a
  client a chunk's collision counts as resident only after that snapshot is applied, until then the owner's mover
  holds the character.
- GPU uploads are visual only, budgeted per frame, and only for the local camera's space and surroundings.
- **Frozen entities**: the engine tag `Simulated` (a flag in the slot state, not a component; not saved, rebuilt on
  load) is maintained in phase 6 from the simulation radii, `alwaysSimulate` (simulated while resident) and `Player`,
  re-evaluated round-robin together with the tick interval (§9.2). World-state and space-state entities are always
  simulated; on clients every replicated entity is `Simulated`. Inside phases 2-5 `world.view<...>()` skips entities
  without the flag (`ownedBy` too); `world.viewAll<...>()` is the explicit opt-out (replication, extract, save,
  streaming); `behaviourSystem`, event delivery and physics activation use the same tag. Un-freezing calls no
  `onCatchUp`, `sinceTick` continues. Before S24 every entity is `Simulated`; the API (`viewAll`, the tag) exists from
  S08.
- The resident chunk set is not saved: loading rebuilds the chunks around the players present in the loaded state
  synchronously before the first step (none for a save directory from S33, §15.4; joining players arrive through
  `requestArrival`).
- Host load: each scattered player adds a simulation island; the budgets of §17.10 assume up to 4 islands, and games
  may limit how far players spread (policy). `maxSimulationIslands` is a soft limit: exceeding it emits
  `IslandLimitEvent {count}` and a statistic, the engine keeps simulating; with `WorldConfig::hardIslandLimit` set,
  `requestArrival` returns false. A Steam Deck can host small sessions (default: 4 players, 2 islands).

### 12.4 Spaces and transitions [planned: S21]

- The open world is a height field. Caves, dungeons and interiors are **separate spaces** (enclosed scenes): authored
  from prefabs or generated, with their own chunk grid, static mesh geometry and colliders, ambient light and fog
  settings; `SpaceKind::Enclosed` has no sky and no height map.
- An **entrance** connects two spaces: an entity with `SpaceLink {SpaceId target; Vec3 arrival; Quat
  arrivalRotation}` (a prefab-stamped entity from S23; a piece that is an entrance is promoted to an entity).
  Interacting with it starts a **transition**:
  1. the host (`World::requestTransition(player, link)`, called by the game; the engine offers an optional
     `spaceLinkSystem` (phase 4, host) that calls it for an `InteractEvent` whose target has `SpaceLink`) starts an
     arrival (§12.3) in the target space;
  2. the player's peer shows a **loading screen** (game UI; progress = resident fraction of the required chunks,
     colliders and GPU data around the arrival point);
  3. when the host's arrival completes (§12.3), the host moves the player with `moveToSpace` without waiting for the
     client; the client keeps its loading screen and its mover holds the character until its own chunk snapshots are
     applied (§12.3); the entity is frozen and invisible during the transition. For a
     client-owned player `moveToSpace` sends a reliable `OwnerCommand{MoveToSpace}` and increments the entity's teleport
     counter (§13.3); the host switches grid, interest and physics system immediately and drops poses with an older
     counter; the owner applies the command in phase 1.
- Players can be in different spaces at the same time; the host simulates every space that contains a player (each
  space is a simulation island, §12.3). Spaces without players are unloaded like chunks (their state goes to the chunk
  store; entities are unloaded, not destroyed, §9.6).
- Each peer renders only the space of its own camera; switching the space swaps the resident render data, no
  rendering of two spaces at once.
- **Before S23** (S21): there is one space instance per `spaceTemplates` entry (`SpaceId` = template index, seed =
  hashCombine(world seed, index)); authored enclosed geometry comes with prefabs in S23, its colliders in S28. **Before S24**:
  there is no arrival mechanism; `requestTransition` completes in the next step (simulation tiles are generated
  synchronously on demand, §12.2); the loading screen waits for the visual data only on the local peer and does not
  affect the simulation.
  **Before S25**: no space unloading; the second space (height map or empty) keeps all its chunks resident.

### 12.5 Placed pieces (tier 1p) [planned: S30, extended in S31]

- Per chunk arrays of pieces: mesh, transform (double in the simulation data, region-relative float for rendering),
  `InstanceKey`, owner (`characterId`), an optional light (piece types with a light source, e.g. torches, §10.7), and a
  small state record (health, flags, game data as a fixed-size `visit`-able struct per piece type).
- Placement, removal and state changes are chunk modifications applied on the host and replicated live.
- Rendering through instance sets (updated ranges), colliders through the chunk's compound shape, far representation
  per region rebuilt after edits.
- Engine queries for game rules: neighbors and contacts of a piece, pieces in a box; structural stability is game
  policy (§2.3).

```cpp
struct PieceKey { ChunkId chunk; InstanceKey key; };
template <class State> void PieceTypeList::add(PieceTypeId id, const char* name, const char* mesh,
                                               const PieceTypeInfo& info);   // collider, light, state struct
PieceKey World::placePiece(SpaceId space, PieceTypeId type, const Vec3& position, const Quat& rotation,
                           uint64_t ownerCharacter);   // owner: Player::characterId
void World::removePiece(PieceKey key);
template <class State> State& World::pieceState(PieceKey key);             // marks the piece changed
std::vector<PieceKey> World::piecesInBox(SpaceId space, const Vec3& min, const Vec3& max) const;
std::vector<PieceKey> World::pieceNeighbours(PieceKey key) const;           // touching pieces
template <class T, class... C> Entity World::promote(ChunkId chunk, InstanceKey key, C&&... initial);   // tier 1
template <class T, class... C> Entity World::promote(PieceKey piece, C&&... initial);                    // tier 1p
PieceKey World::demote(Entity e, PieceTypeId type);                         // entity -> placed piece
void World::tombstone(ChunkId chunk, InstanceKey key);                      // tier 1 (S23)
void World::editGrass(SpaceId space, const Vec3& center, double radius, double density);   // S31: edited grass mask
```
- A piece with behavior (a crafting station with a timer) is promoted to an entity while it is active (game decides).

## 13. Multiplayer (co-op)

### 13.1 Model [planned: S15 transport, S16 skeleton, S32 and S33 production]

- **Host-authoritative state replication.** The host runs the full simulation; clients receive replicated state and run
  presentation and the systems registered `RunOn::Client` / `RunOn::Both`.
- Clients are trusted (§2). Single player is a host without clients (same code path).
- **Skeleton (S15, S16)**: `Transport` interface and `LoopbackTransport`, `NetOwner`, `Replicate` modes, `RunOn`
  filtering, `ownedBy`, `NetId` mapping, reliable spawn/destroy, `Despawn`, `OwnerCommand` with the teleport counter,
  `RenderPose` and the interpolation buffer, full-state snapshots of all entities (§13.4), host + client scenario
  variant.
- **With the features that need them** (over `LoopbackTransport`): the world handshake and chunk snapshots (S26), live
  terrain edits (S22), tombstones (S23; sent like the S22 live terrain modifications to every client until S26), pieces
  (S30).
- **S32** interest radius, snapshot rate, cosmetic events, loss and reordering; **S33** UDP transport, sessions.

```cpp
enum class Reliability { Unreliable, Reliable };
enum class Channel : uint8_t { Control, State, ChunkData };    // independent channels; reliable messages ordered per channel
struct TransportEvent { enum Type { Connected, Disconnected, Packet } type; uint8_t peer; Channel channel;
                        std::vector<uint8_t> packet; };
class Transport   // engine/net/
{
   public:
    virtual ~Transport() = default;
    // large reliable messages (ChunkData, Control) are fragmented by the transport; unreliable packets are <= 1200
    // bytes (§13.4); a chunk snapshot never blocks Control messages
    virtual void send(uint8_t peer, Channel channel, const std::vector<uint8_t>& packet, Reliability r) = 0;
    virtual bool poll(TransportEvent& event) = 0;   // non-blocking
};
// LoopbackTransport (S15):
static std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>>
LoopbackTransport::createPair(double dropRate = 0, int delaySteps = 0, uint64_t seed = 1, int maxReorderSteps = 0);
// maxReorderSteps: parameter added in S32. Each reliable packet gets a seeded extra delay of 0..maxReorderSteps,
// raised if needed so that it is not delivered before the previous reliable packet of its channel; unreliable packets get an independent seeded extra delay of
// 0..maxReorderSteps, so a state update can arrive before an earlier reliable spawn. dropRate applies to unreliable
// packets only; reliable packets are never dropped.
// delay counted in poll() rounds of the receiver (one round per step); drops come from a seeded Rng (reproducible)
void LoopbackTransport::disconnect(uint8_t peer);   // [S33] both ends receive Disconnected (tests)
```

### 13.2 The world over the network [planned: S26; live terrain edits: S22]

The world is generated from the seed on every machine (§12.1). The handshake sends the seed and the pinned generator
versions and compares the protocol version and a hash of the schema table (§15.2); a mismatch rejects the join with a
message. Per chunk or region level a client loads, the host sends the chunk's placement version (region level: the
summary's version list) and the modification state at that level (chunk modifications or region summary); while the
client has it loaded, new modifications are sent as reliable messages. Base snapshots are sent only for retired
generator versions (§15.3). Generated content is not verified at run time; CI tests that generation reproduces the
committed content hashes on every compiler (§12.1). Clients generate terrain, tier 1 and stamped pieces locally but
never stamped entities: those exist only on the host and are replicated. A modification snapshot carries the chunk's
modification sequence number; every live modification carries the number it produces. A client queues live modifications
until the host's modification snapshot (not only its local base) is applied, drops those with a number <= the
snapshot's, and treats a gap as an error that re-requests the snapshot. Live terrain modifications carry the integer
result as a difference rectangle (new heights minus base), never the brush; clients do not evaluate brushes.

### 13.3 Ownership [planned: S16; moving platforms: S40]

- Every replicated root entity has `NetOwner {uint8_t peer}` (host = 0); ownership is defined on roots, descendants
  inherit their root's owner, `NetOwner` on a child asserts. Entities without `NetOwner` belong to the host. Each
  player's character is owned by its peer.
- Components registered `Replicate::OwnerAuthoritative` (`Transform`, velocity, `PlayerInput`, ...) are written by the
  owner and sent to the host; the host does not write them and does not send them back to the owner. On the host, a
  client-owned character is kinematic: `Transform` and the kinematic body use the same pose, the latest received pose
  extrapolated to the current step by its velocity (clamped); the host's interpolation buffer feeds only `RenderPose`.
  Owners send poses every step (unreliable, stamped with the estimated host step, §13.4).
- Effects of the host on an owned entity (impulse, teleport, attach) are reliable `OwnerCommand` messages applied by the
  owner in phase 1. Every `OwnerCommand` that sets the pose (teleport, arrival, `MoveToSpace`) increments the entity's
  teleport counter (`uint8_t`, sent with every owner pose); the host applies the pose immediately and drops poses with
  an older counter.
- **No runtime ownership transfer**: ownership is set when an entity is created (player characters at join,
  `setOwner`) and never changes. Vehicles, platforms and carried objects stay host-owned; a driving client sends its
  input (`PlayerInput`, `PlayerAction`) and the game's vehicle control system on the host moves the vehicle, so steering
  has one round trip of latency (§20). A client holds its character while the collision chunk below it is not resident.
- **Moving platforms** (boats, rafts, lifts) [planned: S40]: a character standing on a moving body gets
  `RideFrame {Entity platform; Vec3 localPosition = Vec3::Zero(); Quat localRotation = Quat::Identity();}`
  (registered `OwnerAuthoritative`, saved; set and cleared by the mover from ground contact).
  - State: while `RideFrame` is set, the authoritative state of the character is its pose relative to the platform.
    It is sent relative to the platform; each peer derives `Transform` from it and its own platform pose (host: the
    platform's `Transform`; clients: the proxy, i.e. the interpolated render pose; platforms are always host-owned).
  - Movement: the mover runs the character controller in the platform's local frame (platform shape static, gravity
    and platform acceleration transformed into the frame). The local-frame move is followed by a world-space
    collide/sweep of the resulting pose against all other layers (platform body filtered out), so riders collide with
    docks, terrain, other boats and characters.
  - Leaving: ground contact on a non-platform body clears `RideFrame`. Entering and leaving the frame converts pose
    and velocity.

### 13.4 Replication [planned: S16 full state; S32 interest, rate, cosmetic events]

- Components are registered `Replicate::Never | Host | HostReliable | OwnerAuthoritative`; events
  `Replicate::Never | Cosmetic`. `HostReliable` is for large, rarely changing components (inventories, container
  contents): sent reliably on `Channel::State` with the spawn and whenever they change (in step order with the
  snapshots' host step), never in snapshots.
- **Snapshot size per entity**: the unreliable snapshot state of one entity (its `Host` and `OwnerAuthoritative`
  components) is at most 1100 bytes after serialization (quantized); serialization asserts and names the entity and the
  largest component otherwise. Each packet has a header of at most 16 bytes (host step, flags) including a 1-byte
  flag compressed/raw; the host sends a packet raw when compression does not shrink it, so a complete packet is at most
  1200 bytes in every case. A component that can exceed it is registered `HostReliable`.
- **Full-state snapshots**: at the snapshot rate (default 20 Hz; S16: every step) the host serializes each replicated
  entity once per snapshot with `visit` (quantized per field options; `Host` components and `OwnerAuthoritative`
  components, not `HostReliable`), shared by all clients; per client it concatenates the blobs of its interest set, leaving out the
  `OwnerAuthoritative` components of entities that client owns, splits the result at entity boundaries into packets of
  <= 1200 bytes (bound above), each carrying the host step, compressed with zstd on its own (or raw) and decodable alone,
  and sends them
  unreliable on `Channel::State`; a lost packet loses only its entities until the next snapshot.
  A lost snapshot is replaced by the next one; there are no per-client baselines or acknowledgements. When the network
  budget of §17.10 is exceeded, delta compression against acknowledged snapshots is the planned extension (§20).
- Interest: an entity enters a client's interest within `SpaceTemplate::interestRadius` of its player and leaves
  beyond `interestRadius + interestMargin` (`SpaceTemplate`, default 64 m) (spatial grid, closed over `Parent`;
  asserted `interestRadius >= clientPhysicsRadius`). Entities beyond it are not replicated; showing distant players
  (map markers) is a game message (policy).
- `Entity` fields are sent as `NetId` (0 = none) and resolved through `NetId -> Entity`; the client keeps unresolved
  `NetId`s and resolves them when the target arrives. Spawn (with `classId` and all replicated components) and destroy
  are reliable messages on `Channel::State`; a snapshot entry for an unknown `NetId` is ignored (the reliable spawn
  follows and the next snapshot carries the full state anyway). Leaving the interest set is a `Despawn` (no
  callbacks).
- Clients render remote entities from an interpolation buffer (~100 ms) that writes `RenderPose` (`Replicate::Never`,
  `Saved::No`).
- **Time**: `stepIndex` is local to each peer. Every snapshot carries the host step. A client keeps an estimate
  `hostStep = f(local stepIndex)` from snapshots and the round-trip time measured in local steps (never wall clock) and
  slews it (never jumps, except on a resync). The interpolation buffer, proxies and `world.time()` on clients are
  expressed in estimated host time; owner poses are stamped with the estimated host step; the host clamps extrapolation
  of received poses to 6 steps.
- Cosmetic events are sent unreliably to clients whose interest set contains their position and delivered to
  `RunOn::Client` systems.

### 13.5 Game objects on clients [planned: S16]

A client never calls `onSpawn`, `onLoad`, `onCatchUp`, `onDestroy`, `update` or the event handlers. It calls
`configure()` and `onReplicatedSpawn()` when an entity arrives and `onReplicatedDestroy()` when it is destroyed (not on
`Despawn`).

### 13.6 Transport [planned: S15 loopback; S33 network]

Internet sessions need NAT traversal: candidates in S33 are Steam Networking Sockets (relay, NAT traversal) and ENet
(LAN, tests), both behind the `Transport` interface (§13.1). `using MessageId = uint16_t;` with an `EngineMessage` enum
(game messages from 64); messages are sent with `world.send(peer, message)` on `Channel::Control` and handled by the
handler registered in `MessageList::add`. `ScenarioWorld` creates both worlds, connects them with
`LoopbackTransport::createPair`, steps the host world, then the client world, every step; `spawnPlayer` creates the
player entity on the host and calls `setOwner(player, clientPeer)`; S16 sends a snapshot every step.

### 13.7 Sessions [planned: S33]

Drop-in / drop-out: a joining client receives the seed, generator versions, world time and weather, and the state around
its player; a leaving player's character is exported (§15.4) and sent to the client, then its player entity is removed
(the game decides whether something stays behind, e.g. a grave). On `Disconnected` (no clean leave) the host removes the
player's entity group in the same step with the engine-internal unload command (no `onDestroy`; other clients receive a
destroy) after exporting the character from its own state into the next commit (not sent: the client is gone), and emits
`PlayerDisconnectedEvent {characterId, position}` so the game can leave a body or grave (policy). When the host opens a
world save directory (§15.1, from S33), it contains no player entities (the character files are authoritative); joining
players import their character again and arrive through `requestArrival`. `World::save()`/`load()` blobs keep player
entities (scenario round trips, §17.4). After a host crash the world and the characters roll back to the last commit
(accepted; the commit interval is game policy; on rejoin the loaded commit's copy of a character wins, §15.4). Host
migration is not supported; the host's world save is the world.

### 13.8 Determinism and multiplayer [planned: S16]

Reproducibility (§8.6) is a property of `ScenarioWorld` runs, used for tests, not for lockstep. Clients are not
deterministic replicas.

## 14. Content and data [exists: S11 (data files); planned: S23 (prefabs)]

- Tunable data: one JSON file per type, `data/<game>/<snake_case(T::name)>.json`, an object of named entries
  (`{"wolf": {...}, "bear": {...}}`); loaded in the `World` constructor from `WorldSetup::dataDirectory` (R-DATA-1;
  `engine/data/`: `DataTypeList`, `DataStore`, `readDataFile`/`readJsonFile` with the same rules for other files);
  `world.data<T>(name)` throws `DataError` on a missing name. `configure()` reads its parameters there; systems read
  them per step through a `DataRef<T>` field (§7.2). `world.reloadData()` re-reads the files and applies them only if
  all are valid and no entry was removed (entries are updated in place, so references and `DataRef`s see the new values;
  members set in `configure()` change when the objects are configured again: the `reload_data` tool reloads the world
  for that). Data are not saved. Worlds of one setup can share the parsed store (`WorldSetup::sharedData`).
- Prefabs (authored places): data files with terrain edits, placed pieces, entities, entrances and (for enclosed
  spaces) static geometry, with their
  components (`visit` JSON), loaded and stamped by the generation pipeline (§12.1; one file per prefab version, §15.3,
  not the data file mechanism above). Authored in an external tool or by hand; no
  editor is planned.
- Meshes: procedural or model files (glTF via assimp) [planned: S18 (static glTF import, flat-color face duplication,
  palette indices; `saiga/vulkan/scene/MeshImport.h`); skinned meshes: S34]; impostor and LOD baking as an offline tool
  (S27).

## 15. Persistence [exists: S11 (save blobs, schema, migration); later stages add their state as stated in their sections]

### 15.1 Save content and layout

- **World save = a directory**: a header (schema table, §15.2; seed; world id: random, created with the world; pinned
  generator versions), a world-state file
  (entity indices, generations and slot states, so `Entity` fields stay valid without remapping; all saved components
  (`Saved::Yes`) of resident entities through `visit`, excluding player entity groups from S33 (§15.4; until S33 players
  are saved here); the RNG state; pending events; step index and time; from S33 the latest export of every character that has played in the world,
  §15.4) and a
  **chunk store** [planned: S25] (region files with the
  chunk states: modification state, stored generated base, entity groups). Binary, zstd compressed.
- Dirty chunk states are written incrementally on the streaming pool; unloaded chunks live on disk, not in memory.
- **Commits** [planned: S25]: a save is a sequence of commits (the commit interval is game policy, default 5 min). At
  the commit step the main thread freezes, between steps, the world-state snapshot and immutable copies (copy-on-write)
  of all chunk states dirty since the last commit; the pool writes these chunk records with commit number c, flushes
  them, and only then writes the world-state file (temporary file, flush, rename) naming c; later changes go into c + 1.
  Chunk records are appended (never overwritten in place); loading ignores chunk records newer than that commit; the
  previous commit is kept as a backup; compaction runs on the streaming pool. The world-state snapshot is serialized
  between steps on the main thread into memory (budget: one step time, measured in S25), then compressed and written on
  the pool.
- `world.save()` / `load()` (byte blobs) remain for scenario worlds, tests and the world API's snapshots. [exists: S11]
  Blob layout (`engine/save/SaveFormat.h`): magic, format version, sections `HEAD` (engine version, step rate, step,
  time), `SCHM` (schema table), `ENTS` (slot table and free list), `COMP` (per type the values in the pool's dense
  order, so views iterate in the same order after a load), `EVNT` (pending events), `ACTN` (queued actions), `STAT`
  (RNG, `Persistent` counter); uncompressed (S25 compresses files). `save()` is deterministic. Not saved: `Saved::No`
  components, game objects (recreated), the grid, hierarchy order, child counts, tick evaluation, change flags (rebuilt
  by the load); paused, time scale and accumulator (app state). A load keeps the time of the save (`time()` stays
  `steps * dt` when the step rate and the saved time agree; otherwise the saved time + steps since the load * dt).
  Saves are untrusted input: the schema's type strings are bounded (256 characters, nesting 8, arrays of at most 2^16)
  and decoding work is bounded (elements without bytes share a budget).
- `tryLoad` checks everything before the world changes (a rejected save leaves the world unchanged: format, schema,
  slot table, free list, every value, `Parent` targets and cycles, unique `PersistentId`s, class IDs), then replaces all
  entities (removes and creates in the structural log; load listeners get the log range), rebuilds the derived state and
  recreates the game objects in three passes (create all, `configure` all, `onLoad` all; `onSpawn` is not called; during
  a load `configure` makes no structural changes and `onLoad` adds or removes only `Saved::No` components of its own
  entity: refused in every build, an assert and an exception that becomes a hook error); a load is refused (an error,
  the world unchanged) inside a step, a game hook or a view; a hook that throws removes its entity with its descendants without hooks and is reported in
  `LoadResult::hookErrors` (a throwing load listener in `listenerErrors`; the others still run). A state entity freed by
  a load (its components are no longer registered as state) must have no children, `PersistentId` or game object, and
  state entities never have game objects.
- JSON dump (`world.dumpJson()`) for debugging: the JSON form of `VisitJson.h` (exact numbers; 64-bit values beyond 2^53
  as strings). `diffSaves(a, b)` compares two blobs by name (fields, entities and components on one side, world-level
  state; filters and paging) without building worlds; `World::diff` formats it.

### 15.2 Schema and migration (long-lived worlds across patches)

- The save header holds all **schema tables** ever used by this save, numbered; every chunk record and character file
  stores its schema table number and is migrated lazily when it is loaded. A schema table lists per component, event,
  class and message its name, version and field list (name, type), plus name tables for enum values, meshes, piece types
  and generator slots. Data is loaded by name and IDs are remapped through the name tables: fields that no longer exist
  are dropped, new fields get their defaults. A field type change without a version increment is a load error.
- Nested `visit` structs (e.g. `ItemStack` in an inventory vector) and piece state structs have `name` and `version`,
  appear in the schema table and take `MigrationList` entries like components (R-SAVE-1 applies to them). Every chunk
  record section (heights, tombstones, grass, pieces, region summaries, entity groups) has a
  section tag and format version; unknown sections are a load error.
- A removed component, class or piece type needs a registered `Removed {name, policy}` entry (drop the entity, drop the
  component, or replace with another type); otherwise loading fails with a message naming it.
- A component whose meaning changed increments `version` (R-SAVE-1) and registers a migration that rewrites the loaded
  field values from an older version before they are applied.
- Within one build, save -> load restores all saved state (an equivalent world: equal `save()` bytes and `hash()`; the
  engine's systems continue identically in the tests, which is not a guarantee, §8.6); across versions it is as exact
  as the migrations.
- Implementation (S11): a type whose saved schema equals the registered one is read straight from its binary form;
  changed types (fields added or removed, enum values added or reordered, an older version) are decoded by the saved
  field list into the JSON form, migrated (`FieldMap`, nested structs first), stripped of fields that no longer exist and
  merged into a default value; a field type change without a version increment, a version gap without a migration and
  a save from a newer version are load errors; a stored enum value whose name was removed is an error. Types without
  saved values need no `Removed` entry; pending events of unregistered event types are dropped (logged); a component
  now registered `Saved::No` is dropped. Cross-component migrations (split, merge) come with the save directory (S25)
  when a game needs them. A committed golden save (`data/tests/engine/save_v1.bin`) and the committed engine schema
  (`data/tests/engine/schema.json`) make R-SAVE-1 violations of engine types fail in CI.

### 15.3 Generator versions [planned: S23]

- A world **pins** the world-plan and terrain generator versions it was created with, so terrain stays continuous
  across chunk borders and under existing bases.
- The set of generated chunks stores per chunk the placement generator version it was generated with (uint16; prefab
  versions come from the pinned plan); newer versions apply only to chunks generated later.
- All generator versions a shipped game ever used stay callable (the `version` argument of the `Generator`
  functions, §12.1), for all layers. A chunk's
  base is regenerated from (seed, pinned plan/terrain version, its placement version); the chunk store holds
  modifications, entity groups and, for chunks of retired versions, their stored base.
- **Prefabs are versioned content**: a shipped prefab file is never edited; a change adds `prefabs/<name>.v<N>.json`.
  The first version of a prefab is `prefabs/<name>.json` (v1). The world plan stores per site the prefab name and
  version; they are part of the pinned plan version, so every chunk of a site, its height patch (`height()`) and its far
  mesh use the same prefab version. A chunk's placement version does not select prefab versions. A new prefab version
  reaches existing worlds only through a new plan version (never for an existing world). All shipped prefab versions
  stay in the game data.
- Retiring a version is an explicit one-time conversion on load that stores the affected chunks' generated bases;
  only such chunks are sent to clients as base snapshots (§13.2).

### 15.4 Characters [planned: S33]

A player's character (the entity group of the player entity: its components, children and inventory) is a **character
file** on the player's machine, with the same header and migrations as world saves, so characters move between worlds
and versions; the game decides what is kept. On join the client sends it (`Channel::ChunkData`) and the host imports it;
the host exports the connected characters at every commit step from the same frozen snapshot as the world (§15.1),
and a disconnected character from the host's own state into the next commit (§13.7). Every commit stores the latest
export of every character that has played in the world (carried forward from the previous commit, part of the save
directory). The host sends a connected client its export only after the world-state file of that commit has been
renamed into place (the client keeps the newest), and on leave. The host keeps the
latest export of every character in memory until a commit stores it: the export of a clean leave (the same file the
client receives) and the export of a disconnect. **Rejoin**: if the host has a copy for the joining `characterId`
(in memory, else in the loaded commit), the host copy is authoritative: it is imported and sent back to the client.
Without a crash this is the leave-time export, so nothing is lost; after a host crash (§13.7) it is the loaded commit's
copy, so the character rolls back together with the world (no item exists twice). Otherwise (a new character, or the
first visit of this world) the client's file is imported normally. `World::exportCharacter(Entity player)` [S33] exports and sends one character at any time, so
a game can export more often (e.g. on inventory changes, policy). On import into the world that exported the character,
its `Persistent` IDs are kept if still free; otherwise (and in other worlds) they are reassigned from the target world's
counter and remapped inside the group; `PersistentId` fields pointing outside the group become null. On leave the player
entity is removed after the export is acknowledged. The save directory contains no player characters from S33 (§13.7);
`World::save()`/`load()` blobs keep them.

## 16. Tooling: MCP and running the game

### 16.1 Running the game

| Step | How | Status |
|---|---|---|
| Configure | as in `AGENTS.md`, plus `-DSAIGA_BUILD_SAMPLES=ON` for the sample executables | [exists] |
| Build | `cmake --build build --target sample_vulkan_world` | [exists] |
| Start with MCP | in a working directory: `SAIGA_MCP_PORT=8765 <build>/samples/vulkan/sample_vulkan_world`; `config.ini` is created there (`enableValidationLayer`, `vsync`) | [exists]; from S13 command-line options instead of `config.ini` (§8.3) |
| Start without a display | `--headless` | [planned: S13] |
| Connect | `.mcp.json` server `saiga-world` (`http://127.0.0.1:8765/mcp`); scripts can POST JSON-RPC | [exists] |
| Ready | `tools/list` answers | [exists] |
| Stop | MCP tool `quit` [planned: S14]; now: SIGTERM by the full executable path (the main thread is renamed `Saiga::main`, `pkill -x` by name does not match) | [exists] |
| Benchmark | `--benchmark <name>` (§17.7) | [exists] (S03: `sample_vulkan_world`; S13: `EngineApp`) |

### 16.2 MCP tools

| Tool | Contract | Status |
|---|---|---|
| `screenshot`, `get_state`, `set_camera`, `move_camera`, `set_settings`, `terrain_query`, `measure_performance` | prototype; in S14: `screenshot`, `measure_performance`, `set_camera`, `move_camera` become engine tools (without `height_above_ground` and `walk_mode`, which move to `game_set_camera`); `get_state` -> `game_get_state`, `set_settings` -> `game_set_settings`, `terrain_query` -> `game_terrain_query` | [exists] |
| `describe_world`, `world_status` | the registrations (components with fields, types, replication, global/space state; classes; events; operations); step, time, paused, time scale, counts, hash | [exists: S10] |
| `list_entities` | `{class?, components?: [names], near?: {space?, position, radius}, include?: [names], limit? (100, <= 1000), after?}` -> entity, class, position, component names in index order, the values of the `include` components; near selects by the grid of the last step (`grid_step`) and reports that `grid_position` and the `distance` from it next to the live `position`; `next` when more exist | [exists: S10] |
| `inspect(entity, components?)` | class, flags, parent, children, every component as JSON | [exists: S10] |
| `set_component`, `add_component`, `remove_component`, `set_transform` | applied at once (between steps); merges the given fields (missing keys keep values; unknown keys and wrong types change nothing; the error lists the fields); Transform through its setters (a teleport by default; refused for children: their `LocalTransform` decides); Parent with the cycle check (no world-state parent or child); Behaviour: the tick flags (the interval is derived again; writing `tickInterval` or `classId` is an error); not `Behaviour` (spawn) or `Persistent` (makePersistent); writes must be finite (nan and inf are errors; quaternions are normalized); `set_component` and `set_global` return the new `values` and the `previous` writable ones (an undo payload `set_component` accepts) | [exists: S10] |
| `spawn(class?, position, rotation?, scale?, space?, components?)`, `destroy(entity)` | initial components as JSON, checked before anything is created and applied after `onSpawn` (merged into components the object added itself); returns the entity; destroy returns the destroyed descendants (onDestroy runs); world-state entities cannot be destroyed | [exists: S10] |
| `get_global`, `set_global` | world-state (or with `space` space-state) components | [exists: S10] |
| `emit_event(event, values)`, `queue_action(player, action, aim?)` | the gameplay path: readable in the next step (e.g. `DamageEvent` reaches `onDamage`) | [exists: S10] |
| `pause`, `resume`, `step(n)`, `time_scale(s)` | `step` only while paused, runs n steps (stops at the request's time budget, half the tool timeout: `ran`, `stopped_early`; `apply` shares one budget and stops between operations) | [exists: S10] |
| `changes_since(cursor?, include_moved?, limit?)` | created, destroyed (its components folded in), component added/removed with sequence numbers and steps (a journal of 100000; `truncated`, `more`; `limit` per call, default 200); with `include_moved` the entities moved since the cursor (once each, with the handle they moved with; a move window's members are fixed when it opens (a snapshot shared by the cursors paging it; the newest 16 windows are kept, an older continued window reports `moves_truncated`) and paged by entity index with `moved_more`, so every entity that moved in it is delivered; moves made meanwhile come in the next window; tracked from the first such call); the cursor is opaque (`0`: everything kept, `"now"`: from now on); one of another session (a random epoch per API) is an error | [exists: S10] |
| `get_profile(steps?, scopes?)`, `reset_profile`, `measure_performance(steps)` | step profile (§17.1), filtered by scope path prefixes; measure (at most the 600 samples of the history) runs n steps while paused or waits for n steps while running (refused at time scale 0) | [exists: S10; S13 adds frame and GPU scopes] |
| `logs(since?, category?)` | log lines with a sequence number > `since` (at most 500, `first`, `truncated`; category `api`: the edits made through the tools, logged after they ran with `ok` or `failed:`; an empty result names the known `categories`) | [exists: S10] |
| `apply(operations)` | a batch `[{op, ...arguments}]` in order; stops at the first failing operation (the earlier ones stay applied); available read-only when every operation reads | [exists: S10] |
| `save_snapshot(name, overwrite?)`, `load_snapshot(name, paused?)`, `list_snapshots`, `delete_snapshot(name)`, `diff_snapshot(name, against?, entity?, components?, limit?, after?)`, `export_snapshot(name, file, overwrite?)`, `import_snapshot(name, file, overwrite?)` | in-memory saves of the world (16, 256 MB) to roll back experiments and compare them (`diffSaves`: summary per kind and component, paged differences with class and position of one-sided entities); files only in the directory set by the app (`WorldToolsOptions::snapshotDirectory`, plain names); a load is one `world_loaded` entry in `changes_since` (cursors from before stay valid and answer `reloaded: true`); `WorldApi::rebind(World&)` moves the API and the tools to another world | [exists: S11] |
| `get_data(type, name?)`, `reload_data(reconfigure?)` | the data files (§14): an entry or the entry names; re-read after edits (nothing changes on an error), optionally reconfiguring every game object (a save and load) | [exists: S11] |
| `run_scenario(name)` | replaces the live world with the scenario, runs it, returns the checks | [planned: S14] |
| `quit` | | [planned: S14] |

S10 implementation: the operations are `WorldApi` calls (`engine/api/WorldApi.h`: JSON arguments and results by
names, errors instead of asserts, only between steps, edits logged under `api`; S12's scenarios (`ScenarioWorld::api`) and an editor
can use the same API); `registerWorldTools(server, world, {allowEdits})` (`engine/tools/WorldTools.h`) registers one tool
per operation plus `measure_performance` and returns the shared API (the caller keeps it while the tools are used;
`allowEdits = false` registers only the reading tools for observing agents). Components and events are read and
written through `visit` (`engine/ecs/VisitJson.h`: 64-bit integers beyond 2^53 as strings, non-finite doubles as
`"nan"`/`"inf"`, enums by name, quaternions `[w, x, y, z]`). Unknown argument names are errors (the schemas set
`additionalProperties: false`); integers must be integers in range (no strings). Results are compact JSON with exact
numbers (integers in full, other numbers in the shortest form that reads back to the same double; profile times
rounded to 1 ns, with a `top` list of the slowest systems). The API detaches when its world is destroyed. An exception from a game hook during an operation becomes an error result
(an `apply` batch reports the operations before it). `McpServer::addPollListener` lets a deferred tool check its
deadline in every poll (`measure_performance` answers although the world stops stepping, and fails on a profile reset). Entities in JSON are `{"index": i, "generation": g}` (the
tools also accept `"i:g"`); errors are tool results with `isError: true` and a message. Tool tests use the `Poller`
(`tests/mcp_test_helper.h`, S10). Tools that need rendered frames (`screenshot`, the frame part of
`measure_performance`) register a frame callback and respond later; they do not render inside the handler.

## 17. Verification

### 17.1 Logging and profiling [exists: S06; SAIGA_LOG_E: S07; profiler: S10]

`SAIGA_LOG(category, format, ...)` (printf style, S06) and `SAIGA_LOG_E(category, entity, format, ...)` (S07) into a
ring buffer (MCP `logs`) and stdout; `SAIGA_ASSERT` messages name the entity and component. The ring
(`engine/core/LogRing.h`, process-wide `engineLog()`, default 4096 lines) numbers its lines 1, 2, ... and returns the
lines after a sequence number (`since(seq)`, for `logs(since)` and scenario failure output); `setLogEcho(false)` turns
the stdout echo off (tests, benchmarks).

Profiler (`engine/core/Profiler.h`, S10; `world.profiler()`): scopes in a tree by path (`step`, `step/phase:<name>`,
`step/phase:<name>/system:<name>`, `step/phase:<name>/commands`) and counters (`entities`, `game_objects`,
`objects_updated`, `commands_applied`, `structural_records`, `grid_items`), one sample per step, each with a ring of the
last 600 samples; `report(n)` gives count, avg, min, p50, p95, max (ms) and the share of the parent per scope (the MCP
tools `get_profile` and `measure_performance`). Wall-clock time (steady clock) only for these numbers, never for the
simulation (R-DET-1); `WorldSetup::profiling = false` makes no clock calls; an aborted step's sample is discarded. From
S13 the app adds `frame`, `extract`, `recording` and the GPU passes (timestamps) to the same profiler, and its ImGui
window "Engine statistics" (history plots, per-phase and per-system tables like the prototype's "World Statistics")
reads it.

### 17.2 Scenario tests [exists: S12; terrain, streaming, render sink and clients with their stages]

```cpp
// tests/test_engine_scenarios.cpp (Wanderer: a test-only game object of this file)
void registerEngineScenarios(ScenarioList& s)
{
    s.add("wanderer_flees_when_damaged", [](ScenarioWorld& w)
    {
        auto wanderer = w.spawn<Wanderer>({0, 0, 0});
        auto player   = w.spawnPlayer({5, 0, 0});
        w.stepSeconds(1.0);
        w.damage(wanderer, 10.0, player);                 // emits DamageEvent: the real path
        w.check(w.stepUntil([&] { return w.distance(wanderer, player) > 20.0; }, 600), "flees from the attacker");
    });
}
```

- `ScenarioWorld(const WorldSetup&, const ScenarioOptions& = {})` [exists: S12]: headless, reproducible (§8.6: the
  setup's seed and worker thread count, profiling off, synchronous chunk loading from S24), flat terrain (seeded
  terrain from S20), `RecordingRenderSink` (from S19); every object ticks every step (`World::setTickEveryStep`) unless
  `w.enableDistanceTicking()`; helpers `step(n)`, `stepSeconds`, `stepUntil(pred, maxSteps)`, `spawn<T>`,
  `spawnPlayer(position, peer = 0)` (entity with `Transform`, `Player{peer, characterId 1, 2, ...}`, `PlayerInput`,
  `CharacterIntent`; in the host + client variant owned by the client) and `spawnPlayer<T>(position, peer, initial...)`
  (a game object player without `CharacterIntent`), `move(player, direction)` (sets `PlayerInput.move`, length <= 1),
  `input(player, action, aim)` (queues a `PlayerAction`; a message from S15; both from the owning peer; `spawnPlayer`
  steps both worlds until the client has the entity, helpers translate host `Entity` to the client's through `NetId`),
  `damage(target, amount, source)` (a `DamageEvent`), `check(cond, msg)` (collects failures with the step; works under
  gtest and MCP), `dumpJson(entity)`, `trace<T>(entity)`, `distance`, `count<T>()`, `api()` and `call(op, args)` (the
  world API, so a scenario can act exactly like an agent; it follows the save/load swap through `WorldApi::rebind`).
  `world()` returns the current `World`: the save/load variant replaces it, so scenario code calls `world()` each time
  and never keeps the reference across a step helper (entity handles stay valid; replaced worlds live until the
  scenario ends). Steps are observed by a step listener, so `world().step()`, `advance()` and the API's `step` count
  too; `ScenarioOptions::maxSteps` (default 36000) ends a scenario that runs longer, by any of these paths (through
  the step helpers and `call`, which throw the abort; a direct `api().call("step", ...)` returns it as an error, and a
  loop that ignores errors does not end: use `call`). The save/load variant swaps the world at the first step helper
  or `call` at or after half of the plain run's steps (after the call); a scenario in which no step runs on the
  replacement (steps after the middle only through `world()` or `api()`, or one `call` that runs all of them) fails the
  variant. A scenario
  adds its own step listeners with `ScenarioWorld::addStepListener` (they move to the replacement world; a listener
  added to `world()` directly fails the save/load variant), and never steps inside a step listener (a failure: the
  world's listener dispatch is not re-entrant: `World::step`/`advance` and the world API's edits are refused inside a
  step listener in every build).
- `ScenarioWorld` overrides the radii of every `SpaceTemplate` with small defaults (`fullRadius = simulationRadius =
  interestRadius = clientPhysicsRadius = 128`, `prefetchMargin = 0`), which a scenario can raise; it requests no visual
  tiles or far content, and all worlds of one test process share a cache of generated base chunks (content is
  deterministic, R-CONTENT-1).
- A failing scenario prints the failed checks with their steps, the log lines since it started (at most 200),
  `dumpJson` of the involved entities (spawned through the helpers, at most 10) and the traces.
- Runner (`engine/tools/ScenarioRunner.h`, S12): `runScenario(setup, entry, variant, options)` and
  `runScenarios(setup, list, filter, options)` (every variant) return `ScenarioResult {passed, failures, report, steps,
  seconds}`; exceptions from the scenario or a game hook are failures, a `SAIGA_ASSERT` stops the process.
- Files: `engine/tools/ScenarioWorld.h`, `engine/tools/ScenarioList.h`, `engine/tools/ScenarioRunner.h`,
  `engine/sync/RecordingRenderSink.h` (S19).
- The saiga tests test the engine only: engine scenarios use small test-only game objects and systems defined in
  `tests/` (e.g. `Wanderer` of S12) and run in `tests/test_engine_scenarios.cpp` as a value-parameterized gtest over the
  scenario list x variants. Games are samples (`samples/games/<name>/`, §18.1); their scenarios
  (`EngineAppSetup::scenarios`) run with the same runner and variants through `game_<name> --scenarios` (from S13;
  headless, exit code 0 if at least one scenario ran and all pass, else 1 (a filter that matches nothing prints `no
  scenarios matched`), the failure count is printed) or `run_scenario` (MCP), not in ctest; a stage that changes a game
  runs them locally and reports the result in the pull request.

### 17.3 Multiplayer variant [planned: S16]

Every scenario not registered local-only also runs as host + clients (`ScenarioWorld` supports N clients; the default
variant uses one) over `LoopbackTransport` (S16: no loss, `delaySteps` 0 and 2, and an asynchronous-stepping variant:
the client starts with an offset and runs a seeded 0, 1 or 2 steps per host step; from S32 additionally with loss):
checks run on the host. After the scenario host and clients run network-only steps (`stepNetworkOnly`, phases 1 and 8;
owners send their final owner-authoritative state in phase 8) until each client has applied a snapshot sent after the
scenario ended and the host has received the owners' final state (bounded by 600 steps; without loss at most `2 *
(delaySteps + maxReorderSteps) + 3` steps, asserted; during these final steps the asynchronous variant steps host and
client 1:1), then compares the host-authoritative components on the client with the host, and the `OwnerAuthoritative`
components on the host with the last state received from the owner (`stepNetworkOnly` writes received poses without
extrapolation), for the scenario's entities, within the quantization tolerances.

### 17.4 Repeat and save/load tests [exists: S08 (hash), S12 (variants)]

- `world.hash()` (FNV-1a 64 over the `visit` bytes) covers all saved components (`Saved::Yes`) of the alive entities in
  index order (with index and generation, `Transform` first, then the components in ID order), the free list (it decides
  future entity indices), the RNG state, the pending events, the step index and (from S25) the chunk states. It is a
  debugging aid and the check of the repeat variant.
- Variants of every scenario [exists: S12]: (0) **plain**: once; (a) **repeat**: the scenario runs twice and the
  per-step hashes, the step count and the final hash must be equal (catches wall-clock time, unordered containers and
  uninitialized memory); on a mismatch the runner runs both again up to the first differing step and prints
  `diffSaves` of the two saves; with `ScenarioOptions::varyThreads` the second run uses another worker thread count
  (1 <-> 3; the engine's own scenarios use it: the engine's `parallelEach` systems must not depend on the thread count,
  games need not, §20); (b) **save/load**: after a plain run that counts the steps, `ScenarioWorld` saves at the first
  step-helper boundary at or after half of them (at the end for scenarios with fewer than 2 steps or that step only
  outside the helpers), loads the save into
  a fresh `World` of the same setup and continues with it; the loaded world's `save()` must equal the saved bytes
  (`diffSaves` explains a mismatch), hook errors of the load fail, and the scenario's checks must still pass (catches
  state outside components, R-STATE-1); with `ScenarioOptions::strictContinuation` the final hash must also equal the
  plain run's (the engine's own scenarios use it; it catches member state that no check looks at). Paused, time scale,
  the advance accumulator, the worker thread count, the shared data store, the tick setting, the listeners of
  `ScenarioWorld::addStepListener` and the world API carry over to the new world. The save/load variant runs single-world only; a host
  reload with connected clients is a session restart (§13.7). A loaded world is not required to continue
  bit-identically (§8.6).

### 17.5 Rendering tests [planned: S18, extended in S19]

Headless Vulkan tests (lavapipe in CI) assert numeric render statistics (instances drawn per LOD, terrain nodes).
Counts compared across implementations (GPU vs CPU reference, S19 vs prototype) are equal except for instances within
1e-3 relative of a frustum plane or LOD threshold, as determined by the CPU reference.

### 17.6 Golden images [exists]

- `bool compareGolden(const TemplatedImage<ucvec4>& image, const std::string& name, const std::string& deviceName)` in
  `tests/golden_image.h` (header only; `deviceName` from `VkPhysicalDeviceProperties::deviceName` decides the gating);
  PNGs in `data/tests/golden/`.
  - Setup: clay mode, `freezeTime` (time = 0; both in `WorldSettings` until S19, then in
    `RenderSceneParameters::shading` / `freezeTime`), fixed seed, camera, 512 x 512, captured after 5 frames; small
    scenes (<= 1 km, <= 10^4 instances).
  - Golden mode uses a fixed exposure and forces a full (non-incremental) update of every incremental effect (volumetric
    fog and cloud history, horizon and cloud shadow maps, shadow atlas) in the captured frame [planned: by the stage of
    each effect].
- S02 goldens (`VulkanWorldGolden.WorldOverview` / `.WorldForest` in `tests/test_vulkan_world.cpp`, `WorldParameters`
  with `terrain.size = 513` (1 km), 5000 trees, seed 1): `world_overview` (camera 300 m above the ground 100 m from the
  west edge at the center z, looking 45 degrees down along +x, so terrain and trees fill the image) and `world_forest`
  (camera 2 m above max(ground, water) at the edge of a clearing 28 m west and 80 m north of the center, looking along
  +x and slightly down: a close tree with alpha-to-coverage foliage, grass, trunks, the forest at mid distance and the
  sky); the tests also `EXPECT_EQ` the drawn tree instances per LOD and the terrain node count against committed numbers
  (used by S19). The counts gate only on lavapipe like the images (the placement uses std random distributions and float
  comparisons: other standard libraries, compilers or drivers may differ by a few trees).
- Pass: mean absolute difference < 1/255 and at most 0.5% of the pixels differ by more than 16/255. It gates only when
  the device name contains `llvmpipe` (else it prints the metrics). Gray (clay) goldens are stored as 8-bit grayscale
  PNGs.
- On a gating failure (also a missing, unreadable or differently sized golden) `golden_failed/<name>_actual.png` and
  `_diff.png` (only when the golden can be compared) are written to the working directory
  (`build/tests/` under ctest, git-ignored; a passing comparison removes stale ones); CI uploads them (`actions/upload-artifact`, `if: failure()`, artifact
  `golden-failed-<job>`; the actual images are saved like goldens) and prints the metrics of every run from ctest's log (step
  "Golden image metrics"). The committed goldens are rendered by a local lavapipe (Mesa with LLVM 21.1.8); CI's
  lavapipe (Mesa 25.2.8, LLVM 20.1.2) differs slightly and passes within the tolerance (reference margin, run
  43754502053: `world_overview` mean 0.005/255, 0.011% of the pixels > 16/255; `world_forest` mean 0.14/255, 0.19%); a
  Mesa update on the CI runner may need the regeneration from the CI artifact below.
  Goldens are only written on lavapipe, in a dedicated commit: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
  SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_vulkan_world --gtest_filter='VulkanWorldGolden.*'` (it writes the PNGs
  instead of comparing and does not check the counts), then the printed `trees per lod` counts are copied into the `ExpectCounts` calls. If CI fails after a Mesa update, the goldens and the committed counts are taken from the failed
  run: the `_actual.png` images of the artifact (it exists only for a run whose comparison failed) and the
  `trees per lod` lines of the step "Golden image metrics", and the Mesa/LLVM versions named here and in AGENTS.md are
  updated (the goldens then come from CI's lavapipe); the exact commands are in AGENTS.md.

### 17.7 Benchmarks [exists: flyover, forest (S03); later benchmarks planned as marked]

- `--benchmark <name>` [exists]: parsed by `sample_vulkan_world.cpp` (needs a display); paths, statistics and the result
  in `src/saiga/vulkan/world/WorldBenchmark.h` (`WorldBenchmark::pose`, `summarize` (median, nearest-rank p95), the
  runner); moved to `engine/app/` in S13.
  Benchmarks `flyover` (one revolution in 600 frames on a circle of radius 3 km around the world center (the prototype's
  center (4096, 4096) until S27; from S27 `(sizeChunks * 128 m, sizeChunks * 128 m)` on the 32 km world), at the ground
  height below the camera + 300 m, looking at the center) and `forest` (2 m above max(terrain, water level) from center
  minus 300 m to center plus 300 m along x in 600 frames, looking along +x); default `WorldParameters` (8 km, 200k
  trees); 1920 x 1080, MSAA 4; 60 warm-up frames excluded; `freezeTime`; `--benchmark` turns validation, vsync, ImGui
  and defragmentation off; pass keys are `WorldRenderer::gpuPassName` (from S19 `RenderScene::gpuPassName`); `quality`
  is `"default"` until S37.
  Writes `benchmark_<name>.json` `{gpu, quality, build: {type, asserts, archnative}, frames, gpu_frames, width, height,
  cpu_ms: {median, p95}, cpu_work_ms: {median, p95}, gpu_ms: {<pass>: {median, p95}}, stalled_frames}` to the working
  directory and exits; the exit code is non-zero if the benchmark did not finish, the result could not be written, or
  the rendered size changed or a frame stalled during the measured frames (`stalled_frames`: interval above 4 x the
  median and above 50 ms, e.g. a minimized window or a swap chain recreation; an old result file is deleted at the
  start, a bare `--benchmark` and unknown arguments are errors).
  - `cpu_ms` is the frame interval (one `transfer()` to the next): it includes waiting for the GPU and presenting, so
    it is at least the GPU frame time. `cpu_work_ms` is the CPU work of the world in the frame (`transfer()` and the
    command recording in `render()`).
  - `gpu_frames`: frames with GPU times; `gpu_ms` is null without GPU times (no timestamp support).
  - `width`/`height`: the rendered size, the swap chain extent of the measured frames (`VulkanRenderer::surfaceSize()`;
    a warning is printed if it is not 1920 x 1080; a resize during the warm-up does not count).
  - `build`: the CMake configuration of `saiga_vulkan` (`SAIGA_BUILD_CONFIG` = `$<CONFIG>`, right with multi-config
    generators too; without a build type `release (NDEBUG)` or `debug (no NDEBUG)`), `SAIGA_ASSERTS`, and whether
    `-march=native` was really used (`SAIGA_ARCHNATIVE_ACTIVE`, set by `Saiga_compiler_flags.cmake` where it adds the
    flag); baselines come from the release build of AGENTS.md (asserts and `-march=native` off).
  - `SAIGA_SCREENSHOT` is ignored in benchmark mode (its automatic screenshot would close the window at frame 60).
  - No GUI and no MCP during a benchmark, and the default world (`SAIGA_WORLD_TREES` is ignored). From S13 the result
    also has `step` and `recording`, from S19 `extract` (each `{median, p95}`).
- Per-frame values come from `struct WorldRenderer::FrameTimes { double cpuMs; double cpuWorkMs; int cpuFrame; double
  gpuMs; std::array<double, PASS_COUNT> gpuPassMs; int gpuFrame; }; const FrameTimes& WorldRenderer::lastFrameTimes()
  const;` (unsmoothed; also passed to `WorldRenderer::frameCallbacks`) [exists]. The CPU values of frame N are known
  when frame N + 1 is prepared, its GPU times when it has completed (a few frames later): samples are matched by frame
  number. Every completed frame delivers its GPU times (one per `transfer()`, oldest first).
- From S27 the benchmarks run on the 32 km world, from S37 at each tier's resolution of §17.10.
- `--benchmark host_step` [S29, extended in S32]: headless, 8 scripted players in 4 islands with physics, one island
  containing 50 terrain-edited chunks with 30 container entities and 20 NPCs (from S30 additionally a base of placed
  pieces in those chunks; the stage that is done later re-measures); S29 runs
  it without clients, S32 adds 7 loopback clients and reports the upload per client; reports step ms (median, p95)
  against §17.10 and `streamingStalls`.
- Terraforming benchmark [S22, extended in S28]: 8 brushes at 10 edits/s; S22 measures the integer result and the CPU
  tile/pyramid update, S28 re-runs it with collider patches; the step time stays within §17.10.
- ECS benchmark: `tests/test_engine_ecs_benchmark.cpp` (non-gating) prints the total ms of create, views over one
  component, `Transform` plus one and three components, random reads, remove + re-add and destroy at 10^5 entities
  [exists: S07].
- `data/tests/benchmarks.json` [exists] maps GPU name -> benchmark -> result; a regression of more than 5% must be
  explained. Local only. The S03 baseline is for the development GPU (an NVIDIA RTX 5080 Laptop GPU at 1920 x 1080):
  it is not the budget reference of §17.10 (an RTX 3060 / RX 6600 class card at 1440p); a baseline on that class is
  added in S37, which checks the budgets. A benchmark run turns ImGui and memory defragmentation off, so the result
  does not depend on `config.ini`. Run-to-run noise (S03, release build, two runs each): the `cpu_ms` median 0.5-0.7%,
  its p95 1.5-1.9% (frame pacing makes the p95 the noisier value), the GPU pass medians about 1% (passes below 0.1 ms
  up to 5%, timer resolution); compare medians, and repeat a run before reporting a regression of the frame interval.

### 17.8 Not verified automatically

| What | Required action |
|---|---|
| Look of the colored rendering, lighting, weather | MCP `screenshot` on a GPU; describe the result in the report |
| GPU performance and budgets (§17.10) | benchmarks on the reference GPUs |
| Real network conditions beyond the loopback simulation | manual co-op session |
| Input devices, window behavior, Steam Deck controls | manual |

### 17.9 Quality tiers [planned: S37, parameters from S17]

`QualityTier {Low, Medium, High}` selects MSAA samples, render scale (spatial upscaling below 1), shadow cascades and
resolution, shadowed local lights and re-rendered faces per frame, light budget, instance and grass densities,
far-content budgets and draw distance (High: to the horizon), fog mechanism (analytic or volumetric), clouds (2D layer
or volumetric, ray-march resolution), ambient occlusion, depth of field, water reflections and caustics, particle budget
(`RenderSceneParameters::quality`, the single source; `QualityTier` is defined in `RenderSceneTypes.h`); games can
override each value. **Tiers never change simulated content**: they only skip rendering of grass (1b) and of tier-1
generator slots flagged `cosmetic` (no collider, not promotable), by counter as in §10.4. Default Steam Deck tier: Low
with a reduced render scale, analytic fog, 2D cloud layer, 0-1 shadowed local lights, no depth of field.

### 17.10 Budgets (defaults; games may tighten them)

| Target | Tier | Resolution | Frame time | GPU memory |
|---|---|---|---|---|
| Desktop reference (RTX 3060 / RX 6600 class) | High | 1440p | <= 16.6 ms (60 fps) | <= 6 GB |
| Steam Deck | Low | 1280 x 800 | <= 25 ms (40 fps) | <= 3 GB |
| Desktop reference | Medium | 1440p | no frame-time budget (report only) | <= 6 GB |

GPU time per pass at the High tier on the desktop reference (defaults, checked from S37): shadows (cascades, far
terrain, local atlas) <= 2.5 ms; depth pre-pass + AO <= 1.5 ms; opaque <= 4.5 ms; sky <= 0.2 ms, water <= 0.8 ms,
volumetric fog <= 0.7 ms, volumetric clouds <= 0.8 ms (sky: S37; water and transparents: S39; volumetric fog: S38;
particles: S41; volumetric clouds: S42; the rest: S37); transparents <= 0.4 ms, particles <= 0.6 ms; post chain <= 1.5
ms; rest (UI, culling, uploads) <= 1 ms (sum 14.5 ms, headroom for jitter). Steam Deck: the same split scaled to 22 ms
at Low. Pass budgets apply to the median; the frame time p95 must stay within the frame time of the table.

Network (defaults): steady-state host upload <= 256 kbit/s per client (8 players, 4 islands); join data <= 20 MB for 50
edited chunks (incl. a placed-piece base once S30 is done). Upload is measured by `host_step` (S32), join data by the
S33 two-process session (local). These are reported, not gates: exceeding the upload budget is the trigger to add delta
compression (§13.4, §20).

Budgets are measured in the release configuration (`SAIGA_ASSERTS` off).

Host CPU: one simulation step with 8 players in up to 4 simulation islands <= 4 ms on the desktop reference CPU; on
the Steam Deck as host with 4 players / 2 islands <= 6 ms. Main thread per frame (steps + extract + animation +
recording): <= 10 ms desktop, <= 18 ms Steam Deck. Benchmarks report `cpu_ms` split into step, extract and recording.
Benchmarks report against these budgets; from S19 they run on both targets when the hardware is available (otherwise
the missing target is reported as open).

## 18. Files, names and cookbook

### 18.1 Files and names

| What | Path | Name | Namespace / macro |
|---|---|---|---|
| engine code | `src/saiga/engine/<area>/` | `CamelCase.h/.cpp` | `Saiga::Engine`, `SAIGA_ENGINE_API` on non-template classes and functions |
| engine component | `src/saiga/engine/components/<Name>.h` | struct `<Name>`, ID in `EngineIds.h` | |
| engine system | `src/saiga/engine/<area>/<Name>System.cpp`, declared in `<area>/Systems.h` | `<name>System` | `SAIGA_ENGINE_API` |
| engine tests | `tests/test_engine_<area>.cpp` | `saiga_test(test_engine_<area>.cpp "saiga_engine")` in `if (MODULE_ENGINE)` | |
| game | `samples/games/<name>/` (executable `game_<name>`, `saiga_game(<name>)`, `main.cpp`; headers included as `"games/<name>/<File>.h"`; content in `content/`) | IDs in `<Name>Ids.h` | `Saiga::Game`, no export macros |
| code shared by games | `samples/games/common/` (compiled into each game) | | `Saiga::Game` |
| game scenarios | in the game, `EngineAppSetup::scenarios`; run with `game_<name> --scenarios` (not in ctest, §17.2) | | |
| shaders | `shader/vulkan/<area>/` | `snake_case.vert/.frag/.comp`, `*.glsl` includes | |
| game data, prefabs | `data/<game>/`, `data/<game>/prefabs/` | `snake_case.json` | |
| golden images | `data/tests/golden/<test>.png` | | |

### 18.2 Registration lists

| Kind | Engine | Game |
|---|---|---|
| components | `registerEngineComponents(Registry&)` | `WorldSetup::components` |
| events | `registerEngineEvents(EventRegistry&)`, `engine/world/Events.h` | `WorldSetup::events`, `<game>/Events.h` |
| systems | `registerEngineSystems(SystemList&)` | `WorldSetup::systems` |
| game object classes | – | `WorldSetup::classes` |
| scenarios | in `tests/test_engine_scenarios.cpp` with test-only game objects (§17.2) | `EngineAppSetup::scenarios` (`--scenarios`, `run_scenario`) |
| data file types | – | `WorldSetup::dataTypes` |
| network messages | `registerEngineMessages(MessageList&)` | `WorldSetup::messages` |
| generators, prefabs | – | `WorldSetup::generators` |
| piece types | – | `WorldSetup::pieceTypes` |
| meshes, materials | – | `WorldSetup::renderAssets` (into the `World` by name; forwarded to the sink) |
| collision meshes and shapes | – | `WorldSetup::collisionAssets` (`CollisionAssetList`, by name) |
| migrations, removed types | – | `WorldSetup::migrations` |
| MCP tools | `registerEngineTools(McpServer&, EngineApp&)` | `EngineAppSetup::gameTools` (`game_*`) |

### 18.3 Cookbook

- **Component**: struct with `name`, `id` (add to the module's ID enum), `version` and `visit` in its own header
  (initialize every math field); add to the component list with its replication mode; a `visit` round-trip test for
  custom field types; a migration when existing saves are affected (R-SAVE-1).
- **System**: function in its own file with the header comment (reads/writes); add it to the phase list with `RunOn`
  and a comment why at this position; scenario test.
- **Game object class**: subclass without constructor parameters with `static constexpr ClassId id` and `name`, state in
  `<Name>State`, `configure` / `onSpawn` / `onLoad` / `onReplicatedSpawn`; `classes.add<T>()`; scenario test (R-TEST-1,
  R-NET-1).
- **Event**: struct with `name`, `id` and `visit` in `Events.h`; register (with replication mode); emitted in step k,
  read in step k+1; deliver to game objects through a game system (§9.2).
- **Data file type**: struct with `name` and `visit`, register, file `data/<game>/<name>.json`, load test incl. the
  unknown key error.
- **MCP tool**: function in `engine/tools/` or the game's tool list (`game_*`); description with reads/changes; test
  through `handleMessage` with the `Poller`.
- **Render feature**: in `saiga_vulkan`, data through `RenderScene` or pass parameters; headless test; golden image if
  visible; benchmark and budget check if in a hot path; quality tier settings.
- **Scenario**: a function `void(ScenarioWorld& w)` added to a `ScenarioList` (engine: `tests/test_engine_scenarios.cpp`;
  game: `EngineAppSetup::scenarios`): spawn through the helpers, `step`/`stepSeconds`/`stepUntil`, `w.check(...)`; call
  `w.world()` each time (the save/load variant replaces the world), add step listeners with `w.addStepListener` and
  never step inside one; it runs in the plain, repeat and save/load variants.
- **Debugging a failing scenario**: read the printed checks and logs; add `trace<T>` for the suspicious component; run
  it live with `run_scenario`, `pause`, `step`, `inspect`; for repeat or save/load failures read the `diffSaves` lines
  of the report.

### 18.4 Tech demo games [exists]

Three small games show different aspects of the engine. They were added in S04, before the engine modules they need,
so they are simple and built only on what exists (saiga_core double math, the forward renderer, the world prototype).
Later stages port them and use them as tech demos: the stage table (§22) names the demo work of each stage, and each of
these stages keeps the demos working and shows its feature in them (screenshots in the pull request).

| Game | Shows | Directory, executable |
|---|---|---|
| Flight simulator | the large world and efficient rendering: a small airplane over the 8 km world prototype (CDLOD terrain, 200k GPU culled trees); flight model with lift, drag, stall and crashes; the demo mode's autopilot flies a circuit | `samples/games/flight/`, `game_flight` |
| Platformer | character animation: one level of platforms, a humanoid of boxes with procedural skeletal animation (idle, walk and run cycles blended by speed, jump and fall poses, landing squash; blend weights sum to 1), a kinematic controller (coyote time, jump buffer, variable jump height, sub-stepped box collision); the demo runner plays the level | `samples/games/platformer/`, `game_platformer` |
| Racing | physics and world: a car (rigid body on four spring-damper wheels, tire friction circle) on a closed track placed on gentle, tree-poor land of the world prototype; checkpoints and lap times; the demo mode's AI driver races | `samples/games/racing/`, `game_racing` |

- **Running**: `./build/samples/games/game_<flight|platformer|racing> [--demo]` from a directory with a
  `config.ini` (like the other samples). `--demo` simulates a few seconds before the first frame, so
  `SAIGA_SCREENSHOT=<file.png>` (frame 60) shows the action. The controls are listed in each game's window. CI
  compiles the executables in the Clang job (samples ON) but does not run them (they need a window).
- **Structure**: each game is one executable (simulation in double, fixed step of 1/60 s in `MainLoop::update`,
  rendered per frame with interpolation; `main.cpp` maps the keyboard); the window, loop and frame wiring is shared
  (`samples/games/common/GameSampleHost.h`, header only). `samples/games/common` holds the shared helpers (mesh
  renderer, procedural meshes, chase camera, input, `GameMath.h`) and is compiled into each game. The games
  are samples and have no tests in `tests/` (the saiga tests test the engine, R-TEST-1); from S13 they have game
  scenarios (`--scenarios`). The simulation classes take the ground as a function (`GroundHeight`).
- **Transitional parts** (not the engine design; replaced when the stages arrive):

  | Today | Replaced by | Stage |
  |---|---|---|
  | game classes without `World` (own state, own fixed-step loop in `MainLoop::update`), `GameSampleHost` | game objects in a `World`, `EngineApp` main loop, game scenarios | S13 |
  | `MeshRenderer` (flat colored meshes, one draw per object, `shader/vulkan/games/`; all three games) | `RenderScene` instance sets and materials | S19 |
  | `ChaseCamera` (all three games) | the camera rig (third-person and chase cameras) | S35 |
  | procedural box animation of the platformer | skinned meshes, animation state and pose LOD | S34 |
  | platformer `CharacterController`, racing `Vehicle` | `CharacterVirtual`, Jolt's vehicle constraint | S29, S28 |
  | `GameInput` key helpers | the input action map | S13 |

## 19. Migration of the prototype

| Today | Becomes | Stage |
|---|---|---|
| `VulkanForwardRenderer` (MSAA, reversed-Z), `MainLoop`, `StandaloneWindow` | not used by engine apps (own loop in `EngineApp`, own frame in `RenderScene`); kept for the other samples or retired later | S13, S17 |
| gray look | `WorldSettings::shading = WorldShading::Clay` (enum with one value) [exists] | S01 (done) |
| wall-clock `time`, OpenMP reduction in `placeTrees` | `WorldSettings::freezeTime`; serial sum (tests in `tests/test_vulkan_world.cpp`) [exists] | S01 (done) |
| `sample_vulkan_world.cpp` | `--benchmark` [exists] (S03, done); the world game `samples/games/world/` (`game_world`) on `EngineApp`, `--headless` (S13) | S03 (done), S13 |
| `WorldMcp` | engine tools in `engine/tools/`; walk mode etc. as `game_*` tools | S14 |
| `WorldBenchmark` | benchmark runner in `engine/app/` (`--benchmark` of `EngineApp`) | S13 |
| `WorldCommon.h`, `common.glsl` (world uniforms) | uniforms per pass in `saiga_vulkan`, helpers stay | S19 |
| `ForestRenderer`, `tree_cull.comp`, `tree.vert`, `foliage.frag` (alpha-tested leaves, also used by the grass) | `RenderScene` instance sets, generic culling and instance shaders | S18, S19 |
| `WorldRenderer`, `WorldStatistics.cpp` | passes and statistics in `RenderScene`; settings in the game | S19 |
| `TerrainRenderer`, `HeightField`, `terrain.vert`, `terrain.frag`, `heightmap.glsl` | integer multi-level tiles + pyramid, edits, double camera and region-relative tile data (the shaders are camera relative already) | S20, extended in S22 |
| `GrassRenderer`, `grass.vert` | region-relative cells, mask from the prototype generator (S20); generated masks (S23); edited masks (S31) | S20 |
| `sky.vert`, `sky.frag`, `shading.glsl` (clay lighting, height fog), water plane (a mode of `terrain.vert` / `terrain.frag`) | double camera position, sea level from `SpaceTemplate` (S20; the shaders are camera relative already); sky and lighting replaced in S36, water in S39 | S20 |
| `FoliageGenerator`, `HeightField::generate`, `placeTrees` | generation pipeline with integer math (R-CONTENT-1) | S23 |
| `tests/test_vulkan_world.cpp` | render tests + engine tests | S19 |

The sample keeps running and all tests stay green after every stage.

## 20. Decisions and alternatives

- **Mechanism, not policy** (R-MECH-1): several games of the genre share the engine; game-level choices stay in games.
- **Hybrid instead of full ECS** (Bevy, Unity DOTS): unique behavior stays local in classes; data and bulk work in
  systems; R-STATE-1 keeps the option to move a type to pure systems.
- **Data-oriented back-ends from the start**: classic OO engines all added them later (instanced meshes, Mass Entity,
  servers, DOTS).
- **Own ECS with enum IDs** instead of EnTT / flecs: small, readable, DLL safe, no type lookups, with reflection and a
  deterministic iteration order.
- **Double precision state, float only for rendering**, made safe on the GPU by camera-relative rendering.
- **Tiny-eigen**: saiga's default; small, fast to compile, extendable in this repository.
- **Host-authoritative replication with trusted owner-authoritative clients** instead of lockstep or dedicated servers:
  works with players far apart in a streamed world and stays simple because cheating is not a concern.
- **Cross-machine deterministic generation**: clients generate the world locally and receive only modifications;
  worlds regenerate unmodified chunks after patches. This is the one place where bit-exact results across machines are
  required.
- **Height field only, caves and dungeons as separate spaces**: keeps terrain, streaming, collision and the massive
  view distance simple; enclosed scenes behind a loading screen give caves and dungeons without volumetric terrain.
- **Separate tier for placed pieces**: player bases with 10^4-10^6 pieces do not fit the entity tier.
- **Schema-based saves with migrations**: worlds survive game updates.
- **No scripting language, no modding**: C++ with compiler errors and tests gives coding agents better feedback; JSON
  and prefabs cover data.

**Simplifications** (each can be added later if it turns out to matter):

- **Reproducible only in `ScenarioWorld`** (§8.6): no thread-count independence and no bit-identical continuation after
  load required of games (the engine's own scenarios check both, §17.4), no Jolt snapshot or engine-allocated body IDs, streaming results apply when ready. Add later if lockstep or
  replays of live sessions are needed.
- **R32F GPU height tiles** (§12.2): no 16-bit tiles with per-tile scales and border rounding rules. Add later if
  terrain GPU memory becomes the limit.
- **Portable character files only** (§15.4): no characters stored in the world save. Add later as a policy option.
- **`alwaysSimulate` only while the chunk is resident** (§9.6): no streaming around important entities. Add later if a
  game needs traveling NPCs that interact with the world far from players.
- **Host-owned vehicles** (§13.3): no runtime ownership transfer; driving clients have one round trip of steering
  latency. Add later (ownership change with epochs) if that latency is a problem.
- **Games are samples** (§5.3, §17.2): one executable per game in `samples/games/`, no game libraries; the saiga
  tests test the engine with test-only game objects, games check themselves with their scenarios (`--scenarios`).
- **No input record/replay** (`InputLog`): scenarios are the reproducible path. Add later together with live
  determinism if bugs from live sessions need replays.
- **Plan-defined spaces only** (§12.1): no space instances created or deleted at run time. Add later (IDs from a
  saved monotonic counter, never reused) if a game needs them.
- **Round-robin tick/`Simulated` re-evaluation** (§9.2, §12.3): no incremental grid-cell rules. Add later if the
  1/16 latency matters.
- **Synchronous edits** (§12.2, §11): terrain results, collider patches and compound cell rebuilds run in the
  requesting step. Add later (asynchronous preparation, a shape filter for removed sub-shapes) if the S22/S30
  measurements exceed the step budget.
- **No access declarations per system** (`Reads<>`/`Writes<>` with asserts): systems run serially, R-NET-2 is a
  convention. Add later if systems are scheduled in parallel.
- **Full-state snapshots** (§13.4): no per-client delta baselines and no far-visible replication tier. Add later when
  the network budget of §17.10 is exceeded.
- **No runtime content-hash check** (§13.2): CI tests generation on every compiler. Add later (per-chunk hash with
  download fallback) if mismatches appear in the field.
- **MSAA only** (§10.5): no motion vectors, temporal anti-aliasing or temporal upscaling. Add later if foliage
  aliasing or Steam Deck performance requires it.
- **Global sea plane** (§10.9): no water extents, lakes at other levels or fill on dig. Add later if a game needs dry
  pits below sea level or lakes.

## 21. Open questions

- Impostor type (octahedral vs billboard) and far-content generation cost per region.
- Hi-Z occlusion culling from the previous frame's depth (forests, base interiors): when to add it.
- Jolt: height quantization, static compound cell size and rebuild cost at base scale (10^4 pieces per
  chunk).
- Network library (Steam Networking Sockets and/or ENet, §13.6) and snapshot rate.
- Game UI framework (HUD, inventory, crafting; stage S46) and audio service (stage S47): designed in their stage plans.
  Before S46 the loading screen and debug UI use ImGui.
- Decided in the S44 plan (game content, not engine): what moves with a character file into another world;
  terraforming limits and protected areas; the target number of pieces per 32 m cell; the island limit policy and the
  Steam Deck host limits; the commit interval; what remains when a player disconnects.

## 22. Implementation stages

The work is split into stages; each stage gets its own plan file (`docs/stages/Sxx_<name>.plan.md`) with the details
(files, signatures, tests), derived from the sections of this document. `docs/stages/S05_math_additions.plan.md` is the
template. Written before the implementation: Goal, Depends on, Rules that apply, Scope, Out of scope, Design notes,
Tests, Done when, Verification (commands; the AGENTS.md recipes where they apply) and Files. Filled in when the stage is
done: Status, Design changes (with the reason and the updated sections of this document) and Results. A stage starts
when all stages it depends on are done; stages without a dependency between them can be worked on in parallel. Status
markers `[planned: Sxx]` in this document name the stage that implements an item ("Sxx, extended in Syy" when two stages
contribute). Stage numbers identify stages; they do not define an order beyond the dependencies. An item that needs two
stages without a dependency between them is implemented and tested by the stage that is done later; the earlier one
leaves the documented stub (e.g. S36 renders piece lights if S30 is done, otherwise S30 adds them to the light list; S41
consumes the cosmetic events of S32 the same way; S36 adds impostor shadows if S27 is done; S27 bakes skinned impostors
if S34 is done, otherwise S34 does; S27 bakes the smooth-textured material parameters if S43 is done, otherwise S43
extends the baking; S23 re-snaps/tombstones tier-1 instances on terrain edits if S22 is done, otherwise S22 does; S35's
camera collides with pieces through the physics raycast, so S30 needs no camera change). "CI" = headless checks in
GitHub Actions; "local" = GPU checks with MCP.

| Stage | Name | What is implemented | Depends on | Done when |
|---|---|---|---|---|
| S01 | Prototype baseline (done) | Commit the prototype; clay shading enum, `freezeTime`, serial `placeTrees` sum (§19) | – | prototype committed, sample runs, existing tests green (CI) |
| S02 | Golden image tests (done) | `compareGolden`, lavapipe goldens `world_overview` and `world_forest` of the prototype, CI upload of failed images (§17.6) | S01 | goldens pass in CI |
| S03 | Benchmark runner (done) | `--benchmark` with `flyover` / `forest`, unsmoothed frame times, baseline file (§17.7) | S01 | baseline in `data/tests/benchmarks.json` for the development GPU (local) |
| S04 | Engine module setup and tech demos (done) | `saiga_engine` in CMake, export macro, module defines, test registration, Clang and Windows/MSVC (shared libraries) CI build jobs (§5.3); three tech demo games as samples in `samples/games/`: flight simulator, platformer, racing, with `samples/games/common` (§18.4) | – | `Module enabled: Engine` printed; engine library and a smoke test build and run on all CI compilers (CI); the games compile in the Clang job (CI); each demo runs with `--demo` without validation errors (local) |
| S05 | Math additions (done) | Missing tiny-eigen functions and double helpers (§7.10, R-MATH-2) | – | tiny-eigen tests for every addition (CI); `test_core_math_functions` builds and passes with Eigen (local) |
| S06 | Core utilities (done) | `JobPool`, `Rng`, `Hash`, logging ring buffer with `SAIGA_LOG` (§8.5, §8.6, §17.1) | S04 | unit tests: parallel-for coverage with the caller participating, RNG and hash reference values, log ring (CI) |
| S07 | ECS core (done) | `Entity`, registry, pools, component IDs, `visit`, views, `SAIGA_LOG_E`, `ChunkId`/`RegionId`/`SpaceId`/`InstanceKey`/`PersistentId` value types (§7.1-§7.5) | S04, S05, S06 | unit tests: stale handles, deterministic order, `visit` round trip; non-gating 10^5-entity benchmark (CI) |
| S08 | World and step loop (done) | `World(WorldSetup)`, phases, `SystemList`, commands, structural log, events, `Transform` hierarchy, `advance`/`step`, `hash` (§7.6-§7.10, §8.1-§8.3, §8.6) | S07 | unit tests: deferred destroy, commands during iteration, structural log consumers, event latency, `advance` cap, hierarchy; hash equal for 2 runs (CI); step time with 10^5 idle entities (benchmark) |
| S09 | Spatial grid (done) | Grid per space, `queryRadius` (§9.5) | S08 | query tests incl. deterministic order (CI) |
| S10 | Game objects (done) | `GameObject`, `Behaviour`, behavior and event systems, ticking policy, engine gameplay basics incl. local `PlayerAction` queue, kinematic mover, world-state entities, `Persistent`, the world API (`WorldApi`), the MCP world tools and the profiler (§7.8, §8.4, §9.2-§9.4, §9.6, §16.2, §17.1) | S09 | lifecycle tests (spawn with initial components, commands during a step, destroy), ticking tests, `global`/`spaceState`, `PlayerAction` delivery order, `characterMoveSystem`, `makePersistent`/`resolve`, JSON visitors, every API operation and every tool through `handleMessage` (CI) |
| S11 | Save, data and migration (done) | Save/load blobs with schema tables and migrations, save of `Behaviour` (factory, `configure`, `onLoad`), `Persistent` counter and world-state entities, `dumpJson`, `diff`, data files (§14, §15.1, §15.2) | S10 | save/load round trip incl. a migration, a removed type and an entity with a `GameObject` (`configure` + `onLoad` called, cache rebuilt); data file load incl. the unknown-key death test (CI) |
| S12 | Scenario framework (done) | `ScenarioWorld`, scenario list and runner, repeat and save/load variants, `tests/test_engine_scenarios.cpp` with the test-only game object `Wanderer` (§17.2, §17.4) | S10, S11 | `wanderer_flees_when_damaged` passes incl. the repeat and save/load variants (CI) |
| S13 | Engine app | `EngineApp` main loop, `createVulkanDevice`, `RenderScene` frame shell (swap chain / offscreen, MSAA + depth pass, ImGui, screenshots), window and headless mode, MCP polling, input map (keyboard, mouse), transitional render hooks, benchmark runner moved to `engine/app/`, sample on `EngineApp` (§8.3, §17.7); tech demos: flight, platformer and racing ported to `EngineApp` and to game objects in a `World` (§18.4) | S12, S02, S03 | headless `EngineApp` test: `runFrames` advances, also with rendering disabled; input map test (CI); sample runs and `--benchmark flyover` works (local); `--scenarios` runner; each tech demo has a scenario that passes with `--scenarios` and runs on `EngineApp` (local) |
| S14 | Engine MCP tools | App tools (`screenshot`, camera, the frame part of `measure_performance`, `run_scenario`, `quit`), prototype tools renamed to `game_*`; the world tools exist from S10 (§16.2) | S13 | every tool has a `handleMessage` test; a headless `screenshot` through `handleMessage` (CI) |
| S15 | Transport and messages | `Transport`, `LoopbackTransport`, channels, message registration and `world.send` (§13.1, §13.6) | S08 | loopback tests: channels, ordering, delay and loss, fragmentation (CI) |
| S16 | Replication skeleton | Ownership (fixed at creation, teleport counter), replication modes, `ownedBy`, `NetId`, full-state snapshots every step, `RenderPose`, host time estimate on clients, host + client scenario variant (§13.3-§13.5, §13.8, §17.3); tech demos: co-op variants (a second plane, character, car of a client) | S12, S15 | all scenarios pass in the host + client variant without loss, delay 0 and 2, and with asynchronous stepping (CI); the tech demo co-op variants run host + client over loopback (local) |
| S17 | HDR frame | Shader target Vulkan 1.3 / SPIR-V 1.6, HDR target, pre-exposure, custom resolve, tone map in the `RenderScene` frame (§10.1, §10.5) | S13 | headless frame test and a tone-map golden (CI) |
| S18 | Instances and culling | Mesh and material registry, instance buffer, offset table, dynamic instances with GPU interpolation, `setCamera`/`setInterpolation`, GPU culling with compaction and indirect draws, material data model, static mesh import (glTF via assimp into `MeshLods`, flat-color face duplication, palette indices; `saiga/vulkan/scene/MeshImport.h`) (§10.1, §10.2, §10.8, §14) | S17 | culling counts per LOD equal a CPU reference (§17.5 tolerance); dynamic instance interpolation test; an imported glTF mesh renders with face-duplicated flat normals (CI) |
| S19 | Extract and render sink | `RenderSink`, `RecordingRenderSink`, extract with interpolation (`RenderPose`), depth pre-pass, prototype passes inside `RenderScene`, trees as instance sets (§10.3, §10.5); tech demos: all three (flight, platformer, racing) render through `RenderScene` (`MeshRenderer` removed) | S18, S16, S14 | S02 goldens within tolerance with equal drawn counts per LOD (§17.5 tolerance); client extract from `RenderPose` flags (CI); culling + instance passes <= the S03 `trees` + `upload + cull` time, total <= the S03 baseline + 5 % (local); tech demo screenshots reviewed (local) |
| S20 | Height tiles | `SpaceTemplate` and `WorldConfig::spaceTemplates` (one entry, space 0), `prototypeHeight` hook, `seaLevel`, `kHeightUnit`, terrain surface rule, integer multi-level tiles, R32F GPU tiles, pyramid, CDLOD terrain with a double camera position and region-relative tile data (the prototype shaders are camera relative already), grass and water plane, visual tile streaming (§7.9, §12.2); tech demo: flight over the height tiles | S19 | terrain CPU+GPU memory <= 768 MB for 32 km at 1 m base spacing (`samples/games/world`); origin offset 0 vs 30 km within golden tolerance; a 1 mm camera move moves no vertex > 0.5 px; `terrainHeight` equals the integer rule at seeded random points incl. cell diagonals, and the finest CDLOD level has one vertex per sample with its rendered height matching within float tolerance; adjacent tiles of different levels show no background pixel along their border (CI); the S02 goldens are regenerated in a dedicated commit after review; flight demo screenshots reviewed (local) |
| S21 | Spaces and transitions | Further space instances, `moveToSpace`, entrances, loading-screen transitions (ImGui until S46) (§12.4) | S16, S20 | use-a-`SpaceLink` scenario (player ends in space 1 at the arrival pose) incl. save/load and host + client (CI) |
| S22 | Terraforming | Terrain brushes, synchronous integer result (collider patch from S28), asynchronous visual and nav data, live edit replication to every client (integer results), clients keep all edited tiles (§12.2) | S20, S16 | terraform scenario incl. save/load; host + client end with equal heights, delay 0 and 2 (CI); terraforming benchmark (integer result, CPU tile/pyramid update) within budget (local) |
| S23 | Generation pipeline | `Generator` interface, world plan with sites and space instances, chunk generation, prefab stamping (terrain patches, entities, static geometry; pieces in S30), tombstones, content hashes, pinned versions, the hash test on all CI compilers (§12.1, §15.3) | S20, S21, S22 | committed chunk hashes reproduced on all CI compilers; stamp scenario (prefab entity and terrain patch present); a tombstoned instance is hidden on the client in the host + client variant (CI) |
| S24 | Streaming | Simulation radii, prefetch, results applied when ready (synchronous in `ScenarioWorld`), arrivals, `Simulated` tag, edits of non-resident or pending chunks (§12.3) | S23, S22 | scenarios with streaming incl. the repeat and save/load variants; unit test with the asynchronous pool and a held result: an edit on a pending chunk survives when the result is applied, an edit on a non-resident chunk is in its modification state (CI); a headless run with the asynchronous pool and a player at 30 m/s has 0 streaming stalls (local); step time with 10^5 resident and 10^4 simulated entities and one player at 30 m/s within §17.10 (benchmark, local) |
| S25 | World save directory | Chunk store with commits, entity unload/reload with group remapping, `onCatchUp`, space unloading; players saved in the world-state file until S33 (§9.6, §15.1) | S24, S11 | unload/reload scenario; the save directory survives a kill at any point of a commit, also with an unload between two commits (CI) |
| S26 | World over the network | Handshake with schema hash, chunk messages and snapshots, modification sequence numbers, live modifications over loopback (§13.2) | S24, S16 | host + client variant with streaming; a join with a different schema hash is rejected; a client that reloads a chunk while an edit is in flight ends with the host's heights (CI) |
| S27 | Far content | Region quadtree with hierarchical placement, modification summaries in `save()` blobs and chunk messages (in chunk store records once S25 is done), relightable impostors, baking tool (§10.4); tech demo: the flight demo shows far content to the horizon over 32 km | S26 | flight across 32 km at 200 m/s: p99 <= 1.5x median, no frame > 33 ms, 0 streaming stalls (local); far set equals the filtered full chunk incl. a site; a tombstoned area stays cleared at all levels; a region with mixed placement versions (CI); flight demo flight to the horizon reviewed (local) |
| S28 | Physics core | Jolt required by the engine (CMake default, all engine CI jobs), `PhysicsWorld`, terrain and compound colliders, edit patches, bodies (recreated from components on load), `collisionAssets`, contacts (§7.9, §11); tech demo: the racing car on Jolt (vehicle constraint) | S24 | physics scenarios incl. the repeat and save/load variants; an edit on a chunk corner without a rebuild; Jolt's diagonal equals the terrain surface rule (CI); terraforming benchmark re-run with collider patches within §17.10 (local); `AGENTS.md` updated (`SAIGA_WITH_JOLT` default); the racing demo drives on Jolt (local) |
| S29 | Characters and proxies | `CharacterVirtual` movers, client proxy bodies (§11, §13.3); tech demo: the platformer character on `CharacterVirtual` | S28, S16 | character scenarios incl. host + client (CI); `--benchmark host_step` without clients within §17.10 and with 0 streaming stalls (local); the platformer demo runs on `CharacterVirtual` (local) |
| S30 | Placed pieces | Piece types, per-chunk piece arrays, stamped pieces of prefabs, rendering, merged far meshes per region and summary input, colliders, light descriptions, queries, live modifications (§12.5) | S27, S28 | build-a-wall scenario incl. save/load and host + client (CI); synchronous cell rebuild for a piece edit in a full base cell within the step budget; `host_step` re-run with the placed-piece base (local) |
| S31 | Promotion and grass edits | `promote`/`demote`, tombstones from interaction, `editGrass` (§6, §12.5) | S30 | chop-a-tree scenario incl. save/load and host + client (CI) |
| S32 | Network production | Interest radius, snapshot rate (20 Hz), cosmetic events, loss and reordering handling, snapshot packets (§13.4) | S29, S26 | all scenarios pass in the host + client variant with seeded loss and reordering (CI); `--benchmark host_step` with 7 clients within the step budget, upload per client reported against §17.10 (local) |
| S33 | Sessions and transport | UDP / Steam Networking transport, drop-in/out, character files (§13.6, §13.7, §15.4) | S32, S25, S30 | character file round trip incl. a rejoin of the same world (`Persistent` IDs kept) and a rejoin with the file of an older commit (receives the loaded commit's character); deposit, disconnect, commit, crash, rejoin: no duplicate item; leave and rejoin before the next commit: no item lost; a loopback disconnect removes the group and emits `PlayerDisconnectedEvent` (CI); two-process co-op session over UDP, join data reported against §17.10 (local) |
| S34 | Skinned characters | Skinned meshes, GPU skinning, animation state and replication, pose evaluation and LOD (§10.6); tech demo: the platformer character skinned, with the animation state machine (replaces the procedural box animation) | S19, S16 | animation scenario (state replication, pose LOD) (CI); screenshots reviewed (local); platformer demo screenshots reviewed (local) |
| S35 | Camera rig | Third-person camera with collision, gamepad input (§8.3, §10.6); tech demos: all three on the camera rig (third-person for the platformer, chase cameras for the car and the airplane; `ChaseCamera` removed) | S29, S34 | camera collision tests (CI); reviewed in a session (local); tech demo cameras reviewed (local) |
| S36 | Lighting | PBR shading, sun/moon cascades, far terrain shadows, clustered lights with cached shadow atlas, environment map, sky with 2D clouds, `WorldTime`/`Weather` (§10.7, §10.11); tech demos: lighting and shadows shown in the flight demo (day and night) | S19, S20 | golden images per time of day (CI); flight demo screenshots reviewed (local) |
| S37 | Post chain and tiers | AO, DOF, bloom, exposure, grading, render scale with spatial upscaling, quality tiers and budgets incl. Steam Deck (§10.5, §17.9, §17.10); tech demos: post chain and tiers in all three demos | S36, S27 | per-pass budgets met on the desktop reference (or reported as open per §17.10 if no such card is available) and, when available, the Steam Deck (local); benchmark baselines in `data/tests/benchmarks.json` for the desktop reference class (RTX 3060 / RX 6600, §17.10; or reported as open per §17.10 if no such card is available) (local); golden images per tier (CI); tech demo screenshots per tier reviewed (local) |
| S38 | Volumetric fog | Froxel fog with light shafts, fog volumes, analytic fog to the horizon (§10.5.1); tech demo: flight through volumetric fog | S37 | golden images with fog and local lights (CI); fog within its budget (local); flight demo screenshots reviewed (local) |
| S39 | Water | Global sea plane, waves, `waterHeight`, buoyancy, exclusion volumes, rendering incl. the transparent pass, underwater (§10.9); tech demos: flight over the sea, racing along the coast | S28, S37 | boat-floats scenario incl. host + client; a pit dug below sea level shows water; golden images above and below water (CI); water within its budget (local); tech demo screenshots reviewed (local) |
| S40 | Moving platforms | `RideFrame`, local-frame movement, vehicle control through the driver's input, replication relative to platforms (§13.3) | S39, S29 | passenger stays on a host-owned boat steered by a client (turning boat, delay 2, driver and passenger on two different clients) and step-from-boat-to-dock scenarios (CI) |
| S41 | Particles | GPU particles in the transparent pass, emitters, weather particles (§10.10) | S39 | particle golden images (CI); particle budget (local) |
| S42 | Volumetric clouds | Depth-aware cloud ray march, fog coupling, cloud shadows (§10.11); tech demo: flight through and above the clouds | S38 | golden images below, inside and above the cloud band (CI); cloud pass within the High-tier budget (local); flight demo screenshots reviewed (local) |
| S43 | Materials and look | Full material set, colored look (§10.8) | S36 | golden images for both material models (CI); screenshots reviewed (local) |
| S44 | First gameplay | Player, NPCs and a boss in `samples/games/world` | S31, S35, S43, S46, S47 | a game scenario per feature, passing with `game_world --scenarios` (local); screenshots reviewed (local) |
| S45 | Navigation (optional) | Navigation module for path-finding NPCs (§9.7) | S30, S20 | path-finding scenarios incl. terrain and piece edits (CI) |
| S46 | Game UI | UI framework for HUD, menus, inventory and the loading screen; designed in its plan (§21) | S13 | UI layout tests headless (CI); screenshots reviewed (local) |
| S47 | Audio | Audio service: positional sources, music, mixing by game policy; designed in its plan (§21) | S13 | headless mixer tests (CI); listened to in a session (local) |
