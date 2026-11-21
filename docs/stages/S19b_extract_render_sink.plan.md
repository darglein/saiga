# S19b: Extract and render sink

Status: done.

Sections follow the stage plan template of S05.

## Goal

The engine renders entities: a `RenderSink` interface between the `World` and rendering (`VulkanRenderSink` over
`RenderScene`, `RecordingRenderSink` for tests), mesh and material name tables owned by the `World`, the
`RenderInstance` component, and the extract that sends interpolation pairs of changed poses (from `RenderPose` when
the entity has one, else `Transform`) to the sink, also for edits between steps. One double view camera
in the app. The three tech demos render their entities through it (`MeshRenderer` and `PoseHistory` removed). Split from
S19 in planning (S19a: the prototype passes).

## Depends on

S19a (the prototype world inside `RenderScene`), S18 (instances, explicit IDs, the GPU interpolation formula), S16 /
S16b (`RenderPose`, roles, `--coop`, the demos' driver entities and render caches), S14 (camera hold, camera tools,
`screenshot`), S11 (saves, load listeners).

## Rules that apply

R-STATE-1 (the extract and the sink hold no state the save needs), R-MATH-1, R-DET-1, R-PAR-1, R-REG-1
(`WorldSetup::renderAssets`), R-NET-2 (clients render remote entities from `RenderPose`), R-SAVE-1 (`RenderInstance` is a
new saved component), R-TEST-1, R-TEST-2, §5.2 (`engine/sync/` depends on `saiga_vulkan`, never the reverse).

## Scope

### 1. Sink and name tables (`engine/sync/`, `World`)

- `RenderSink` as §10.1 (the S18 subset), `VulkanRenderSink` (over `RenderScene`, in `saiga_engine`), and
  `RecordingRenderSink`: keeps state (live handles, last poses, counters) with an opt-in call log, so long scenarios do
  not grow; it never changes `world.hash`.
- `World::registerMesh(name, std::shared_ptr<const MeshLods>, MaterialId)`, `registerMaterial(name, MaterialData)`,
  `findMesh(name)` (§7.10): the World assigns the IDs (the S18 explicit IDs) and keeps the data by `shared_ptr` (no
  duplicate for late forwarding); `setRenderSink(sink)` (any time, null allowed) forwards every registration and
  creates the render instances of existing entities. `WorldSetup::renderAssets(World&)` registers them at world
  creation; games build each `MeshLods` once per process (a static cache of `shared_ptr<const MeshLods>`), so the many
  scenario worlds share them.
- ID ranges: content registered directly on `RenderScene` (S19a's tree meshes and material, the level geometry of §5;
  `VulkanRenderSink::scene()` gives the game the scene) takes IDs from the game range `1 .. kGameReservedIds - 1`
  through `RenderScene::reserveIds`; the World assigns mesh and material IDs from `kGameReservedIds` upward; S30's
  generated meshes take the top. `VulkanRenderSink` accepts an identical (ID, name) re-registration as a no-op (a
  second `setRenderSink` of the same sink) and asserts on a different one. The table hash (names and IDs) is part of what S26's handshake compares; S19b asserts equal tables in the
  host + client variant.

### 2. `RenderInstance` (`engine/components/RenderInstance.h`)

`RenderInstance {std::string mesh; /* cached MeshId, not visited */}` (registered `HostReliable`: rarely changing,
names stay out of the snapshots; `world.hash` hashes the name, so it does not depend on registration order). An empty
name means "tracked, not drawn": the extract keeps its pose pair for `interpolatedPose` (the platformer hero until
S34, S16b's driver entities) without a sink instance.

- Every write of the name goes through one engine-internal path, `World::renderMeshWritten(e)`, which validates the
  name and appends a structural record: `World::setRenderMesh(e, name)`, the world API's `set_component` /
  `add_component` of `RenderInstance` (an unknown name is an API error, nothing changes), and the `add` apply path
  (spawn initial components, command flushes, S16's client `Spawn`, `ComponentChange` and `Resync` apply). `get` /
  `tryGet` and mutable views of `RenderInstance` are refused at compile time (static_assert, as for `Parent` and
  `Persistent`). An unknown name in the `add` path is refused in every build: logged (`SAIGA_LOG_E`) and stored as the
  empty name (tracked, not drawn), so a game's own save stays loadable. The record is a remove + add pair of
  `RenderInstance` (accepted: replication presence and `changes_since` report it so).
- Loads: a new loader step `checkRenderMeshes` over the staged `RenderInstance` block, before `commit` (validate-then-
  commit, §15): an unknown name is a load error naming the entity. A game renames a mesh with
  `World::registerMeshAlias(old, name)`, consulted by `checkRenderMeshes` and `renderMeshWritten` (`MigrationList` is
  keyed by the engine component's version, which a game cannot bump).
- Scale: the extract uses `scale.x`; a non-uniform scale on an entity with `RenderInstance` is rejected by the world
  API's `set_transform` (an error), and logged once per entity (`SAIGA_LOG_E`) when game code sets it; tints are
  materials.

### 3. Extract (`engine/sync/ExtractSystem.{h,cpp}`, owned by the `World`)

- Placement: `World` calls the extract explicitly after S16's interpolation call at the end of phase 8 (a fourth
  explicit call after the phase-8 flush, not a `SystemList` system), with its own profiler scope `extract` (§17.1).
  Its state commits with the step: if the step aborts, prev/curr are not shifted (they shift when the rerun completes).
  `World::extract()` gives access (catch-up, `interpolatedPose`) also without an app. Not run by `stepNetworkOnly`
  (the next real step or catch-up consumes those structural records). In `--coop` only the client world has a sink.
- Source: "has `RenderPose` → `RenderPose`, else `Transform`" (§10.3: every peer; the host uses `RenderPose` for
  client-owned entities); adding or removing `RenderPose` switches the source with a snap. Remote roots (replicated
  and owned by another peer: on a client every root not owned by `localPeer()`, on the host the client-owned roots)
  without `RenderPose` yet are tracked without an instance until their first sampled `RenderPose` (no pop back by the
  delay). Children of a `RenderPose` root compose the root's `RenderPose` with their local chain (`LocalTransform`s),
  so they stay with the delayed parent. S16b's
  driver system writes `RenderPose` in phase 4 on clients; "changed this or last step" covers it.
- Per step: for entities whose pose changed in this or the previous step: prev = old curr, curr = new pose (teleported:
  prev = curr), chunk-relative, `setDynamic`. Instances are created and freed from the structural log (spawn, destroy,
  add/remove, the name record); creation always sends the current pose as a snap (a spawn in phase 7 or 8 is not in
  `changedTransforms()`). Changes in phases 7-8 arrive one step late; late teleports use `wasTeleportedLastStep`. A load
  needs nothing extra: it appends removes and adds to the structural log and marks all transforms changed, so consuming
  the log recreates the instances and snaps all poses.
- At the end of each step's extract, entities whose source changed in phases 7-8 (flagged, not in the phase-6 list)
  get their catch-up baseline (the last observed source pose, kept apart from curr) refreshed, so catch-up never
  mistakes them for edits; curr stays, and the next step's pair (old curr, new pose) carries the move (no snap).
- Between steps (`ExtractSystem::catchUp()`, called by `EngineApp` before rendering, and by tools that render): when
  the frame ran no step and (the world is paused, or `timeScale() == 0`, or an MCP tool or a load ran, or the
  structural log has unread records for the extract's consumer); it consumes the structural log and
  compares each tracked entity's source pose bitwise with the catch-up baseline (as `PoseHistory::interpolated` does;
  the change flags cannot be used: a setter's flag stays set until the next step ends, and `RenderPose` flags stay set
  after each step) and handles a pending `wasTeleported()` once; a difference is sent as a snap. Cost: O(tracked entities)
  per frame while paused; no work in frames between running steps.
- `bool ExtractSystem::interpolatedPose(Entity, double alpha, WorldPose& out, bool* teleported)` with `WorldPose {SpaceId
  space; Vec3 position; Quat rotation; double scale;}` in world space (for cameras): false for an untracked entity (dead,
  spawned since the last extract, a remote not yet valid; `out` untouched); `teleported` = the current pair is a snap
  (stateless, the same for every reader until the next pair); plus `toChunkPose(WorldPose) ->
  {ChunkId, DynamicPose}` for cosmetic `setDynamic`; the formula is S18's GPU interpolation (`interpolatePose` in
  `RenderSceneTypes.h`, nlerp with the sign flip) in double.

### 4. Camera (`EngineApp`)

`EngineApp::setViewCamera(SpaceId, Vec3 position, Quat rotation, double fovy, bool snap = false) -> bool` is the single
camera, the camera of the next rendered frame used as is (no interpolation by the app: the games set it per frame from
`interpolatedPose`; `snap` only reaches later consumers such as the S35 rig). While a tool holds the camera (S14) the
app ignores non-tool calls and returns false, so games need no branch; on the release edge it calls the game's camera
reset (`EngineAppSetup::onCameraReleased`, the chase camera's `reset()`). S14's `set_camera` / `move_camera` write it
(the ±1e6 m range stays); the `WorldMcpContext` setter of `game_set_camera` counts as a tool write (not blocked, no
hold), and `game_world`'s walk mode adjusts the height through it also while held (S14's behaviour); the free camera integrates its motion in double. `app.camera()` becomes `const
PerspectiveCamera&` derived from it (remaining readers: the input map's heading, old GUI code). `game_set_camera` and
the `fovy` of `game_set_settings` (in `WorldMcp`, `saiga_vulkan`, no engine header) reach it through getter/setter
callbacks in the `WorldMcpContext` that `game_world` sets. S14's camera tests that change are listed in the PR.

### 5. Render hooks and games

- `GameRenderHooks` keeps `init` / `shutdown` (device-bound game state: tree sets, level instance sets, cosmetic
  dynamics), `gui`, `groundHeight` (until S20), and gains `beforeRender(EngineApp&, double alpha)`, called after
  `catchUp()` and before recording, not with `--no-render`; `transfer`, `render` and `passTimes` are removed (§8.3).
- Level geometry (the platformer level, the race track with its cones and banners) stays constant context and becomes
  instance sets the game creates in `init` after the sink is set (like the trees): one instance per mesh in region
  level 0 at the level's origin region, region-relative float vertices; its bounds drive culling, and its LOD has a
  single level with an end distance beyond the level's extent (no LOD-out by the distance to its origin); the game owns
  the handles; scenario worlds get none (they have no sink beyond the recording one and no game init).
- Cosmetic dynamics (racing wheels, the platformer's limbs until S34): a `common/` helper `CosmeticDynamics` reconciles
  `Entity → handles` in `beforeRender` over a view (spawns, destroys, loads, client despawns and resyncs), frees the
  handles of dead or missing entities, clears everything in `shutdown`, and calls `setDynamic(prev = curr)` each
  frame from `interpolatedPose`. Wheels: one mesh per radius (no non-uniform scale).
- The demos: their meshes (`MeshBuilder` → `MeshLods`, colors as materials) are registered in `renderAssets`; the
  vehicles are entities with `RenderInstance`; the hero has an empty-name `RenderInstance` (tracked) until S34.
  `MeshRenderer`, `PoseHistory` and `shader/vulkan/games/` (if unused) are removed; S16b's client caches use
  `interpolatedPose`.

## Out of scope

Height tiles (S20), skinning (S34), the camera rig (S35), lighting (S36), the post chain (S37), particles and
transparents (S39, S41), far content (S27).

## Design notes

- The extract sends changes only (this or the previous step), so idle entities cost nothing per frame (§10.3); the
  between-step catch-up keeps the paused-world editing that `PoseHistory` supported.
- `RenderInstance` by name like `DataRef` keeps `visit` free of World context, and `HostReliable` keeps names out of the
  snapshots.

## Tests

- `tests/test_engine_extract.cpp` (new, no device; `RecordingRenderSink`): registrations forwarded when the sink is set
  late; `setRenderSink(null)` and a replacement free the old handles and the consumer; instances created and freed from
  the structural log (spawn, destroy, add/remove, `setRenderMesh`, `set_component` of the name through the world API,
  a client's `ComponentChange`); creation sends a snap; prev/curr pairs for a moving entity, the stop after two idle
  steps, the teleport snap, chunk-relative poses across a chunk border; a paused world: an edit through the world API
  and a spawn reach the sink once over N frames (not every frame), `load_snapshot` recreates the instances (the live
  handle count equals the entity count); a client frame without a step sends nothing; an aborted step does not shift
  prev/curr twice (a sink that throws mid-extract: the unread structural record is redelivered, the pairs are
  not shifted twice); phase 7-8 movers are not sent as snaps by a catch-up after a tool call; the trigger at time scale
  0; a remote root drawn only after its first sampled `RenderPose`, a child of a `RenderPose` root drawn with it; the
  `RenderPose` switch with a snap and no drawing before the first sample (S16 over loopback);
  `RenderInstance` save by name, `checkRenderMeshes` rejecting an unknown name, the hash independent of registration
  order, the schema golden regenerated; the recording sink does not change the hash; `interpolatedPose` matches S18's
  `interpolatePose` within 1e-6 m / 1e-6 rad; a non-uniform `set_transform` on a rendered entity is an API error; an
  empty name is tracked, not drawn; an unknown name through `add` is stored empty and logged; `registerMeshAlias`;
  `interpolatedPose` false for untracked entities; a second `setRenderSink` of the same sink; reserved ID ranges are
  skipped; host and client tables equal.
- `tests/test_engine_app.cpp`: `setViewCamera` and S14's camera tools on it, the hold returning false to the game and
  the release callback, `beforeRender` after catch-up, the removed hooks, a `--coop` client with a sink.
- `ScenarioWorld` gives every world it creates (the host, each client, the save/load replacement, repeat runs) its own
  `RecordingRenderSink` and removes its consumer on teardown (§17.2), so the extract runs in every scenario.
- Benchmark (R-TEST-2, `tests/test_engine_ecs_benchmark.cpp`): 10^4 moving entities with `RenderInstance` and a
  recording sink: the step's `extract` scope and a paused-frame catch-up with nothing changed, against the §17.10
  main-thread budget.

Locally: the three demos' `--scenarios` and `--demo` screenshots compared with S19a's; `--coop --demo` screenshots;
`game_world --benchmark flyover` and `game_world --benchmark forest` unchanged within 5 %.

## Done when

- The S19b part of the S19 row of §22: client extract from `RenderPose` flags (CI); tech demo screenshots reviewed
  (local).
- `docs/engine_design.md` updated: §10.1 (`RenderInstance {std::string mesh}`, the recording sink's role,
  `RenderSink::registerMesh` with `shared_ptr<const MeshLods>`, the ID ranges), §7.10's `World` sketch, §10.3 (as
  implemented, catch-up, `interpolatedPose`), §7.4 (flag readers between steps), §7.6 (the extract consumer, the name
  record), §7.9/§7.10 (`renderAssets`, `registerMesh` with `shared_ptr<const MeshLods>`, `setRenderSink`,
  `setRenderMesh`), §13.4 (`RenderInstance` as a `HostReliable` example), §15 (`checkRenderMeshes`), §16.2
  (`game_set_camera`, the `set_camera` range), §8.3 (hooks, `setViewCamera`), §17.2 (the
  recording sink), §18.4 (the demos' transitional parts: `MeshRenderer`, `PoseHistory` gone), §19, §3, §22 (S19b);
  `AGENTS.md` (`engine/sync/`, the new test, the test count).

## Verification

The CI recipe of AGENTS.md with samples ON; `ctest -R "test_engine_(extract|app|scenarios)"`; the demos as in Tests.

## Files

`src/saiga/engine/sync/{RenderSink.h,VulkanRenderSink.{h,cpp},RecordingRenderSink.{h,cpp},ExtractSystem.{h,cpp}}`,
`src/saiga/engine/components/{RenderInstance.h,EngineComponents.cpp,EngineIds.h}`, `src/saiga/engine/world/World.{h,cpp}`,
`src/saiga/engine/api/WorldApi.cpp`, `src/saiga/engine/save/WorldLoad.cpp`, `src/saiga/engine/tools/WorldTools.cpp`,
`src/saiga/engine/net/Replication.cpp` (the client apply), `src/saiga/engine/app/EngineMain.cpp` (`--coop`: a sink on
the client only), `src/saiga/vulkan/scene/RenderScene.*` (`reserveIds`),
`src/saiga/engine/app/{EngineApp,EngineAppSetup,AppTools}.*`, `src/saiga/engine/tools/ScenarioWorld.*`,
`src/saiga/vulkan/world/WorldMcp.*` (`game_set_camera` callbacks), `samples/games/` (incl. the removed
`common/{MeshRenderer.*,GameEngine.h}` parts and `shader/vulkan/games/`), `tests/test_engine_ecs_benchmark.cpp`, `tests/test_engine_extract.cpp`,
`tests/test_engine_app.cpp`, `tests/CMakeLists.txt`, `data/tests/engine/schema.json`, `docs/engine_design.md`,
`AGENTS.md`, this file.

## Design changes

- **The sink and the consumer belong together**: without a render sink the extract tracks nothing and has no structural
  log consumer (a world without a sink, e.g. the `--coop` host or `--no-render`, pays nothing); `interpolatedPose` is
  false there. `setRenderSink` adds the consumer and creates the instances of the existing entities (snaps); a
  replacement or null removes the old instances and the consumer. The World destroys the extract before its log (the
  sink is not called from the destructor). `World::renderSink()`, `ExtractSystem::instanceOf` / `instanceCount` /
  `trackedCount` and `World::meshName`, `meshCount`, `materialCount`, `resolveRenderMesh` were added for tests and tools.
- **Catch-up trigger**: `markDirty()` (every world API edit, a committed load) makes the next `catchUp()` run also in a
  frame that ran steps, unless a step committed after it (the step's extract covers edits made before it). The plan's
  "the frame ran no step and (... a tool or a load ran ...)" missed a tool call after the step tool in the same
  `mcp.poll()`. The step's commit clears the mark. Tested by `ExtractCatchUp.LateMoversAreNotSentByACatchUp`.
- **Source evaluation**: the extract processes phase 6's `changedTransforms()` plus a list of the tracked entities
  whose source is not `Transform` (RenderPose, composed children, pending remotes), evaluated every step by value
  (pose differs from curr, or the pair is not yet (curr, curr)) instead of reading the `RenderPose` flags: this covers
  the plan's "changed this or last step" rule, the composed children (their root's `RenderPose` changes without a flag
  on them) and the pending-to-valid transition. A `RenderPose`, `Parent` or `NetOwner` record re-evaluates the sources
  (O(tracked) on such steps; a client's spawns). The phase 7-8 movers are found through `changedDuringLastStep()` at
  the commit (the extract retains the step changes while a sink is set): O(changed), no scan of the tracked entities.
- **Teleports**: a `Transform` teleport snaps in the step it is seen; a late one (`wasTeleportedLastStep()`) snaps only
  if the extract did not already handle it in the previous step (otherwise every teleport would snap twice); a teleport
  that the catch-up sent stays a snap only there (the next step interpolates from it).
- **Scale and logging**: the non-uniform scale is logged by the extract (once per entity, `SAIGA_LOG_E` category
  `extract`), not by `Transform` (no World there); the world API rejects it in `set_transform` and in `spawn` with a
  `RenderInstance` initial component.
- **`setRenderMesh` on an entity without `RenderInstance`** adds it (through the add path); during a step it is a
  command. An unknown name asserts (refused in every build).
- **Unknown mesh names are kept** (local review): the plan's "stored as the empty name" (the `add` path) and "an
  unknown name is a load error" (loads) became: the name is kept with the cached ID 0 (tracked, not drawn), logged
  (per write; one line per load), and a later `World::registerMesh` creates the instance; a game's own save stays
  loadable. The world API and `setRenderMesh` still refuse an unknown name.
- **Name records on the client**: `Replication::applyComponent` calls `renderMeshWritten` with the previous name, so an
  unchanged name (a `ComponentChange` of other bytes, never for `RenderInstance` today) appends no record.
- **Camera**: `ViewCamera` (`EngineApp.h`) with `cameraLookRotation(dir)` (double); `setViewCameraByTool(..., hold)` is
  the single tool path (set_camera / move_camera hold, `game_set_camera`, `game_set_settings`' fovy and the walk mode
  do not); the tools keep the view camera's field of view. The free camera keeps the GLFW camera's mouse look (the
  rotation is taken only when the mouse turned, so a tool's exact orientation stays) and integrates WASD / space /
  shift in double. `WorldMcpContext::camera` became `getCamera` / `setCamera` with `WorldMcpCamera {position, rotation,
  fovy}`; `game_set_camera` keeps the rotation exactly.
- **`GameRenderHooks::groundHeight`** (§8.3's sketch) never existed in the code: not added (the benchmark reads the
  height field itself); the sketch is updated.
- **Benchmark JSON**: `game_world --benchmark` also reports `extract` (`{median, p95}` of the measured steps' extract
  scope, §17.7, §17.10); `cpu_ms` is measured at `beforeRender` (right before `renderFrame`; S19a: the transfer hook);
  `recording` stays S19a's `cpu_work_ms` and an incomplete run still fails (`WorldBenchmark::complete()`).
- **Demos**:
  - Meshes: `samples/games/common/RenderAssets.{h,cpp}`: `meshLods(MeshData)` (one LOD; the linear vertex colors as sRGB8
    color words: the colors stay vertex colors, not one material per color), `demoMaterial()` (flat color, two-sided:
    some builders are not consistently wound, the normals follow the winding, so `flat.frag`'s `gl_FrontFacing` flip
    lights the visible side as `mesh.frag` did), `registerDemoMesh` (a static cache by name, behind a mutex).
  - Level geometry: `LevelGeometry` (one mesh and one instance set per level, registered on the scene with
    `reserveIds`, reused by name on a second init); the racing level merges the track, the cones and the banner, the
    platformer's the boxes and the flag.
  - `CosmeticDynamics` (`common/CosmeticDynamics.{h,cpp}`): `begin` / `set(entity, meshes, poses)` / `end` per frame
    (the parts of entities not set are freed), `clear` in shutdown.
  - Wheels: one mesh per wheel radius of the vehicle data (`racing/wheel/r<radius>`); the platformer's 11 body parts are
    meshes `platformer/hero/part<j>` registered in `renderAssets` (`heroPartMesh` moved to `PlatformerWorld.cpp`).
  - The platformer has no world prototype: its meshes use RenderScene's fixed S18 light (sun `normalize(0.4, 1, 0.7)`,
    ambient 0.35) instead of its own (sun (0.4, 0.8, 0.35), hemisphere ambient 0.35): slightly different shades.
  - `HeldCamera` is gone (`EngineAppSetup::onCameraReleased` calls the renderer's camera reset); the chase cameras keep
    updating while a tool holds the camera (the app ignores their writes), so the release needs no glide.
  - `RenderCaches<State>` keeps only the state ring (the pose comes from `interpolatedPose`); its snap detection reads
    the `RenderPose` teleport flag or `wasTeleportedLastStep()` directly. `WorldLighting.h` keeps only `worldGround`.
- **Files**: no change was needed in `WorldTools.cpp`, `EngineMain.cpp` and `RenderScene.*` (`reserveIds` existed
  since S19a): the world API marks the extract dirty for every edit (also the tools'), the app creates the sink.

## Results

Tests (`VK_DRIVER_FILES=.../lvp_icd.json ctest --parallel 8`, RelWithDebInfo, GCC 15): 67 of 68 pass; only
`test_opengl_window` fails locally (no display, as before). `test_vulkan_*`, `test_engine_app` and
`test_engine_extract` (15 tests) also pass on the RTX 5080 Laptop GPU with the validation layer (no validation
message).

- `tests/test_engine_extract.cpp` (new, 19 tests, no device): registrations forwarded when the sink is set late, the ID
  range from 256, a second `setRenderSink` of the same sink; replacement and null free the handles and the consumer;
  host and client tables equal; instances from the structural log (spawn, remove / add, `setRenderMesh`, the world
  API's `set_component` with a known and an unknown name, `add_component` and `spawn` with an unknown name, destroy);
  the creation snap; the empty name (tracked, not drawn); an unknown name through `add` (kept, logged; drawn once the mesh is registered later, see Local review);
  `registerMeshAlias`; pairs of a moving entity, the stop after the idle steps, the teleport snap, chunk-relative poses
  across the 256 m border; `interpolatedPose` against `interpolatePose` (1e-6 m / rad) and false for untracked
  entities; an aborted step (a sink that throws in `setDynamic`: the rerun's prev is the committed curr; one that throws
  in `addDynamic`: the record is delivered again); the recording sink keeps the hash; the paused catch-up (an API edit
  and a spawn reach the sink once over 5 frames, `load_snapshot` recreates one instance per entity); time scale 0; phase
  7 movers not sent by a catch-up after a tool call; a client frame without a step sends nothing (and draws the remote
  entity from `RenderPose`); a client's `ComponentChange` of the name; the host draws a client-owned root only after its
  first `RenderPose` and its child composed from it; saves by name (another registration order: equal hash; an unknown
  name: loaded and kept, see Local review; an alias); the non-uniform scale (API error, logged once).
- `tests/test_engine_app.cpp` (44 tests): `ViewCameraHoldAndRelease` (the double camera, the hold returns false to the
  game, tool writes without a hold, the release callback once, `move_camera` on it), `BeforeRenderAfterCatchUp`,
  `CoopClientHasTheSink`; `FramesStepsScreenshotsAndProfiles` counts `beforeRender` instead of `transfer` / `render`,
  `GpuSectionsAndGamePasses` became `GpuSections` (no game passes). S14's camera tests (`SetAndMoveCamera`,
  `InputMovesRelativeToAHeldCamera`) are unchanged and pass on the double camera. `test_vulkan_world`'s `GameTools`
  uses the camera callbacks and checks that `game_set_settings`' fovy reaches the camera.
- Schema golden regenerated (`RenderInstance {mesh: string}` added; nothing else changed).
- Game scenarios (`--scenarios`, with the host + client variants and a recording sink in every world): `game_flight`,
  `game_platformer`, `game_racing` 18 passed, 0 failed each.
- Screenshots (`SAIGA_SCREENSHOT`, `--headless`, RTX 5080, with `--validation`: no message): `--demo` of the three
  demos next to the S19a tip's: the same views (flight: the plane over the forest; racing: the car with its wheels on the
  track; platformer: the hero on a platform with the flag); the platformer's shades differ slightly (the scene's fixed
  light). `--coop --demo`: flight both planes, racing both cars, platformer the client's hero. `game_world`'s start view
  is pixel-equal to the S19a tip's in the 3D view (only the ImGui panel differs).

Benchmarks (release build of AGENTS.md with tests, RTX 5080 Laptop GPU, 1920 x 1080; before = the bb89f993 tip built
from a copy in a scratch directory, after = this stage; 5 runs each, before and after interleaved; medians of the run
medians, ms, ranges in parentheses):

| | flyover before | flyover after | forest before | forest after |
|---|---|---|---|---|
| `cpu_ms` (frame interval) | 2.494 (2.48-3.36) | 2.423 (2.35-2.47) (-2.9 %) | 2.410 (2.34-2.96) | 2.357 (2.34-2.88) (-2.2 %) |
| `cpu_work_ms` (recording) | 0.181 | 0.179 (-1.2 %) | 0.158 | 0.151 (-4.3 %) |
| `gpu_frame_ms.gpu` | 2.279 (2.12-2.31) | 2.125 (2.12-2.36) | 1.828 (1.81-1.91) | 1.822 (1.80-1.84) (-0.3 %) |
| `scene.instances` | 1.431 | 1.431 (0.0 %) | 0.466 | 0.466 (-0.1 %) |
| `cull` | 0.070 | 0.070 | 0.033 | 0.033 |
| `scene.terrain` / `scene.grass` | 0.094 / 0 | 0.095 / 0 | 0.052 / 0.688 | 0.051 / 0.687 |
| `step` (us) | 3.7 | 4.0 | 3.2 | 3.6 |

- `game_world` is within 5 % everywhere except the step, which has no entities: +0.4 us per step (the extract's run and
  commit with nothing tracked, and the step-change collection of the flag shift that the extract retains while a sink
  is set). The flyover's GPU frame is bimodal between batches as in S19a (both builds have runs at 2.12 and 2.31-2.36).
- ECS benchmark (release, 3 runs each, interleaved): the existing numbers are unchanged (e.g. `World::step` of 10^5
  moving entities 1.87-2.02 ms before, 1.84-1.89 after; the spatial grid step 2.12-2.14 vs 2.10-2.12; a first
  whole-suite run showed save / load +8-12 %, which the isolated save test does not reproduce: before 34-42 / 110-117
  ms, after 35-36 / 110-113 ms). New `ExtractTenThousandEntities`: 10^4 moving entities with `RenderInstance` into the
  recording sink: the extract scope 0.67-0.68 ms of a 0.98 ms step (67 ns per entity, including the recording sink's
  map lookups); the same entities idle: 0.0000 ms; a paused frame's catch-up without changes 0.10 ms (10 ns per tracked
  entity). Against §17.10's main-thread budget (steps + extract + recording <= 10 ms per frame on the desktop
  reference) this is 7 % for 10^4 moving entities per step.
- `data/tests/benchmarks.json` is not updated (no baseline after S18 is accepted yet).

Local review (after the stage's commits; each fix with a regression test that fails without it, in
`tests/test_engine_extract.cpp`, now 26 tests):

1. **Late movers settle**: an entity moved in phase 7-8 got the pair (old, new) in the next step and kept it forever
   (it was in phase 6's list of that step only). The commit keeps a settle list (the entities whose committed pair is
   not (curr, curr), and the late movers the step's extract did not see), processed by the next step's extract; a
   paused world's catch-up (or time scale 0) sends the late movers the pair the next step would send (a late teleport
   snaps). Tests `ExtractPairs.LateMoverPairSettles`, `ExtractCatchUp.PausedAfterALateMove`.
2. **Spawn merge of `RenderInstance`**: the world API's `spawn` merged an initial `RenderInstance` into the one the
   class added in `onSpawn` without `renderMeshWritten` (the cached ID and the sink kept the class's mesh); the merge
   and `set_global` now write the name through it (`set_global` also checks the name; `RenderInstance` is never a
   global today, so that path has no test). Test `ExtractInstances.SpawnMergeOfAnInitialRenderInstanceIntoTheClassOne`.
3. **Snaps as an edge**: `ExtractSystem::snapSequence(Entity)`, the World-unique number of the entity's last snap pair
   (a re-created instance after a load gets a new one). The platformer reset its chase camera from the level of
   `interpolatedPose`'s `teleported` every frame while the hero stood still after a spawn or restart (Q/E did not turn
   the camera; confirmed headless: 90 of 90 frames, now 1); racing reseeded the wheels and reset the camera every frame
   while a car stood. Flight, platformer and racing (per car in `RenderCaches::Entry::snapsSeen`) react once per
   snap. Test `ExtractPairs.SnapSequenceChangesOncePerTeleport`.
4. **One snap per teleport**: an entity tracked while its `Transform` teleport was pending (created and teleported
   before the step's extract, or before a sink was set) snapped again in the next pair; a teleport the catch-up sent
   followed by an idle step snapped again when the entity moved (the idle return dropped `snapStep`). `track()` marks
   the teleport handled (in a step: `snapStep`; between steps: like a catch-up's), the idle return keeps `snapStep`.
   Test `ExtractPairs.TeleportSnapsOnce`.
5. **Free camera keys over ImGui**: W/A/S/D stopped while the mouse was over an ImGui window (`dt` 0); now only the
   mouse look pauses (`freeCamera(dt, mouseLook, keys)`). No test (GLFW input; no interactive session here).
6. **Late mesh registration**: an unknown name in the `add` path (and replication) is kept with the cached ID 0 and
   logged per write, instead of stored empty, so `World::registerMesh` after an entity named it creates its instance
   (`ExtractSystem::meshRegistered` could never find one before). The API (`set_component`, `add_component`, `spawn`)
   and `setRenderMesh` still refuse an unknown name. Loads keep one too (`checkRenderMeshes` resolves aliases, caches 0
   for an unknown name and the load logs one `render` line with the names and the entity count after the commit)
   instead of rejecting the load, so a game's own save stays loadable and a later registration draws the entity. Tests
   `ExtractInstances.UnknownNameIsKeptAndDrawnWhenRegisteredLater` (replaces
   `UnknownNameThroughAddIsStoredEmptyAndLogged`), `ExtractSave.UnknownNameLoadsAndIsDrawnWhenRegistered` (a save with
   an unregistered name loads into a world without the mesh, one log line, a later `registerMesh` draws both entities;
   `ByNameIndependentOfRegistrationOrder` no longer expects the load error).
7. **A failed first `setDynamic` is retried** (found by S19a's reviewer): when `setDynamic` threw on an instance's
   creation snap after `addDynamic` succeeded, the handle stayed and the re-delivered record sent nothing, so an idle
   entity's instance never got a pose. The creation marks the pose pending (`Tracked::posePending`, cleared once the
   snap is sent) and keeps the entity on a retry list until then; a re-delivered record, the next step's extract or the
   next catch-up (in any frame: not an edit) sends it. Test `ExtractPairs.FailedFirstSetDynamicIsRetried` (in a step's
   extract and in a paused catch-up).

After the fixes: `ctest --parallel 8` on lavapipe 67 of 68 (only `test_opengl_window`, no display); `test_engine_*` and
`test_vulkan_*` on the RTX 5080 Laptop GPU with the validation layer 29 of 29 (no validation message); the goldens are
unchanged; `--scenarios` 18 passed, 0 failed for each demo; `--headless --demo --validation` screenshots of the
platformer (the hero on a platform, the flag) and racing (the car with its wheels on the track) as before, no
validation message.

Left open: CI has not run on this branch (the MSVC job builds the new `sync/` code and `test_engine_extract` there for
the first time); an interactive window session (the free camera's double motion, the chase cameras' release under a
real mouse) was not run (no interactive window here); the extract's per-entity cost with `VulkanRenderSink`
(`RenderScene::setDynamic`) is only covered by the demos (a handful of entities), not by a benchmark of 10^4 rendered
entities; the Steam Deck target of §17.10 is not measured (no hardware).

