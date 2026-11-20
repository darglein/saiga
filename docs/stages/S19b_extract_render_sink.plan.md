# S19b: Extract and render sink

Status: planned.

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
  get their stored curr refreshed (a snap in the next step's pair is not needed: the pair carries them), so catch-up
  never mistakes them for edits.
- Between steps (`ExtractSystem::catchUp()`, called by `EngineApp` before rendering, and by tools that render): when
  the frame ran no step and (the world is paused, or `timeScale() == 0`, or an MCP tool or a load ran, or the
  structural log has unread records for the extract's consumer); it consumes the structural log and
  compares each tracked entity's source pose bitwise with the stored curr (as `PoseHistory::interpolated` does; the
  change flags cannot be used: a setter's flag stays set until the next step ends, and `RenderPose` flags stay set after
  each step) and handles a pending `wasTeleported()` once; a difference is sent as a snap. Cost: O(tracked entities)
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
`game_world --benchmark` unchanged within 5 %.

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
