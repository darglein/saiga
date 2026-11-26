# S24: Streaming

Status: planned (written against `master` after S36c and the ZipStreamWriter change, f92a2a42d).

Sections follow the stage plan template of S05 (§22); "Current state", "Decisions", "Interfaces for S25 and S26",
"Implementation steps", "Risks and open questions" and "Design changes to engine_design.md" are added for this stage.
The decisions of this plan are numbered G1-G22 (the letters A-F are taken by the S20-S23 plans). Before step 1 the
plan is checked against the code of the day (a plan update in the stage branch, R-DOC-2).

## Goal

Chunks become resident around the players and the simulation runs only where it is needed (`docs/engine_design.md`
§12.3, §12.4, §9.2, §9.6):

- per space a **resident chunk set** around the streaming centres (players and pending arrivals): the level-0 base
  heights pinned in the terrain store, the tier-1 placement and the stamped chunk entities; requested in phase 7 with a
  prefetch margin, generated on the streaming pool, **applied when ready** in the phase 7 after the delivery, dropped
  with an unload margin; synchronous in `ScenarioWorld`, so scenarios stay reproducible;
- **edits of non-resident or pending chunks** keep working: they build what they need synchronously and update the
  modification state, and a pending result never undoes an edit;
- the engine tag **`Simulated`** is maintained from the simulation radii, `Behaviour::alwaysSimulate`, `Player` and the
  state entities, exactly and incrementally in phase 7 after the residency update (G8); frozen entities are skipped by the views of phases 2-5;
- the kinematic mover **holds** an entity on a missing chunk, and the steps in which a player stood on a missing chunk
  are counted (`streamingStalls`);
- **arrivals** (`World::requestArrival`): the target island is preloaded, the entity is held (frozen, invisible) and is
  moved in phase 1 of the step after the island is complete; space transitions become arrivals; the island limits of
  §12.3 (`IslandLimitEvent`, `maxSimulationIslands`, `hardIslandLimit`); the loading screen starts at request time;
- streaming state in the world API, MCP and the debug UI; the flight demo streams its 32 km world.

Acceptance (§22): scenarios with streaming incl. the repeat and save/load variants; a unit test with the asynchronous
pool and a held result: an edit on a pending chunk survives when the result is applied, an edit on a non-resident
chunk is in its modification state (CI); a headless run with the asynchronous pool and a player at 30 m/s has 0
streaming stalls (local); step time with 10^5 resident and 10^4 simulated entities and one player at 30 m/s within
§17.10 (benchmark, local).

## Depends on

- **S23b** (and through it S23a): `GenerationPipeline` (`chunkBase(ChunkId)`, `chunkPlacement(ChunkId, version)`,
  `contentSource(SpaceId)` with `ContentSource::requestPlacement(pool, chunk, version, priority, onMainThread)`,
  `stampList`, `height`), `ContentCache` (thread safe; a synchronous requester never waits, D8), `ChunkRecords`,
  `World::effectivePlacementVersion` / `onRecordChanged`, `chunkStampSystem` (phase 7, host), `ContentRenderSystem`
  (`ContentCentres::{None, View, Players}`), `World::tombstone` (validates by generating the placement
  synchronously), `World::spaceReadiness`, the content test setup `tests/content_test_setup.h`.
- **S22**: `TerrainEditStore` (deltas, per-chunk sequence; never evicted), `applyTerrainEdit` (the base gathered through
  `TerrainSystem::sample`: a resident level-0 tile, else the height source, counted in `baseHookSamples`), D3 (edits
  never change base tile data; deltas are merged at every read and at the visual send point), the S22 messages and
  the minimal snapshot (every client keeps every modification).
- **S21**: `requestTransition`, `PendingTransition`, `spaceTransitionSystem`, `transitionsRequestedInStep`,
  `moveToSpace`, the loading screen (`SpaceLoading`, `loadEnds`, `EngineAppSetup::loadingRadius`).
- **S20**: `TerrainSystem` / `TerrainStore` (tiles with LRU per level, `sample`, `updateView` / `completeView`, the
  single send point), `StreamingPool` (owned by `TerrainSystem`, lazy, `drain`, `runOneOnCaller`, `cancel`).
- **S16**: roles, `RunOn`, `ownedBy`, the replication modes, the host + client scenario variants.
- **S10/S12/S14/S19b/S19c**: `Behaviour` (`alwaysSimulate` exists, unused), `tickSystem`, `characterMoveSystem`,
  `ScenarioWorld` and the runner variants, the world API and `registerWorldTools`, `ExtractSystem`, `DebugUi`.

## Rules that apply

- R-MECH-1: whether a world streams at all, the radii, the margins, `alwaysSimulate`, the island limits and what
  happens to frozen entities are game policy (WorldConfig / SpaceTemplate / Behaviour); the engine provides the
  mechanism (G1).
- R-DET-1, §8.6: the live game is not reproducible (results apply when ready); `ScenarioWorld` loads synchronously, and
  every simulation-visible rule depends only on the requested set, never on pool timing or unload hysteresis (G7).
- R-PAR-1: generation runs on the streaming pool with copies (`ContentSource`, never the World); results are applied on
  the main thread in phase 7; the store mutations happen outside `parallelEach` (G16).
- R-CONTENT-1: a resident base tile equals the point-evaluated height bit for bit (the same generator call), so
  residency never changes a height; `TerrainSystem::sample` stays exact either way.
- R-STATE-1/2: residency and the `Simulated` evaluation are derived data (not components, not hashed); the
  modification state stays World state (S22, S23b). Arrivals are the saved component `PendingTransition`.
- R-NET-1/2: every streaming scenario runs in the host + client variants; clients compute their own residency and hold
  their own movers; they never stamp (G10).
- R-SYS-1/2, R-FILE-1, R-REG-1: the systems are free functions registered with phase and `RunOn`; arrivals during a step
  are commands; one class per file under the new directory `engine/streaming/`; `IslandLimitEvent` in `Events.h` with
  an appended `EngineEvent` ID.
- R-MCP-1: the new tools have descriptions saying what they read and change, and `handleMessage` tests.
- R-SAVE-1: no component field changes (`PendingTransition`'s fields are unchanged; only its replication mode, G13);
  no save format change (G15).
- R-TEST-1/2: engine scenarios, unit tests, a non-gating benchmark; streaming and terrain are performance relevant:
  the benchmark is run before/after on the release build, `game_world --benchmark flyover|forest` confirm the
  `TerrainStore` change costs nothing (streaming off there).
- R-DOC-1/2: every signature below that is not in the design document is listed under "Design changes to
  engine_design.md" and applied in the stage's change.

## Current state (grep of `src/saiga/engine/`, f92a2a42d)

| What | Where | State before S24 |
|---|---|---|
| `Registry::kSimulated` | `ecs/Registry.h`, `Registry::activate` | set on every activated entity (`s.flags = kSimulated`); never cleared |
| The view filter | `World::view` / `ownedBy` (`world/World.h`), `simulatedOnly(Phase)` (`world/Phase.h`) | phases 2-5 pass `kSimulated` as a required flag; `viewAll` passes 0 |
| Other users of the flag | `behaviourEventSystem` (skips unsimulated targets), `WorldApi::inspect` (`flags.simulated`), `WorldSave` (the ENTS slot byte), `WorldLoad::checkSlots` (accepts it) | `World::hash()` does not hash slot flags (`Registry::hashInto`) |
| `Behaviour::alwaysSimulate` | `gameplay/Behaviour.h` | a saved field, read by nothing |
| Ticking | `tickSystem` (phase 6, after `spatial_grid`) | exact: all objects re-evaluated when the sorted player cells change, else the changed list (no round-robin, unlike §12.3's wording) |
| Mover | `characterMoveSystem` (phase 4, `ownedBy(localPeer)`) | moves and snaps y to `terrainHeight`; no notion of residency |
| Simulation heights | `TerrainSystem::sample` | a level-0 tile of the space's `TerrainStore`, else the generator point evaluation (`baseHookSamples`); exact either way |
| The store | `TerrainStore` (per space, owned by `TerrainSystem`) | filled only by the visual path: `requestTile`'s completion stores a tile only for the view space; `evict` keeps at most `tilesPerLevel` per level (LRU); mutations asserted on the main thread "between steps" |
| The pool | `StreamingPool`, `TerrainSystem::ensurePool` / `streamingPool()` | lazy; drained in `updateView` / `completeView` (the app, between steps) and by the content render sets; priorities: pyramid 0, tiles `1..8008`, masks `100000 + d` |
| Content | `GenerationPipeline::chunkBase` / `chunkPlacement` (sync), `ContentSource::requestPlacement` (pool) | no pool form for `ChunkBase` |
| Stamping | `chunkStampSystem` (phase 7, host; registered with sites or a non-legacy generator) | stamps every unstamped chunk with stamps within `fullRadius` of a `Player` (its own player loop) |
| Content render sets | `ContentRenderSystem` | visual; `View` (app camera) / `Players` (scenario worlds); removed beyond `fullRadius + 256 m` |
| Transitions | `requestTransition` (`void`), `PendingTransition` (`Replicate::Never`, saved), `spaceTransitionSystem` (phase 1, host) | completes in phase 1 of the next step, unconditionally |
| Templates | `SpaceTemplate` (`world/SpaceTemplate.h`) | has `fullRadius` (1000; scenarios lower it to <= 128); no `simulationRadius`, `prefetchMargin`, `unloadMargin`; `WorldConfig` has no island fields |
| Loading screen | `EngineApp::updateSpaceLoading`, `LoadingScreen.h` | starts on a view-space switch with `spaceReadiness < 1`; visual only |
| Phase 7 | `Phase::Streaming` | only `chunk_stamp` |

## Decisions

- **G1 Streaming is a world policy.** `WorldConfig::streaming` (bool, default `false`). Off: exactly S23b's behaviour
  (every entity `Simulated`, no residency, no hold, S21's one-step transitions, S23a's stamping rule); every existing
  test, scenario and save behaves as before (world hash values change once, by the appended event registration of
  G15, not by behaviour). On: everything of this plan. Reason: R-MECH-1 (a small level, the racing
  track and the platformer level fit in memory and want every entity simulated), and freezing every entity of a world
  without players would break most unit tests, which create worlds without a `Player`. The flight demo turns it on
  (G22).
- **G2 What is resident in S24.** Per chunk of a streaming world: (a) **Base**: the level-0 base heights, pinned in the
  space's `TerrainStore` (only for spaces with terrain); (b) **Placement**: the chunk's `ChunkPlacement` at its
  effective placement version, held by `shared_ptr` (only for spaces whose generator has slots); (c) the **stamped
  entities** (host): `chunkStampSystem` stamps a chunk once it is resident. Colliders are S28's part (the part list is
  open, Interfaces). The modification state (S22's deltas and sequences, S23b's records) stays in memory for every
  chunk and is never unloaded (§12.3 "S24 scope"). Entities are never unloaded (S25). "Unloading" a chunk in S24 drops
  (a) and (b) only. Grass masks, tiles above level 0, the pyramid and the content render sets stay visual (S20, S23b:
  the view camera in the app, `Players` in scenario worlds) and keep their own rules.
- **G3 Three distances** per space template, all measured from a centre to the nearest point of the chunk's square in
  x/z (the stamp system's rule): **required** `d <= fullRadius`, **requested** `d <= fullRadius + prefetchMargin`,
  **kept** `d <= fullRadius + prefetchMargin + unloadMargin`. A chunk is requested when it enters the requested
  distance of a centre and unloaded when it is beyond the kept distance of every centre. Deviation from §12.3 ("unloaded
  when it leaves `fullRadius + unloadMargin`"): with the default margins (prefetch 256 > unload 128) that rule would
  unload prefetched chunks at once; the hysteresis must lie beyond the prefetch ring.
- **G4 Streaming centres.** Host: every alive entity with `Player` or with the new engine component `StreamingAnchor`
  (an empty tag, `Saved::Yes`, `Replicate::Never`: a game marks a player-less vehicle or camera rig as a centre, e.g. the
  flight demo's autopilot plane, G22) at its world position (also held ones, at their current position), and the target
  of every pending arrival of a `Player` root (`PendingTransition`; arrivals of other entities are no centres, G12).
  Client: the `Player` and `StreamingAnchor` entities it owns (`ownedBy(localPeer)`) and their replicated
  `PendingTransition` targets. The view camera is never a centre (visual only, §12.3 "GPU uploads are visual only").
- **G5 Results apply in phase 7 only.** Pool completions only move their result into the residency's delivery queue
  (they may run in any `drain`, also `updateView`'s between steps); the queue is applied in phase 7 by the `streaming`
  system. Between steps only `World::completeResidency` (the app's complete points, the load) applies results.
  Synchronous mode (`World::setStreamingSynchronous(true)`, set by `ScenarioWorld`): phase 7 builds every requested,
  missing chunk inline (the cache's synchronous path) in the requesting step; no pool is created.
  **Centres changed between steps**: at the start of every `step()` the centres are computed (a handful of entities:
  every `Player` / `StreamingAnchor` and pending player arrival) and compared with those of the last update; if they
  differ (for whatever reason: `spawn`, `teleport`, a direct `Transform` setter, `moveToSpace`, `requestArrival`, a
  component add or remove; no flag at mutation points that could be missed), the **step-start refresh** runs before
  phase 1: the residency's request part (new requests, unloads; in synchronous mode the inline builds; no delivery queue
  is applied, so asynchronous results still apply in phase 7 only), the islands (G14) and the simulation evaluation
  (G8); stage hooks run after it (S25's reloads). So a between-step change is seen by the next step's phase 1 exactly as
  after a load (whose `completeResidency` does the same). One latency rule: a change between steps is resident for the
  next step; a change inside step k (commands at a flush) is resident from phase 7 of step k, i.e. for step k + 1.
- **G6 The re-patch rule holds by construction.** A residency result is base content only: the base tile and the
  placement are never patched (S22 D3), the deltas are read from `TerrainEditStore` at every sample and merged at the
  visual send point, tombstones are applied by every reader from `ChunkRecords`. So "applied together with the
  modification state as it is at that moment" needs no code; the result records the chunk's sequence at the request
  and the apply compares it: a newer sequence counts `appliedAfterModification` (S28 adds a listener there for its
  collider patch; S24 adds no hook without a consumer). A changed effective placement version (a load, a client record
  message) drops a pending result and requests again (the content sets' rule); for a `Resident` chunk it drops the held
  placement, the chunk goes back to `Requested` (`ready` false, the base stays pinned) and the placement is requested at
  the new version.
- **G7 Simulation-visible rules use the requested set.** `alwaysSimulate` (G8) and the mover hold (G11) read
  "requested and resident", never "resident" alone, and the requested set is a pure function of the current centres.
  In synchronous mode requested implies resident, so the unload hysteresis (which depends on history) is invisible to
  the simulation: the repeat and save/load variants stay exact (`strictContinuation`).
- **G8 `Simulated` is exact and incremental, by 32 m grid cells.** A system `simulationSystem` (phase 7, right after
  `streaming`, so phases 2-5 of the next step see flags computed from the current positions and the current ready set;
  deviation from §12.3's "phase 6", which would lag the ready set by a step) keeps, per space, the coverage of the
  spatial grid's 32 m cells by the centres' simulation discs (a cell is covered by a centre cell when `(dx^2 + dz^2) *
  32^2 <= simulationRadius^2`, cell index differences) and re-evaluates only the entities of cells whose coverage
  flipped, the moved entities, the entities of chunks whose ready state flipped (accumulated from the residency's
  `onReadyChanged` listener since the last evaluation) and the entities whose `Player`, `StreamingAnchor`, `Parent` or
  `PendingTransition` changed. **New entities** are evaluated when they become alive (a between-step `spawn` at once,
  a spawn command at its flush, stamped entities at the flush after phase 7), and an entity moved by `moveToSpace` or
  an arrival's completion is re-evaluated at once (`refreshSimulated` of its root), so flags only lag a move made by a
  system's own `Transform` writes inside phases 2-5, until that step's phase 7. Arrival targets are residency centres
  only, not simulation centres (entities at a target do not tick before the player arrives). The rule for a root, in
  order: world-state or space-state entity: simulated; a held player root (G23 with `PendingTransition`): frozen; a
  player root or a `StreamingAnchor` subtree: simulated; its cell covered: simulated; `Behaviour::alwaysSimulate` and its chunk requested and
  resident: simulated; else frozen. Descendants take their root's flag: a root whose flag flips (or that gains a child)
  sets its subtree through the hierarchy's child lists (no pass over all of `hierarchyOrder`). Deviation from §12.3
  ("re-evaluated round-robin together with the tick interval"): S10 made ticking exact by cells; doing the same here
  avoids a 1/16 latency and is cheap (a player crossing a cell flips about 50 cells at 384 m). The full evaluation runs
  only when the set is not initialised (the first update, a load); an empty coverage map is a valid state.
- **G23 Player root.** A root whose subtree contains an alive `Player` (flight's player is a driver entity next to the
  aircraft; S25's groups use the same definition). Its position is the root's world position. Everywhere this plan says
  "player" for centres, holds, islands, arrivals and simulation it means a player root.
- **G9 The flag is not saved** (§12.3). The ENTS slot byte keeps its layout but `save()` writes `kSimulated` set for
  every alive entity (so saves of streaming-off worlds stay byte-identical to S23b's and a saved bit never depends on
  derived state); a load runs `completeResidency` and a full evaluation before `tryLoad` returns, and a world with
  streaming off sets the flag on every entity. (A saved derived bit would be stale whenever centres changed between
  steps or the ready set differs in an asynchronous world, and would break the save/load byte check.)
- **G10 Clients.** A client's own residency (centres of G4) generates base and placement locally (R-CONTENT-1); its
  chunk counts as resident when both are ready: in S24 every client receives every modification (S22, S23b), so no
  per-chunk modification snapshot gates it (that is S26, Interfaces). On a client every entity is `Simulated` except the
  held ones (`PendingTransition`, G13); clients never stamp and never evaluate radii.
- **G11 Mover hold and stalls.** In a streaming world `characterMoveSystem` writes nothing for an entity whose current
  or destination chunk is not requested-and-resident (no movement, no y snap); it runs on the owner, so the hold is the
  owner's (§12.3). Positions are clamped to the space's chunk range before the lookup (the stamp and `terrainHeight`
  rule), so an entity beyond the edge reads the edge chunk instead of holding forever. The `streaming` system counts, after its apply, the `Player`s whose chunk is not resident on this
  peer (host: every player; client: its own): `streamingStalls` (steps with at least one such player) and
  `playerStallSteps` (the sum).
- **G12 Arrivals reuse `PendingTransition`.** `requestArrival(e, space, position, rotation?)` adds `PendingTransition
  {link = Entity{}, target, position, rotation}`; `requestTransition(e, link)` is an arrival with the link (and now
  returns `bool`). In a streaming world `spaceTransitionSystem` completes the arrival of a `Player` root in phase 1 only
  when the arrival's island (every chunk intersecting the disc of `simulationRadius` around the arrival position) is
  requested and resident on the host; meanwhile the entity is held: frozen (G8), its render instance and those of its
  descendants (incl. their `LightSource`s) hidden by the extract, its pose unchanged (the host drops owner states of a
  held client-owned root in `ownerPoseSystem` and `stepNetworkOnly`). The arrival of any other root (an NPC teleport)
  completes with S21's rule (the next step, no centre, no hold, no island check); it lands frozen unless a player is
  near. Without streaming the S21 rule is unchanged (the next step). A completed arrival emits `SpaceTransitionEvent` as
  before (`link` null for a plain arrival; `from == to` for a same-space arrival).
- **G13 `PendingTransition` becomes `Replicate::Host`.** The owning client sees its player's pending arrival: its
  loading screen starts at request time (G19), its residency preloads the target (G4) and its views skip the held
  player (G10). The fields are unchanged (no migration; replication modes are not part of the save schema).
- **G14 Islands.** Per space, the connected components of the player roots (G23, entity-index order, union of pairs
  with horizontal distance `<= 2 * simulationRadius`); the island count is their sum over the spaces, computed by the
  host's `streaming` system. `IslandLimitEvent {uint32_t count}` is emitted in a step whose count exceeds
  `WorldConfig::maxSimulationIslands` (default 4) while the last reported count did not, or exceeds it and is larger
  than it; `islandLimitSteps` counts the steps above the limit. The last reported count is **state**: the engine
  component `IslandState {uint32_t reportedCount}` on the world-state entity of streaming worlds (`Saved::Yes`,
  `Replicate::Never`), written by a command at the phase-7 flush together with the event, so an aborted phase drops both
  and a load continues exactly (`StreamingSave.IslandLimitNotRepeated`). With `WorldConfig::hardIslandLimit`,
  `requestArrival` returns `false` (and logs) when the arrival would raise the count above the limit. `maxPlayers` of
  §7.9 stays S33's (nothing in S24 would read it).
- **G15 No save format change, no new message.** Residency is derived and rebuilt on load; `Simulated` stays in the
  slot byte (G9); `PendingTransition` is saved since S21. The appended `EngineEvent::IslandLimit` and the new component
  `StreamingAnchor` change `world.hash()` values of every world against S23b builds the way S22's event did (no
  committed world hash exists; the S22 kernel hash is not a world hash), and the committed engine schema
  (`data/tests/engine/schema.json`) is regenerated in the stage (AGENTS.md: `SaveFormat.EngineSchemaIsCommitted` with
  `SAIGA_UPDATE_GOLDEN=1`).
- **G16 One pool, priorities, the phase-7 drain.** Residency jobs run on `TerrainSystem`'s `StreamingPool` (one pool
  per World, §8.5). Existing priorities: the visual pyramid 0, the content render sets' jobs 0
  (`ContentRenderSystem.cpp`), tiles `1..8008`, masks `100000 + d`. Residency: required chunks `-100000 + int(d)`
  (before everything), prefetch chunks `50000 + int(d)` (after the visual tiles, before the masks); at most
  `StreamingSettings::maxInFlight` (8) residency jobs in flight, the rest wait in the residency's own ordered queue. The
  pool cannot re-prioritise: a queued (not yet running) prefetch job whose chunk becomes required is cancelled and
  resubmitted at the required priority; a running one is left alone. `StreamingPool::cancel` of a queued job returns
  true and its completion runs `done(true)` at once: the in-flight count drops then; a running job keeps its worker,
  `cancel` returns false and its completion runs with the cancelled flag at the next drain: the in-flight count drops at
  that completion (either flag), never earlier. Phase 7 drains the pool (every completion, also the
  visual ones: they run on the main thread outside `parallelEach`, which is all they need). `TerrainStore`'s mutation
  rule becomes "main thread, not inside `parallelEach`" (phase 7 pins tiles inside a step; `sample` readers in
  `parallelEach` run in other phases).
- **G17 One residency job per chunk.** The residency's own job lambda (on the pool, with a `ContentSource` copy) runs
  `jobHookForTests`, then the base (`ContentSource::baseInJob`, new: the cache's base entry, generated in the job) and,
  if wanted, the placement (`placementInJob`, which reads that base from the cache instead of generating it a second
  time), and delivers both `shared_ptr`s together. No separate `requestBase` (the base would be generated twice:
  `generatePlacement` calls the synchronous `generateBase`, `GenerationPipeline.cpp:483`, and the cache has an
  in-flight map only for placements). The pinned store tile is a `TerrainTileData` sharing the base's `heights` pointer
  (no copy; min/max computed once). The visual path's own level-0 requests find a pinned tile in the store and request
  nothing; a visual job already running for the same tile finds the pinned tile at its completion and drops its result
  (equal data). Readers of placements (`World::tierOnePosition`, `tombstone`'s check) consult the residency's held
  placement before the cache, so resident chunks never regenerate when the cache's 256 MB budget evicts.
- **G18 Tools and readouts.** World API operations / MCP tools `streaming_status` (read) and `request_arrival`
  (change, host); `chunk_info` gains `residency`, `world_status` a `streaming` block, `inspect` keeps
  `flags.simulated` and gains `held`; the Engine tab gets a "Streaming" section.
- **G19 Loading screen at request time.** While the app's local player has a `PendingTransition` (host world, or the
  replicated one on a client), `EngineApp` shows the loading screen for the target space with the arrival progress;
  after the move the S21 rules continue the same load (no flash), and on a client it also waits until the player's
  chunk is resident (its mover holds until then).
- **G20 Edits of non-resident or pending chunks.** The S22/S23b paths are already synchronous and residency-independent
  (`applyTerrainEdit` gathers the base through `sample`, `tombstone` generates the placement through the cache); S24
  creates no residency for an edited chunk and adds no hook or statistic without a consumer (the tests read S20's
  `baseHookSamples` and the cache's generation counters; a stage that evicts modification state adds its load hook).
- **G21 The benchmark is a non-gating test binary** (`tests/test_engine_streaming_benchmark.cpp`, the S22 pattern): a
  short CI form, and a local form with `SAIGA_STREAMING_BENCHMARK=full` (10^5 entities, real-time pacing, the stall
  count). No window (the no-windows rule), no game.
- **G22 Tech demos.** `game_flight` turns streaming on (its 32 km generated world, the S27 target; its scenarios and the
  ruin screenshot must stay unchanged). Stamping follows the residency centres incl. anchors (Scope 4), so the
  autopilot plane now stamps the ruins it passes (intended: content around every streaming centre exists; the
  autopilot scenario's checks are about the flight path, the hash scenario is unaffected). Its `--demo` / `--coop` plane is an `Autopilot` without `Player`
  (`samples/games/flight/main.cpp:93`), so the game adds `StreamingAnchor` to every plane it spawns (demo, autopilot
  scenario `autopilot_flies_the_circuit`, the co-op client's plane is a `Player`). Flight keeps its `fullRadius = 3000`
  (`FlightWorld.cpp:262`, its stamping and content distance) and sets `simulationRadius 1000`, `prefetchMargin 1000`
  (a plane at 200 m/s crosses 1 km in 5 s), `unloadMargin 256`: the kept radius is 4.26 km, about 870 chunks of 256 m,
  each a pinned 257^2 int32 base (264 KB): about 230 MB pinned CPU memory, within §12.2's 768 MB terrain budget with
  the visual tiles; the placements add about 20 MB. Measured and reported in the Results; if the load at 200 m/s
  stalls, flight lowers `prefetchMargin` before anything else changes. `game_world` (no `Player`: a camera and lamps),
  `game_racing` (AI cars along a track) and `game_platformer` (one small level) stay off; each keeps its behaviour.

## Scope

### 1. Configuration (`engine/world/World.h`, `engine/world/SpaceTemplate.h`, `engine/tools/ScenarioWorld.cpp`)

- `WorldConfig`: `bool streaming = false;` (G1), `int maxSimulationIslands = 4;`, `bool hardIslandLimit = false;`
  (G14).
- `SpaceTemplate`: `double simulationRadius = 384;`, `double prefetchMargin = 256;`, `double unloadMargin = 128;`
  (§7.9; `fullRadius` exists). `validateSpaceTemplates` adds (fatal in every build, naming template and field): every
  radius and margin finite; `fullRadius > 0`; `0 < simulationRadius <= fullRadius`; margins `>= 0`.
- `ScenarioWorld`'s `scenarioSetup` lowers `simulationRadius` to at most 128 and sets `prefetchMargin = 0` (after the
  existing `fullRadius <= 128`; §17.2), and calls `setStreamingSynchronous(true)` on every world it creates (host,
  clients, the save/load replacement, repeat runs).
- `StreamingAnchor` (`components/StreamingAnchor.h`, an empty engine component, `Saved::Yes`, `Replicate::Never`,
  appended `EngineComponent` ID; G4).
- Runtime settings (not saved, not policy): `struct StreamingSettings { int maxInFlight = 8; bool synchronous = false;
  };` with `World::setStreamingSettings` / `streamingSettings()` and the shorthand `setStreamingSynchronous(bool)`.

### 2. The resident set (`engine/streaming/ChunkResidency.{h,cpp}`, `engine/streaming/Systems.h`)

```cpp
enum class ChunkState : uint8_t { None, Requested, Resident };
enum ResidencyPart : uint8_t { PartBase = 1, PartPlacement = 2 };   // S25: PartStored = 4; S28: PartColliders = 8
struct StreamingCentre { SpaceId space; Vec3 position; Entity entity; bool arrival; };
struct StreamingStats
{
    // cumulative
    uint64_t requested = 0, applied = 0, resultsDropped = 0, appliedAfterModification = 0, unloaded = 0;
    uint64_t syncBuilt = 0, streamingStalls = 0, playerStallSteps = 0, islandLimitSteps = 0, moverHolds = 0;
    uint64_t evaluations = 0;
    // current
    int centres = 0, requestedChunks = 0, residentChunks = 0, pendingJobs = 0, queuedChunks = 0, islands = 0;
    int simulated = 0;
    size_t residentBytes = 0;
    double lastStreamingMs = 0, lastSimulationMs = 0, lastLoadRebuildMs = 0;
};
class SAIGA_ENGINE_API ChunkResidency   // owned by the World (World::residency())
{
   public:
    explicit ChunkResidency(World& world);
    void update();                                   // phase 7 (the streaming system); between steps: completeResidency
    bool complete(double timeoutSeconds);            // World::completeResidency
    ChunkState state(const ChunkId& chunk) const;
    bool requested(const ChunkId& chunk) const;      // within the requested distance of a centre (G3)
    bool ready(const ChunkId& chunk) const;          // requested and resident (G7)
    uint8_t partsReady(const ChunkId& chunk) const;
    const std::vector<StreamingCentre>& centres() const;
    bool kept(const ChunkId& chunk) const;           // within the kept distance of a current centre (pure, G3)
    double islandProgress(SpaceId space, const Vec3& position) const;   // arrival readiness (G12), in [0, 1]
    int addListener(ResidencyListener listener);
    void removeListener(int id);
    void dropAll();      // a load, replaceSources: every chunk to None, no listener called (consumers reset themselves)
    const StreamingStats& stats() const;
    std::function<void(const ChunkId&)> jobHookForTests;   // runs in the residency's job before it generates
};
// Called synchronously inside update() / complete(), in ChunkId order per kind; a consumer that needs the changes at a
// later phase accumulates them itself (no "this step" lists that a second update() would overwrite).
// Listeners never write World state (they run inside update(), also between steps and inside a load): a consumer
// accumulates and acts in its own system.
struct ResidencyListener
{
    std::function<void(const ChunkId&)> onUnload;                  // before the chunk's data is dropped (-> None)
    std::function<void(const ChunkId&, bool)> onReadyChanged;      // ready() flipped (G7; SimulationSet)
};
```

The S24 class has only what S24 uses. The agreed extensions of the consuming stages ("Interfaces for S25 and S26") are
added by those stages: S25 the listener members `onResident` and `onKeptChanged` and the part registration
`addPart(ResidencyPartInfo {bit, wanted, build (synchronous), request (asynchronous), drop})`; S26
`setModificationGate` and `forEachChunk`. Base and Placement are one built-in job (G17), not parts of that shape.

- **Per chunk** (`std::map<ChunkId, Entry>`): state, the parts wanted (Base iff the space has terrain, Placement iff its
  generator has slots) and ready, the request counter (incremented by every request and unload; a result with another
  counter is dropped), the effective placement version and the S22 sequence at the request (G6), the pool job IDs, the
  held `shared_ptr<const ChunkBase>` / `shared_ptr<const ChunkPlacement>`, the nearest centre distance. A chunk that
  wants no part (flat or enclosed spaces, slot-less generators without terrain) becomes resident in the update that
  requests it, without a job (worlds with flat spaces start no pool).
- **`update()`** (phase 7, both roles; also called by `complete`):
  1. the centres (G4), sorted by (space, entity index);
  2. per space the requested chunks (squares within the requested distance of a centre, clamped to `[0,
     sizeChunks)^2`) and the required ones (G3); new ones become `Requested`; queued prefetch jobs of chunks that became required are resubmitted (G16); chunks beyond the kept
     distance of every centre are unloaded (G2: jobs cancelled, the tile unpinned, the placement released, `onUnload`
     listeners);
  3. synchronous mode: every requested chunk that is not resident is built inline (`chunkBase`, `chunkPlacement` at
     the effective version, in `ChunkId` order; `syncBuilt`); asynchronous mode: missing parts are submitted by
     priority (G16) up to `maxInFlight`;
  4. `TerrainSystem::drainStreamingPool()` (fatal on a job exception, as S20), then the delivery queue in `ChunkId`
     order: counter and version checked (G6), parts stored; then every `Requested` chunk whose wanted parts are ready
     (also one that wants no part) becomes `Resident`
     (the tile pinned, `onResident` listeners); `onReadyChanged` for every flip of `ready`;
  5. the stall counters (G11) and, on the host, the islands and `IslandLimitEvent` (G14);
  6. the statistics and the profiler scope `streaming`.
- **The step-start refresh** (G5, centres differ from the last update's): the request part of `update()` (steps 1-3;
  no delivery queue), the islands and the simulation evaluation, before phase 1.
- **`complete(timeout)`** (between steps): `update()` without the stall/island part, then runs the pool on the caller
  (`runOneOnCaller`) and waits until every required chunk of every centre is resident; false at the deadline (what is
  resident is applied). Called by `World::completeResidency(double timeoutSeconds = 120)`.
- **Systems**: `streamingSystem` (`Phase::Streaming`, `RunOn::Both`, registered by the World constructor before
  `chunk_stamp`, only with `WorldConfig::streaming`) calls `residency().update()`. Results applied in a step whose later
  phase aborts stay applied (derived data; the next step continues).
- **Destruction order**: the completions capture a `shared_ptr` to the delivery queue, never the residency; the World
  destroys the pool (`TerrainSystem`) before the residency.
- `World::residency()` (engine-internal; tests, tools), `World::chunkReady(SpaceId, double x, double z) const` (G7;
  `true` in a world without streaming: everything is available on demand), `World::completeResidency`.

### 3. Base heights resident (`engine/terrain/TerrainStore.*`, `engine/terrain/TerrainSystem.*`, `engine/content/GenerationPipeline.*`)

- **Store membership and visual membership are separate.** `TerrainStore::Tile` gains `bool pinned` (the residency
  holds it) and `bool visual` (the visual path holds it: stored by a visual completion, or a pinned tile that the
  visual path desires). The visual budget counts `visual` tiles only (`tilesAtLevel(k)`), so the GPU atlas
  (`tilesPerLevel`) is unchanged. `evict` clears `visual` (LRU among visual tiles) and removes the tile only if it is
  not pinned; `dropStreamed` (the old view space on a view-space switch, `TerrainSystem.cpp:1002`) clears `visual` on
  every tile and removes the unpinned ones, so a player's resident chunks survive the camera switching spaces.
  `TerrainSystem::sendUpdates` removes a sent tile when it is no longer `visual` (today: when the store no longer holds
  it, `TerrainSystem.cpp:584-592`) and adds desired tiles that are stored, marking them `visual`. Without this a pinned,
  once-sent level-0 tile would stay sent after it left the desired set and overflow the atlas
  (`TerrainAtlas.cpp:351`). `cpuBytes` reports pinned bytes separately (`pinnedBytes`).
- `TerrainStore::pinTile(region, TerrainTileData)` / `unpinTile(region)` (removes the tile unless `visual`) are store
  primitives; the decisions that need `sentTiles` or the desired set stay in `TerrainSystem`.
- The thread rule becomes "main thread, not inside `parallelEach`" (G16); `checkThread` (which asserts the thread only)
  stays; the "between steps" wording of its comment is corrected.
- `TerrainSystem::pinSimulationTile(ChunkId, shared_ptr<const ChunkBase>)` / `unpinSimulationTile(ChunkId)` for any
  space (not only the view space; the visual completion keeps its view-space check for its own tiles);
  `drainStreamingPool()` (the `drainPool` with S20's fatal exception rule, callable from phase 7).
  `replaceSources` (a host load with other pins) drops the pinned tiles too (the residency is dropped and rebuilt by the
  load, Scope 9).
- `sample` is unchanged: a pinned tile is a resident level-0 tile. `baseHookSamples` keeps counting the point
  evaluations, so a test proves that reads inside the resident set use tiles.
- `ContentSource::baseInJob(ChunkId)` (G17: the cache's base entry generated inside a pool job; the cache gains an
  in-flight entry for bases like the placements' `running` map, so a visual or render-set job of the same chunk waits
  for it instead of generating it twice) and the residency's job lambda.

### 4. Placement and stamping (`engine/content/ChunkStampSystem.cpp`, `engine/content/ContentRenderSystem.cpp`)

- The Placement part requests `ChunkPlacement` at the chunk's effective placement version through
  `ContentSource::requestPlacement` (asynchronous) or `GenerationPipeline::chunkPlacement` (synchronous); the residency
  holds the pointer; `World::tierOnePosition` and `tombstone`'s check consult it before the cache (G17), so resident
  chunks never regenerate (S28's colliders use it too). The content render sets keep using the cache (visual).
- `chunkStampSystem` in a streaming world: the candidate chunks are the **resident** chunks within `fullRadius` of a
  centre (player roots, `StreamingAnchor` entities and arrival targets, G4) that have stamps and are not stamped, in `ChunkId` order (stateless, as S23a's
  rule). In synchronous mode this is the same step and the same set as S23a's rule for players, so S23a's and S23b's
  scenarios would not change if run with streaming on. Without streaming the system is unchanged.
- The content render sets keep their S23b centres and hysteresis (G2: visual). No change.

### 5. Edits of non-resident or pending chunks (`engine/terrain/TerrainEdit.cpp`, `engine/content/Tombstones.cpp`, `engine/net/Replication.cpp`)

- No code change in the edit paths (G20): they are synchronous and residency-independent already. S24 adds tests only.
- An edit or tombstone creates no residency and changes no chunk state. A pending chunk's result applies with the
  current modification state (G6); its sequence comparison counts `appliedAfterModification`.

### 6. The `Simulated` tag (`engine/streaming/SimulationSet.{h,cpp}`, `engine/world/SpatialGrid.h`)

- `SimulationSet` (owned by the World): per space the centre cells (sorted), the coverage `std::map<CellKey,
  uint16_t>` (G8) and the disc offsets per space (from `simulationRadius`); its own structural log consumer (created
  entities, `Player`, `StreamingAnchor`, `Parent` and `PendingTransition` adds and removes); the dirty set (entity
  indices, ordered); the dirty chunks accumulated from the residency's `onReadyChanged`; `initialised`.
- `simulationSystem` (`Phase::Streaming`, `RunOn::Both`, registered right after `streaming` and before `chunk_stamp`
  only with `WorldConfig::streaming`; G8):
  - host: the centre cells; for each removed / added centre cell the disc's cells change coverage; cells that flip
    between 0 and > 0 mark their entities dirty (`SpatialGrid::forEachInCell(space, cx, cz, fn)`, new); the changed
    transforms (`World::changedTransforms()`), the log's records and the entities of the 8 x 8 cells of every
    accumulated dirty chunk are marked dirty; dirty roots are evaluated with G8's rule (`Registry::setFlags`); a root
    whose flag flipped or whose child list changed sets its subtree through the child lists; a full evaluation instead
    while not `initialised` (first update, a load);
  - client: only held toggles of player roots (`PendingTransition` added: frozen; removed: simulated), the host's rule.
- New entities: the World calls `SimulationSet::evaluateCreated` after each command flush and at once for a
  between-step `spawn` (G8), so the "simulated until the next evaluation" window does not exist.
- `World::simulated(Entity) const`; `World::refreshSimulated(Entity)` (re-evaluates one root at once; the world API's
  `set_component` / `add_component` / `remove_component` on `Behaviour` call it, so an `alwaysSimulate` edit applies
  without a move).
- `activate` keeps setting `kSimulated`; in a streaming world `evaluateCreated` corrects it before any system runs.
- Views: unchanged mechanism (`view` and `ownedBy` require the flag in phases 2-5; `viewAll` for replication, extract,
  save, streaming, ticking). `behaviourSystem` skips frozen objects already: `sinceTick` does not grow while frozen
  and continues when un-frozen; no `onCatchUp` (§12.3). Game systems that read events check `world.simulated(target)`
  (documented in §9.4; the engine's `behaviourEventSystem` does).

### 7. Mover hold and stalls (`engine/gameplay/CharacterMoveSystem.cpp`)

- In a streaming world: before writing, `w.chunkReady(space, x, z)` for the current position and for the destination
  (`p + v * dt`), both clamped to the space's chunk range (G11); if either is false the entity is not written
  (`moverHolds++`). Without streaming unchanged.
- The stall counters of Scope 2 step 5 (G11). `streamingStalls` is also a profiler counter (`streaming/stalls`) and is
  reported by `streaming_status`, `world_status` and the benchmark.
- A direct `moveToSpace` / `ownerTeleport` into non-resident chunks is allowed; the entity is held there until the
  chunks arrive (documented: far teleports use `requestArrival`).

### 8. Arrivals and islands (`engine/streaming/Arrivals.cpp`, `engine/world/SpaceTransitionSystem.cpp`, `engine/streaming/Islands.{h,cpp}`, `engine/sync/ExtractSystem.cpp`)

```cpp
bool World::requestArrival(Entity e, SpaceId space, const Vec3& position,
                           std::optional<Quat> rotation = std::nullopt);   // rotation: the root's current one
bool World::checkArrival(Entity e, SpaceId space, const Vec3& position, std::string* error) const;   // world API
bool World::requestTransition(Entity e, Entity link);                      // S21's, now an arrival; returns bool
std::optional<double> World::arrivalProgress(Entity e) const;              // none: no pending arrival
int World::islandCount() const;
struct IslandLimitEvent { uint32_t count; };                               // EngineEvent::IslandLimit, Never
```

- `requestArrival`: host only (a client call asserts, as `requestTransition`); the S21 checks (alive, a root, not a
  state entity: asserts); refused with a log line and `false` in every build: a space out of range, a non-finite
  position, a position outside the space's x/z range, a pending `PendingTransition` (the first request wins),
  `hardIslandLimit` exceeded (G14, `Player` roots only: the islands recomputed with the entity at the target). Between
  steps applied at once (the next step's refresh sees it, G5), during a step a command (`transitionsRequestedInStep` as S21). In a
  streaming world a `Player` root's flag is cleared at once (held).
- `spaceTransitionSystem` (phase 1, host): a `PendingTransition` requested in an earlier step completes when the world
  does not stream, when the root has no `Player` (G12), or when `residency().islandProgress(target, position) == 1` (every chunk intersecting the
  `simulationRadius` disc of the arrival position is requested and resident). Completion as S21 (`moveToSpace`,
  `SpaceTransitionEvent {entity, link, from, to}`, the component removed); a pending one is re-checked every step.
  In synchronous mode the island becomes resident in phase 7 of the requesting step (a request during a step) or in
  the step-start refresh (a request between steps, G5), so the move happens in phase 1 of the next step in both cases
  and after a load (S21's latency: `uses_a_space_link` keeps its "2 steps after the interact step").
- The arrival target is a streaming centre from the request on (G4), so the whole surroundings start loading, the
  island first (required priority).
- Held entities are invisible: the extract treats a held root (a `Player` root with `PendingTransition` in a streaming
  world) and its descendants (render instances and `LightSource`s) like S19b's `Pending` source (tracked, no sink
  instance; removed when the component is added, recreated as a snap when it is removed, driven by the structural log
  records of `PendingTransition`).
- The host's `ownerPoseSystem` and `stepNetworkOnly` drop the owner states of a held client-owned root (no pose applied
  while held; the counter rule of §13.3 is unchanged).
- Client-owned players: the host moves them with `moveToSpace` (S21's `OwnerCommand`); the client's residency already
  preloads the target (G4, G13); if its chunks are not resident yet its mover holds (G11).
- `IslandLimitEvent` and `islandLimitSteps` (G14) from the host's `streaming` system; `Islands::count(world)` is a pure
  function over the players (tests).
- `WorldConfig::streaming == false`: `requestArrival` adds the component and the S21 rule completes it in the next step
  (no hold, no hiding, no island check); `requestTransition` behaves exactly as in S21.

### 9. Save and load (`engine/save/WorldLoad.cpp`, `engine/world/World.cpp`)

- No new section and no format change (G15). `WorldSave` writes the ENTS flags byte with `kSimulated` set for every
  alive entity (G9).
- A committed load in a streaming world: `residency().dropAll()`, then `completeResidency` for the required chunks of
  the loaded centres (synchronously, before `tryLoad` returns; `lastLoadRebuildMs`), the `SimulationSet` marked for a
  full evaluation before `tryLoad` returns (G9; `IslandState` is loaded like any component, G14); prefetch chunks follow
  asynchronously. If `completeResidency` times out the load still commits (residency is derived), logs a line naming
  the missing chunks, and the rest arrives asynchronously (movers hold, G11). A pending arrival is loaded with its
  `PendingTransition` and completes as in Scope 8. In a world without streaming the load sets `kSimulated` on every
  alive entity.
- The resident set is not saved (§12.3); the modification state of non-resident chunks is saved by S22's `TERR` and
  S23b's `CREC` as before.

### 10. Networking in S24 (`engine/components/EngineComponents.cpp`, `engine/streaming/*`)

- `PendingTransition` is registered `Replicate::Host` (G13).
- Clients run `streamingSystem` (their own residency, G4/G10) and `simulationSystem` (held toggles only); stamping
  stays host-only; no new messages (G15).
- The host + client scenario variants compare the replicated components (incl. `PendingTransition` during an arrival)
  and S22/S23b's modification state as before; residency is derived and not compared. The end condition is unchanged.

### 11. App: complete points and the loading signal (`engine/app/EngineApp.*`, `engine/app/LoadingScreen.*`)

- Every fatal complete point (the first frame, captures, the benchmark frames) calls `world.completeResidency(timeout)`
  before `completeView` (its time is part of the stream time subtracted from the frame time, S20).
- `SpaceLoading` gains `bool arrival` and `Entity entity`. `updateSpaceLoading` first: when `localPlayer()` is alive
  and has a `PendingTransition` on the app's world, a load of the target space runs (started at the request frame,
  progress = `arrivalProgress` scaled to [0, 0.5]); after the move the S21 rules continue the same load with the view
  space's readiness scaled to [0.5, 1], ending when the readiness is 1, no terrain upload is pending and
  `world.chunkReady` holds at the player's position (on a client: its own residency, G19). `loadingMaxFrames` counts
  from the move only (the arrival part has no frame cap: the player is held and hidden until it completes, and ending
  the screen earlier would show nothing); it keeps ending a stuck load with a log line. With `--coop` the client world's local player decides (S21).

### 12. World API, MCP tools, debug UI (`engine/api/WorldApi.*`, `engine/tools/WorldTools.*`, `engine/app/EngineAppDebugUi.cpp`)

| Tool | Reads / changes |
|---|---|
| `streaming_status(space?, chunks?: bool, limit? (100, <= 1000), after?)` | reads: `enabled`, `synchronous`, the stats of Scope 2, per space the counts of requested / resident / pending chunks and the centres (entity, position, `arrival`), the islands and the limit, `simulated` (count), the pool's queued / running jobs; with `chunks` the chunks in `ChunkId` order (state, parts ready, the nearest centre distance), paged |
| `request_arrival(entity, space?, position: [x, y, z], rotation?)` | changes: `World::requestArrival` between steps on the host; errors on a client, for a refused request (with the reason: pending, island limit, range) and inside a step; returns `accepted`, the target and the island progress |
| `chunk_info` (S23b) | gains `residency: {state, parts, requested}` |
| `world_status` | gains `streaming: {enabled, resident, pending, stalls, islands, simulated}` |
| `inspect` | gains `held` (a pending arrival in a streaming world) next to `flags.simulated` |

- The Engine tab gets a section "Streaming" (`engine/Streaming`, collapsed by default): enabled / synchronous, resident
  / requested / pending chunks per space with a centre, stalls (total and of the last 600 steps), islands vs the
  limit, simulated entities, the last `streaming` and `simulation` times, resident bytes. The console reaches the two
  new tools automatically (the tools are its commands).

### 13. Tech demo (`samples/games/flight/`)

- `game_flight` sets `config.streaming = true` (G22) with its radii (full 3000 m as today, simulation 1000 m, prefetch
  1000 m, unload 256 m) and `StreamingAnchor` on every plane it spawns; the pinned memory is reported. Its scenarios (`--scenarios`, incl. `flight_content_hashes` and the ruin) pass in
  every variant; `--demo` and `--coop --demo` screenshots are unchanged (the ruin stamped, the plane flying); the Engine
  tab's Streaming section shows the resident set following the plane. Other games unchanged (G22).

## Out of scope

- Entity unload / reload, the chunk store, commits, `onCatchUp`, space unloading, the pending list of roots on
  non-resident chunks (S25, §9.6). In S24 entities stay in memory, frozen.
- Per-chunk modification snapshots on `ChunkData`, client chunk subscriptions, the queueing of live modifications until
  a snapshot, the re-request on gaps, the handshake (S26, §13.2).
- Colliders and physics activation (S28; the part list and the listener are ready). Far content and region levels
  (S27). Pieces (S30). The `host_step` benchmark (S29).
- Streaming around entities other than players and arrivals (`alwaysSimulate` keeps no chunk loaded, §9.6, §20).
- `maxPlayers` (S33).

## Design notes

- **Why the residency changes no height.** `terrainHeight` was already exact without tiles (point evaluation); the
  base tiles make it cheap. With flight's generator a point costs about 0.8 µs (50 ms per 65 792 samples), so 10^4
  movers would cost about 25 ms per step without tiles and below 1 ms with them.
- **Why `Simulated` by cells.** The cell rule is the same quantization ticking uses since S10 (rings of cells), it is
  exact and incremental, and its cost is proportional to what changed: a player crossing a cell border at 384 m flips
  about 50 cells; at the benchmark's density (about 11 entities per cell) about 550 evaluations per crossing.
- **Why the requested set and not residency for simulation rules (G7).** It makes the repeat, save/load and
  host + client variants independent of pool timing and of the unload hysteresis, and in the live game it only means
  that an `alwaysSimulate` entity of a chunk that is still resident but no longer requested freezes a little earlier.
- **Why arrivals reuse `PendingTransition`.** It is saved since S21 (an arrival survives a save), it already carries
  target and pose, and `spaceTransitionSystem` already completes it in phase 1. One path for teleports, respawns, joins
  and space links.
- **Why `requestTransition` returns `bool` now.** The hard island limit can refuse it; the S21 callers ignore the value.
- **Results never wait in a step.** Nothing in a step blocks on the pool (§12.3 "nothing waits for them"); the only
  blocking calls are `completeResidency` (between steps: load, the app's complete points, tests) and the synchronous
  mode of `ScenarioWorld`.

## Interfaces for S25 and S26

The extensions below are agreed here and implemented by the consuming stage, with its tests (S24 ships no hook
without a consumer). S25 (world save directory, entity unload/reload, space unloading) builds on:

1. `ChunkResidency::addListener(ResidencyListener)`: S25 adds the members `onResident` (its reload and record
   triggers) and `onKeptChanged(ChunkId, bool)` (the `kept` predicate flipped: its entity unload trigger) and uses
   `onUnload` / `onReadyChanged`; it accumulates and acts in its own systems (listeners never write World state).
2. `ChunkResidency::addPart(ResidencyPartInfo {bit, wanted, build, request, drop})`, added by S25 for `PartStored`
   (the chunk store's record); S28 adds `PartColliders` through it.
3. `ChunkResidency::state` / `requested` / `ready` / `kept`, `centres()` per space (S25's space unloading: a space
   without centres). S25 unloads entities beyond the kept distance of every current centre (`kept`, a pure predicate
   of the current centres, so the hysteresis needs no history and stays exact across a load) and reloads them when
   their chunk is ready.
4. `SimulationSet::evaluateCreated` (reloaded groups are created entities, so their flags are evaluated at the flush,
   G8) and `World::refreshSimulated(Entity)`; `Simulated` toggles call no hook, S25's reload calls `onCatchUp`.
5. `World::completeResidency` and the step-start refresh of G5 with a hook list run after it (`World::addStepStartHook`,
   added by S25 for its reloads, so entities of chunks made ready between steps exist before phase 1).

S26 (the world over the network) builds on:

1. The client centres (G4) and `forEachChunk(space, fn)` over the `Requested` and `Resident` chunks (added by S26): S26
   diffs them into the client's chunk subscribe / unsubscribe messages at request time (a chunk cannot become resident
   on a client before its snapshot).
2. `ChunkResidency::setModificationGate` (added by S26; re-checked in every `update()` for a `Requested` chunk whose
   parts are ready): S26 sets one on clients that is true once the host's per-chunk modification snapshot is applied,
   and a chunk without it is not `Resident` (the client's mover holds, §12.3 "Clients").
3. S26 adds its per-chunk queue in its own apply paths (S24 adds no hook there, G20).
4. `PendingTransition` is replicated (G13): S26's join arrives through `requestArrival` (§12.3) and the joining
   client's loading screen uses G19 unchanged.
5. `dropAll()` for a client whose generation is replaced (S26's pin adoption and `Resync` call it, then the next update
   requests again).

## Implementation steps

Each step builds (warnings as errors) and passes ctest with the tests named in it.

1. **Reconcile** with the code of the day (the names of the "Current state" table, `TerrainStore`'s thread rule,
   `SpatialGrid`'s cell map, the World's system registration in the constructor).
2. **Configuration**: `WorldConfig` / `SpaceTemplate` fields and validation, `StreamingSettings`, `ScenarioWorld`'s
   overrides; `StreamingConfig.*`.
3. **Residency, synchronous**: `ChunkResidency` with the three distances, trivial chunks, the synchronous build,
   `TerrainStore` pins, `pinSimulationTile`, `chunkReady`, the `streaming` system, stamping on resident chunks;
   `Residency.*` (synchronous cases).
4. **Residency, asynchronous**: the residency job with `baseInJob`, priorities, the delivery queue, counters and versions,
   `drainStreamingPool`, `completeResidency`, statistics, `jobHookForTests`; the remaining `Residency.*` incl. the two
   acceptance tests.
5. **Edits**: no code (G20); `Residency.NonResidentEditInModificationState`.
6. **`Simulated`**: `SimulationSet`, `SpatialGrid::forEachInCell`, `simulationSystem`, `simulated` / `refreshSimulated`;
   `Simulated.*`.
7. **Mover hold and stalls**: `characterMoveSystem`, the counters; `Residency.StallsCounted`, `Simulated.MoverHolds`.
8. **Arrivals and islands**: `requestArrival`, `checkArrival`, `requestTransition` returning `bool`, the completion
   rule, the extract's held state, `IslandLimitEvent`, `Islands`; `PendingTransition` `Replicate::Host`; `Arrival.*`.
9. **Save/load**: the load rebuild and the flags rule; `StreamingSave.*`.
10. **Clients**: client centres, held toggles; `StreamingNet.*`.
11. **App**: complete points, the loading signal, the Engine tab section; `EngineApp.Streaming*`,
    `DebugUi.StreamingSectionRenders`.
12. **Tools**: `streaming_status`, `request_arrival`, the extensions; their `handleMessage` tests.
13. **Scenarios**: `registerStreamingScenarios` in every variant.
14. **Benchmark**: `test_engine_streaming_benchmark.cpp`; the local full run (release build of the bench recipe).
15. **Flight**: `streaming = true`, its scenarios, the demo screenshots.
16. **Docs**: the amendments below, `AGENTS.md` (layout line for `engine/streaming/`, the test list and count, the test
    budget if a binary exceeds 2 s), §3's row.

## Tests

All engine tests use `tests/streaming_test_setup.h` (new): `streamingTestSetup()` = `contentTestSetup(2)` (overworld
128 chunks at 2 m with `test_tree` / `test_rock` slots, sites 0 and 1, `test_valley`, `test_dungeon`) with
`config.streaming = true`, an appended flat template `test_flat` and small radii (`simulationRadius 64`, `fullRadius
128`, `prefetchMargin 64`, `unloadMargin 64`; the cell quantization makes the effective simulation radius 64 +- one cell
per axis, which the brute-force tests share), plus the test-only game objects `Ticker` (counts its `update` calls in a saved component `TickCount`) and
`Drifter` (writes a seeded `CharacterIntent` in `update`), and a phase-4 test system `countSystem` (increments
`TickCount::systemSteps` over `view<TickCount>`). Asynchronous tests hold jobs with `ChunkResidency::jobHookForTests`
and a latch (the `latchState` helpers of `tests/terrain_test_hooks.h`) and poll with bounded loops (no sleeps). Tests
that count no generations, cache hits or jobs share one `ContentCache` per process (`sharedStreamingCache()`, the
AGENTS.md rule; named in the test file's header comment: the `Simulated.*`, `Arrival.*` except
`HeldUntilIslandResident`, `StreamingConfig.*` and `StreamingNet.*` cases); the `Residency.*` cases that count jobs or
generations use their own.

`tests/test_engine_streaming.cpp` (new; budget about 1 s alone, at most 1.5 s, no Vulkan):

- `StreamingConfig.*`: the validation death tests (`simulationRadius > fullRadius`, a negative margin, a NaN radius:
  `SAIGA_EXIT_ERROR` naming template and field, every build); a world without streaming registers neither `streaming`
  nor `simulation`, keeps every entity simulated with a far player, creates no residency entry and no pool, completes
  an arrival in the next step, and its moved entities are never held (the unchanged existing tests and scenarios are
  the rest of G1's proof); `ScenarioWorld` lowers the radii and sets `synchronous`.
- `Residency.*`:
  - `RequestSetFollowsPlayer` (sync): after one step the requested set equals the chunks within `fullRadius +
    prefetchMargin` (brute force over the space) and all are resident; after a 1 km move the chunks beyond the kept
    distance are `None` (tile unpinned, placement released, `onUnload` called), those between requested and kept distance
    are still resident (hysteresis) but not `ready`;
  - `TrivialChunksNeedNoJobs`: a world of only `test_flat` and `test_dungeon` (no terrain, no slots): resident in the
    requesting update, `poolCreated()` false;
  - `BetweenStepChangeSeenNextStep` (sync, G5): a player teleported 1 km between steps: the next step's phase 1 sees
    its new chunks ready (a test system in phase 1 reads `chunkReady`), as a world loaded right after the teleport does;
  - `AsyncAppliesInPhase7`: a delivered result (pool idle, drained by an `updateView`-style drain between steps) leaves
    the chunk `Requested` until the next step's phase 7;
  - `StaleResultDropped`: a held job of a chunk that is unloaded and requested again: its result (old counter) is
    dropped (`resultsDropped == 1`) and the new request's result applies;
  - `PlacementVersionChangeReRequests`: `ChunkRecords::injectForTests` gives a pending chunk another placement version
    and `onRecordChanged`: the held result is dropped, the new request uses the new version;
  - **`PendingChunkEditSurvives`** (acceptance, asynchronous): the player's chunk C held on the latch (Base and
    Placement); C is `Requested`, the step counts a stall and the mover holds the player (position unchanged with
    `move`); `editTerrain` (Add, 8 m, a box that excludes the chosen tree) on C and `tombstone` of C's first accepted
    `test_tree` outside that box: `terrainSample` shows the edit, the sequence is the one before + 2; the latch opens; within a bounded loop of steps C becomes `Resident` in a phase 7: the
    pinned tile holds the **base** heights (equal to the height source), `terrainSample` / `terrainHeight` still show
    the edit (equal to the integer rule over base + delta), the sequence is unchanged, `appliedAfterModification == 1`,
    `isTombstoned` holds and the `Players` content set has the instance at scale 0; the stalls stop and the player
    moves again;
  - **`NonResidentEditInModificationState`** (acceptance): an edit and a tombstone in a chunk 10 km from every centre:
    sequence 2, deltas present, the record has the key, the chunk's state stays `None` (no request), `baseHookSamples`
    and the cache's placement generations grew, a save / load round trip keeps them; after `requestArrival` there and the arrival the
    chunk is resident and the edited heights and the hidden instance are what the player sees;
  - `CompleteResidency`: returns true with every required chunk resident; with a held job and a 0.05 s timeout false
    (what was ready is applied);
  - `ResidentTilesServeSamples`: `terrainHeight` inside resident chunks adds nothing to `baseHookSamples`, outside it
    adds three per call;
  - `VisualKeepsPinnedTiles`: `updateView` with `tilesPerLevel = 4` and a camera elsewhere evicts no pinned tile; an
    unpinned, unsent tile is removed at once, a sent one by the LRU;
  - `PinnedTilesDoNotOverflowAtlas` (blocker regression): a recording sink, `tilesPerLevel = 4`, the camera and the
    player sweep together across 12 pinned chunks: the sink never holds more than 4 level-0 tiles, every pinned tile
    that left the desired set is removed from the sink but stays in the store;
  - `ViewSwitchKeepsPinnedTiles`: a player resident in space A while the view switches to `test_valley`: A's pinned
    tiles stay in A's store, `terrainHeight` in A adds no `baseHookSamples`;
  - `PromotedPrefetchResubmitted`: a queued prefetch job whose chunk becomes required (player moves toward it) is
    cancelled and resubmitted at the required priority (job hook log), the in-flight count drops at the cancel;
  - `ResidentVersionChange`: `injectForTests` on a resident chunk: `ready` false, the placement re-requested at the new
    version, the base stays pinned;
  - `OutOfRangeClamped`: a player at x = -5 m and one beyond the far edge: `chunkReady` reads the edge chunk, the mover
    does not hold;
  - `StampOnResident` (asynchronous): a site chunk within `fullRadius` is not stamped while held, and is stamped in the
    phase 7 in which it becomes resident (the stamped entities exist after that step's flush);
  - `PriorityRequiredFirst`: with `maxInFlight = 2` the submission order (job hook log) has every required chunk before
    every prefetch chunk, nearest first;
  - `StallsCounted`: two held players: `streamingStalls` +1 per step, `playerStallSteps` +2.
- `Simulated.*` (plain worlds with the test setup, sync):
  - `FarEntitiesFreeze`: `Ticker`s at 30 m and 300 m: the near one's update and system counts grow, the far one's stay
    0; `viewAll` in phase 4 sees both; `ownedBy` skips the far one; `behaviourEventSystem` does not deliver a
    `DamageEvent` to the far one;
  - `MatchesBruteForce` (G8 exactness): 400 steps with a seeded walk of two players (one changes space twice), 2000
    `Drifter`s, spawns, destroys, teleports and parent changes: after every step the flags equal a brute-force
    evaluation of G8's rule for every entity;
  - `AlwaysSimulateWhileRequestedAndResident`: an `alwaysSimulate` `Ticker` beyond the simulation radius in a
    requested chunk ticks; after the player leaves (chunk kept by the hysteresis but not requested) it freezes; an
    `alwaysSimulate` edit through `set_component` applies without a move (`refreshSimulated`);
  - `PlayersAndStateEntities`: a far player is simulated; the world-state and space-state entities of a space without
    players are simulated;
  - `ChildrenFollowRoot`: a child of a frozen root is frozen, re-parented under a simulated root it is simulated
    after that step's phase 7;
  - `NewEntitiesEvaluatedAtCreation`: a far `Ticker` spawned between steps and one spawned by a command in phase 4 are
    never updated (count 0 after several steps); a near one spawned by a command ticks from the next step;
  - `AnchorIsCentre`: a `StreamingAnchor` root without `Player` keeps its chunks resident and is simulated; a `Player`
    child of a moving root is a centre at its world position and its root is simulated;
  - `UnfreezeContinuesSinceTick`: a frozen `Ticker`'s `sinceTick` keeps its value and resumes; no `onCatchUp`;
  - `MoverHolds`: a `Drifter` moving into a held chunk stops at the border, `moverHolds` grows.
- `Arrival.*`:
  - `HeldUntilIslandResident` (asynchronous, latch): `requestArrival` 3 km away between steps: `PendingTransition`
    present, the entity frozen at once, the recording sink has no instance for it, its pose unchanged,
    `arrivalProgress` < 1; steps with the latch closed keep it held; after the latch opens it is moved in phase 1 of the
    step after the island became resident (`SpaceTransitionEvent` with a null link, `from == to`), visible again as a
    snap, simulated after that step's phase 7;
  - `SyncCompletesNextStep`: in synchronous mode the move happens in the next step (S21's latency);
  - `TransitionIsArrival` (asynchronous, latch): `requestTransition` through a `SpaceLink` to `test_valley` (terrain,
    so the island has jobs) returns true and waits for the valley's island while the latch holds; a second request
    returns false;
  - `NonPlayerArrivalUsesS21Rule`: an NPC root's `requestArrival` 3 km away completes in the next step without a centre
    or hold and lands frozen;
  - `IslandLimit`: `maxSimulationIslands = 2`, three players far apart: one `IslandLimitEvent {3}`, `islandLimitSteps`
    grows; a fourth far player: a second event `{4}`; `hardIslandLimit`: `requestArrival` to a new island returns false
    (logged), into an existing island true, `requestTransition` refused returns false;
  - `Refusals`: a client call asserts (death test with asserts), an unknown space, a non-finite position, a position
    outside the space: false with a log line;
  - `StreamingOffUnchanged`: without streaming an arrival completes in the next step with no hold or hiding.
- `StreamingSave.*`:
  - `RebuildBeforeLoadReturns` (asynchronous): a world saved with a pending arrival and partly resident chunks; the load
    rebuilds the required chunks before `tryLoad` returns (resident, no step); the arrival completes; `save()` of the
    loaded world equals the saved bytes;
  - `SavedFlagsAreAllSet`: `save()` of a streaming world with frozen entities writes `kSimulated` for every alive entity;
  - `FlagsContinueExactly` (synchronous, the flag equality): after a save between steps the loaded world's flags equal
    the source's, and after each of 50 further steps of both worlds they stay equal (also with a `requestArrival` and a
    teleport made between the save and the next step in both worlds);
  - `IslandLimitNotRepeated`: a world above `maxSimulationIslands` saved and loaded emits no second `IslandLimitEvent`;
  - `StreamingOffSetsEveryFlag`: a streaming-off world's load sets every flag.
- `StreamingNet.*` (loopback host + client): the client's requested set is around its own player only, the host's
  around both players; the client generates its chunks itself (cache generations on the client world); the client's
  mover holds its player on a held chunk while the host's chunk is resident; `PendingTransition` reaches the client and
  its residency requests the target before the host moves the player; the host drops the held client player's owner
  states (its host pose unchanged while held); replicated far entities are simulated on the
  client, the held player is not; no message type is added (the loopback log).

`tests/test_engine_streaming_scenarios.cpp` (new binary; `test_engine_scenarios` is at 1.07 s today, and S24, S25 and
S26 each add streaming scenarios, so they get their own binary with a budget of 1.5 s): `registerStreamingScenarios`
on `streamingScenarioSetup()` (the test setup above; the
scenario runner's overrides apply), every variant (plain, repeat with `varyThreads`, save/load with
`strictContinuation`, host + client with delay 0 and 2 and asynchronous stepping):

- `streaming_follows_player`: a player (`moveSpeed 32`) walks 256 m east from (1150, 1400) in chunk (4, 5), past site
  0 toward site 1 at (1536, 1536); a step listener checks on the player's owner (host, and the client in the network
  variants) that every chunk within `fullRadius` is resident (`streamingStalls` 0 is implied in synchronous mode; the
  stall counting itself is tested asynchronously in `Residency.*`); the stamped entities of every
  site chunk it reaches exist from the step in which the chunk became resident, never before (host); the chunks behind
  it beyond the kept distance are `None`.
- `far_entities_freeze`: `Ticker`s at 40 m and 200 m; the far one's count stays 0 until the player walks to it, then
  grows while the near one, now far, stops.
- `arrival_far_teleport`: `call("request_arrival", ...)` 3 km away: the player is at the target exactly one step later
  (synchronous), the target's chunks are resident, the old ones unloaded, `SpaceTransitionEvent` read in the step
  after.
- `edit_far_chunk_then_visit`: `editTerrain` and a tombstone at 2 km (not resident), then an arrival there: host and
  client `terrainHeight` equal the edited surface, every peer's recording sink hides the tombstoned instance.
- `space_link_with_streaming`: S21's interact chain into `test_dungeon` on the streaming setup (the move 2 steps after
  the interact step, as S21).

Measured budget: the five scenarios take at most 0.6 s in the new binary (the content cache is shared per run).

`tests/test_engine_world_api.cpp`: `streaming_status` (counts, centres, the chunk list with paging, a world without
streaming: `enabled: false`), `request_arrival` (host accepted with the progress, a client error, a refused second
request with its reason, the island limit refusal, inside a step refused), `chunk_info.residency`,
`world_status.streaming`, `inspect.held`, through `handleMessage`.

`tests/test_engine_app.cpp` (headless, `--no-render` where no frame is needed):

- `EngineApp.StreamingLoadingAtRequest`: a streaming world whose local player gets a held arrival: the loading screen
  is active from the request frame with the target space and a progress below 0.5; it continues after the move and
  ends at readiness 1 with the player's chunk ready;
- `EngineApp.StreamingCompletePoint`: the first frame's complete point leaves every required chunk resident.

`tests/test_engine_debug_ui.cpp`: `DebugUi.StreamingSectionRenders` (the Engine tab with the section open, a streaming
world and a world without streaming).

`tests/test_engine_streaming_benchmark.cpp` (new, non-gating, G21; prints, asserts only correctness):

- `StreamingBenchmark.StepTime`: a streaming world on the prototype `HookGenerator` at 1 m (no slots; realistic base
  cost) with the asynchronous pool; entities spread uniformly over a 2.15 km square (10^5 locally, 10^4 in CI) of which
  `Drifter`s are 10 % (so about 10^4 locally / 10^3 in CI lie within the simulation radius), one player walking at 30
  m/s through the square; reports the step time (median, p95, max) and its `streaming` / `simulation` / `tick` /
  `behaviour` parts (the `tick` reassignment when the player crosses a 32 m cell, `TickSystem.cpp:95-101`, is the
  expected spike and is reported separately), the simulated count, `syncBuilt`, resident bytes, stalls. CI: radii
  scaled down (`fullRadius 128`, `prefetchMargin 0`, `simulationRadius 64`: 4 chunks of 257^2 hook samples), 10^3
  entities, 60 steps unpaced after a `completeResidency` (budget 0.5 s, measured in step 14). Locally with `SAIGA_STREAMING_BENCHMARK=full`: 3600 steps paced at 60 Hz (real time, 2 pool workers);
  prints `streamingStalls` (the local acceptance: 0) and the step numbers against §17.10's 4 ms.
- `StreamingBenchmark.Rebuild`: `completeResidency` around a player from cold and warm caches (ms), the load rebuild;
  in CI with the scaled radii, once each (budget 0.5 s).

Benchmarks (R-TEST-2, local, release build of the AGENTS.md bench recipe): the full streaming benchmark before (S23b:
streaming off, all simulated) and after; `game_world --benchmark flyover` and `forest` within 5 % (streaming off; the
`TerrainStore` pin check costs nothing).

## Done when

The S24 row of §22:

- **Scenarios with streaming** incl. the repeat and save/load variants: `registerStreamingScenarios` passes in every
  variant (CI).
- **Unit test with the asynchronous pool and a held result**: `Residency.PendingChunkEditSurvives` and
  `Residency.NonResidentEditInModificationState` pass (CI).
- **0 streaming stalls** in the local full run of `StreamingBenchmark.StepTime` (asynchronous pool, a player at 30 m/s,
  real time), reported in the Results (local).
- **Step time** with 10^5 resident and 10^4 simulated entities and one player at 30 m/s within §17.10's step budget
  (4 ms, release, the development CPU), reported in the Results with the before/after numbers (local).

Also: every other test above (CI); the existing engine tests, scenarios, goldens and saves unchanged (streaming off);
`game_flight --scenarios` passes and its demo screenshots are reviewed (local); `docs/engine_design.md` and `AGENTS.md`
updated.

## Verification

- The CI recipe of AGENTS.md (`SAIGA_WARNINGS_AS_ERRORS=ON`; samples ON once for the games' compile check).
- `ctest -R "test_engine_(streaming|scenarios|world_api|app|debug_ui|save|content|terrain.*|spaces|replication)"
  --output-on-failure`, then the whole `ctest --output-on-failure --parallel 4`.
- `./build/tests/test_engine_streaming --gtest_print_time=1` and `./build/tests/test_engine_streaming_scenarios
  --gtest_print_time=1` against the budgets above.
- Local: `SAIGA_STREAMING_BENCHMARK=full ./build-bench/tests/test_engine_streaming_benchmark` (configure the bench
  build with `-DSAIGA_BUILD_TESTS=ON` for this binary), `game_world --benchmark flyover|forest` (bench build),
  `./build/samples/games/game_flight --scenarios`, `SAIGA_SCREENSHOT=/tmp/flight.png ./build/samples/games/game_flight
  --demo` and with `--coop` (headless screenshot path; no window is opened by the tests themselves).

## Files

New:

- `src/saiga/engine/streaming/ChunkResidency.{h,cpp}`, `SimulationSet.{h,cpp}`, `Islands.{h,cpp}`,
  `StreamingStats.h`, `StreamingSettings.h`, `Systems.h` (`streamingSystem`, `simulationSystem`),
  `StreamingSystem.cpp`, `SimulationSystem.cpp`, `Arrivals.cpp` (the World's arrival members and `arrivalReady`).
- `tests/test_engine_streaming.cpp`, `tests/test_engine_streaming_scenarios.cpp`, `tests/test_engine_streaming_benchmark.cpp`,
  `tests/streaming_test_setup.h`.
- `src/saiga/engine/components/StreamingAnchor.h`, `IslandState.h`.

Changed:

- engine: `world/World.{h,cpp}` (config fields, `StreamingSettings`, `residency()`, `simulated`, `refreshSimulated`,
  `chunkReady`, `completeResidency`, `requestArrival`, `checkArrival`, `arrivalProgress`, `islandCount`,
  `requestTransition` returning `bool`, the step-start refresh, the conditional system registration, destruction
  order), `world/SpaceTemplate.h` (fields, validation), `world/SpaceTransitionSystem.cpp` (the arrival rule),
  `world/SpatialGrid.h` (`forEachInCell`), `world/Events.h` and the event registration (`IslandLimitEvent`),
  `components/EngineComponents.cpp` (`PendingTransition` `Replicate::Host`), `components/PendingTransition.h`
  (comment), `gameplay/CharacterMoveSystem.cpp` (hold), `gameplay/Behaviour.h` (comment), `terrain/TerrainStore.*`
  (pins, thread rule), `terrain/TerrainSystem.*` (`pinSimulationTile`, `unpinSimulationTile`, `drainStreamingPool`,
  `replaceSources`), `terrain/TerrainEdit.cpp`, `content/Tombstones.cpp`, `content/ContentMessages.cpp`,
  `content/GenerationPipeline.*` / `content/ContentCache.*` (`baseInJob`, the base in-flight entry),
  (`baseInJob`), `content/ChunkStampSystem.cpp` (resident chunks in streaming worlds),
  `sync/ExtractSystem.*` (held entities), `save/WorldLoad.cpp` (flags, rebuild), `api/WorldApi.*`,
  `tools/WorldTools.*`, `tools/ScenarioWorld.*` (radii, synchronous), `app/EngineApp.*`, `app/LoadingScreen.*`,
  `app/EngineAppDebugUi.cpp`.
- tests: `tests/CMakeLists.txt` (the three binaries inside `if (MODULE_ENGINE)`), `tests/test_engine_streaming_scenarios.cpp`,
  `tests/test_engine_world_api.cpp`, `tests/test_engine_app.cpp`, `tests/test_engine_debug_ui.cpp`.
- games: `samples/games/flight/main.cpp` (or its world setup: `config.streaming = true`).
- docs: `docs/engine_design.md` (below), `AGENTS.md`, this plan (Status, Design changes, Results).

## Risks and open questions

- **Test churn from `PendingTransition` replication (G13).** The S21 host + client scenarios now compare the component
  while a transition is pending; they end without one, so the end comparison is unchanged, but the write detector and
  the snapshot sizes change slightly. Fallback if this causes trouble: a separate replicated marker component
  (`ArrivalPending {SpaceId target; Vec3 position}`) instead of changing the mode.
- **Pool contention.** Residency and the visual path share two workers; the required-chunk priority goes first. If the
  local stall run shows stalls caused by visual jobs, `TerrainViewSettings::workers` becomes 3 for streaming worlds (a
  documented default change) before anything else.
- **Phase-7 drain of visual completions inside a step.** They mutate the store and the sink on the main thread; the
  extract (phase 8) already calls the sink inside the step. If a visual completion turns out to assume "between
  steps" (an assert), the phase-7 drain is limited to the residency's jobs (a per-job tag in the completion queue).
- **Cell quantization of the simulation radius (G8).** Cell-index distance differs from the true distance by up to one
  cell per axis (about 45 m at 32 m cells); games set radii with that in mind. Acceptable (ticking has the same property).
- **Hidden held entities and the camera.** A held entity has no sink instance, so `interpolatedPose` returns false
  for it (as for S19b's `Pending` source); a game camera that follows the local player must keep its last pose while
  that happens. The tech demos' chase cameras are checked on the flight demo's respawn path (local); a camera that
  does not cope gets the fix in the game, not an engine exception.

## Design changes to engine_design.md

Applied in the stage's change (R-DOC-2); listed with the decision that causes them.

- §3: a row "Streaming (S24)" (the residency, `Simulated`, arrivals, islands, tools) and the S21 row's
  `requestTransition` note.
- §7.5: "every new entity has the flag until S24" -> "in a streaming world a new entity's flag is evaluated when it
  becomes alive; worlds without streaming keep every entity simulated (G1)".
- §7.9: `WorldConfig::streaming`, `maxSimulationIslands`, `hardIslandLimit` [exists: S24]; `maxPlayers` [planned: S33]
  (G14); `SpaceTemplate::simulationRadius`, `prefetchMargin`, `unloadMargin` [exists: S24] with the validation;
  `StreamingSettings` and `setStreamingSynchronous`.
- §7.10: `requestTransition` returns `bool`; `requestArrival(e, space, position, std::optional<Quat> rotation = {})`
  [exists]; new: `checkArrival`, `arrivalProgress`, `islandCount`, `simulated`, `refreshSimulated`, `chunkReady`,
  `completeResidency`, `residency()` (engine-internal); the component `StreamingAnchor` (G4).
- §8.1: phase 7: "the residency update (requests, unloads, the pool's completions applied), the `Simulated` evaluation,
  then stamping (§12.3)"; before phase 1 the step-start refresh when centres changed between steps (G5); phase 1: `spaceTransitionSystem` completes arrivals whose island is
  resident.
- §8.2: the registration of `streaming` and `simulation` (phase 7, both, in this order) by the World constructor when
  `WorldConfig::streaming` is set (as `chunk_stamp`).
- §8.5: the residency's jobs on the shared pool with their priorities, the phase-7 drain, `TerrainStore`'s thread rule
  (G16).
- §8.6: "synchronous chunk loading" is `World::setStreamingSynchronous` set by `ScenarioWorld`; G7's rule.
- §9.2 / §9.6: `alwaysSimulate` is read from S24 ("requested and resident", G7); un-freezing calls no `onCatchUp`.
- §9.3: the mover hold in streaming worlds (G11).
- §9.4: event consumers check `world.simulated(target)`.
- §12.2: the pinned level-0 tiles of the residency in `TerrainStore` ("S24 adds the players' residency" made concrete,
  G17); "Simulation radii and the chunk store follow in S24 and S25" -> what S24 did.
- §12.3: rewritten as implemented: G1 (a policy), G2 (what is resident), G3 (three distances, replacing "fullRadius +
  unloadMargin"), G4 (centres), G5/G6 (apply in phase 7, re-patch by construction), G7, G8 (exact cells instead of
  round-robin), G9 (the flag in the slot byte instead of "not saved"), G10 (clients), G11 (stalls), G12 (arrivals via
  `PendingTransition`), G14 (islands).
- §12.4: arrivals and transitions (G12, G13), "Before S24" paragraphs kept as history with "S24:" notes; the loading
  screen's request-time signal (G19).
- §13.2: "S24 (before S26)": clients stream their own residency without a modification gate (G10); the gate and the
  queueing are S26's (Interfaces).
- §15.1: `save()` writes `kSimulated` set for every alive entity (G9, not saved); the load rebuilds the residency and the
  flags (Scope 9); `IslandState` on the world-state entity of streaming worlds (G14).
- §16.2: `streaming_status`, `request_arrival`, the `chunk_info` / `world_status` / `inspect` additions (G18).
- §17.2: `ScenarioWorld` lowers `simulationRadius` and `prefetchMargin` and loads synchronously; the streaming
  scenarios.
- §17.7: the streaming benchmark (`test_engine_streaming_benchmark`, the full local form, G21).
- §17.11 / AGENTS.md: the new binaries and their budgets.
- §20: the simplification "Round-robin tick/`Simulated` re-evaluation" becomes "`Simulated` by 32 m cells, exact" (G8);
  a new simplification "Streaming is opt-in per world" (G1).
- §22: the S24 row's status when done.
