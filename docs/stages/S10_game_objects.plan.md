# S10: Game objects, ticking and the world API

Status: done.

Sections follow the stage plan template of S05.

## Goal

Games are written as game objects on the S08 world: entities with a `GameObject` (lifecycle hooks, `update`, event
handlers) owned by a `Behaviour` component, spawned by class, ticked by grid-cell rings around the players, with the
engine's gameplay basics (players, health, input, actions, a kinematic mover), world-state entities and `Persistent`
IDs (docs/engine_design.md §7.8, §8.4, §9.2-§9.4, §9.6). Every piece of state is reachable by name from outside:
the world API (`WorldApi`) and its MCP tools let agents list, inspect, change, spawn and destroy entities, emit
events, queue player actions and control time (moved forward from S14, §16.2). S11 (saves) and S12 (scenarios) build
on it.

## Depends on

S09 (grid: `near` queries of the API), S08 (World, commands, events, structural log), S07 (visit, field lists),
`McpServer` (saiga_core).

## Rules that apply

R-STATE-1 (state in components; members only configuration and `cache`), R-OO-1, R-REG-1/2 (strings only at the API
and MCP boundary), R-SYS-1/2, R-DET-1, R-MECH-1 (`TickPolicy`), R-MCP-1 (generic names, descriptions with reads and
changes, a `handleMessage` test per tool), R-FILE-1, R-DOC-1/2, R-TEST-1, R-TEST-2.

## Scope

### 1. Game objects (`engine/gameplay/`)

- `GameObject` (§9.2): hooks `configure`, `onSpawn`, `onLoad`, `onCatchUp`, `onReplicatedSpawn`,
  `onReplicatedDestroy`, `update(dtSinceLastTick)`, `onDamage`, `onInteract`, `onContact`, `onDestroy`,
  `tickPolicy()`; helpers `entity()`, `world()`, `read/write/add/remove<T>`, `spawn`, `spawnIn`, `destroy`,
  `transform()`, `transformWrite()`.
- `Behaviour` component: `std::unique_ptr<GameObject>`, `classId`, `tickInterval`, `sinceTick`, `alwaysSimulate`,
  `alwaysTickEveryStep`; `visit` covers all but the pointer (the object is recreated by its factory).
- `ClassList::add<T>()` (`T::id`, `T::name`, factory), `WorldSetup::classes`; contiguous IDs and unique names like
  components (`EngineClass` 0..63, game classes from 64).
- `world.spawn<T>(position, initial...)`, `spawnIn<T>(space, ...)` (space 0 until S21): between steps immediately,
  during a step reserved and applied at the flush; `Transform`, `Behaviour`, the initial components, then
  `configure()` and `onSpawn()`. `destroy` calls `onDestroy()` before the components are removed (also for descendants
  of the cascade, children first).

### 2. Systems and ticking

- `behaviourEventSystem` (phase 2, host): the previous step's `DamageEvent`, `InteractEvent`, `ContactEvent` to the
  target's handler, in emit order (dead or object-less targets skipped).
- `behaviourSystem` (phase 3, host, serial): `sinceTick += dt`; objects with `sinceTick >= tickInterval` get
  `update(sinceTick)` and `sinceTick = 0`, in the dense order of the `Behaviour` pool.
- Ticking by grid cells (changed from §8.4's distances, decided in planning):
  `struct TickPolicy { double ringSize = 32; int nearRing = 1; int midRing = 6; double midInterval = 0.1,
  farInterval = 0.5; }` in `WorldConfig` (a class overrides it with `tickPolicy()`). The ring of an object is the
  Chebyshev distance in `ringSize` cells (x, z) between its cell and the nearest player's cell in the same space:
  `<= nearRing` every step (the 3 x 3 cells around the player: 96 m), `<= midRing` every `midInterval` (13 x 13 cells,
  about 200 m), else `farInterval`; another space or no player: `farInterval`. `tickSystem` (phase 6, after the grid):
  builds the table cell -> ring for the cells around the players; re-evaluates the objects in the changed list and the
  new ones; all objects when the table changed (a player crossed a cell border, joined or left). Players and
  `alwaysTickEveryStep` tick every step; `sinceTick` starts at `(hashCombine(0, index) % 1024) / 1024 * tickInterval`.
  Exact and deterministic (no round-robin delay).

### 3. Engine gameplay basics (§9.3)

- Components `Player {peer, characterId, moveSpeed}`, `Health {hp, maxHp}`, `PlayerInput {move, look,
  heldButtons}`, `CharacterIntent {desiredVelocity, target}`; events `DamageEvent`, `InteractEvent`, `ContactEvent`,
  `ActionEvent` (engine event IDs).
- `world.queueAction(PlayerAction{player, action, aim})`: before S15 a local queue; delivered in phase 1 as
  `ActionEvent` in (peer, sequence) order, exactly once (S15 replaces the queue by the message).
- `playerControlSystem` (phase 4, `RunOn::Both`, `ownedBy<PlayerInput, CharacterIntent>`): `desiredVelocity = move *
  moveSpeed`. `characterMoveSystem` (phase 4, `RunOn::Both`): `position += desiredVelocity * dt`, `y = 0` (flat until
  S20).
- `localPeer()` (0), `isHost()` (true), `ownedBy<C...>(peer)` (all entities until S16).

### 4. World state (§7.8)

- One world-state entity and one space-state entity for space 0, created by the World; `Registry::registerGlobal<T>`
  and `registerSpaceState<T>` add their components; `global<T>()`, `spaceState<T>(space)`, `readGlobal`,
  `readSpaceState`. Flagged in the slot; not destroyable (asserts; the API returns an error), excluded from the grid
  and from ticking; visible in views and in the API (agents see and edit them like other entities).

### 5. Persistent IDs (§9.6)

- Engine component `Persistent {PersistentId id}`; `makePersistent(e)` (next value of a world counter, idempotent);
  `resolve(pid)` (`Entity{}` if not alive) through an index kept from the structural log (own consumer).

### 6. Reflection by name (`engine/ecs/`, `engine/world/`)

- `VisitJson.h`: `toJson(value)` and `fromJson(json, value, error)` over `visit`, with the field rules of the binary
  visitors: missing keys keep their values, unknown keys, wrong types and out-of-range numbers are errors (nothing is
  changed then); `Entity` as `{"index", "generation"}`, `Vec3` as `[x, y, z]`, `Quat` as `[w, x, y, z]`, enums by
  name. S11 reuses it for `dumpJson` and data files.
- Type-erased access by name, engine-internal: `Registry::findComponent(name)` (pool with JSON read, merge-write, add,
  remove and the field list for schemas), `ClassList::find(name)`, `EventQueues::find(name)` (emit from JSON).
  Strings are used only here (R-REG-2).

### 7. World API (`engine/api/WorldApi.h/.cpp`)

The C++ API that MCP tools, later the scenario helpers (S12) and an editor call; all calls between steps, results
instead of asserts (an agent's mistake must not stop the game), every change logged under the `api` category.

```cpp
struct ApiResult { bool ok; std::string error; json11::Json value; };
class WorldApi
{
    explicit WorldApi(World& world);
    ApiResult describe() const;                                    // components, classes, events, globals + fields
    ApiResult status() const;                                      // step, time, paused, time scale, counts, hash
    ApiResult list(const json11::Json& filter) const;              // class, components, near, flags, limit, after
    ApiResult inspect(Entity e, const std::vector<std::string>& components = {}) const;
    ApiResult setComponent(Entity e, const std::string& component, const json11::Json& values);   // merge
    ApiResult addComponent(Entity e, const std::string& component, const json11::Json& values);
    ApiResult removeComponent(Entity e, const std::string& component);
    ApiResult setTransform(Entity e, const json11::Json& pose);    // position, rotation, scale, teleport
    ApiResult spawn(const std::string& cls, SpaceId space, const Vec3& position, const json11::Json& components);
    ApiResult create(SpaceId space, const Vec3& position, const json11::Json& components);   // without a class
    ApiResult destroy(Entity e);                                   // returns the destroyed entities (cascade)
    ApiResult setGlobal(const std::string& component, const json11::Json& values, int space = -1);
    ApiResult emitEvent(const std::string& event, const json11::Json& values);
    ApiResult queueAction(Entity player, uint32_t action, const Vec3& aim);
    ApiResult pause(bool), step(int n), timeScale(double);
};
```

### 8. MCP tools (`engine/tools/WorldTools.h/.cpp`)

`registerWorldTools(McpServer&, World&)`: thin wrappers of the API (arguments validated, results as JSON, errors as
`isError` results). Handlers run in `McpServer::poll()`, which the app calls between steps (S13); tests call
`handleMessage`.

| Tool | Arguments -> result | Reads / changes |
|---|---|---|
| `describe_world` | -> components (name, id, replicate, saved, fields with types), classes, events, globals, space states | reads registrations |
| `world_status` | -> step, time, paused, time scale, entity and object counts, hash | reads |
| `list_entities` | `class?`, `components?: [names]`, `near?: {space?, position, radius}`, `limit = 100`, `after?` (entity index) -> `[{entity, class, position, components}]`, `next` | reads; sorted by index, paged |
| `inspect` | `entity`, `components?` -> class, flags, parent, children, every component as JSON | reads |
| `set_component` | `entity`, `component`, `values` (only the given fields) -> the new value | changes one component (bypasses game rules) |
| `add_component`, `remove_component` | `entity`, `component`, `values?` | structural change |
| `set_transform` | `entity`, `position?`, `rotation?`, `scale?`, `teleport = true` | moves through the setters |
| `spawn` | `class?` (none: plain entity), `position`, `space = 0`, `components?: {Name: values}` -> entity | creates |
| `destroy` | `entity` -> destroyed entities (descendants included) | destroys; `onDestroy` runs |
| `get_global`, `set_global` | `component`, `space?`, `values?` | reads / changes world or space state |
| `emit_event` | `event`, `values` | gameplay path: handlers run in the next step (e.g. damage through `onDamage`) |
| `queue_action` | `player`, `action`, `aim?` | delivered as `ActionEvent` in the next step |
| `pause`, `resume`, `step`, `time_scale` | `step(n)` only while paused | time |
| `logs` | `since`, `category?` -> lines with sequence numbers | reads the log ring |

Also (decided in planning): `apply` (a list of operations of the tools above, all validated first, then applied in
order; one result per operation), `changes_since` (`cursor` -> created, destroyed and changed entities since the
cursor, the new cursor; from an API consumer of the structural log and the changed list), and the profiling tools of
section 9. Entities are written `{"index", "generation"}` or `"12:3"` (results always use the object form);
`registerWorldTools(server, world, {allowEdits})` (false: only the reading tools); paging with `limit` (default 100,
at most 1000) and `after`. API edits are logged, not recorded as a replayable script (S12 may add that).

S14 keeps the app tools (`screenshot`, `set_camera`, `move_camera`, `run_scenario`, `quit`) and the `game_*` renames;
its frame and GPU numbers join the profiler of section 9.

### 9. Profiling (`engine/core/Profiler.h/.cpp`)

The numbers of the sample games' statistics window (frame, CPU work, GPU passes, counts), as data for the API, MCP and
later an ImGui window:

- `Profiler`: named scopes in a tree (`step` > `phase:<name>` > `system:<name>`, `commands:<phase>`), each with a ring of
  the last 600 samples (10 s at 60 Hz) -> count, avg, min, p50, p95, max (ms) and the share of the parent; named
  per-step counters (alive entities, game objects, objects updated per tick ring, commands applied, events emitted,
  structural records, grid items) with the same history. Wall-clock time (steady clock) only for these numbers, never
  for simulation (R-DET-1). `WorldSetup::profiling = true` (off: no clock calls).
- The World times every step, phase, system and command flush and writes the counters.
- From S13 the app adds `frame`, `extract`, `recording` and the GPU passes (timestamps) to the same profiler; S13's
  ImGui window "Engine statistics" (history plots, per-phase and per-system tables like the sample's "World
  Statistics") reads it.
- Tools: `get_profile` (`scopes?`, `steps?` (window, default all 600)) -> the tree with the statistics and the counters;
  `reset_profile`; `measure_performance` (`steps` (1..6000)): while paused runs that many steps and reports the
  statistics of exactly those steps; while running reports after that many steps have run (a deferred response,
  answered before the tool call times out).

Tests and docs:

- `tests/test_engine_gameobject.cpp` (lifecycle, classes, ticking, events, actions, mover, world state, Persistent),
  `tests/test_engine_world_api.cpp` (JSON visitors, every API call incl. errors, every tool through `handleMessage`),
  `tests/mcp_test_helper.h` (the `Poller`, moved from `test_core_mcp_server.cpp`), profiler tests (statistics, the tree,
counters, `measure_performance` paused and running), the benchmark.
- `docs/engine_design.md` §3, §7.8, §7.9 (`TickPolicy`), §7.10, §8.2, §8.4 (cell rings), §9.2-§9.6, §16.2 (tools
  moved to S10), the S10 and S14 rows; `AGENTS.md`.

## Out of scope

Saving `Behaviour`, `Persistent` and world state (S11), `ScenarioWorld` and `run_scenario` (S12), real input and the
app loop (S13), app tools (S14), messages (S15), replication and real `ownedBy` (S16), terrain height (S20), more
spaces (S21), unloading and `onCatchUp` calls (S25), Jolt characters (S29).

## Design decisions (planning)

- Ticking by grid-cell rings (32 m) instead of distances.
- The MCP entity tools move from S14 into S10, with the world API under them.
- Entities in tools: the object form or `"12:3"`; `apply` for batches; `changes_since` now; a read-only mode; paging
  with `limit` (<= 1000) and `after`; edits are logged, no replay script yet.
- Profiling (section 9) with its tools now; the ImGui window with the app (S13).

## Done when

All tests pass in CI on GCC, Clang and MSVC; every tool has a `handleMessage` test; the benchmark prints its numbers;
the design describes the implemented API.

## Verification

The CI recipe of AGENTS.md, `ctest -R engine`, full ctest; AddressSanitizer + UBSan and ThreadSanitizer builds of the
new tests.

## Design changes

- §8.4: ticking by grid-cell rings (`TickPolicy {ringSize, nearRing, midRing, midInterval, farInterval}`) instead of
  distances with a round-robin cursor; `tickSystem` re-evaluates everything when the player cells change, else the
  moved objects; a class overrides rings and intervals, not the cell size.
- §16.2: the world tools move from S14 to S10, on `WorldApi`; new tools `describe_world`, `world_status`,
  `add_component`, `remove_component`, `set_transform`, `get_global`, `set_global`, `emit_event`, `queue_action`,
  `changes_since`, `get_profile`, `reset_profile`, `apply`; S14 keeps the app tools. §17.1: the profiler.
- §7.8: the state entities exist only when components are registered for them. §7.9: `WorldSetup::classes`,
  `profiling`. §7.10: `queueAction`, `profiler()`. §9.2, §9.3, §9.6: implementation notes (spawn as a command,
  `onDestroy` before any removal, `Behaviour`/`Persistent` not addable with `add`, the Persistent index kept in
  add/remove/destroy rather than from the structural log, `move` shortened to length 1, idle characters not written).
- `World::addStepListener` (after every completed step; the API's change journal and `measure_performance`), type-erased
  `Registry::findComponent/addDefault/removeById`, `PoolBase` JSON virtuals, `EventQueues::find` and `emitJson`.
- Review round 1 (two reviewers: gameplay core; API and tools): `makePersistent` idempotent within a phase and for
  reserved entities; `onDestroy` collected again until no new hook runs (an entity attached to a doomed parent in a
  hook no longer survives with a dead parent); world-state entities refuse `Parent` and the removal of their components
  in every build; operations of hooks during a flush on entities reserved later in the phase follow the activation;
  `Behaviour::tickOffset` staggers the first tick (update got the stagger as elapsed time); game objects stay alive
  while their hooks run (retired, released when no hook runs); a throwing `onDestroy` aborts the destroy cleanly;
  `Persistent` read-only through `get` and views; the step time of an aborted step is discarded (timed by hand); the
  `structural_records` counter counts this step's records; the action hash uses the queue order. API: unknown
  arguments and wrong-typed numbers are errors; `changes_since` with an opaque cursor (`seq@step`; a foreign one is an
  error), destroyed entities fold their components, moves coalesced per entity and opt-in (`include_moved`; tracked in
  the World's flag shift, `World::setTrackStepChanges`); `set_transform` refused for children; Behaviour edits derive
  the interval again and check the times; `step` and the paused measurement have a wall-clock budget;
  `measure_performance` at most 600 steps and refused at time scale 0; `apply` allowed read-only for reads; `logs`
  reports `first` and `truncated`; `get_profile` filters scopes; the API detaches when the world is destroyed
  (`World::addDestroyListener`); JSON: the 2^53 boundary in the integer domain, float range, unknown fields list the
  valid ones; tool results are compact JSON with exact numbers; the entity schema uses a type list instead of `oneOf`.
  Not changed: `apply` is not atomic (documented); actions have no name registry yet (S13 input map); the spawn
  initial-component vs `onSpawn` add convention is documented (§9.2).
- Review round 2 (robustness; agent experience by driving the tools as an agent): `add<Parent>` during a flush waits for
  a parent reserved later in the phase; one `onDestroy` pass, then entities attached by hooks are destroyed without
  their hooks (a respawning child looped forever), released survivors get their flag reset; nested destroys from
  hooks are queued (a 50000-link chain overflowed the stack); a failed spawn (exception in configure/onSpawn/initial
  adds) destroys the half-started entity; double deferred adds/removes are refused in every build (out-of-bounds write
  without asserts); a throwing step listener keeps the others; `queueAction`/`makePersistent` refuse dead entities in
  every build; step-change tracking is reference counted and the API removes its destroy listener;
  `measure_performance` answers when the world is destroyed while it waits. API: the tool schemas list exactly the
  operation arguments (test) and `apply` stays in read-only mode; moved entities are paged by entity index in the
  cursor (moved_more returned the same page forever); the cursor carries a random session epoch and accepts "now";
  `changes_since` has a `limit` (default 200); writes from the API must be finite and quaternions are normalized;
  `spawn` applies initial components after `onSpawn` (merging into components the object added: an agent could
  trigger the double-add assert) and takes rotation and scale; `list_entities` returns `include`d values, `distance`
  and `grid_step`; `set_component`/`set_global` return `previous`; writing `Behaviour.tickInterval` is an error; edits
  are logged after they ran with the outcome; `time_scale` at most 1000; integers printed in full; profile times
  rounded with a `top` list. Not changed: `step` while running (an agent pauses first), `measure_performance` still
  answers only from steps (it answers when the world is destroyed or at the deadline), no action-name registry yet.
- Review round 3 (portability, regressions of round 2, a build without asserts, docs): the API's `spawn` applies
  rotation and scale only when given (it reset a class's own) and reports initial components it could not apply any
  more (`skipped`; an initial `Parent` whose entity `onSpawn` destroyed asserted); a failed spawn inside `onDestroy` is
  destroyed at once (the queue was dropped by the propagating exception); a deferred `add<Parent>` to a dead parent is
  refused at record time in every build; unit quaternions stay bit-exact through JSON (normalized only when not unit,
  zero refused only for API writes); the moved list is not repeated while structural changes are paged;
  `std::random_device` failures fall back to the clock; includes, docs (`tickOffset`, the two spawn orders). Built and
  tested without asserts (Release, `SAIGA_ASSERTS=OFF`) by the reviewer: passes.
- CodeRabbit (PR #52): near results report the grid position and distance the selection used; moves are recorded with
  their handle (a reused slot is not reported); the move window is fixed while it is paged (moves behind the page
  index or after the window come in the next one) and independent of structural paging (cursor
  `epoch.seq.from.to.index`); `previous` holds only writable fields (an undo payload); a hook exception during an
  operation becomes an error result (apply reports the earlier results); quaternion lengths without overflow; tool
  numbers locale-independent (`std::to_chars`); `measure_performance` fails on a profile reset and answers at its
  deadline from `McpServer::addPollListener` too; `removeByIdNow(Behaviour)` retires the object; a failed spawn removes
  its subtree without hooks; step listeners all see a committed step before the first exception is rethrown;
  `alwaysTickEveryStep` is not folded into `tickInterval`.
- The test event `DamageEvent` and the test component `Health` of `test_engine_world.cpp` and the ECS benchmark are
  renamed (`HitEvent`, `Vitality`): the engine registers those names now.

## Results

- `tests/test_engine_gameobject.cpp`: 32 tests, `tests/test_engine_world_api.cpp`: 29 tests (GCC 15); the S07-S09
  tests unchanged. Full ctest (CI options, GCC 15): 57/57.
- Benchmark (GCC 15, RelWithDebInfo, laptop): 10^4 game objects with one player: `World::step` 0.09 ms standing (411
  updates per step), 0.21 ms when every object is re-tiered each step. The S08 benchmark with 10^5 moving entities:
  1.9 -> 2.3 ms (R-TEST-2: three more engine systems run their (empty) views and the tick system scans the changed list;
  idle 0.51 ms, unchanged).
- AddressSanitizer + UBSan: the game object, world API and world tests are clean. ThreadSanitizer reports races only inside
  `McpServer`'s handoff between `poll` and `handleMessage` in the uninstrumented `saiga_core` (its synchronization is
  not visible to TSan; the world is only touched by the polling thread while the caller blocks).

## Files

`src/saiga/engine/gameplay/*`, `src/saiga/engine/api/WorldApi.*`, `src/saiga/engine/tools/WorldTools.*`,
`src/saiga/engine/core/Profiler.*`, `src/saiga/engine/ecs/{VisitJson.h,ComponentPool.h,Registry.h,Registry.cpp,Ids.h}`,
`src/saiga/engine/components/{Player,Health,PlayerInput,CharacterIntent,Persistent}.h`, `EngineIds.h`,
`EngineComponents.cpp`, `src/saiga/engine/world/{World,Events,EventQueues,SpatialGridSystem,TransformSystem}.*`,
`tests/test_engine_gameobject.cpp`, `tests/test_engine_world_api.cpp`, `tests/mcp_test_helper.h`,
`tests/test_core_mcp_server.cpp`, `tests/test_engine_world.cpp`, `tests/engine_world_second_tu.cpp`,
`tests/test_engine_ecs_benchmark.cpp`, `tests/CMakeLists.txt`, `docs/engine_design.md`, `AGENTS.md`, this file.
