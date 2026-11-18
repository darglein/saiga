# S08: World and step loop

Status: done.

Sections follow the stage plan template of S05.

## Goal

The engine has a `World` (docs/engine_design.md §7.6-§7.10, §8.1-§8.3, §8.6) that owns the S07 `Registry` and runs
fixed steps through the eight phases of §8.1: systems registered per phase, structural changes during a step as
commands applied after each phase, the structural log for later consumers (render sync, physics, network), events
readable one step later, the `Transform` hierarchy, `advance`/`step`, `world.random()`, `parallelEach` and
`world.hash()`. Game objects (S10), saves (S11) and scenarios (S12) build on it.

## Depends on

S07 (and S06: `JobPool`, `Rng`, `Hash`, logging).

## Rules that apply

- R-SYS-1 (systems are free functions registered with phase, order and `RunOn`), R-SYS-2 (structural changes during a
  step are commands; events only for broadcasts), R-STATE-3 (`Transform` only through its setters), R-PAR-1
  (`parallelEach` writes only its own entity; no structural changes, commands, events or RNG calls inside, asserted),
  R-DET-1 (`world.random()` and the step `dt`), R-REG-1/2, R-FILE-1, R-DOC-1/2, R-TEST-1, R-TEST-2 (ECS and step
  benchmarks).

## Scope

`src/saiga/engine/ecs/` (changes to S07, from the S07 "Notes for S08"):

- `Registry`: slots and `Transform`s in stable pages (4096 per page, `PagedArray<T>`), so reserving an entity never
  moves an existing `Transform` or slot while a view runs; `reserve()` (a `Reserved` slot, skipped by views, may be
  called during a view) and `activate(e)`; the iteration guard is atomic (nested views inside `parallelEach`);
  `PoolBase::appendBytes(index, out)` (type-erased binary form for the hash and, in S11, saves); `hashInto(Hasher&)
  const` (index order, saved components in ID order, then the free list); the default flag `Simulated` on every entity
  (the tag of §12.3; streaming clears it from S24); a `JobPool*` for `parallelEach`.
- `View`: a required-flags filter (`Simulated` in phases 2-5) and `parallelEach(fn, chunk)` on the registry's job pool
  (fixed chunks over the dense array of the smallest pool or the slots; asserts that the body makes no structural
  change, command, event or RNG call).

`src/saiga/engine/world/` (namespace `Saiga::Engine`):

- `Phase.h`: `enum class Phase { Input = 1, Events, Behaviour, Gameplay, Physics, Transform, Streaming, Extract }`
  (§8.1), `Phase::None` between steps, `kFirstPhase`/`kLastPhase`.
- `SystemList.h/.cpp`: `using SystemFn = void (*)(World&, double dt)`; `add(Phase, const char* name, SystemFn,
  RunOn = RunOn::Host)`; systems run per phase in registration order (engine list first). Names are unique.
- `EventRegistry.h`, `EventQueues.h/.cpp`: events with `name`, `id` (`EventId`, `EngineEvent` 0..63, game 64..),
  `visit`; `EventRegistry::add<E>(Replicate = Never)` (unique ID and name, contiguous IDs checked like components);
  `world.events.emit(e)` appends to the pending queue; at the start of each step the pending events become readable
  (`read<E>()`, emit order) and the old readable ones are dropped: events emitted in step k or between steps k and
  k+1 are readable in all phases of step k+1, then cleared. Pending events are part of the hash.
- `Events.h`: `enum class EngineEvent : uint16_t { Count }` (grows with S10).
- `StructuralLog.h/.cpp`: records `{Entity entity; ComponentId component; bool added}` for every applied structural
  change, also between steps (create adds a `Transform` record, destroy one removed record per component incl.
  `Transform`, in ID order); consumers register (`addConsumer()` returns an index), read from their cursor
  (`consume(consumer, fn)`); at the end of every step the log is truncated up to the smallest cursor (everything if
  there are no consumers).
- `Commands.h` (engine-internal): during a step `create` reserves and returns the entity; `add` (on alive or
  reserved entities), `remove` and `destroy` are recorded (the component value is moved into the command) and applied
  at the end of the phase in recording order; a reserved entity is activated (gets its `Transform`, the log record) when
  its create command is applied. `get`/`read`/`has` on a reserved entity assert until then.
- `World.h/.cpp` (`SAIGA_ENGINE_API`): `WorldSetup` (`seed`, `workerThreads`, `WorldConfig {stepsPerSecond = 60,
  maxStepsPerAdvance = 4}`, `components`, `events`, `systems`), `explicit World(const WorldSetup&)` (engine lists
  first, then the game's, then `validateRegistration` for components and events); the §7.10 subset: `create`,
  `destroy`, `alive`, `add`, `remove`, `get`, `tryGet`, `read`, `tryRead`, `has`, `view` (only `Simulated` entities in
  phases 2-5), `viewAll`, `mask`, `events`, `random()`, `time()`, `stepIndex()`, `dt()`, `alpha()`, `advance(frameDt)`
  (accumulator x time scale, at most `maxStepsPerAdvance` steps, the backlog beyond is dropped and counted:
  `droppedSteps()`), `step(n)`, `setPaused`, `setTimeScale`, `setWorkerThreads`, `hash()`; engine-internal:
  `registry()`, `structuralLog()`, `changedTransforms()`, `currentPhase()`.
- Children of `Transform` (§7.4): engine components `Parent {Entity entity}` and `LocalTransform {position, rotation,
  scale}` (`EngineComponent::Parent`, `LocalTransform`), `registerEngineComponents(Registry&)`; the engine system
  `transformSystem` (phase 6): children in a depth-sorted list (rebuilt when `Parent` is added or removed) get their
  pose from the parent and `LocalTransform` through the setters (a teleported parent teleports its children), then the
  list of changed entity indices (`changedThisStep` or `changedLastStep`, sorted) is built for the grid and extract;
  destroying a parent destroys its descendants. At the end of each step the flags shift (`changedLastStep =
  changedThisStep`, `changedThisStep = teleported = 0`). `registerEngineSystems(SystemList&)`.
- `RunOn`: before S16 the world is the host: `Host` and `Both` systems run, `Client` systems do not.
- `world.hash()` (§17.4): FNV-1a 64 over the saved components (`Saved::Yes`) of the alive entities in index order (with
  index and generation), the free list, the RNG state, the pending events and the step index.

Tests and docs:

- `tests/test_engine_world.cpp`; `tests/test_engine_ecs_benchmark.cpp` gains the step time with 10^5 idle entities.
- `docs/engine_design.md` §3, §7.6-§7.10, §8.1-§8.3, §8.6, §17.4, the S08 row; `AGENTS.md`.

## Out of scope

- Game objects, `Behaviour`, ticking, `spawn`, world-state entities, `global`/`spaceState`, `Persistent` (S10); saves,
  `dumpJson`, `diff` (S11); `ScenarioWorld` (S12); `EngineApp` (S13); the spatial grid (S09); `moveToSpace` (S21);
  replication, `ownedBy` filtering, `stepNetworkOnly` (S16); the render sink (S19).

## Design notes

- Stable storage instead of reserving ahead: a `Transform&` handed to a view callback must survive a `create()` in the
  same callback (it reserves a slot); pages of 4096 never move.
- The command buffer stores type-erased closures (`std::unique_ptr<Command>`, `ClosureCommand<F>`, so a move-only
  component can be moved into it). The order of application is the recording order; changes made while a flush runs
  apply immediately (nothing is recorded during a flush).
- The structural log is written by `World` around the `Registry` calls (the `Registry` stays a plain store).
- The hash walks the slots in index order (dense pool order depends on the history) and appends each saved component's
  bytes through `PoolBase::appendBytes`; free-list order is included because it decides future entity indices.
- `parallelEach` runs on the world's frame `JobPool` (`WorldSetup::workerThreads`, default 1 = only the caller); the
  chunk boundaries are fixed, so the result of a body that only writes its own entity does not depend on the thread
  count.

## Tests

`tests/test_engine_world.cpp`:

- Registration: engine components and systems first; contiguous IDs for components and events; unique system names.
- Phases: systems run in phase order, registration order inside a phase; `currentPhase()`; `RunOn::Client` systems do
  not run on the host.
- Commands: create during a step reserves (views skip it, `get` asserts), is applied at the end of the phase and
  visible in the next phase; destroy and remove during a view are deferred; add on a reserved entity; recording order;
  between steps changes apply immediately.
- Stable references: a `Transform&` from a view stays valid while the callback creates entities (stable pages).
- Structural log: records for create, add, remove, destroy (one per component, incl. `Transform`), also between steps;
  two consumers with different cursors; truncation at the end of the step.
- Events: emitted in step k readable in step k+1 only; emitted between steps readable in the next step; emit order;
  unregistered event asserts.
- Hierarchy: a child follows its parent (translation, rotation, scale), grandchildren in depth order, a teleported
  parent teleports the children, destroying a parent destroys its descendants, unchanged parents cause no flags; the
  changed list; the flag shift at the end of the step.
- Time: `advance` runs whole steps (at most 4), drops the backlog and counts it, `alpha()` in [0, 1); pause, time scale;
  `step(n)`.
- `parallelEach`: every entity visited once for several worker counts and chunk sizes; the same result as `each`;
  death tests for a command, an event, `random()` and a structural change inside the body.
- `Simulated`: in phases 2-5 `view` skips entities without the flag, `viewAll` does not.
- Hash: two worlds with the same seed and history have equal hashes after every step; a changed field, a changed RNG
  state, a pending event or a different free-list order change it; unsaved components (`Saved::No`) do not.

Benchmark (non-gating): `World::step` with 10^5 idle entities (no systems besides the engine ones) and with 10^5
entities moved by a system.

## Done when

All tests pass in CI on GCC, Clang and MSVC; the benchmark prints its numbers; the design describes the implemented
API.

## Verification

- The CI recipe of AGENTS.md, `ctest -R "engine|compile_fail"`, full ctest.
- AddressSanitizer + UBSan and ThreadSanitizer builds of the world tests (standalone compile, as in S06/S07).

## Design changes

- §7.6: the S08 implementation (stable pages for slots and `Transform`s, reserve during a view, commands as
  type-erased closures, commands on entities destroyed earlier in the flush are dropped, the structural log records
  including `Transform` on create and destroy, consumer API).
- §7.7: events need `version` and noexcept moves (like components); registration checks.
- §7.5: `parallelEach(fn, chunk)` details, the `Simulated` required flag.
- §7.9: `WorldSetup::workerThreads` counts the calling thread (1 = serial).
- §8.3: `droppedSteps()`, `step(n)` also when paused. §17.4: the exact hash content incl. the free list.
- `RunOn::Client` systems do not run before S16 (the world is the host).
- §3: a row for the world; the S08 row is done.
- Review round 1: `add<Parent>` rejects cycles (asserts; refused without asserts) and a dead parent (asserts between
  steps; during a flush the child is destroyed, like a child of a destroyed parent); `Parent` is read-only through
  `World::get`/`tryGet` (`static_assert`; change it with remove + add); the destroy cascade uses one sorted
  parent-to-children list and a visited set; a throwing system or command aborts the step (pending commands dropped,
  reserved entities released, the step index does not advance, `hash()` asserts while commands are pending); teleports
  made in phases 7 and 8 reach the children at the end of the step; `add`/`remove` during a step check the entity and
  the component when recorded; `StructuralLog::removeConsumer`; `advance` ignores NaN, treats +inf as a long frame and
  counts only the discarded whole steps. §7.4, §7.6 and §8.3 updated.
- Review round 2: late teleports are propagated before extract (phase 7 teleports) and through a new
  `teleportedLastStep` flag in the next phase 6 (phase 8 teleports), also when the hierarchy changed in the same phase;
  `destroy` skips the cascade for entities without children (child counts per slot; it sorted all `Parent` links per
  destroy, 0.33 ms with 20,000 links); `Parent` only as `const Parent` in World views; an aborted step puts its readable
  events back (the repeated step sees them); `add<Parent>` during a step asserts on a dead parent when recorded; the
  time scale must be finite; `destroy` re-checks each doomed entity (S10 `onDestroy`). §7.4 updated.
- Review round 3: whether an entity has the component is checked when a command is applied (the record-time check made
  remove + add of `Parent` inside a system assert); late teleports through a list of children whose parent was
  teleported after their last update (collected after phase 8, teleported in the next phase 6) instead of
  `teleportedLastStep`, and the pass before extract recomputes every child of a teleported parent (a second teleport in
  one step reached the child as a normal move); `add<Parent>` to a dead parent between steps is refused also without
  asserts; `rebuildHierarchyState()` recounts the children from the `Parent` pool (for S11 loading); a finite
  `stepsPerSecond`, the dropped-step count clamped; §7.4, §7.6, §7.10 (`mask` is static), §8.2 (the transform system),
  §8.3 updated.
- CodeRabbit (PR #49): every structural change reserves its log record before the change and appends it after
  (`StructuralLog::prepare`; the log is a vector), `destroy` snapshots, reserves and commits after `Registry::destroy`,
  `add<Parent>` allocates the parent's count slot first; a consumer's cursor moves after its callback returned (a
  throwing callback gets the record again) and `consume` stops when the callback removes the consumer; views keep their
  traversal state local (nested calls on workers); `release` asserts inside `parallelEach`; duplicate component and
  event IDs stop the program in every build, event access checks the type size; the depth and cycle walks are bounded by
  the slot count (a valid chain deeper than 1000 kept its order); `rebuildHierarchyState` validates before it replaces
  the counts; `abortStep` leaves the step first and cleans up best effort. §7.6 updated.
- CodeRabbit, second pass: `StructuralLog::prepare` counts outstanding reservations (a nested change cannot use the
  slot promised to the outer one); `consume` rejects reentry for the same consumer; `rebuildHierarchyState` returns
  false without asserting for stale parents and also rejects cycles; `abortStep` runs the event restore and each
  release independently.
- Event queues and component pools store `typeid(T)` and check it with `std::type_info` equality before every
  downcast (`isRegisteredType`, `ecs/TypeCheck.h`): two types with the same ID, name and size are told apart (the
  size check was not enough). Not the typeid name: types of the same name in the unnamed namespaces of two translation
  units have the same name on GCC and Clang (test `EventOfTheSameNameInAnotherTranslationUnit`, second translation
  unit `tests/engine_world_second_tu.cpp`). `type_info` equality works across DLLs (the standard libraries compare the
  mangled names), but Clang with libstdc++ also finds those two equal; names with `_GLOBAL__N_` (unnamed namespace on
  the Itanium ABI) therefore also need the same `type_info` object.
- `StructuralLog::prepare` returns a guard (`Prepared`) that appends the records and gives back the slots not appended
  (a mutation that failed after prepare); the count rises only after the reserve succeeded.
- `abortStep` does not allocate: the registry's free list is a vector FIFO whose capacity always covers the releases
  of all reserved slots (`Registry::reserve` makes room, `destroy` keeps it, so a destroy applied before the abort
  cannot use it up); the event restore is a swap (the readable events become pending again, the aborted step's own
  events are dropped like its commands).

## Results

- `tests/test_engine_world.cpp`: 46 tests pass (GCC 15), incl. 11 death tests; `tests/test_engine_ecs.cpp`: 32 tests
  (the reworked `Registry`/`View` keep the S07 behaviour; new entities are `Simulated`).
- Benchmark (GCC 15, RelWithDebInfo, laptop): `World::step` with 10^5 entities 0.48 ms idle (the phase-6 scan),
  0.97 ms with a system that moves all of them; the S07 ECS numbers are unchanged (create 14.4 ms, view<3> 0.61 ms).
- AddressSanitizer + UBSan and ThreadSanitizer (5 repeats, `parallelEach` with 1, 2 and 4 threads): no reports.
- Full ctest (CI options + samples, GCC 15): 54/54 pass.

## Files

`src/saiga/engine/ecs/{Registry.h,Registry.cpp,View.h,ComponentPool.h,PagedArray.h}`,
`src/saiga/engine/components/{EngineIds.h,Transform.h,Parent.h,LocalTransform.h,EngineComponents.h,
EngineComponents.cpp}`, `src/saiga/engine/world/{Phase.h,SystemList.h,SystemList.cpp,EventRegistry.h,EventQueues.h,
Events.h,StructuralLog.h,StructuralLog.cpp,Commands.h,World.h,World.cpp,TransformSystem.h,TransformSystem.cpp}`,
`tests/test_engine_world.cpp`, `tests/test_engine_ecs_benchmark.cpp`, `tests/CMakeLists.txt`, `docs/engine_design.md`,
`AGENTS.md`, this file.
