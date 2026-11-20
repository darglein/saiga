# S12: Scenario framework

Status: done.

Sections follow the stage plan template of S05.

## Goal

Engine gameplay is tested by scenarios: short scripted runs of a headless, reproducible world with typed helpers
(spawn, players, input, damage, stepping, checks), each run in three variants (plain, repeat, save/load) that catch
nondeterminism and state outside components (docs/engine_design.md §17.2, §17.4, R-STATE-1, R-DET-1). Engine scenarios
with test-only game objects run in ctest; the same runner serves the games' `--scenarios` (S13) and `run_scenario`
(S14). From this stage on, every engine gameplay mechanism has an engine scenario (R-TEST-1).

## Depends on

S10 (game objects, gameplay basics, `WorldApi`), S11 (`save`/`tryLoad`, `diffSaves`, `dumpJson`), S06 (`LogRing`).

## Rules that apply

R-TEST-1 (engine scenarios from S12), R-STATE-1 (checked by the save/load variant), R-DET-1 (checked by the repeat
variant), R-REG-1 (scenario lists), R-FILE-1, R-DOC-1/2.

## Scope

### 1. `ScenarioList` (`engine/tools/ScenarioList.h/.cpp`)

```cpp
using ScenarioFn = std::function<void(ScenarioWorld&)>;
class ScenarioList
{
   public:
    void add(const char* name, ScenarioFn run, bool localOnly = false);   // name: unique, snake_case (asserted)
    struct Entry { std::string name; ScenarioFn run; bool localOnly; };
    const std::vector<Entry>& entries() const;
    std::vector<const Entry*> match(const std::string& filter) const;   // substring; empty: all
};
```

`localOnly` is stored for S16 (no host + client variant).

### 2. `ScenarioWorld` (`engine/tools/ScenarioWorld.h/.cpp`)

```cpp
class ScenarioWorld
{
   public:
    ScenarioWorld(const WorldSetup& setup, const ScenarioOptions& options = {});
    World& world();       // the current World: call it each time, never keep the reference across a step helper
                          // (the save/load variant swaps in a loaded World)
    WorldApi& api();      // created on first use, rebound on the swap (WorldApi::rebind, S11): cursors, snapshots
                          // and pause state survive (scenarios can drive the world as agents do)

    template <class T, class... C> Entity spawn(const Vec3& position, C&&... initial);
    Entity spawnPlayer(const Vec3& position, uint8_t peer = 0);   // Transform, Player{peer, characterId 1, 2, ...},
                                                                  // PlayerInput, CharacterIntent (§17.2)
    template <class T, class... C> Entity spawnPlayer(const Vec3& position, uint8_t peer, C&&... initial);
                                                    // a game object player (S13b): Player and PlayerInput, no
                                                    // CharacterIntent (the game's controller moves it)
    void move(Entity player, const Vec3& direction);              // PlayerInput.move (length <= 1)
    void input(Entity player, uint32_t action, const Vec3& aim = Vec3::Zero());   // world.queueAction
    void damage(Entity target, double amount, Entity source = {});               // emits DamageEvent (the real path)
    void enableDistanceTicking();   // default: every object ticks every step (§8.4, §9.2)

    void step(int n = 1);
    void stepSeconds(double seconds);   // rounded to whole steps
    bool stepUntil(const std::function<bool()>& predicate, int maxSteps);   // false: not reached (no failure itself)

    bool check(bool condition, const std::string& message);   // records a failure (with the step), returns condition
    double distance(Entity a, Entity b);                        // Transform positions; a check fails for dead entities
    template <class T> size_t count();
    template <class T> void trace(Entity e);                    // records T's JSON every step (shown on failure)
    std::string dumpJson(Entity e);

    const std::vector<std::string>& failures() const;
};
struct ScenarioOptions
{
    uint64_t maxSteps = 60 * 600;     // 10 min at 60 Hz
    bool varyThreads = false;         // Repeat: second run with 1 thread if the setup has more, else 3
    bool strictContinuation = false;  // SaveLoad: the final hash must equal Plain's
};
```

- Headless and reproducible: the seed, worker thread count and step rate of the setup; `profiling` off (no clock
  calls); no MCP; flat terrain (seeded terrain from S20, `RecordingRenderSink` from S19, the radius overrides of §17.2
  from S24). Ticking: every object every step (`World::setTickEveryStep(true)`, a new World switch read by
  `tickSystem`: it overrides the world's and the classes' `tickPolicy()` and also applies without players) unless
  `enableDistanceTicking()` turns it off.
- Steps are observed through a World step listener (`addStepListener`), so `world().step()` and the API's `step` are
  covered too (hashes, traces, the step count). The save/load point is taken only inside the step helpers; a scenario
  that steps only outside them gets the round trip at its end.
- Exceeding `maxSteps` (or the runner stopping a rerun at a step) makes the step observation throw `ScenarioAbort`
  (it registers itself again, since the world removes a throwing listener); the world API turns it into an error
  result, so `ScenarioWorld::call` and the step helpers check the abort flag again; the runner catches it.
- Entities created through the helpers are "involved": a failure report dumps them (at most 10).

### 3. Runner and variants (`engine/tools/ScenarioRunner.h/.cpp`)

```cpp
enum class ScenarioVariant { Plain, Repeat, SaveLoad };
struct ScenarioResult
{
    std::string scenario; ScenarioVariant variant; bool passed;
    std::vector<std::string> failures;   // checks, exceptions, variant mismatches
    std::string report;                  // failures, log lines, dumps, traces, diff
    uint64_t steps; double seconds;
};
ScenarioResult runScenario(const WorldSetup& setup, const ScenarioList::Entry& entry, ScenarioVariant variant,
                           const ScenarioOptions& options = {});
std::vector<ScenarioResult> runScenarios(const WorldSetup& setup, const ScenarioList& list,
                                         const std::string& filter, const ScenarioOptions& options = {});
```

- **Plain**: runs the scenario once; passes when no check failed and nothing threw. The world's hash after the last step
  and after the scenario function returned are recorded.
- **Repeat** (§17.4 a): runs twice; the per-step `world.hash()`, the final hash and the step count must be equal. With
  `varyThreads` the second run uses another worker thread count (1 <-> 3; opt-in, used by the engine's own scenarios:
  the engine's `parallelEach` systems must not depend on it, while §8.6 and §20 do not require it of games). On a
  mismatch at step k, the runner runs both again up to step k, saves both and reports `diffSaves`.
- **SaveLoad** (§17.4 b): first runs Plain to count the steps N (so the variant runs the scenario twice), then runs it
  again; at the first step-helper boundary at or after N / 2 (inside a `step(n)` call: between its steps), the world is
  saved, a fresh `World` is constructed from the setup and loaded with `tryLoad` (every game object is recreated
  through its factory with `configure` + `onLoad`), and swapped in for the old one. The runner carries over what is not
  world state: paused, time scale, worker threads, the tick-every-step switch, its own step listener and the API
  (`rebind`). The loaded world's `save()` must give the same bytes (else the report shows `diffSaves`), `hookErrors`
  are failures, and the scenario continues and must pass its checks; with `strictContinuation` the final hash must
  also equal Plain's (state in members changes later behavior, and this catches it even when no check looks at it).
  Entity handles stay valid (indices and generations are kept). A scenario with N < 2 does the round trip at its end.
- A failure report has the failed checks with their steps, the log lines since the scenario started (`LogRing`, at
  most 200), `dumpJson` of the involved entities and the traces.
- An exception from the scenario or a game hook is a failure with its message; a `SAIGA_ASSERT` stops the process
  (death tests check that separately).

### 4. Engine scenarios (`tests/test_engine_scenarios.cpp`)

A value-parameterized gtest over the scenario list x the three variants (a name generator: `wanderer_flees_when_damaged_
Repeat`, ...), with `varyThreads` and `strictContinuation` (the engine's systems continue identically after a load;
games need not, §8.6). Test-only game objects and systems are defined in the test file (R-TEST-1):

- `Wanderer` (state in `WandererState {Entity threat; double fleeSpeed}`, `Health`, `CharacterIntent`): `onDamage`
  lowers `Health.hp` and stores the source as the threat; `update` sets `desiredVelocity` away from a live threat.
- Scenarios for the existing mechanisms:
  - `wanderer_flees_when_damaged` (§17.2);
  - `player_walks_with_input` (`move` -> `playerControlSystem` -> `characterMoveSystem`, the distance after 2 s);
  - `action_reaches_listener` (`input` -> `ActionEvent` in the next step, exactly once, in queue order);
  - `far_objects_tick_slower` (`enableDistanceTicking`; update counts near / mid / far from a player match
    `TickPolicy`);
  - `destroy_runs_on_destroy_children_first`;
  - `persistent_reference_survives` (a `PersistentId` field resolves after the save/load point);
  - `api_edits_reach_gameplay` (`api().call("set_component", ...)` on `Health`, `emit_event` of a `DamageEvent`: the
    agent path behaves like the C++ path).

### 5. Runner tests (`tests/test_engine_scenarios.cpp`, a separate suite with its own lists)

- A failing check: the result lists it with its step, the log lines and the dump of the involved entities.
- A nondeterministic scenario (a game object reading a process-global counter) fails Repeat with a `diffSaves` naming
  the component and field.
- A game object with state in a member (not in components) fails SaveLoad (R-STATE-1 is enforced), by a check and,
  without one, by `strictContinuation`.
- SaveLoad keeps the API: a cursor, a snapshot and `pause` from before the swap work after it.
- A system whose result depends on the thread count fails Repeat with `varyThreads` only.
- A scenario that throws, one that exceeds `maxSteps` (also when stepping through `api().call("step")`), one that
  steps through `world().step()`, a 0-step scenario, `match` filters, duplicate names (death test).
- `setTickEveryStep` with a class that overrides `tickPolicy()` and without players.

## Out of scope

The host + client variant (S16), `ScenarioWorld` with streaming and terrain (S20, S24), `RecordingRenderSink` (S19),
`--scenarios` (S13), `run_scenario` and live scenarios over MCP (S14), scenarios written as JSON by agents (a possible
S14 addition: `apply` operations plus checks).

## Design notes

- SaveLoad loads into a fresh `World` (as §17.4 says) rather than in place: engine state that a load forgets to reset
  or rebuild would survive an in-place load and hide the bug. The price is that scenario code must not keep `World&`
  across step helpers (`world()` each time); entity handles stay valid.
- The step listener, not the helpers, observes steps, so no stepping path escapes the repeat checks.
- On a Repeat mismatch the runner reruns to the differing step instead of keeping a save per step (cheap normal runs;
  the rerun is deterministic for the first run's side, and the report says if the rerun diverges elsewhere).
- The helpers are typed C++ (fast, checked at compile time); `api()` gives the JSON path for scenarios that test the
  API or act exactly like an agent (§16.2: "S12's scenario helpers ... use the same API" becomes "can use").

## Tests

Sections 4 and 5; all scenarios x variants pass, the runner tests detect each failure kind.

## Done when

- The S12 row of §22: `wanderer_flees_when_damaged` passes incl. the repeat and save/load variants (CI).
- §8.4/§9.2's `enableDistanceTicking` and §17.4's variant rules (thread variation, strict continuation) documented.
- The engine scenarios of section 4 and the runner tests of section 5 pass on GCC, Clang and MSVC.
- `docs/engine_design.md` (§3, §16.2, §17.2, §17.4, §18.3 cookbook, the S12 row) and `AGENTS.md` updated.

## Verification

The CI recipe of AGENTS.md, `ctest -R engine`, the full `ctest`; a ThreadSanitizer build of
`test_engine_scenarios` (Repeat with `varyThreads` runs 3 worker threads).

## Files

`src/saiga/engine/tools/{ScenarioList.h,ScenarioList.cpp,ScenarioWorld.h,ScenarioWorld.cpp,ScenarioRunner.h,
ScenarioRunner.cpp}`, `src/saiga/engine/world/World.{h,cpp}` and `src/saiga/engine/gameplay/TickSystem.cpp`
(`setTickEveryStep`), `tests/test_engine_scenarios.cpp`, `tests/CMakeLists.txt`, `docs/engine_design.md`, `AGENTS.md`,
this file.

## Design changes

- `ScenarioWorld::call(op, args)`: an API call that also ends the scenario after `maxSteps` (the world API catches
  every exception, so the abort flag is checked again after the call).
- `spawnPlayer` numbers the character IDs 1, 2, ... instead of drawing them from `world.random()` (§9.3 updated): the
  RNG stream of a scenario is not changed by the helper.
- `World::setTickEveryStep` (not saved): the scenario runner carries it to the replacement world.

- Review round 1: the swap happens only between step helpers at depth 0 (a step listener that stepped replaced the
  World while it ran its listeners: a crash) and replaced worlds live until the scenario ends (a kept `World&` stays
  valid); `ScenarioWorld::addStepListener` re-attaches listeners after the swap and a world with other listeners fails
  the swap loudly (they were lost silently); the advance accumulator, the worker thread count and the data store
  (`WorldSetup::sharedData`) carry over; `count<Transform>` excludes the state entities; `ScenarioList::find` (exact
  lookup); the scenario's world API has no wall-clock budget (machine speed must not change a scenario); the report
  lists each failure once, then the context (log, dumps, traces; the second run's in Repeat); strict continuation and
  final-hash mismatches show `diffSaves`. Not changed: per-step hashes in Repeat (fine for engine scenarios; game
  scenarios may need sampling later), `enableDistanceTicking` takes effect from the next tick evaluation.

- Review round 2: stepping inside a step listener is a scenario failure (and `World::step`/`advance` assert inside
  one): the world's listener dispatch is not re-entrant, a nested step skipped listeners; the step observation throws
  `ScenarioAbort` past `maxSteps`, so loops over `world().step()` end too; the tick setting carries over from the
  world; the final save of a run is guarded; the thread-dependence test waits for a worker instead of a fixed delay.

- Review round 3: `World::step`/`advance` inside a step listener are refused in every build (an exception; they
  corrupted the listener dispatch without asserts), the world API refuses edits inside a step listener (a `step` there
  aborted the process); the step observation re-registers itself after it threw (later steps were neither counted nor
  stopped); `ScenarioList::add` refuses bad names in every build (R-REG-1).

- CodeRabbit: a failing baseline run fails the save/load variant (it was dropped); `call` swaps the world like a step
  helper, and a scenario that steps after the middle only through `world()`/`api()` fails the variant (the round trip
  ran only at the end, nothing ran on the loaded world; also when one `call` ran every step after the middle); a listener's callable is shared by its registrations (its
  captured state survives the swap); `saveLoad` inside a step listener is a failure; §17.2 says that `maxSteps` ends a
  loop of `api().call("step")` only through `call`.

Updated: §3, §9.3 (`characterId` in scenarios), §16.2, §17.2, §17.4, §18.3, §20, the S12 row; AGENTS.md.

## Results

- `test_engine_scenarios`: 8 engine scenarios x 3 variants (with `varyThreads` and `strictContinuation`) and 15 runner
  tests pass (42 tests); 62 ctest tests.
