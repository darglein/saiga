# S16b: Co-op tech demos

Status: done.

Sections follow the stage plan template of S05.

## Goal

The three tech demos run as co-op over loopback in one process: `--coop` starts a hidden host world and a client world
in the app, the client's player gets a plane, car or character next to the host's AI one, both move, the client renders
remote entities from `RenderPose`, and every demo scenario passes the S16 host + client variants (or is local-only
with a reason), including at least one scenario per demo with a client-owned driver or character. Split from S16 in
planning.

## Depends on

S16 (roles, ownership, replication, `RenderPose`, the host + client variant, `networkVariants`), S15 (loopback), S14
(app tools), S13b (the demos).

## Rules that apply

R-NET-1, R-NET-2 (the demos' systems and GUI write only what the local peer owns), R-STATE-1, R-SAVE-1 (the demo
components' fields move: versions bumped, no migration since no demo saves exist outside tests, stated in the PR), R-MECH-1 (who gets what
in co-op is game policy), R-DET-1, R-TEST-1 (game scenarios through `--scenarios`), R-DOC-1/2, R-FILE-1.

## Scope

### 1. Ownership and replication of the demos (§13.3)

§13.3 keeps vehicles host-owned: a driving client sends its input and the host's vehicle system moves the vehicle (one
round trip of steering latency, §20). The demos follow it with driver entities.

- `Driver {Entity vehicle}` (`samples/games/common/`, `Replicate::Host`, never owner-written) on the driver entity, and
  `VehicleSeat {Entity driver; int slot}` (`Host`) on the vehicle, kept by the host (the reverse link the vehicle systems read).
  AI vehicles carry `Driver{self}` and `VehicleSeat{self}`. A vehicle without a live driver gets neutral controls (it
  coasts and lands or stops; S16b's demos respawn it as today).
- `spawnDriven<T>(World&, uint8_t peer, int slot) -> {Entity driver, Entity vehicle}` (common): creates the driver
  entity first (`Player`, `PlayerInput`, the game's control component), spawns the vehicle `T` (host-owned) with the
  initial component `VehicleSeat{driver, slot}`, then writes `Driver.vehicle` and calls `setOwner(driver, peer)`. The
  slot is a grid offset that `respawn` keeps (the platformer offsets the hero's start the same way). `CarBase` /
  `AircraftBase::onSpawn` add `Driver{self}`, `VehicleSeat{self}` and the controls only when no `VehicleSeat` was passed
  (AI vehicles, and a vehicle spawned as the player: the regression scenario `checkpoint_out_of_range`).
- One layout everywhere: the single-player start hook also calls `spawnDriven(startWorld(), 0, 0)` and
  `setLocalPlayer(driver)`, so the shipped single-player game, the co-op game and the scenarios use driver + vehicle.
- `ScenarioWorld::defaultPeer()` (added here to S16's helpers): the peer `spawnPlayer` would use; the scenarios pass it
  to `spawnDriven`.
- Replication modes of every demo component:

  | Demo | Component | Mode | Writer |
  |---|---|---|---|
  | flight | `AircraftControls` (sticks, `throttleChange`; on the driver) | `OwnerAuthoritative` | the driver's owner (`applyInput`, the owner-side demo pilot: `throttleChange = clamp((target - AircraftState.throttle) * k)` from the replicated lever) |
  | flight | `AircraftState` (velocity, rates, crash; the throttle lever `throttle` moves here from the controls) | `Host` | `flightModelSystem` (`RunOn::Host`) integrates the lever from the driver's `throttleChange`; the host `Autopilot` (its own driver) writes the lever directly; `FlightModel::step(controls, throttle, dt)` takes the lever separately |
  | racing | `VehicleControls` (on the driver) | `OwnerAuthoritative` | the driver's owner |
  | racing | `VehicleState`, `LapState` | `Host` | `vehicleSystem`, `lapTimerSystem` (`RunOn::Host`, from the host's own simulation) |
  | racing | `DriverState` (AI progress) | `Never` | the AI on the host |
  | platformer | `CharacterMotion`, `Transform` of the hero | `OwnerAuthoritative` | `characterControllerSystem` (`RunOn::Both` over `ownedBy(localPeer)`) |
  | platformer | `RunState` | `Host` | the hero's game object |
  | all | `Driver`, `VehicleSeat` | `Host` | the host |

  The flight and racing vehicle systems iterate vehicles and read the controls of `VehicleSeat.driver`.
- Actions: `AircraftBase::update` / `CarBase::update` map an action's player (the driver) through `Driver.vehicle` to
  the vehicle. The GUI buttons (respawn, back to the track, restart) queue the action on the local player
  (`queueAction`) instead of calling the game object; on the host with a focused entity that has no local player
  (`--demo`) they call the game object directly as today. A queued action acts only when steps run (not while
  paused; the button says so).
- Platformer restart and goal: the host part (`RunState`, `ownerTeleport` of the hero) and the owner part (reset
  `CharacterMotion`, `jumpWasHeld` from the local `PlayerInput`) are split; the host sends `RestartRun` (reliable,
  `Control`) before the `OwnerCommand` teleport, both on `Control`, so the owner applies both in phase 1 of the same
  step. The host decides the kill height and the goal from the received owner pose (extrapolated up to 6 steps: the
  goal time of a client-owned hero has that error; `demo_reaches_goal` stays host-owned).
- Respawn of a host-owned vehicle stays a host `teleport` (S16's snap counter).
- The driver entity's `Transform`: an owner-side system (`RunOn::Both` over `ownedBy(localPeer)`, phase 4 after the
  vehicle system) copies the vehicle's pose into it: `Transform` on the host (the host's own driver of a host-owned
  vehicle has no `RenderPose`), `RenderPose` on clients; tools, the grid and S32's interest see the player near its
  vehicle (a plain `setPose`; owner teleports are not snaps in S16, nothing renders the driver).

### 2. `--coop` (engine: `EngineMain`, `EngineApp`)

- `--coop [delay]` (delay 0..30, default 2; an argument that is not an integer is not consumed): the app creates a
  hidden host world and its client world over a loopback pair; `app.world()` is the client world (rendered, input,
  MCP); `EngineApp::hostWorld()` (null without `--coop`), `EngineApp::startWorld()` (the host with `--coop`, else the
  world). The app owns the transports and destroys the worlds before them. `--coop` with `--scenarios` or
  `--benchmark` is an error.
- Stepping: one accumulator for both worlds; the app computes the frame's step count once (`World::stepsForAdvance(dt)`,
  a new hook that applies the time scale, the cap and the dropped-step rule without stepping) and interleaves host one
  step, client one step; `stepAll(n)` does the same for n steps (the `--demo` prewarm). So the client never gets
  several host steps in one round from frame batching, and one-step pulses (landings, jumps) survive.
  `app_status.coop` reports both step counts, the delay and the dropped steps.
- MCP in `--coop` (the world tools are bound to the client world; the "both worlds" operations go through
  `WorldToolsOptions` hooks the app sets): `pause`, `resume`, `time_scale`, `reload_data` act on both worlds; `step`
  and `measure_performance(steps)` while paused step through `stepAll`; the snapshot tools are refused (a client cannot
  save); `run_scenario` runs (its own worlds, borrowing the host world's `sharedDataStore()`; S14's `canSave()` check
  becomes "not inside a step or a hook" of the host world in co-op; the variant names include S16's three) but refuses
  `keep` and `load`; `measure_performance`, `get_profile` and `world_status` gain a `host` block in co-op (the client
  runs no vehicle, lap or behaviour systems; without it the numbers would hide the simulation's cost); `set_input` drives the client's player and
  counts client steps; the other tools read the client world (read-only for replicated entities, S16).
- The client's local player: when it is invalid on a client, the app looks for the entity with `Player.peer ==
  localPeer()` each frame (the player arrives 1 + delay steps after the host's join step) and clears it on `Resync`;
  `setLocalPlayer` with an entity of another peer asserts in `--coop`. The app's fallback uses `localPeer()`, not 0.
- Join policy: `EngineAppSetup::onPeerJoined(EngineApp&, World& host, uint8_t peer)` and `onPeerLeft`, called by the app
  from a host step listener for the peers of that step's `peersJoined()` / `peersLeft()` (S15: the lists are replaced at
  the next receive, so step listeners see them). Scenario worlds do not run it (scenarios spawn with `spawnDriven`).
  Each demo's policy: `spawnDriven` for the joined peer (with `DemoPilot`, below, when `--demo` is set); a leaving
  peer's driver is removed and its vehicle stays empty; the platformer's hero is destroyed (R-MECH-1; a common helper
  `removeDriver`).
- The games' `main.cpp`: the start hook spawns on `app.startWorld()` and calls `setLocalPlayer` only without
  `--coop`; the load listeners stay on `app.world()`; the host keeps its demo AI vehicle (autopilot, AI driver, demo
  runner) next to the client in `--coop` with and without `--demo`.
- `--coop --demo`: the join policy adds `DemoPilot {bool suspended; ...}` (`Replicate::Never`, saved; the pilot's small
  state: racing progress, platformer waypoint) to the driver; it reaches the owner through S16's presence list. An
  owner-side `RunOn::Client` system runs the existing `FlightAutopilot` / `DriverModel` / `PlatformerDemo` for drivers
  with `DemoPilot`, reading the replicated vehicle state; `applyInput` sets `suspended` while device input is active.
  The pilot state resets when the vehicle's `Transform::wasTeleported()` is set on the client (the snapshot apply uses
  `teleport()`) and on `RestartRun`. The platformer hero with `DemoPilot` restarts after the goal like the `DemoRunner`
  (the host part sees the component; `RestartRun` + `ownerTeleport`). The dead time is about 2 · (1 + delay) steps;
  `--coop --demo` clamps the delay to 8 with a message (the demo pilots' gains stay stable).

### 3. Rendering on the client

- The pose caches (`PoseHistory` and the racing wheels, platformer landings) read `RenderPose` for non-owned entities
  on clients (and take snaps from its teleport flag), `Transform` for owned and local entities.
- The renderers draw every vehicle or character (per-entity wheel caches, one animator and landing tracker per
  character) and follow the local player's vehicle (driver → `Driver.vehicle`); GUI labels say whose vehicle is shown.
- Latency seen by the client: about 1 + 2 · delay + `interpolationDelaySteps` (6) steps between a key and the vehicle's
  rendered reaction.
- Render-relevant state besides the pose (racing wheels; the platformer's velocity, grounded flag and landing speed)
  comes from snapshots without the interpolation delay; the render caches keep a per-entity ring of it, read
  `interpolationDelaySteps` client steps late and flushed on the `RenderPose` teleport flag, so wheels and animation
  match the drawn body approximately (exact matching needs the sampled host step, not exposed by S16; accepted).

### 4. Demo scenarios

`EngineAppSetup::networkVariants` is turned on for the three games (S16). Each scenario's change (in the PR):

| Demo | Scenario | Change |
|---|---|---|
| flight | `autopilot_flies_the_circuit` | none (host AI) |
| flight | `climbs_with_full_throttle` | `spawnDriven` with `w.defaultPeer()`; the lever written on the host-owned plane, the sticks through `edit` on the driver; `stepUntil` |
| flight | `crashes_into_terrain` | as above; the automatic respawn 2 s after the crash is checked as today (host-side); a second part sends `Respawn` on the driver and checks the driver → vehicle mapping with `stepUntil` |
| platformer | `jumps_onto_platform` | client-owned hero (`w.defaultPeer()`); the jump decided inside `edit` from the owner's own `Transform` (not the host's extrapolated pose) and the held bit written there |
| platformer | `demo_reaches_goal` | none (host-owned demo runner) |
| platformer | `held_jump_after_a_restart` | client-owned hero; the fall trigger an owner-side teleport through `edit`; the restart through `RestartRun` + `ownerTeleport`; the key release held 5 host steps (the async bound is ±2) so the owner sees it; `stepUntil` instead of the fixed `step(10)` for the jump checks |
| racing | `accelerates_on_track` | `spawnDriven`, controls through `edit`; the fixed `stepSeconds(3)` becomes a `stepUntil` on the progress window (the controls arrive up to 8 steps late) |
| racing | `ai_completes_a_lap` | none (host AI) |
| racing | `checkpoint_out_of_range` | the car spawned as the player is its own driver (`onSpawn` adds the links); the `LapState` edit stays on the host; unchanged otherwise |

Scenarios that still cannot run networked are registered local-only with the reason.

## Out of scope

Client-side prediction of vehicles (§20), a second process (S33), rendering the host's view, the camera rig (S35).

## Design notes

- Driver entities instead of client-owned vehicles keep §13.3 unchanged (no ownership transfer, vehicles always on the
  host) and match how S40 (moving platforms) and the game later handle vehicles.
- `--coop` is an engine feature (any game gets it) so agents and the developer can test co-op without a second process
  before S33.

## Tests

No ctest tests for the demos (R-TEST-1). `tests/test_engine_app.cpp` gets the engine parts with a test game: `--coop`
parsing (delay range, the errors with `--scenarios` / `--benchmark`); a `--coop --no-render` app: host and client
advance, `hostWorld()`, `startWorld()`, `stepAll` interleaving, the client's local player found after it arrives late
and cleared on `Resync`, `setLocalPlayer` of another peer's entity asserting, pause / step / time scale /
`measure_performance(steps)` on both worlds, `run_scenario` without and with `keep`, the snapshot tools refused,
interleaved stepping at a low frame rate (the client sees every host step), `app_status.coop`, `onPeerJoined` called
once per join. The leave path (`removeDriver`) is tested at world level in a demo-independent engine test (destroy the
client world, then its transport end; the host's `peersLeft()` and the helper). Locally: each demo's `--scenarios` with
the network variants; `SAIGA_SCREENSHOT=/tmp/<game>_coop.png timeout 60 ./game_<game> --coop --demo` shows both
vehicles; a short interactive `--coop` session per demo.

## Done when

- The S16b row of §22 (new; the demo part of S16's row): the tech demo co-op variants run host + client over loopback
  (local); the demo scenarios pass the host + client variants or are local-only with a reason.
- `docs/engine_design.md` updated: §8.3 (`--coop`, `hostWorld`, `startWorld`, `stepAll`, the client's local player),
  §16.1/§16.2 (MCP in co-op), §13.3 (a game control component on the driver as the input next to `PlayerInput`), §17.2
(`defaultPeer`), §18.4 (the driver-entity pattern and the demos' replication table), §22 (S16b row; S19,
  S29 and S34 depend on S16b); `AGENTS.md` (`--coop` with an example: `SAIGA_SCREENSHOT=/tmp/racing_coop.png
  ./build/samples/games/game_racing --coop --demo`).

## Verification

The CI recipe of AGENTS.md with samples ON (the Clang job builds the demos); `ctest -R test_engine_app`; locally the
demos as in Tests.

## Files

`src/saiga/engine/app/{EngineApp,EngineMain,EngineAppSetup,AppTools}.*`, `src/saiga/engine/tools/{ScenarioWorld,
WorldTools}.*`, `src/saiga/engine/api/WorldApi.*`, `src/saiga/engine/world/World.*` (`stepsForAdvance`),
`samples/games/{flight,platformer,racing,common}/`, `tests/test_engine_app.cpp`, `docs/engine_design.md`, `AGENTS.md`, this file.

## Design changes

- Common component IDs: registration IDs must be dense (`Registry::validateRegistration`), so `Driver`, `VehicleSeat`
  and `DemoPilot` (`samples/games/common/DrivingComponents.h`) take the IDs 64-66 in every demo and the demos' own enums
  start at `kFirstDemoComponent` (67). The demos' component IDs shift (no demo saves exist outside tests; the save
  format maps by name anyway).
- `DemoPilot` is one common component `{suspended, progress, waypoint}` (racing progress, platformer waypoint; the
  flight autopilot has no state) instead of a per-game field list.
- The leave path: `removeDriver` (common) calls the new engine mechanism `destroyOwnedBy(host, peer)`
  (`src/saiga/engine/gameplay/DestroyOwnedBy.*`: the peer's roots with their descendants) and clears the seats of
  vehicles whose driver is gone; the platformer uses the same helper (its hero is the peer's only entity). Reason: the
  plan's demo-independent engine test of the leave path cannot use sample code (R-TEST-1); which entities a leaving
  peer loses stays the game's call (R-MECH-1).
- `EngineAppSetup::checkOptions(AppOptions&, error)` (new): the engine does not know `--demo`, so the clamp of the co-op
  delay to 8 with `--demo` is the games' `clampCoopDemoDelay`, called after the command line was parsed.
- The co-op hooks of the MCP tools live in `WorldApi` (`setCoop(CoopHooks {host, stepAll})`) instead of
  `WorldToolsOptions`, so the operations inside an `apply` batch act on both worlds too; `measure_performance` (a
  `WorldTools` tool) reads them from the API. All seven snapshot tools are refused in co-op (also `list_snapshots`,
  `delete_snapshot`, `diff_snapshot`: the client's API has no snapshots).
- `run_scenario` accepts the host + client variants also without `--coop` (refused for local-only scenarios).
- Resync: no explicit hook; a `Resync` destroys the client's replicas, so the app drops the dead local player handle
  and takes the new replica with `Player.peer == localPeer()` when it arrives (tested).
- The vehicle systems iterate all vehicles (`view<Transform, VehicleState>` / `<Transform, AircraftState>`) and read
  the controls through `seatControls<C>(world, vehicle)`; the owner-side pilot systems: racing and flight in phase
  Gameplay, the platformer's in phase Behaviour (before the controller, as `DemoRunner::update`).
- Render caches: `PoseHistory` lost its own listeners; `RenderCaches<State>` (`GameEngine.h`) keeps one `PoseHistory`
  and a state ring per entity, recorded by one step listener per renderer, with a pulse (`Entry::pending`, the
  platformer's landing speed) collected when a state becomes current.
- Slots: racing rows 8 m behind the line, alternating 3 m left / right; flight 60 m behind and 20 m inside the circuit
  per slot (so both planes are in the `--coop --demo` screenshot); platformer 1.5 m steps next to the start, three per
  row. `RunState` gains `slot` (a new field with a default: no version bump); `AircraftControls` (the lever left) and
  `AircraftState` (the lever arrived) are version 2, `VehicleControls` version 2 (it moved to the driver). No
  migrations (R-SAVE-1: no demo saves exist outside tests).
- Scenarios: `accelerates_on_track` waits until 25 m along the track (at most 6 s), then checks the speed (> 12 m/s),
  the window (< 40 m) and the wheels; `crashes_into_terrain`'s second part levels the sticks, waits until the plane is
  100 m from the respawn point and checks that the driver's `Respawn` brings it back (< 5 m); `held_jump_after_a_restart`
  also checks that the hero is back at the start after the restart. The platformer's `jumps_onto_platform` teleports
  the hero with `ownerTeleport` before its spawn (a plain teleport in every variant). No scenario is local-only.
- The GUI's own-vehicle buttons queue the action for the local player only when the shown vehicle is the local
  player's; the label says "(when steps run)" while paused.

## Results

- Tests: `tests/test_engine_app.cpp` 37 tests (6 new): `AppCommandLine.Coop` (the delay, an argument that is not
  consumed, the range, the errors with `--scenarios` / `--benchmark`, `checkOptions`),
  `EngineAppCoop.InterleavedStepsAndTheClientsPlayer` (`hostWorld`, `startWorld`, the join once, the local player
  found 1 + delay steps later, interleaved host / client steps at a low frame rate with every host snapshot applied,
  the input to the host through the owner state, `app_status.coop`, the local player after a `Resync`),
  `EngineAppCoopDeathTest.SetLocalPlayerOfAnotherPeer`, `EngineAppCoop.ToolsActOnBothWorlds` (pause, step, time scale,
  `measure_performance(steps)` with `host`, `world_status` / `get_profile` with `host`, the snapshot tools refused,
  `run_scenario` Plain + HostClient, `keep` refused, `reload_data` through the host),
  `EngineAppCoop.StepsForAdvanceMatchesAdvance`, `EngineCoopLeave.PeersLeftAndDestroyOwnedBy` (destroy the client
  world, then its transport end: the host's `peersLeft()`, the helper keeps host-owned entities).
- Full ctest (RelWithDebInfo, lavapipe): 65 of 66 pass; `test_opengl_window` fails locally because of the display (as
  before).
- Game scenarios (`--scenarios`, now with the host + client variants): `game_flight`, `game_platformer`, `game_racing`
  18 passed, 0 failed each (3 scenarios x 6 variants). Scenario changes: flight `climbs_with_full_throttle` and
  `crashes_into_terrain` (driver of `defaultPeer()`, the lever on the plane, the sticks through `edit`, `stepUntil`, the
  respawn action of the driver); platformer `jumps_onto_platform` and `held_jump_after_a_restart` (a client's hero, the
  jump decided inside `edit`, the fall an owner-side teleport, the restart through `RestartRun` + `ownerTeleport`, the
  release held 5 host steps, `stepUntil`); racing `accelerates_on_track` (`spawnDriven`, controls through `edit`, a
  progress window); unchanged: `autopilot_flies_the_circuit`, `demo_reaches_goal`, `ai_completes_a_lap` (host AI,
  the recorded positions, 7.25 s and 62.07 s still hold), `checkpoint_out_of_range` (the car is its own driver).
- `SAIGA_SCREENSHOT=<file> ./game_<game> --coop --demo --headless`: racing shows both cars, flight both planes (the
  client's 60 m behind the autopilot), the platformer the client's hero (the host's demo runner runs the same path a
  few steps ahead and is mostly hidden behind it); the GUI says "co-op client (peer 1)" with the vehicle count.
- Instead of an interactive window session (no display here): headless `--coop --no-render` sessions over MCP per demo:
  racing `set_input throttle` moved the client's car 34 m in 3 s through the host and back (the driver entity sits at
  its car), flight `set_input throttle, pitch` moved the lever to 1 and climbed 78 m, platformer `set_input move_z`
  moved the client's hero and its `queue_action restart` went through the host (`RestartRun` + `ownerTeleport`) back to
  its start slot.
- Not covered: an interactive session with a window and a keyboard (no display in this environment).

