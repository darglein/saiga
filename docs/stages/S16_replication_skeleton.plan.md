# S16: Replication skeleton

Status: planned.

Sections follow the stage plan template of S05.

## Goal

A host world and client worlds stay in sync over the S15 transport: roles, ownership fixed at creation, `RunOn`
filtering, `NetId`s, reliable spawn, destroy and hierarchy changes, full-state snapshots every step, owner-authoritative
state from the owners, owner commands with the teleport counter, `RenderPose` from an interpolation buffer, a host time
estimate on clients, `PlayerAction` as a message, and the host + client scenario variant that every non-local-only
engine scenario passes (delay 0 and 2, asynchronous stepping). Split from S16 in planning: the tech demos' co-op port
and `--coop` are S16b.

## Depends on

S15 (transport, messages incl. raw messages, phase 1 receive with per-handler command flush, flush after phase 8,
abort and load semantics), S12 (scenario variants), S11 (saves: a host load resyncs clients), S10 (game objects,
`ownedBy`, `queueAction`), S08 (hierarchy, structural log, transform flags), S07 (`visit` with `Quantize`).

## Rules that apply

R-NET-1 (every gameplay feature passes the host + client variant unless local-only), R-NET-2 (who writes what; from
this stage checked by the variant's write detector and in review; the host's `ownerTeleport` write and the phase-5 owner pose receiver
are its explicit exceptions), R-STATE-1 (`onReplicatedSpawn` rebuilds `cache` from replicated components only), R-DET-1
(the variant is reproducible: loopback seeded, client stepping seeded), R-PAR-1, R-REG-1/2, R-TEST-1, R-TEST-2
(`engine/net/`, `ownedBy` filtering in views: benchmarks), R-MATH-1, R-SAVE-1 (`Transform` gains `Quantize` field
options: a visit change without a layout change, no migration), R-DOC-1/2, R-FILE-1.

## Scope

### 1. Roles and filtering (`World`, `SystemList`)

- `WorldSetup::role` (`NetRole::Host` default, `Client`) and `localPeer` (§7.9; asserted equal to
  `transport->localPeer()` when a transport is set); `World::isHost()`, `localPeer()`.
- Systems run by `RunOn` (host: `Host`, `Both`; client: `Client`, `Both`). The engine's systems: `behaviour_events`,
  `behaviour` `Host`; `player_control` `Both` over `ownedBy(localPeer)`; `character_move` `Both` ported to
  `ownedBy(localPeer)` (R-NET-2; today it views all entities and is safe only because `CharacterIntent` stays zero on
  non-owners); `transform`, `spatial_grid`, `tick` `Both` (ticking computes due flags; the `update` calls are in the
  `Host` behaviour system). `GameObject` hooks on clients are only `configure`, `onReplicatedSpawn` and
  `onReplicatedDestroy` (§13.5).
- Client restrictions (asserts with `SAIGA_ASSERTS`, errors in the world API): no `destroy` of replicated entities, no
  `setOwner`, no `save`/`load` (`canSave()` false), no `queueAction` for players it does not own (it sends
  `PlayerAction` for its own). Entities a client creates are always local (never replicated). The client world API is
  read-only for replicated entities except their owned `OwnerAuthoritative` components.

### 2. What is replicated (`engine/net/NetIds.h`, `Registry`)

- `Transform` is `OwnerAuthoritative` (§9.3; its mode is already stored, `Registry.cpp`). Its fields get `Quantize`
  options (position 1e-4 m, rotation 1e-5); the field list keeps the `Quantize` step (today `FieldList.h` drops it,
  `Visit.h` promises it for S16); the save schema is unaffected (`TypeSchema.h`).
- `Parent` never travels as a component: it is the spawn header field and the `Hierarchy` message only (excluded from
  snapshots and `ComponentChange`; applied through the hierarchy internals, `addNow<Parent>` with child counts and
  `hierarchyDirty`). `LocalTransform` is a normal `Host` snapshot component; its copy in the `Spawn` / `Hierarchy`
  header only makes the hierarchy change atomic.
- Peer-local components (`Replicate::Never`, saved, e.g. `CharacterIntent`, computed on each peer, §9.3): the `Spawn`
  carries a presence list of the entity's `Never` saved components, which the client default-constructs, so its
  owner-side systems (`player_control` over `ownedBy<Player, PlayerInput, CharacterIntent>`) find the entity; later
  additions and removals of such components are `ComponentChange` presence entries. `Persistent` is applied through its index path;
  `makePersistent` asserts on a client (its counter would collide with the host's IDs).
- "Spawned": an entity passed its first committed phase-8 replication (with or without connected peers; its `NetId` is
  assigned then, and a late joiner's spawn set reuses it), so the gates below behave the same in single-player, the
  plain variant and the host + client variant.
- A replicated entity is one created on the host that is alive at phase 8 and not marked local (`NetLocal` tag
  component, `Replicate::Never`, saved; for host-side helpers, cameras, editor objects; adding or removing it after the spawn was sent asserts; a
  `NetLocal` entity's descendants are local too). Every other host entity is replicated with its `Host`, `HostReliable`
  and `OwnerAuthoritative` components.
- World-state and space-state entities: bound by fixed `NetId`s (1 for the world state, 2 + space for space states,
  `SpaceId` is 16 bit; dynamic `NetId`s start at 2^17) to the client's own world-state and space-state entities
  (created by the client `World` at construction when globals are registered, §7.8); spawns, snapshots and `Resync`
  overwrite their components and never create or destroy them; a bound spawn for a state entity the client lacks is a
  setup mismatch (assert in every build).
- Children: the `Transform` of an entity with `Parent` is derived (phase 6 from `Parent` + `LocalTransform`) and is
  left out of snapshots and owner states; owner states cover owned roots (their descendants' `OwnerAuthoritative`
  components ride with the root's state).

### 3. Ownership (`engine/net/NetOwner.h`, §13.3)

- `NetOwner {uint8_t peer}` (`Replicate::Host`, saved) on roots only (asserted on a child);
  entities without it belong to the host. `World::setOwner(root, peer)` on the host before the entity's spawn is sent
  (asserted); no transfer later. A `Player` entity's `Player.peer` equals its `NetOwner` peer (`setOwner` asserts it).
- `World::ownedBy<C...>(peer)` (exists as all entities) filters by the root's owner (a cached owner per entity slot;
  a `Parent` or `NetOwner` change updates the subtree; rebuilt after a load next to `rebuildHierarchyState`), so the
  filter is one lookup. Reparenting never changes the owner: once spawned, the old and the new root must have the same
  owner (assert), `add<Parent>` on an entity with `NetOwner` asserts, and `setOwner` on a root with children sets the
  subtree (before the spawn).
- Before an entity's spawn is sent, the host may write its `OwnerAuthoritative` components (`onSpawn` initial state);
  afterwards the host does not, and the host world API refuses such edits on client-owned entities with an error
  naming the owner.
- `World::ownerTeleport(root, position, rotation)` (host; asserts a root): before the entity is spawned, or for a
  host-owned entity, a plain `teleport` (no command, no counter); for a spawned client-owned entity the host applies
  the pose with `teleport()` (the R-NET-2 exception; so other clients snap, below), increments the entity's command
  counter and sends `OwnerCommand {Teleport, pose, counter}` (reliable, `Control`); the owner applies it in phase 1 and
  stamps its owner states with the counter; the host drops owner states with an older counter (serial-number compare).
- Two counters, both outside components (not saved, not in snapshots): the command counter per client-owned root (host
  table and the owner's owned-entity table; `uint8_t`, wrapping) for owner states, and the snap counter per replicated
  root (host table; bumped in phase-8 replication when the root's `teleported` flag of this step is set and not yet
  counted; sent in snapshots; `Resync` resets both). `NetOwner` is `{uint8_t peer}`.
- Teleports by the owner of its own entity are not conveyed as snaps (the counter comes only from the host): owner-side
  respawns go through the host's `ownerTeleport` (as S16b's restart).

### 4. NetIds and replication (`engine/net/Replication.{h,cpp}`, `engine/net/NetVisitors.h`, `EngineMessages.{h,cpp}`)

- Host table per entity slot: `NetId` (`uint32_t`, monotonic for the life of the host process, never reused, not reset
  by a load), spawned flag, last-sent bytes hash per `HostReliable` component. Client table: `NetId -> Entity`.
  Snapshots and `Resync` carry an epoch (incremented by every host load); clients drop snapshots of another epoch.
- Network visitors (`NetWriter` / `NetReader`, little endian): `Entity` fields as `NetId` (unresolved on the client:
  `Entity{}` and the component re-decoded when the target arrives, for spawns and `HostReliable` components; snapshot
  components are replaced by the next snapshot); `Quantize{q}` fields as zigzag varints (S15's `Varint.h`) of
  `round(v / q)`; quaternions normalized after reading; other fields as `toBytes`.
- Messages (engine IDs below 64; `Spawn`, `ComponentChange`, `Snapshot`, `OwnerState` are raw messages serialized
  once with the network visitors):

  | Message | Direction, channel, reliability | Content |
  |---|---|---|
  | `Spawn` | host → clients, State, reliable | `NetId`, class ID (0: none), owner, parent `NetId` (a header field, never decoded as the `Parent` component), the `LocalTransform` if parented, all replicated components |
  | `Destroy` | host → clients, State, reliable | `NetId` (the client calls `onReplicatedDestroy`, then destroys; a destroy of an entity the client's own cascade already removed is ignored) |
  | `Hierarchy` | host → clients, State, reliable | `NetId`, new parent `NetId` (0: root), `LocalTransform` (reparenting) |
  | `ComponentChange` | host → clients, State, reliable | a `HostReliable` component whose bytes changed, or an addition/removal of any replicated component |
  | `Snapshot` | host → clients, State, unreliable | header (epoch, host step, part, flags); per entity `NetId`, the root's snap counter (§3; survives several host steps in one batch and S32's loss), its `Host` and `OwnerAuthoritative` components (not the client's own entities' `OwnerAuthoritative` ones, not derived child transforms); split at entity boundaries into messages of at most `kMaxMessagePayload`, each zstd-compressed alone or raw, decodable alone |
  | `OwnerState` | owner → host, State, unreliable | estimated host step, per owned root `NetId`, teleport counter, its `OwnerAuthoritative` components |
  | `OwnerCommand` | host → owner, Control, reliable | `NetId`, kind (`Teleport`), pose, counter |
  | `PlayerAction` | owner → host, Control, reliable | the §9.3 action with the player's `NetId` |
  | `Ping` / `Pong` | client ↔ host, Control, unreliable | the client's step; echoed with the host step |
  | `Resync` | host → clients, State, reliable | new epoch: clients destroy all replicated entities without callbacks (not the bound world/space-state entities) and receive spawns again |

- The per-entity bound of §13.4 (1100 bytes of snapshot state) stops the program in every build naming the entity and
  its largest component (as S15's oversized unreliable send); a
  component that can exceed it must be registered `HostReliable`.
- Phase 8 order: the game's and engine's phase-8 systems (`SystemList`; the game's are appended after the engine's,
  so "last" cannot be a registration), the phase's command flush, then `World` calls replication (host), owner state
  (owners) and interpolation explicitly; only these can abort after the replication bookkeeping was staged.
  `World::runStep` filters systems by `RunOn` and the role (today it hard-codes `Host`).
- Host replication (serial): with no connected peer it consumes the structural log and serializes nothing (every
  single-player world takes this path); a peer in `peersJoined()` gets the full spawn set of the current state (the
  `Resync` path for one peer) before the regular messages; spawns of new replicated
  entities, roots first, then descendants by depth (a parent is always spawned before its children, whatever their
  indices); destroys from the structural log (the `NetId` from the host table; an entity spawned and destroyed in the
  same step sends nothing); hierarchy changes; component changes; then one snapshot of all replicated entities. Its
  bookkeeping (new `NetId`s, spawned flags, last-sent hashes) is staged in preallocated storage and committed when the
  step completes; `abortStep` drops the staging and rewinds the replication consumer of the structural log
  (`StructuralLog::rewind(consumer, cursor)`, new; safe because the log is truncated only after a completed step); S15
  drops the phase's outbox entries. Clients register no structural log consumer.
- Owner, phase 8: `ownerStateSystem` (`RunOn::Client`).
- Receive, phase 1 (S15's order; each handler's structural changes applied before the next): spawns, applied as one
  unit through an internal client path (create the entity, its components, the parent through the hierarchy internals,
  waiting for an unresolved parent instead of destroying the child, `configure`, `onReplicatedSpawn`; never `onSpawn`);
  a throwing `onReplicatedSpawn` removes the object but keeps the entity and its mapping (the message is consumed, S15;
  the entity must stay in sync). `spawn<T>` of a class on a client asserts (a replicated game object only arrives).
  Destroys use an internal path that marks the subtree `destroying` (as `abandonSpawn`) and calls only
  `onReplicatedDestroy` (never `onDestroy`); hierarchy changes, component changes; every
  snapshot of the batch goes to the interpolation buffer (§6), the newest complete host step is applied to the
  components of non-owned entities (owned entities keep their own `OwnerAuthoritative` state; a changed teleport counter
  applies the pose with `teleport()`). `PlayerAction` on the host: the handler translates the `NetId`, drops and counts
  actions for players the sender does not own, and puts the action into the action queue with (sender, sequence), so it
  is delivered as an `ActionEvent` in the next step (exactly once, S15).

### 5. Client-owned entities on the host

In phase 5 `ownerPoseSystem` (`RunOn::Host`) writes the latest received owner state of each client-owned root: its
`OwnerAuthoritative` components, and the pose extrapolated from the last two received poses to the current host step
(at most 6 steps, §13.4; no extrapolation when the last two stamps are equal or not increasing: async stepping sends
0 or 2 states per host step, clock jumps). States with an older teleport counter are dropped. `stepNetworkOnly()`: the engine's receive, `ownerPoseSystem` without extrapolation, phase 6's `transformSystem` (so
derived child transforms follow), replication and owner state; no game systems, no step listeners, no action or event
delivery, the transform flags are not shifted (the replication remembers which `teleported` flag it counted), the log
is not truncated, `stepIndex` does not advance; it commits the staged replication bookkeeping at its end. Game
messages stay queued for the next real step (only engine messages are handled). Snapshot headers
carry a snapshot sequence number separate from the host step; "newest" and the end condition use it.

### 6. Time and interpolation (`engine/net/HostClock.{h,cpp}`, `engine/net/InterpolationSystem.{h,cpp}`)

- Clients estimate `hostStep = localStep + offset` from snapshot host steps and the `Ping`/`Pong` round trip (in local
  steps, every 30 steps): the offset slews by at most 1/8 step per step toward the newest estimate and jumps only when
  the error exceeds 30 steps (a resync); `world.time()` on clients is the estimated host time.
- `RenderPose {Vec3 position; Quat rotation; uint8_t flags}` (`Replicate::Never`, `Saved::No`; flags: changed this
  step, changed last step, teleported; the interpolation system shifts them every step) is written in phase 8 by the
  interpolation system for remote entities on clients and for client-owned entities on the host: the buffer keeps the
  received poses by host step (every snapshot, not only the newest) and samples at `estimated host step -
  interpolationDelaySteps` (default 6, `WorldConfig::interpolationDelaySteps`), linear in position and slerp in
  rotation, holding the last pose when it runs dry; a teleport counter change snaps.

### 7. Host + client scenario variant (`ScenarioWorld`, `ScenarioRunner`, §17.3)

- New `ScenarioVariant` values `HostClient` (delay 0), `HostClientDelay` (delay 2) and `HostClientAsync` with their
  `variantName`s (the enum stays the gtest parameter and the `ScenarioResult` field); `ScenarioOptions::clients`
  (default 1). The runner's default set becomes Plain, Repeat, SaveLoad and the three, when
  `ScenarioOptions::networkVariants` is set; local-only scenarios skip the three.
- `ScenarioWorld` creates the host and client worlds over `LoopbackTransport::createNetwork` (clients with
  `setTickEveryStep`, profiling off and the host's `sharedData`); clients are stepped from the host's step observer
  (which re-attaches itself if a listener throws), before the hash and traces, so `world().step(n)`, `advance` and the
  API's `step` drive them like the helpers; `steps()` and `maxSteps` count host steps: after
  each host step every client steps once (delay variants) or a seeded 0, 1 or 2 times with the cumulative difference
  `clientSteps - (hostSteps - 3)` kept in [-2, 2] by restricting the choice (async; the client starts 3 steps late).
  `world()` is the host world (checks run there).
- Helpers act from the right peer: `spawnPlayer(position, std::optional<uint8_t> peer)` and `spawnPlayer<T>(position,
  std::optional<uint8_t> peer, initial...)` spawn on the host; `Player.peer` is `peer.value_or(0)` in the single-world
  variants (an explicit peer above 0 sets only `Player.peer`) and `peer.value_or(1)` in the host + client variants,
  where the helper also calls `setOwner(player, Player.peer)` (peer 0: the host's own player); it does
  not step: `move`, `input` and `edit` for a client entity that has not arrived yet are queued and applied on arrival.
  `move` writes `PlayerInput` and `input` sends `PlayerAction` on the owner; `edit(entity, fn(World&, Entity))` runs
  fn on the owner's world with the owner's handle (the way a scenario writes owner-authoritative state); `client(i)`
  gives a client's world; `onClient(i, entity)` translates a host handle through `NetId`. Checks that depend on the
  network latency use `stepUntil` with a bound instead of fixed step counts.
- End: `stepNetworkOnly` on all worlds, host and clients 1:1, until each client applied a snapshot sent after the
  scenario ended and the host received the owners' final state (at most `2 * delaySteps + 3` steps without loss,
  asserted; bounded by 600), then the comparison for all replicated entities: for every replicated component the
  authority's value (the host for `Host`/`HostReliable` and for `OwnerAuthoritative` of host-owned entities, the owner
  for client-owned ones) equals every other peer's copy, within 2 `Quantize` steps (quaternions are renormalized per
  hop; exact for unquantized fields); `Entity` fields compare through `NetId`s; the `Transform` of entities with
  `Parent` is not compared (derived; `Parent` through `NetId` and `LocalTransform` are). A difference is a failure
  naming peer, entity, component and field.
- Write detector (scenario worlds): the receivers (snapshot apply on clients, `ownerPoseSystem` on the host) keep a
  hash per (entity, component) of what they last wrote, and every other legitimate writer refreshes it (the host's
  `ownerTeleport`, the client's `Spawn`, `ComponentChange`, `Hierarchy` and `Resync` handlers, the owner's
  `OwnerCommand` apply); before overwriting they compare, and a mismatch (another
  writer touched a component that belongs to a remote authority, R-NET-2) is a scenario failure naming the peer,
  entity and component. The final comparison alone cannot see such writes: the receivers heal them every step.
- The save/load variant stays host-only; a host load in a networked world sends `Resync` (tested separately).
- Until S16b ports the demos, the host + client variants are opt-in: `ScenarioOptions::networkVariants`, which
  `EngineMain` fills from `EngineAppSetup::networkVariants` (false for the games) and `tests/test_engine_scenarios.cpp`
  sets; `game_* --scenarios` keeps its three variants; S16b turns it on. The runner's result-count test (`test_engine_scenarios.cpp`) follows.
- Engine scenarios that change (`tests/test_engine_scenarios.cpp`): `action_reaches_listener` waits with `stepUntil`
  (the action arrives 1 + delay steps later); `player_walks_with_input` checks its distance and its stop with bounds
  that allow the latency (a `stepUntil` for the stop); scenarios writing a player's components on the host use `edit`;
  scenarios that cannot run networked are registered local-only with the reason in a comment. Each change is listed in
  the PR.

## Out of scope

The tech demos' co-op port and `--coop` (S16b), interest management, snapshot rate below the step rate, cosmetic events,
loss and reordering, `Despawn` (S32; §13.1 lists `Despawn` under the skeleton, corrected), sessions, handshake, real
transports (S33), the world and chunks over the network (S26), live terrain edits (S22), client physics proxies (S29),
moving platforms (S40), delta compression (§20), the generic extract from `RenderPose` (S19).

## Design notes

- zstd: `engine/net` is the first direct zstd user outside core (linked PUBLIC through `saiga_core`); the MSVC job
  confirms the link.
- Snapshots of every entity every step are the simplest correct baseline; S32 adds interest, rate and loss handling on
  the same messages; full state each step also heals unresolved `NetId`s in snapshot components. The bandwidth (about
  100 bytes per moving entity per step) is far above §17.10 for large worlds; that is S32's job.
- `NetId`s are monotonic per host process and snapshots carry an epoch, so a host load (also through the world API's
  snapshots) is a clean resync and in-flight snapshots of the old state are ignored.
- The replication bookkeeping commits with the step, mirroring S15's outbox rules, so an aborted step never leaves the
  host believing it sent what it did not.

## Tests

`tests/test_engine_replication.cpp` (new) and `tests/test_engine_net.cpp` (extended):
- Visitors: `Entity` as `NetId` with unresolved targets, `Quantize` round trips at the step boundaries, the 1100-byte
  entity bound (death test naming the entity), zstd/raw choice.
- Roles: `RunOn` filtering of every engine system, client restrictions (death tests and API errors), `ownedBy` with
  owners and descendants (cached owner after reparenting), `setOwner` after the spawn was sent (death test), the
  `Player.peer` assert, the host API refusing `OwnerAuthoritative` edits of client-owned entities after the spawn.
- Replication over a pair: spawn with class (`configure` and `onReplicatedSpawn` called, `onSpawn` not); a parent with
  a higher index than its child; reparenting; destroy (`onReplicatedDestroy`; a cascade-removed child's destroy
  ignored); spawned and destroyed in one step (nothing sent); world-state and space-state bound and overwritten;
  `HostReliable` changes only when bytes change; several host steps in one batch (the buffer gets all, components the
  newest); a host teleport snaps the client's `RenderPose`; owner state on the host (extrapolation clamp, older counter
  dropped); `ownerTeleport` both cases and the counter wrap 255 → 0; the owner ignoring the snapshot counter;
  `PlayerAction` exactly once, in order, the next step, a non-owned player's action dropped; `Ping`/`Pong` and the host
  clock (slew, resync jump, convergence under the async pattern); interpolation buffer (sampling, dry hold, snap);
  `Resync` after a host load (epoch, old snapshots ignored, bound entities kept); an aborted phase 8 after
  `replicationSystem` (bookkeeping rolled back, the spawn sent in the next step); a client-created entity stays local;
  `NetLocal` host entities (and their descendants) are not replicated; a host without peers serializes nothing; a peer
  joining late receives the full spawn set; reparenting across owners and `add<Parent>` on an owned root assert; a
  throwing `onReplicatedSpawn` keeps the entity mapped; `makePersistent` on a client asserts; a client-owned player
  walks (`CharacterIntent` present on the client through the presence list); `ownerTeleport` before the spawn is a
  plain teleport; the host's `ownerTeleport` snaps a third client; a child's animated `LocalTransform` reaches clients.
- `tests/test_engine_scenarios.cpp`: every engine scenario passes all host + client variants (or is local-only with a
  reason); a scenario that writes an owner-authoritative component on the host after the spawn, and one whose host
  system writes it every step, are failures of the write detector; the async variant is reproducible (two runs give equal hashes).
- Benchmarks (R-TEST-2): snapshot serialization and application of 10^4 entities with `Transform`, and `HostReliable`
  change detection of 10^3 entities with a 1 KB component, in
  `tests/test_engine_net_benchmark.cpp` (S15's); `ownedBy` and `RunOn` filtering in
  `tests/test_engine_ecs_benchmark.cpp`.

## Done when

- The S16 row of §22 (engine part): all engine scenarios pass in the host + client variant without loss, delay 0 and
  2, and with asynchronous stepping (CI). The demo co-op part is S16b's row.
- `docs/engine_design.md` updated: §1 (R-NET-2's exceptions), §7.2 (`Quantize` in use, `NetLocal`), §7.4 (applied snapshots and the transform flags, the
  teleport counter), §7.6 (the replication consumer, `rewind`),
  §7.9, §7.10 (`setOwner`, `ownerTeleport`, `stepNetworkOnly`, `queueAction` as a message), §8.1/§8.2 (phase 1, 5, 8
  work; the engine systems' `RunOn`), §9.3, §13.1 (`Despawn` is S32), §13.3-§13.5, §13.6 (helpers), §13.8, §17.2
  (`edit`, `client`, `onClient`, the `spawnPlayer` peer), §17.3 (stepping from the observer, the bounded async pattern,
  the comparison matrix), §16.2 (the world API on clients), §10.3 (`RenderPose` as written), §18.2 (the engine messages), §3, §22 (S16 split into S16 and S16b); `AGENTS.md` (the new
  test).

## Verification

The CI recipe of AGENTS.md and the full ctest run; `ctest -R "test_engine_(net|replication|scenarios)"`; the benchmarks
before and after (ECS views with the filter).

## Files

`src/saiga/engine/net/{NetIds.h,NetOwner.h,NetVisitors.h,Replication.{h,cpp},HostClock.{h,cpp},
InterpolationSystem.{h,cpp},EngineMessages.{h,cpp}}`, `src/saiga/engine/components/{RenderPose.h,NetLocal.h,
Transform.h,EngineIds.h,EngineComponents.cpp}`, `src/saiga/engine/ecs/{Registry,View,FieldList}.*`,
`src/saiga/engine/world/{World,SystemList,StructuralLog}.*`, `src/saiga/engine/gameplay/{CharacterMoveSystem.*,
Systems.h,PlayerAction.h}`, `src/saiga/engine/save/WorldLoad.cpp` (epoch, `Resync`), `src/saiga/engine/tools/Scenario*`
(incl. `ScenarioRunner`), `src/saiga/engine/app/EngineAppSetup.h` (`networkVariants`), `src/saiga/engine/api/WorldApi.cpp`, `tests/test_engine_{net,replication,
scenarios,ecs_benchmark,net_benchmark}.cpp`, `tests/CMakeLists.txt`, `data/tests/engine/schema.json` (new components),
`docs/engine_design.md`, `AGENTS.md`, this file.
