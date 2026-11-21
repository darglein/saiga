# S21: Spaces and transitions

Status: implemented (2026-11-21; steps 1-10 on `stage-s21`, stacked on `stage-s20`).

Sections follow the stage plan template of S05.

## Goal

A world has more than one space: one space instance per `WorldConfig::spaceTemplates` entry (open-world height maps,
flat open-world spaces and enclosed spaces). Entities live in any space (`spawnIn`), move between spaces with
`World::moveToSpace` (root and descendants, a teleport; grid, ticking, extract and replication follow), and players use
**entrances**: an entity with `SpaceLink`, `World::requestTransition(player, link)` (the optional `spaceLinkSystem`
turns an `InteractEvent` into it), completed in phase 1 of the next step (no arrival mechanism before S24). A
client-owned player is moved through a reliable `OwnerCommand` that now carries the space; the host + client variant
ends with equal spaces on every peer. Each peer renders only the space of its view camera; switching the space rebuilds
the GPU terrain set for that space and, when the space is not ready yet, shows a **loading screen** (ImGui, an engine
default the game can replace) until the visual data around the camera is resident. Saves keep spaces, space-state
entities per space and pending transitions; S21 adds the optional save-section mechanism that S22 and S23 reuse. The
platformer demo gets an enclosed cellar behind a door.

## Depends on

S20 (`SpaceTemplate`, `WorldConfig::spaceTemplates` with one entry, `World::spaceTemplate` / `spaceSeed` /
`terrainHeight(space, ...)`, the empty-list default template, `SpaceTemplate::hasTerrain()`, `TerrainSystem`
(`World::terrain()`) with a `TerrainStore` per space, `updateView(space, camera)` with its space-switch path,
`completeView`, `readiness(space, camera, radius)` (D6), the `RenderSink` terrain calls incl. `clearTerrain()` (E1),
`TerrainSystem::terrainUploadsPending()` (forwarded from the sink, 0 without one), `Stats::pyramidsGenerated`, the
complete points and `EngineApp::renderFrame` passing `view.space`), S16
(ownership, `OwnerCommand` with the teleport counter, snapshots, interpolation buffer, `RenderPose`, the host + client
variants), S19b (extract, `interpolatedPose`, `WorldPose.space`, `RecordingRenderSink`, the app's view camera), S16b
(`--coop`, the platformer's client-owned hero and `DemoPilot`), S14 (app tools), S11 (saves), S10 (`spawnIn`,
space-state entities, `InteractEvent`).

S20, S22 and S23a/b/c are planned in parallel and follow the same cross-plan decisions (D1–D14 of review round 1,
E1–E14 of review round 2, which override D where they differ). This plan uses S20's names
(`docs/stages/S20_height_tiles.plan.md`). Where S20 already delivers something (space seed, `spaceTemplate`,
`terrainHeight` per space, the empty-list default, `SpaceTemplate::hasTerrain()` (E6), the GPU terrain set that the sink
drops itself on a pyramid of another space or layout, the explicit sink call `clearTerrain()` for a camera space without
terrain (E1) together with its trigger in `TerrainSystem::updateView`, lazy CPU-cached pyramids,
`TerrainSystem::terrainUploadsPending()`), this plan marks it as existing and only
describes its delta from one space to many. S21 owns the optional save-section mechanism and its extension recipe (D7),
the chunk range and site origin of enclosed spaces (D12, E4) and `World::spaceReadiness` (E8).

## Rules that apply

R-STATE-1/2/3 (pending transitions are a component; `Transform::space` is written only by the World's internal space
primitive), R-SYS-2 (`requestTransition` during a step is a command), R-DET-1, R-NET-1 (the scenarios pass the host +
client variants), R-NET-2 (clients never move replicated entities; the owner applies the host's command; the host never
takes a space from an owner), R-SAVE-1 (new saved components `SpaceLink`, `PendingTransition`; `RenderPose` is
`Saved::No`; no migration), R-REG-1/2 (new components, event, system, MCP tool in the explicit lists), R-MECH-1 (which
spaces exist, how they connect, how a player interacts, the loading radius and what the loading screen looks like are
game policy), R-MCP-1, R-TEST-1, R-TEST-2 (`engine/net/`, `engine/sync/`, `RenderScene`, terrain), R-FILE-1,
R-DOC-1/2, R-MATH-1/2. §5.2: `saiga_vulkan` never includes engine headers (D2).

## Scope

### 1. Space configuration (`World`, §7.9, §12.4 "Before S23")

- `WorldConfig::spaceTemplates` accepts 1..65535 entries; S20's empty list (one flat default template) stays. S20's
  stop for more than one entry, whose message names S21, goes. `SpaceId` = template index; the list is append-only for
  a game's lifetime (saves store `SpaceId`s; Scope 5 checks names and kinds).
- Validated in S20's `validateSpaceTemplates` (the `World` constructor, `SAIGA_EXIT_ERROR` in every build already, D1;
  so the death tests run without `SAIGA_ASSERTS`), extended: non-empty unique template names (at most 256 bytes, the
  save's string bound); an `Enclosed` template must have `hasTerrain() == false` (E6; today: no `prototypeHeight`) and
  `sizeChunks` in 1..512; its `seaLevel` and `baseSpacing` are **ignored**, not checked (no terrain, no sea: S20 draws
  the sea only with `hasTerrain()`). Ignoring `seaLevel` matters because S20's default is `seaLevel = 0`, not
  `std::nullopt`: an enclosed template written by hand (`SpaceTemplate t; t.kind = SpaceKind::Enclosed;`) is valid. The
  hook error names the helper: "space template 'cellar': an enclosed space has no terrain (no prototypeHeight; use
  SpaceTemplate::enclosed(name, sizeChunks))". An `OpenWorld` template keeps S20's rules (spacing 1 or 2, `sizeChunks` a
  multiple of 16 in 16..512, `seaLevel` within `kTerrainHeightLimit`). An `OpenWorld` template **without** a hook is
  legal: flat (y = 0), no terrain drawn, as S20's default template (D1).
- Existing from S20: `World::spaceTemplate(SpaceId)`, `World::spaceSeed(SpaceId)` (space 0 = the world seed, space
  i ≥ 1 = `hashCombine(worldSeed, i)`, S20's amendment A3; D11), `World::terrainHeight(space, x, z)` evaluating the
  space's own template. New: `World::spaceCount()`; the accessors assert `space < spaceCount()` instead of `space == 0`.
  `terrainHeight` of an enclosed space is 0 (the floor plane, as a hook-less open-world space) until its authored
  geometry (S23) and colliders (S28) exist (amendment A2).
- **Chunk range and site origin of an enclosed space** (D12, E4): `[0, sizeChunks)^2`, as for an open-world space;
  the authored site (S23a) is placed with its origin at the centre of the middle chunk, `o = (sizeChunks / 2) * 256 +
  128` metres on both axes (integer division; S23a's formula): `(640, 0, 640)` for 4 chunks. `SpaceTemplate::enclosed(
  std::string name, int32_t sizeChunks = 4)` (static helper, 1 km): `kind = Enclosed`, `seaLevel = std::nullopt` (tidy;
  ignored anyway), no hook; `static int64_t SpaceTemplate::enclosedOrigin(int32_t sizeChunks)` returns `o`. S21 uses the
  range for `describe_world`, `terrainHeight`'s clamp (S20 A16) and the placement of the demo's and the scenarios'
  enclosed content at `o`; S22 rejects edits in enclosed spaces and outside `[0, sizeChunks)^2`, S23 stamps sites
  in the range. Entity positions are not confined before S24 (as in open-world spaces: the platformer's flat space 0
  extends to negative z); S24's arrival requires the arrival chunk to be in range and adds that check (amendment A17).
- New field `SpaceTemplate::background` (`Vec3`, linear RGB, default `(0.02, 0.02, 0.025)`): the clear color of an
  enclosed space (Scope 8). Open-world spaces ignore it (sky). S36's `ambient` / `fog` template data replace it.
- `spawnIn(space, ...)` accepts every `space < spaceCount()` (the assert `space == 0` at `World.cpp:268` and the comments
  at `World.h:153` / `:313` and `World.cpp:142` go); the world API's `spawn` too (Scope 7). `GameObject::spawn` keeps
  spawning in its own object's space.

### 2. `moveToSpace` (`World`, `Transform`, `TransformSystem`, §7.4)

```cpp
// Moves a root and its descendants to `space` at the pose (a teleport: no interpolation; the spatial grid, ticking,
// the extract and the replication follow). Applied at once, between steps and during a step (a space change is a pose
// write, not a structural change: R-SYS-2 commands are for spawn/destroy/add/remove). On the host, for a spawned
// client-owned root, the pose is applied here and sent to the owner as an OwnerCommand with the space and the
// incremented command counter. The same space is a plain teleport.
void World::moveToSpace(Entity e, SpaceId space, const Vec3& position, const Quat& rotation);
```

- Asserts (code errors; the world API returns errors instead): a dead or reserved entity, a child (its parent decides;
  as `set_transform`), a world-state or space-state entity, `space >= spaceCount()`, inside `parallelEach`, on a client
  for a replicated entity (R-NET-2; client-local entities may move).
- **One World-internal primitive** writes the space (private members of `World`, `World.cpp`):
  - `Transform` gets a private `void setSpaceInternal(SpaceId)` (sets `sp` and `changedThisStep`; no teleport flag, no
    `teleports` increment) and `friend class World`.
  - `void World::setSubtreeSpace(Entity root, SpaceId space)`: `setSpaceInternal` on the root and every descendant. The
    subtree comes from `collectSubtree(Entity)`, factored out of `collectDoomed` (one pass over the `Parent` pool,
    parents before children, cycle-safe). Not `hierarchyOrder`, which can be dirty in the middle of a step.
  - `void World::teleportTo(Entity root, SpaceId space, const Vec3& position, const Quat& rotation)`: `if (space !=
    t.space()) setSubtreeSpace(root, space);` then `t.teleport(position, rotation)`. `teleports` is incremented exactly
    once per move (by `teleport`), and the descendants' poses follow in phase 6 because the root `wasTeleported()`
    (S08's path).
- **The complete list of space writers** (all inside `World` or `Replication`, which is a friend of `World` today):
  1. `moveToSpace`: on the host for a spawned client-owned root `replication().ownerMove(root, space, position,
     rotation)` (Scope 4, calls `teleportTo`), else `teleportTo`.
  2. `World::ownerTeleport(root, position, rotation)` stays as a convenience: `moveToSpace` with the root's current space.
  3. The owner's `OwnerCommand` apply (`Replication`, phase 1): `teleportTo` with the command's space.
  4. The client's spawn and snapshot apply (`Replication::applyTransform`): a decoded space that differs from the
     replica's root goes through `setSubtreeSpace` and forces a snap (Scope 4).
  5. **Attaching a child**: `addNow<Parent>` (the template in `World.h`, ~line 955) and `World::reparentNow`
     (`World.cpp:186`) call, after `acceptParent`, `setSubtreeSpace(child, parentSpace)` when the spaces differ, and
     mark the child teleported (`Transform::teleport` at its current pose; phase 6 then writes the parent-relative
     pose, and the flag keeps the extract and the interpolation from blending across spaces). The replication's parent
     paths (`Replication.cpp` ~1409 spawn with a known parent, ~1434 waiting children when the parent arrives, ~1563
     the hierarchy message, the `pendingParents` retry) all go through `addNow<Parent>` and need no code of their own.
  6. `World::makeStateEntity(flag, space, defaults)` (Scope 3; used by `createStateEntities` and by the loader's
     `restoreStateEntities`): `setSpaceInternal(space)` on the new space-state entity.
  Not writers: `WorldLoader` (decodes the saved `Transform` through `visit`; Scope 5 checks the invariant on the staged
  values) and `updateChildren` (a free function in `TransformSystem.cpp` that cannot call private members: it asserts
  `t.space() == parent.space()` with `SAIGA_ASSERTS` and repairs nothing, because every writer above keeps the
  invariant).
- Hierarchy invariant: a child is always in its parent's space, from the moment of the write (amendment A6: §7.4
  "changed only by `World::moveToSpace`" becomes "by `moveToSpace` and the engine's hierarchy and replication paths").
- During a step, a `Parent` added by a pending command is not yet part of the subtree; when the command is flushed,
  writer 5 moves it.
- The spatial grid already keys cells by space (S09, `SpatialGrid::update` moves an entry across spaces); the tick
  system's player cells already include the space (S10). No change, covered by tests.

### 3. Space-state entities per space (§7.8)

- `createStateEntities` creates one space-state entity per space (in `SpaceId` order, after the world-state entity) when
  a component is registered with `registerSpaceState`; each one's `Transform::space` is its `SpaceId` (writer 6; state
  entities are outside the grid, so the field is only their identity). Its `make` lambda (`World.cpp:150`) becomes the
  private member `Entity World::makeStateEntity(uint8_t flag, SpaceId space, StateDefaults* defaults)`: `reg.create()`
  with the `Transform` record (the path of `World::create`, `World.cpp:583`, which the loader cannot call: it is not a
  step-time creation), the flag, `setSpaceInternal(space)` for a space-state entity, and per state component either the
  staged default from `defaults` (the loader, moved in with `addStaged`) or `addDefault` (null: the constructor), each
  with its log record. `WorldLoader` is already a friend of `World` (`World.h:654`). `World::spaceStates` stays indexed
  by `SpaceId`. `spaceState<T>(space)` and `readSpaceState` work for every space. Worlds without space-state components
  still have none (entity counts of plain worlds unchanged).
- Replication needs no change: the host binds `kFirstSpaceStateNetId + index in spaceStates` (`Replication.cpp:709`),
  and the index is the `SpaceId` (kept by creation and by the load's mapping, Scope 5); the client creates its own from
  the same setup. Before S32 every client receives every space-state entity; the interest rule of §7.8 ("clients whose
  player is in that space") applies from S32.

### 4. Replication (`engine/net/`, §13.3, §13.4)

- **Roots carry the space in spawns and snapshots**: `Transform::visit` already sends `space`, but S16's
  `applyTransform` drops it (all entities were in space 0). From S21 the client applies it to the root (writer 4): a
  decoded space that differs from the replica's goes through `setSubtreeSpace` and forces a snap (`teleport()`, also
  without a changed snap counter, so a lost snapshot from S32 cannot interpolate across spaces). The spawn path
  (`applyTransform(e, ..., true, false)`, ~1377) applies it the same way.
- **Children and state entities**: the host omits their `Transform` in spawns and snapshots (`Replication.cpp`
  ~517–568, the snapshot skip at ~1730). A child replica is created in space 0 and takes its parent's space when it is
  attached (writer 5) — with the parent known at the spawn (~1409), later when the parent arrives (~1434: the parent's
  `Transform` with its space is applied before the waiting children are attached), or by the hierarchy message (~1563).
  The whole subtree moves. State entities are bound to the client's own ones, created with their spaces (Scope 3).
- **Initial `RenderPose`** at the spawn (~1416) is written after `applyTransform` and the parent attach, so it carries
  the entity's space.
- **The owner's own root**: its `Transform` is `OwnerAuthoritative`, so the owner never takes its root's space from
  snapshots. The space reaches the owner only through the spawn, a `Resync` (destroys and respawns every replica), and
  the `OwnerCommand`.
- **`OwnerCommand` version 2**: a field `SpaceId space` (sent always; messages are not saved, the version is bumped);
  the kinds stay `kTeleport` only (no separate move kind). `Replication::ownerTeleport` becomes `ownerMove(root, space,
  position, rotation)`: `teleportTo`, then everything as S16 (increment the command counter, forget the stored owner
  record, push the interpolation sample with the space, journal for an aborted step, `noteWritten`). The owner applies
  it in phase 1 (`onOwnerCommand`, `Replication.cpp:1766`) with `teleportTo(e, m.space, m.position, m.rotation)` and
  `noteWritten`. `m.space >= spaceCount()` is rejected with `rejectInput` (network input never asserts) and the pose is
  not applied, but the counter **is** taken (`ce->command = m.counter`, as today after the `if`): otherwise every later
  owner state would carry the old counter and the host would drop all of them as stale (`Replication.cpp:1118`),
  freezing the entity on the host. With a taken counter the host's mismatch rule below reports each state instead
  (`net/rejected`); both sides use the same setup, so this only happens for a broken or hostile host.
- **Owner states** (`onOwnerState`, `Replication.cpp:1081`): the host never takes a space from an owner:
  `ownerPoseSystem` applies position and rotation only. For a root's `Transform` record the decode (`t.visit(nr)` on a
  copy of the host's `Transform`, ~1140) also overwrites `t.space()`; the decoded space is compared with the host's
  `Transform::space` **before** `ownerRecords[e.index]` is created or updated and before the interpolation buffer is
  touched: the `Transform` component of a root record is decoded first (a pre-pass over `rec.components`), and a
  mismatch at the current counter rejects the whole record with `rejectInput` (counted in `net/rejected`, not `stale`;
  impossible for a correct owner, whose states before the command carry the old counter and are dropped as in S16). The
  sample pushed into the buffer (`b.second.push`, ~1162) and the first `RenderPose` (~1163) use the **host's** space,
  never the decoded one. No new counter.
- **Interpolation across spaces**: `InterpolationBuffer::Sample` gets `SpaceId space`; `push(step, space, position,
  rotation, counter)` (every caller: `ownerMove`, `onOwnerState`, the snapshot apply). Sampling never interpolates
  between samples of different spaces (a space change is a break like a counter change: the newer sample is used from
  its step on, with `kTeleported`). `RenderPose` gets `SpaceId space = 0` as its **last** field, after `flags` (version
  2; `Replicate::Never`, `Saved::No`: no migration). The one aggregate initialisation, `RenderPose{t.position(),
  t.rotation(), 0}` (`Replication.cpp:1163`), becomes `RenderPose{position, rotation, 0, hostSpace}`; the spawn path
  (~1416) and `InterpolationSystem` assign the field by name. A grep for `RenderPose{` in `src/`, `samples/` and
  `tests/` is part of step 4 (today: that one site). Without this a remote entity would be drawn at old-space positions
  inside the new space for `interpolationDelaySteps` (amendment A4).
- **Extract**: `sourcePose` takes the space from `RenderPose` when the pose comes from it (also for a composed child:
  the `RenderPose` owner's space), else from `Transform`. A space change is a teleport pair (prev = curr) in the new
  space's chunk; the sink draws only the camera's space (S18 culling compares `chunk.space` / `region.space` with the
  camera's).
- The write detector (S16) treats `teleportTo` / `setSubtreeSpace` from writers 1–5 as legitimate writers
  (`noteWritten`); the final comparison of the host + client variant compares `Transform::space` exactly (it is an
  integer field of the flattened `Transform`).

### 5. Saves (§15.1)

- `Transform::space` is saved already (S11), so entities keep their space. `PendingTransition` and `SpaceLink` are
  ordinary saved components.
- **Optional sections** (D7; `SaveFormat.h`, `SaveReader.{h,cpp}`, `WorldSave.cpp`, `SaveDiff.cpp`). The fixed order of
  `parseSave` (`SaveReader.cpp:96`) becomes a table of `{tag, optional, maxVersion}` entries in `SaveFormat.h`
  (`maxVersion`: the highest format version of an optional section this build reads; 0 for the required ones):
  `HEAD, SCHM, SPCE?, GENR?, TERR?, CREC?, ENTS, COMP, EVNT, ACTN, STAT` (? = optional). S21 adds the mechanism and
  `SPCE`; S22 inserts `TERR`, S23 `GENR` and `CREC` at these positions.
  - The reader walks the table: an optional entry whose tag is not the next section of the blob is absent. An unknown
    tag, a section out of order or twice, and a missing required section stay errors with today's message, naming the
    candidates ("unexpected section TERR (expected SPCE or ENTS)").
  - Each optional section starts with its own format version (u32); a version this build does not read is a load error
    naming the section ("SPCE: section format version 2 (this build reads 1)"). `kSaveFormatVersion` stays 1: older
    builds reject a blob with an optional section as an unexpected section, the existing rule for newer saves.
  - `ParsedSave` gets one member per optional section (`std::optional<SpaceTable> spaces` in S21), filled by a parse
    function per tag with the one signature of the recipe, `parseSpaceTable(BinaryReader&, uint32_t version,
    SpaceTable&, std::string& error)`, bounded like the schema strings (at most 65535 rows, names of at most 256 bytes;
    no bytes left over). `struct SpaceTable { struct Row { std::string name; SpaceKind kind; }; std::vector<Row> rows;
    };` (`SaveFormat.h`; `SpaceKind` from S20's `SpaceTemplate.h`, an unknown kind byte is a parse error).
  - `SectionWriter::beginOptional(tag, version)` writes the tag, the length placeholder and the version.
  - `diffSaves` reports an optional section present on one side only, and `SPCE` differences per space ("space 1:
    'cellar' (enclosed) vs 'cave' (enclosed)").
  - **Extension recipe** for S22 (`TERR`) and S23a (`GENR`, `CREC`); each adds, in its own stage:
    1. a tag constant next to `kTagEntities` in `SaveFormat.h` (`kTagTerrain = saveTag("TERR")`, ...);
    2. its entry in the order table at its D7 position with `optional = true` and `maxVersion` (1 at first);
    3. a `ParsedSave` member: either a parsed value (`std::optional<SpaceTable> spaces`, S21) filled by a parse
       function `parseX(BinaryReader&, uint32_t version, X&, std::string& error)` that bounds every count and leaves no
       bytes over, or, for a large section, a **raw range** `std::optional<ParsedSave::Range> terrain` (`{version,
       begin, end}` into the blob, like `ParsedSave::Block`), checked only for its version and length by `parseSave`
       and decoded by the loader's own pass before the commit;
    4. the writer in `World::save` (`WorldSave.cpp`): `s.beginOptional(kTagX, version)` ... `s.end()` between the
       `SCHM` and `ENTS` blocks, in table order, only when the section has content;
    5. the load check in `WorldLoader` before the commit (the world unchanged on an error);
    6. a diff hook: one case in `SaveDiff.cpp`'s optional-section comparison (a function per tag), and the `dumpJson`
       of the section if it has one.
    The reader's dispatch is a `switch` over the tag in `parseSave`, one case per optional entry (its parse function or
    its raw range).
  - **Merge order with S22.** S22 does not depend on S21. If S22 merges first, it adds this mechanism itself with these
    names (the `{tag, optional, maxVersion}` order table, `SectionWriter::beginOptional`, `ParsedSave::Range`, the
    `parseX(BinaryReader&, uint32_t version, X&, std::string& error)` signature, the `switch` dispatch, the diff hook)
    and `TERR`; S21 then adds only `SPCE`: its tag, its order-table entry, `SpaceTable` and `parseSpaceTable`, its
    writer, `checkSpaces` / the prefix rule and its diff case (step 3's mechanism part becomes a no-op).
- **`SPCE`** format 1: space count (u32), then per `SpaceId` the **template name** of that space (u32 length + bytes)
  and the kind (u8). Names may repeat: from S23a several space instances use one template, and each row names its
  template; the unique-name rule of Scope 1 applies to the templates, not to `SPCE` rows (before S23a a repeated name
  can only come from a hand-built blob: the reader accepts it, `SaveSections.SpaceTableNamesMayRepeat`). Written only
  when `spaceCount() > 1`: a single-space save has no `SPCE`. `data/tests/engine/save_v1.bin` is not regenerated and
  still loads; new single-space saves are not byte-identical to it, because `SCHM` lists the two new components. Load
  (before the world changes), the **prefix rule** (shared with S23a's `GENR`): the save's rows must equal the setup's
  first rows; the save has more spaces than the setup, or a name or kind differs at an index, is a load error naming the
  space and both values ("space 1 is 'cellar' in the save and 'cave' in the setup"); fewer spaces is fine (appended
  templates). The kind is compared because an enclosed space loaded as an open-world one (or back) would put its
  entities on or off a terrain; spacing and size are not compared (S22's `TERR` checks the spacing it depends on; entity
  positions stay valid). A save without `SPCE` has one space; nothing is checked. Amendment A11.
- **Space checks on the staged values** (a new `WorldLoader::checkSpaces()`, after the components are staged and
  `Parent` targets checked, before `commit`; `checkSlots` sees only `ENTS`): every alive entity's space is below the
  save's space count (1 without `SPCE`); at most one space-state entity per space; a child's space equals its parent's
  (the hierarchy invariant). Step 5 adds, with the components: `PendingTransition::target` and `SpaceLink::target` are
  below the setup's `spaceCount()`. Each failure is a load error, the world unchanged. S11's check "more than one
  space-state entity" (`WorldLoad.cpp:110`) goes.
- **`restoreStateEntities` per space** (`WorldLoad.cpp` ~622–672): `w.spaceStates` is resized to `spaceCount()` and
  filled by each loaded space-state entity's `Transform::space` (not by slot order: spaces added to a setup get their
  entities at free slots). For a space without one, when a space-state component is registered, a new space-state
  entity is created at the first free slot with its space through `World::makeStateEntity` (writer 6), S11's rule for
  components registered since the save, now per space; an existing space-state entity that lacks a newly registered
  component gets it as today. When no space-state component is registered any more, every space-state entity of the
  save is freed (`checkFreedStateEntities` already refuses those with children, a `PersistentId` or a game object) and
  `spaceStates` stays empty.
- **Staged defaults per (state entity, component)**: today `stateDefaults[id]` holds one staged default per component
  (built before the commit, `WorldLoad.cpp:799–812`, moved in by `addStaged` at ~662), which serves exactly one state
  entity. It becomes `StateDefaults` = one `std::vector<std::unique_ptr<StagedValues>>(kMaxComponents)` per target:
  index 0 the world-state entity, index 1 + s the space-state entity of space s. Before the commit the loader computes
  from the staged slots and component masks which (target, component) pairs the commit will add (a missing state entity
  needs all its components, an existing one only the missing ones) and builds one default for each pair (a throwing
  constructor still rejects the load before the world changes; `SaveRobustness.StateDefaultsAreBuiltBeforeTheCommit`
  gets a two-space case).
- A `PendingTransition::link` that was destroyed before the save stays a dead handle after the load (entities are
  saved as index + generation, the slot's generation differs); `spaceTransitionSystem` uses it only for the event, and
  `diffSaves` compares it as the raw field. No check needed.
- `World::hash()` covers `Transform::space` and `PendingTransition` through `visit` already; the space table is
  configuration and not hashed.

### 6. Entrances and transitions (`engine/components/`, `engine/world/`, `engine/gameplay/`, §12.4)

```cpp
// engine/components/SpaceLink.h: an entrance to another space (§12.4). Replicate::Host (clients show prompts), saved.
struct SpaceLink
{
    static constexpr const char* name = "SpaceLink";
    static constexpr ComponentId id   = ComponentId(EngineComponent::SpaceLink);
    static constexpr uint32_t version = 1;
    SpaceId target       = 0;
    Vec3 arrival         = Vec3::Zero();         // in the target space
    Quat arrivalRotation = Quat::Identity();
    template <class V> void visit(V& v);         // target, arrival, arrivalRotation
};

// engine/components/PendingTransition.h: a requested transition, completed by spaceTransitionSystem (host, phase 1 of
// the next step; from S24 when the arrival is complete). Replicate::Never, saved: a save between the request and the
// next step keeps it (R-STATE-1).
struct PendingTransition
{
    static constexpr const char* name = "PendingTransition";
    static constexpr ComponentId id   = ComponentId(EngineComponent::PendingTransition);
    static constexpr uint32_t version = 1;
    Entity link;                                 // for the event; may be dead when the transition completes
    SpaceId target  = 0;
    Vec3 position   = Vec3::Zero();              // copied from the link at the request
    Quat rotation   = Quat::Identity();
    template <class V> void visit(V& v);
};

// engine/world/Events.h (EngineEvent::SpaceTransition, Replicate::Never): emitted on the host when a transition
// completes; readable in the next step (games: velocity fix-ups of vehicles, quest triggers)
struct SpaceTransitionEvent { Entity entity, link; SpaceId from = 0, to = 0; };

// World (host): starts a transition of a root to the link's target at its arrival pose. Between steps applied at once,
// during a step a command (makeCommand, applied at the phase flush).
void World::requestTransition(Entity e, Entity link);

void spaceTransitionSystem(World& world, double dt);   // Phase::Input, RunOn::Host, engine-registered (always)
void spaceLinkSystem(World& world, double dt);         // Phase::Gameplay, RunOn::Host, optional (games register it)
```

- `requestTransition` takes any root (the design doc's parameter is named `player`; vehicles and the S16b driver
  entities move too, amendment A7). It copies the link's target and arrival pose at the request (deterministic; a link
  destroyed before completion does not cancel).
  - Asserts (code errors): on a client, a dead or reserved entity, a child, a state entity, a dead link or one without
    `SpaceLink`.
  - Refused with a log line under `world`, in every build (data): the link's target `>= spaceCount()` (a hand-written or
    tool-edited `SpaceLink`); the entity already has a `PendingTransition` (the first request wins).
  - The command applied at the flush re-checks that the entity is an alive root without `PendingTransition` (it may have
    died or got a parent in the same phase) and drops the request with a log line otherwise.
- `spaceTransitionSystem`: added as the first line of `registerEngineSystems` (`TransformSystem.cpp:143`):
  `systems.add(Phase::Input, "space_transition", spaceTransitionSystem, RunOn::Host);` (phase-1 systems run after the
  received messages are applied; no other engine system is in phase 1). For each entity with `PendingTransition` in
  view order it revalidates — the entity is now a child or a state entity, or the target is `>= spaceCount()` (only
  from a tool's `set_component`; the load checks saves) — and then removes the component with a log line (no assert:
  the state may come from a save or a tool). Otherwise: `moveToSpace(e, target, position, rotation)`, emit
  `SpaceTransitionEvent`, remove the component (a command, applied at the phase-1 flush). Before S24 every request
  completes in phase 1 of the step after the request (§12.4 "Before S24"); S24 makes the system wait for
  `requestArrival` and hold the entity.
- `spaceLinkSystem` (optional, R-MECH-1): for each `InteractEvent` of the previous step whose target has `SpaceLink`,
  whose actor is an alive root, and both are in the same space: `requestTransition(actor, target)`. Events across
  spaces are ignored with a log line (a game bug: the interaction was decided on a stale pose). Who interacts with what
  (key, distance, prompts) stays game policy: games emit `InteractEvent`. Events are readable only in the step after
  their emission (`EventQueues`, §7.7), so the system's position within phase 4 changes nothing.
- **Step latency of the interact chain** (fixed; the scenario bounds below depend on it). A `PlayerAction` queued on the
  host between steps k − 1 and k (`queueAction`; a client's action arrives with the transport's delay and is queued the
  same way):
  - step k: `deliverActions` makes it an `ActionEvent`, readable in step k; the game's interact system (phase 4) emits
    the `InteractEvent`;
  - step k + 1: `spaceLinkSystem` reads it and calls `requestTransition` (a command, `PendingTransition` added at the
    phase-4 flush);
  - step k + 2: `spaceTransitionSystem` moves the entity in phase 1; the host's `Transform::space` is the target from
    phase 1 of step k + 2 on, the grid from phase 6;
  - step k + 3: `SpaceTransitionEvent` readable.
  A game that emits `InteractEvent` directly (no action) saves one step; `requestTransition` between steps moves in
  phase 1 of the next step.
- IDs (D10): `EngineComponent::SpaceLink`, `PendingTransition` are appended after `RenderInstance` and
  `EngineEvent::SpaceTransition` after `Action` (`world/Events.h`); S21 and S22 both append, and whichever merges
  first takes the next IDs (existing IDs unchanged). Registered in `registerEngineComponents` /
  `registerEngineEvents`. `data/tests/engine/schema.json` is regenerated in this stage with only the schema test
  (`SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_save --gtest_filter='SaveFormat.EngineSchemaIsCommitted'`; without
  the filter `SaveFormat.CommittedSaveLoads` would also read `SAIGA_UPDATE_GOLDEN` and rewrite `save_v1.bin` /
  `save_v1.json`, which must stay as committed).

### 7. World API and MCP tools (§16.2)

- `spawn(space)`: any `space < spaceCount()` (the error "space: only space 0 exists (S21)" at `WorldApi.cpp:1455`
  goes).
- New operation and tool `move_to_space(entity, space, position, rotation?)` (edit; rotation default: the current one):
  `World::moveToSpace` with the API's errors (dead, child, state entity, unknown space, client refusals as S16's
  `netEditRefusal`; a client-owned root on the host is allowed: it is the engine's command path). Returns `{entity,
  previous: {space, position, rotation}}`; `previous` is the undo payload (a `move_to_space` with it restores the
  entity). Logged under `api`.
- The `space` key is already refused: `set_transform` rejects it as an unknown argument (`WorldApi.cpp` ~595–602) and
  `set_component` refuses `Transform` (`WorldApi.cpp:280`). Only the texts are extended to name `move_to_space`
  ("unknown argument "space" (...; a space change: move_to_space)", "Transform: every entity has one (use
  set_transform; its space: move_to_space)").
- `describe_world.spaces`: per space id, name, kind, `size_chunks`, `has_terrain` (`SpaceTemplate::hasTerrain()`,
  S20/E6), sea level (only with terrain), `site_origin` (enclosed: `enclosedOrigin`, E4). `world_status.spaces`: per
  space the entity count and the players in it. `list_entities(space?)`: a filter without `near` (entities in that
  space, index order).
- `requestTransition` gets no tool: agents use `queue_action` / `emit_event` (the game's interact path, which is what a
  player does) or `move_to_space`.
- App tools: `set_camera(space?)` (default: the current view space; validated against `spaceCount()` of the app's view
  world, which is the client world with `--coop`; the hold and the ±1e6 m range as S14); `app_status.space_loading`
  (`null` or `{space, name, progress, frames}`).
- `game_set_camera` (`WorldMcp`, in `saiga_vulkan`, D2: no engine types) is unchanged and keeps the view camera's
  current space; the engine's `set_camera(space)` switches spaces.

### 8. Rendering per space (`RenderSink`, `RenderScene`, S20's `TerrainSystem`, §12.4 "each peer renders only the space of its own camera")

- `SpaceRenderSettings {bool openWorld = true; vec3 background = vec3(0.02f, 0.02f, 0.025f);}` in
  `saiga/vulkan/scene/RenderSceneTypes.h`; `RenderSink::setSpaceSettings(SpaceId, const SpaceRenderSettings&)` (new;
  `VulkanRenderSink` forwards to `RenderScene::setSpaceSettings`, `RecordingRenderSink` records). The World forwards one
  per `SpaceId` (for `0 .. spaceCount() - 1`, from `spaceTemplate(space)`: `openWorld = kind == OpenWorld`, `background`
  from the template; from S23a several spaces may share a template, and each still gets its own call) whenever a sink is
  set, as the mesh tables (S19b). A space without settings is drawn as open world (the vulkan samples and tests that
  never call it). There are no ambient or fog settings per space before S36: in open-world camera spaces fog, sky and
  sun come from the game's `PrototypeSettings` through S20's `setPrototypeEnvironment`, and the sea level travels in
  `TerrainPyramidData` (S20 A9).
- `RenderScene` uses the camera space's settings. Open world: as S20 (terrain, grass and water when the GPU terrain set
  is this space's; the sky; the game's `PrototypeSettings`). **Enclosed** camera space, with named constants until S36:
  - no terrain, grass, water or sky pass; the scene pass clears to `background` (linear, through the HDR path like
    `clearColor`), which replaces the sky;
  - the instances' lighting block is built from `kEnclosedLighting = PrototypeSettings{}` with `fog = false`: sun
    azimuth 35°, elevation 38°, intensity 1.0, ambient 0.25, albedo 0.55 (the `PrototypeSettings` defaults,
    `PrototypeWorld.h:31`), independent of the game's `setPrototypeEnvironment` (whose sun and fog belong to the
    outdoors); wind phases unchanged (`setTime`);
  - instance sets and dynamic instances of other spaces are already culled (S18).
- **GPU terrain set: only S21's delta** (E1; S20 owns the set, its rebuild with `deferDelete`, the implicit drop of
  tiles and masks when the sink receives a pyramid of another space or layout, the `TerrainSystem` space-switch path
  of `updateView` step 2 with lazily generated, CPU-cached pyramids, and a new sink receiving the streamed space only):
  1. **Terrain-less target**: when the camera's new space has `hasTerrain() == false` (enclosed or hook-less open
     world), `TerrainSystem` sends S20's `clearTerrain()` at the switch (E1; S20's trigger in `updateView` step 2, only
     if a set was sent) and requests nothing; `RenderScene` then holds no set and draws no terrain (the old set goes to
     `deferDelete`). So leaving a terrain space for a terrain-less one sends exactly one `clearTerrain()`, and a switch
     between two terrain-less spaces sends none. No per-tile removal loop and no "everything removed" assert. S21
     changes no `TerrainSystem` code: the trigger, the store per space and the lazy pyramids are S20's; S21 adds the
     multi-space tests.
  2. **Layouts differ between spaces**: space 0 at 1 m and another open-world space at 2 m (or another `sizeChunks`);
     the switch and the switch back each build a set for that layout (S20's rebuild path; S21 adds the multi-space case
     to S20's test, below).
  3. **Pyramid cache reuse**: a space viewed before re-sends its CPU-cached pyramid without a generation; only its tiles
     are streamed again (D5 drops them on every switch). Consequence (stated, not changed in S21): every exit from a
     cave or cellar into an overworld **with** terrain re-streams the overworld's tiles around the camera from the hook
     and shows a short loading screen (readiness < 1, Scope 9); a flat or enclosed target shows none. Keeping the old
     space's tiles is S25's space residency question. S22 keeps its pyramid dirty boxes per space across switches and
     merges its edits at send time (D3), so a switch back re-sends the edited pyramid.
- **Readiness** (E8): `double World::spaceReadiness(SpaceId space, const Vec3& camera, double radius) const` is the one
  extension point; in S21 it returns S20's `terrain().readiness(space, camera, radius)` (the fraction of the desired
  tiles within the radius plus the pyramid that have been sent; 0 while the pyramid is not sent; 1 for a space without
  terrain; monotone only for a fixed camera). S23b adds its content sets inside it, S27 far content; the app never calls
  `TerrainSystem::readiness` directly.

### 9. Loading screen (`EngineApp`, `engine/app/LoadingScreen.{h,cpp}`, §12.4 step 2)

- The app keeps the space of the view camera of the last rendered frame. `renderFrame`: `beforeRender`, then S20's
  `updateView(view.space, view.position)` (or `completeView` at a complete point); when `view.space` differs from the
  last frame's, the app computes `r = world.spaceReadiness(view.space, view.position, setup.loadingRadius)` after the
  update and starts a load **only if r < 1** (D6: an enclosed space, a hook-less space or a still-resident one shows
  nothing, no one-frame flash). `SpaceLoading {SpaceId space; double progress; uint64_t frames;}`, `const SpaceLoading*
  EngineApp::spaceLoading() const` (null when none).
- While a load runs, each rendered frame computes `r` for the current view camera and sets `progress = max(progress,
  r)` (a running maximum: the bar never moves back when the camera moves; E8). The load **ends** at the first of:
  1. `r >= 1` and `world.terrain().terrainUploadsPending() == 0` (S20's forwarding of the sink's value: the
     `RenderScene`'s queue bytes through `VulkanRenderSink`, 0 without a sink; the sent tiles are recorded, so the next
     frame shows them). The rule is a function `bool loadEnds(const SpaceLoading&, double r, size_t pending)` in
     `LoadingScreen.cpp`, so a test pins it with a `RecordingRenderSink` whose `uploadsPendingForTests` is set
     (`LoadingRules.EndsOnlyWithoutPendingUploads`);
  2. a complete point ran this frame (S20: the first frame, the frame before a screenshot capture, an MCP `screenshot`,
     a measured benchmark frame): `completeView` makes the desired set resident and sent for this camera, so `r` is 1
     by definition, and the capture itself waits for `terrainUploadsPending() == 0` (S20); the app ends the load at
     once instead of evaluating rule 1;
  3. `frames >= setup.loadingMaxFrames` (a moving camera whose desired set keeps changing): ended with a log line under
     `app` ("loading of space 'islands' ended after 600 frames at 0.83").
  A change of `view.space` during a load starts the rule again for the new space (a new load if its `r < 1`). There is
  no restart for camera moves or evictions inside the same space. Local and visual only: the simulation does not wait
  (before S24 nothing holds the player; §12.4 "Before S24").
- `EngineAppSetup` (game policy, R-MECH-1): `double loadingRadius = 256`; `uint64_t loadingMaxFrames = 600`;
  `bool showLoadingScreen = true` (false: the load is tracked and reported, nothing is drawn);
  `std::function<void(EngineApp&, const SpaceLoading&)> loadingScreen` (null: the engine's default
  `drawLoadingScreen(app, loading)`), called inside the ImGui frame while a load runs. The default (ImGui until S46): an
  opaque background over the whole viewport through the background draw list, the space's name and a progress bar.
  Hidden with the GUI (G) like every ImGui window; `screenshot(gui: false)` shows the scene.
- Not with `--no-render` (no visual data, `spaceLoading()` stays null); with `--coop` the client world's camera decides
  (the hidden host renders nothing). The "Engine statistics" window (#66's layout) shows the camera space and a running
  load (space, progress, frames); the frame statistics are unchanged (a switch frame's set rebuild shows up in the
  stutter metrics like any long frame).

### 10. Tech demos (§18.4)

- **Platformer: an enclosed cellar** (shows enclosed spaces, entrances, a client-owned hero's transition in `--coop`,
  the per-space camera and render settings).
  - `spaceTemplates = {SpaceTemplate{} (S20's flat "overworld", unchanged), SpaceTemplate::enclosed("cellar", 4)}` with
    a dark `background`. The cellar level sits at the cellar's site origin `enclosedOrigin(4)` = (640, 0, 640) (D12,
    E4): a floor, walls, two platforms (`Level::cellarLevel()`). `PlatformerContext::levels` is indexed by `SpaceId`;
    `characterControllerSystem` and the hero's kill height read the level of the hero's space.
  - Two door entities (`SpaceLink` + `RenderInstance "platformer/door"`): one on the start island (to the cellar), one
    in the cellar (back, arriving next to the first door).
  - An `interact` action (`PlatformerAction::Interact = 2`, key E in `data/platformer/input.json`) and a game system
    `platformerInteractSystem` (host, phase 4) that turns its `ActionEvent` into an `InteractEvent` with the nearest
    door within 1.5 m in the hero's space; the game registers the engine's `spaceLinkSystem` (phase 4; it reads the
    previous step's events, so the chain is the three-step latency of Scope 6).
  - The restart becomes `moveToSpace(hero, 0, start, yaw 0)` (from the cellar back to the start). For a client-owned
    hero the order stays S16b's: `RestartRun` first, then the `OwnerCommand` (now with space 0) in the same step; the
    owner applies both in phase 1 (the pilot's waypoint reset, then the move).
  - `LevelGeometry::create` takes a `SpaceId` (instance sets of the cellar in `RegionId{space 1}`); the renderer passes
    the focused hero's `WorldPose::space` to `setViewCamera` and the cosmetic body parts (the poses already carry the
    space). The cellar shows no loading screen (an enclosed space is ready at once, Scope 9).
- **`--cellar`** (added to `engineMain`'s flag list: `{"--demo", "--cellar"}`) sets `PlatformerContext::demoRoute =
  Level::cellarDemoRoute()`. The route is setup data in the context that the host and the client worlds share (built
  from the same setup in `--coop`), like the level, so it is not a saved component (review round 1 dropped
  `DemoRoute`; R-STATE-1 holds because a save is loaded with the same setup).
  - `struct DemoWaypoint {SpaceId space; Vec3 position; bool door;}`. Without `--cellar` the route is the level's
    waypoints in space 0 (unchanged: `demo_reaches_goal` keeps 7.25 s).
  - **Minimum stay**: the cellar part of the route (arrival, over both platforms, to the exit door) is laid out so that
    the hero spends **at least 240 steps** (4 s) in space 1 in every mode; no wait state is needed (the route is longer
    than 240 steps of running). `demo_runner_visits_the_cellar` and `pilot_visits_the_cellar` assert the stay (steps
    with `Transform::space() == 1` on the host, ≥ 240) and log the entry step and the stay.
  - `runnerInput(w, e, waypoint)` (`PlatformerWorld.cpp:86`, shared by the host's `DemoRunner::update` and the client's
    `platformerDemoPilotSystem`) walks to `route[waypoint]` while the hero is in that waypoint's space. At a door
    waypoint within 1 m it queues `PlayerAction{e, Interact}` every 30 steps while the hero stays in the door's space
    (on the host directly; on a client the pilot sends it like the restart action at `PlatformerWorld.cpp:128`; the
    host drops repeats because the first request wins). The waypoint advances when the hero's `Transform::space`
    equals the next waypoint's space.
  - `RestartRun` resets `DemoPilot::waypoint` to 0 before the `OwnerCommand` (unchanged), so a restart from the cellar
    continues at waypoint 0 in space 0.
  - **Pre-simulation**: frames are not steps: after `app.stepAll(N)` the app renders 60 frames before the
    `SAIGA_SCREENSHOT` capture and steps by wall-clock time meanwhile (about 60 steps at 60 Hz, more or fewer with a
    slow or fast frame). N is therefore pinned to put the expected capture step, N + 60, in the **middle of the stay**:
    `N = entry + stay / 2 − 60`, per mode (both recorded as constants in `main.cpp` and in Results):
    - solo (`--demo --cellar`): `entry` and `stay` of the host's runner from `demo_runner_visits_the_cellar` (Plain
      variant);
    - co-op (`--coop --demo --cellar`): the client's hero, from `pilot_visits_the_cellar` in the HostClientDelay variant
      (delay 2 = `--coop`'s default), which includes the pilot's start lag (its first route step after the join and the
      action's transport delay) and the command delay of the transitions. With a stay of ≥ 240 steps the capture stays
      in the cellar for a frame-to-step drift of ±120 steps.
    Checked locally in both modes: the app's capture log line names the view camera's space (S21 adds it: "screenshot
    of frame 60 (view space 1) written to ..."), which must be 1, and the screenshot shows the cellar.
  - Scenarios (all variants): `cellar_round_trip` (hero of `defaultPeer()`: interact at the door, in space 1 at the
    arrival pose, walk to the exit door, back in space 0 at its arrival pose), `demo_runner_visits_the_cellar` (the
    host's `DemoRunner` with the cellar route reaches the goal after the detour; the time is recorded when implemented)
    and `pilot_visits_the_cellar` (a hero of `defaultPeer()` with `DemoPilot` and the cellar route: in the network
    variants the pilot runs on the client, enters space 1 and is back in space 0 within a bounded step count).
- **`game_world`** (the world prototype on S20's height tiles): a second open-world template `"islands"`: S20's
  `prototypeSpaceTemplate()` (built in `samples/games/common`, D2) with this name and the hook `islandsHeight(seed, x,
  z) = prototypeHeight(seed, x, z) - 40 * 1024` (in `samples/games/world/main.cpp`; the prototype noise lowered by
  40 m, so the sea makes islands). The game's ImGui window gets a "Space" combo that calls `app.setViewCamera(space,
  pose)` at the same pose, and the walk mode uses `world.terrainHeight(cam.space, ...)` (`main.cpp` ~158); the start
  camera and the benchmark keep space 0 (`main.cpp` ~122, ~158), so the islands pyramid is never generated in
  `--benchmark` (D5: lazy). Local check of the set rebuild and the loading screen with real tile streaming: the combo
  or `set_camera {space: 1, ...}`, screenshots during and after the load, `app_status.space_loading` progress.
- **Flight and racing**: no feature change (one space each); they pass the vehicle's `WorldPose::space` to
  `setViewCamera` instead of 0. A flight portal to another space would need the vehicle-and-driver transition plus a
  velocity fix-up (game policy through `SpaceTransitionEvent`) without showing more of the engine than the platformer
  and `game_world` do.

## Out of scope

- The arrival mechanism, held/frozen/invisible entities during a transition, simulation islands, island limits, the
  arrival-chunk range check (S24); space unloading (S25); chunk snapshots and the client's hold until they are applied
  (S26); interest per space and the space-state interest rule (S32).
- Authored enclosed geometry from prefabs and the world plan's space instances (S23); static mesh colliders and one
  Jolt system per space (S28); ambient, fog and the full environment per space (S36); enclosed water (S39).
- Runtime creation or deletion of spaces (§20); rendering two spaces at once (§12.4); GPU terrain data of more than one
  space at a time.
- A game UI loading screen (S46; ImGui until then).

## Design notes

- **`PendingTransition` as a component** instead of a World-side queue: R-STATE-1 (a save between request and
  completion keeps it, checked by the save/load variant and a unit test), it is hashed, visible to tools (`inspect`),
  and S24 extends the same component to "held until the island is complete".
- **`moveToSpace` applies at once** (not a command): it is a pose write like `teleport`, which systems and game objects
  already do during a step; a command would leave the entity in the old space for the rest of the phase and complicate
  `OwnerCommand` ordering. Grid queries in the same step see the old space until phase 6, the existing rule for every
  move (§9.5).
- **One space primitive, a closed list of writers**: `setSubtreeSpace` / `teleportTo` are the only code that writes
  `Transform::space`. Descendants' spaces are set immediately (a child is never observable in another space than its
  parent), their poses in phase 6 (S08's path). Attaching follows the same rule, which also covers every client-side
  parent path, so phase 6 only asserts.
- **The space-state entity's identity is its `Transform::space`**: already saved, needs no new field, and load maps by
  it rather than by slot order.
- **One GPU terrain set** (D5, E1; S20's mechanism): the S20 atlas budget holds for one space; a switch rebuilds the
  set (the layouts may differ) instead of keeping several atlases, and a terrain-less camera space clears it with one
  explicit call. Other spaces cost only their CPU pyramid, generated on first view. The price is a re-stream of the
  overworld's tiles on every exit from an enclosed space (Scope 8), accepted until S25.
- **Loading starts only if the target is not ready** (D6), keyed on the view camera's space, not on a replicated
  "transition started" signal: before S24 the move happens one step after the request, so the camera switch is the
  earliest useful moment on every peer, and it also covers tool cameras and spectators. S24, which inserts a wait before
  the move, adds a signal for the player's peer (e.g. `PendingTransition` replicated to the owner) and starts the
  screen at the request.
- **Per-space render settings through the sink**, forwarded by the World: headless worlds and the host keep the same
  tables, a sink set later gets them (the S19b pattern), and `RenderScene` stays free of engine types. They hold only
  what has a data source before S36 (the kind and the enclosed background).
- **Optional save sections** (D7) keep the golden `save_v1.bin` loading unchanged (no `SPCE` in a single-space save,
  the file is not regenerated) and give S22/S23 a place for their data without bumping the blob version; each section
  versions itself.
- **A space table in the save** guards the append-only rule cheaply until S23 replaces templates by pinned plan
  instances (then the table lists the plan's instances).
- **The demo route is setup, not state**: the host's runner and the client's pilot read the same route from the shared
  context, so no component needs saving or replicating.

## Implementation steps

Each step builds with the CI recipe (and samples ON for steps 9–10), passes `ctest`, and is a reviewable commit.

1. **Space configuration.** Several templates, the extended validation (enclosed via `hasTerrain()`, `seaLevel` and
   `baseSpacing` ignored for enclosed), `SpaceTemplate::enclosed` / `enclosedOrigin`, `background`, `spaceCount`, the
   accessors for every space, `terrainHeight` of enclosed spaces, `spawnIn` / world API `spawn` for every space. S20's
   `Terrain.TemplateValidation` drops its "two templates" and "Enclosed" cases and the constructor stops naming S21.
   Tests: `SpaceConfig.*`, `SpaceSpawn.*` in the new `tests/test_engine_spaces.cpp` (registered in
   `tests/CMakeLists.txt` inside `if (MODULE_ENGINE)`).
2. **`moveToSpace` in one world.** `Transform::setSpaceInternal`, `collectSubtree`, `setSubtreeSpace`, `teleportTo`,
   `moveToSpace` with its asserts, the attach rule in `addNow<Parent>` / `reparentNow`, the phase-6 assert,
   `ownerTeleport` on the same path (single-world part). Until step 4, `OwnerCommand` is still version 1 without a
   space, so `moveToSpace` of a spawned client-owned root **to another space** asserts ("moveToSpace: a client-owned
   root changes space only with OwnerCommand v2 (S21 step 4)"); the same space keeps S16's `ownerTeleport` path. Step 4
   removes the assert. Tests: `MoveToSpace.*`, `MoveToSpaceDeathTest.*`.
3. **Space-state entities and saves.** One space-state entity per space through `World::makeStateEntity`; the
   optional-section mechanism (table with `maxVersion`, `beginOptional`, per-section versions, raw ranges,
   `diffSaves`); `SPCE` with the prefix rule; `checkSpaces` without the transition targets; `restoreStateEntities`
   per space with staged defaults per (state entity, component). Tests: `SaveSections.*` and
   `SaveFormat.CommittedSaveLoadsWithAppendedSpaces` in `test_engine_save.cpp`, `SpaceState.*`, `SpaceSave.*` except
   `PendingTransitionSurvives` and the target cases of `CorruptStates` (step 5). Nothing in this step uses `SpaceLink`,
   `PendingTransition` or `requestTransition`.
4. **Replication.** The root's space in spawn/snapshot apply, `OwnerCommand` v2 and `ownerMove` (the step-2 assert for
   client-owned roots goes), the owner-state space rule, interpolation samples and `RenderPose::space`, the extract's
   space from `RenderPose`. Tests in `test_engine_replication.cpp` (`SpaceReplication.*`) and `test_engine_extract.cpp`
   (`ExtractSpaces.*`). Run `test_engine_net_benchmark` before and after (R-TEST-2).
5. **Entrances and transitions.** `SpaceLink`, `PendingTransition`, `SpaceTransitionEvent` (IDs per D10, `schema.json`
   regenerated), `requestTransition`, `spaceTransitionSystem` (first in `registerEngineSystems`), `spaceLinkSystem`;
   `checkSpaces` gains the `PendingTransition` / `SpaceLink` target checks. Unit tests `Transition.*`,
   `SpaceSave.PendingTransitionSurvives` and the target cases of `SpaceSave.CorruptStates`; the engine scenarios
   `uses_a_space_link` and `objects_change_space` (all six variants) in `test_engine_scenarios.cpp` with
   `spacesScenarioSetup()`.
6. **World API and tools.** `move_to_space`, `spawn(space)`, the `space` key texts, `describe_world.spaces`,
   `world_status.spaces`, `list_entities(space)`. Tests in `test_engine_world_api.cpp` through `handleMessage`.
7. **Rendering per space.** `SpaceRenderSettings`, `RenderSink::setSpaceSettings`, the World's forwarding,
   `RenderScene`'s per-camera-space passes with `kEnclosedLighting`, `World::spaceReadiness`; for S20's `clearTerrain()`
   trigger and space-switch path only tests (no `TerrainSystem` change). Tests: `VulkanSceneSpaces.*`
   (`test_vulkan_scene.cpp`), the multi-space case of S20's `VulkanTerrain.NewSetOnOtherSpaceOrLayout`
   (`test_vulkan_terrain.cpp`), `TerrainSpaces.*` (`test_engine_terrain.cpp`), forwarding in `test_engine_extract.cpp`.
   Run `game_world --benchmark flyover` and `forest` before and after.
8. **Loading screen.** `SpaceLoading` (running maximum, the three end rules in `loadEnds`), `EngineApp::spaceLoading`,
   `EngineAppSetup::{loadingRadius, loadingMaxFrames, showLoadingScreen, loadingScreen}`, the view space in the capture
   log line, `drawLoadingScreen`, `set_camera(space)`, `app_status.space_loading`, the statistics window line.
   Tests in `test_engine_app.cpp`.
9. **Tech demos.** The platformer cellar, `--cellar` and the route, its scenarios; `game_world`'s `islands` template and
   space combo; flight and racing pass the camera space. Local: `--scenarios` of the three demos with all variants,
   screenshots.
10. **Documents.** The amendments below in `docs/engine_design.md` (§3 rows, §7.4, §7.8, §7.9, §7.10, §8.1, §8.2,
    §10.1, §10.3, §12.2, §12.4, §13.3, §13.4, §15.1, §16.2, §17.2, §18.4, §22 S21 row status); `AGENTS.md`: the test
    count from the configure's `Test enabled` lines (67 on the S19b master, plus S20's new binaries, plus
    `test_engine_spaces`), the layout lines for `components/` (`SpaceLink`, `PendingTransition`), `world/`
    (`SpaceTransitionSystem`), `save/` (optional sections), `app/` (`LoadingScreen`), and the `--demo --cellar`
    screenshot command; this plan's Design changes and Results.

## Tests

`tests/test_engine_spaces.cpp` (new, `saiga_test(test_engine_spaces.cpp "saiga_engine")`), a setup with three templates
(space 0 open world with a cheap integer test hook at spacing 2, space 1 `SpaceTemplate::enclosed("cellar", 4)`, space
2 open world with another integer hook at spacing 1):

- `SpaceConfig.CountTemplateAndSeed`: `spaceCount`, names, kinds, `spaceSeed(0) == seed`, `spaceSeed(2) ==
  hashCombine(seed, 2)` (S20's rule, now for three spaces); `enclosedOrigin(4) == 640`, `enclosedOrigin(1) == 128`,
  `enclosedOrigin(5) == 640`. `SpaceConfig.HooklessOpenWorldIsFlat` (D1: valid, `terrainHeight` 0).
  `SpaceConfig.HandWrittenEnclosedIsValid` (`kind = Enclosed` on a default `SpaceTemplate`: S20's `seaLevel = 0` and
  `baseSpacing` are ignored, no error). `SpaceConfigDeathTest.InvalidTemplates` (`SAIGA_EXIT_ERROR`, every build):
  duplicate and empty names, an enclosed template with `prototypeHeight` (the message names
  `SpaceTemplate::enclosed`), enclosed `sizeChunks` 0 and 513 (messages matched).
- `SpaceConfig.TerrainHeightPerSpace`: spaces 0 and 2 follow their own hooks at seeded points (with the space's seed and
  spacing); the enclosed space returns 0; the kinematic mover in space 2 walks on space 2's surface.
- `SpaceSpawn.SpawnInEverySpace`: `spawnIn(1, ...)` gives `Transform::space() == 1`; `GameObject::spawn` from an
  object in space 2 spawns in space 2; `queryRadius(1, ...)` finds it after a step, `queryRadius(0, ...)` does not.
  `SpaceSpawnDeathTest.UnknownSpace`.
- `MoveToSpace.RootAndDescendants`: a root with a child and a grandchild (`LocalTransform`s), moved between steps:
  - at once: every descendant's space is the new one (`setSubtreeSpace`), the root's `teleports` counter grew by
    exactly 1 and the root's `wasTeleported()` is set; the descendants' teleport flags are **not** set yet
    (`setSpaceInternal` sets only the space and `changedThisStep`);
  - after the next phase 6 (read by a test system in phase 7 of the next step): the descendants' poses equal parent pose
    ∘ local and their `wasTeleported()` is set (`updateChildren` teleports a child of a teleported parent); the grid
    finds all three only in the new space; tick player cells move with a moved player.
- `MoveToSpace.DuringAStep`: from a system (phase 4) and from `GameObject::update`; the space (also the descendants')
  is visible to later systems of the step, the descendants' flags after phase 6 of the same step.
  `MoveToSpace.SameSpaceIsATeleport`. `MoveToSpace.AttachTakesTheParentsSpace`: `add<Parent>` (between steps and as a
  command) and `reparentNow` move the child and its own children into the parent's space, flagged as teleported.
- `MoveToSpaceDeathTest.Refusals` (`SAIGA_ASSERTS`): a child, a state entity, an unknown space, a reserved entity,
  inside `parallelEach`, on a client for a replicated entity.
- `SpaceState.OneEntityPerSpace`: with `registerSpaceState<T>`: three entities, `spaceState<T>(i)` distinct and
  writable, each entity's `Transform::space() == i`; a world without space-state components has none.
- `SpaceSave.RoundTrip`: entities in three spaces, a hierarchy across a move, `save` / `load` / `save` equal bytes,
  spaces and grid queries restored. `SpaceSave.SingleSpaceHasNoSpaceTable`: a one-space world's save has no `SPCE`.
  `SpaceSave.PendingTransitionSurvives` (step 5): `requestTransition` between steps, save, load into a fresh world, one
  step: in the target space at the arrival pose, `SpaceTransitionEvent` readable in the step after.
  `SpaceSave.SpaceTable`: a save from the three-space setup loads into a four-space setup (the prefix rule; the new
  space-state entity created), fails in a two-space setup, with a renamed space 1 and with space 2 changed to enclosed
  (the messages name the space and both values); repeated names are covered by the reader-level
  `SaveSections.SpaceTableNamesMayRepeat`. `SpaceSave.StateEntitiesPerSpace`: the mapping by `Transform::space` when the
  save's slots are not in space order; two spaces added since the save each get their entity with their own staged
  defaults (a space-state component with a counter in its default constructor: two distinct values, no shared default);
  a space-state component registered since the save is added to every existing space-state entity; with no space-state
  component registered any more, all of them are freed. `SpaceSave.CorruptStates`: two space-state entities of one
  space, an entity in an unknown space, a child in another space than its parent; from step 5 a `PendingTransition` and
  a `SpaceLink` with a target out of range: load errors, the world unchanged.
- `Transition.CompletesInTheNextStep`: between steps and from a phase-4 system; in phase 1 of the next step the entity
  is in the target space at the link's arrival pose (pose copied at the request: a link edited or destroyed after the
  request does not change it), `PendingTransition` removed, `SpaceTransitionEvent{entity, link, from, to}` readable in
  the following step. `Transition.FirstRequestWins` (two links in one step; the second is logged).
  `Transition.TargetOutOfRangeIsRefused` (a `SpaceLink` written with target 7: logged, nothing added, every build).
  `Transition.RevalidatedAtCompletion`: the entity got a parent between the request and the completion, and a
  `PendingTransition` set by `set_component` with an unknown target: removed with a log line, no move.
  `Transition.SpaceLinkSystem`: an `InteractEvent` with a `SpaceLink` target requests the transition; an event whose
  actor is in another space is ignored and logged; an event without `SpaceLink` is ignored.
  `TransitionDeathTest.Refusals` (`SAIGA_ASSERTS`): on a client, a child, a dead link, a link without `SpaceLink`.

`tests/test_engine_save.cpp`: `SaveSections.OptionalAbsentAndPresent` (a blob with and without `SPCE` parses),
`SaveSections.OrderErrors` (an optional section after `ENTS`, twice, an unknown tag: errors naming the expected
sections), `SaveSections.UnknownSectionVersion` (`SPCE` version 2: the message names it), `SaveSections.DiffReportsSpaces`
(`diffSaves` of saves with different space tables), `SaveSections.SpaceTableNamesMayRepeat` (a hand-built blob whose
`SPCE` has two rows with the same template name parses, `dumpJson` / `diffSaves` show both rows; a reader-level test,
because S21's setups cannot produce it before S23a: template names are unique and `SpaceId` = template index). The
existing `SaveFormat.CommittedSaveLoads` keeps loading the unchanged `save_v1.bin` (not regenerated).
`SaveFormat.CommittedSaveLoadsWithAppendedSpaces` (next to `SaveFormat.CommittedSaveLoads`, the file that owns the
golden setup `WorldSetup{}` + `goldenWorld`): `save_v1.bin` (no `SPCE`) loads into the golden setup with two appended
templates (S20's default `"overworld"` first, then `SpaceTemplate::enclosed("cellar")` and `("cave")`), and the
loaded world's `dumpJson` equals `save_v1.json` (no space-state components in the golden setup, so no new entities).
`SaveRobustness.StateDefaultsAreBuiltBeforeTheCommit` gets the two-space case (a throwing default of a space-state
component with two spaces missing their entity: load error, the world unchanged).

`tests/test_engine_scenarios.cpp` (engine scenarios, all variants: Plain, Repeat with `varyThreads`, SaveLoad with
`strictContinuation`, HostClient, HostClientDelay, HostClientAsync). The two new scenarios use a dedicated
`spacesScenarioSetup()`; `scenarioSetup()` is unchanged. Wiring: they are registered by their own
`registerSpacesScenarios(ScenarioList&)`, and the file's `Case` gets two function pointers, `WorldSetup (*setup)()`
and `void (*registerScenarios)(ScenarioList&)`. `allCases()` collects the cases of `registerEngineScenarios` with
`scenarioSetup` and of `registerSpacesScenarios` with `spacesScenarioSetup`; `EngineScenario.Passes` builds its list
with `GetParam().registerScenarios` and runs `runScenario(GetParam().setup(), *entry, variant, options)` (the test
names stay `<scenario>_<variant>`). The setup:

- space 0 S20's flat default template, space 1 `SpaceTemplate::enclosed("cellar", 4)` (site origin (640, 0, 640), E4);
- `TransitionLog`, a test-only **world-state component** (`registerGlobal`, like the file's `ActionLog`), not a
  process global: each world of each variant run (Repeat's two runs, SaveLoad's loaded world, the host of the network
  variants) has its own, it is saved with the world, and nothing leaks between runs in one binary. Fields: per
  transition `{entity, link, from, to, step}` and the steps at which the interact system saw action 20. A test system
  (phase 4, host) appends from `SpaceTransitionEvent`. The event is emitted in phase 1 of the move's step and readable
  only in the next step (§7.7), so the system records `step = world.stepIndex() - 1` (the step that moved the entity;
  the check "move step minus interact step is exactly 2" uses this value);
- a test-only `interactSystem` (phase 4, host): an `ActionEvent` with action 20 emits an `InteractEvent` towards the
  nearest `SpaceLink` within 2 m in the actor's space (ties by entity index) and records the step;
- the engine's `spaceLinkSystem` (phase 4; its order relative to `interactSystem` does not matter, Scope 6).

Positions are positive (D12):

- **`uses_a_space_link`** (the §22 acceptance): a link at (103, 0, 100) in space 0 to space 1, arrival (640, 0, 640)
  (the cellar's site origin) with a yaw of 90°; a return link at (645, 0, 640) in space 1 to space 0, arrival (103, 0,
  102) with a yaw of 180°. The player of `defaultPeer()` (client-owned in the network variants) stands 10 steps at (102,
  0, 100), then `w.input(player, 20)`; `stepUntil` its host `Transform::space() == 1` (at most 30 steps: the chain of
  Scope 6 moves it 2 steps after the step in which the host's `ActionEvent` is readable; a client's action reaches the
  host's queue 1 + delay steps later, more in HostClientAsync). Checks: on the host, the move's step minus the step
  `interactSystem` saw the action is exactly 2 (`TransitionLog`), in every variant; the arrival position within 1e-3 m
  (the owner state's quantization) and rotation within 1e-4 rad; in the network variants the owner's copy
  (`w.onClient(w.defaultPeer(), player)`) is in space 1 within `2 * delay + 5` steps (`stepUntil` on the client copy);
  after the next step `queryRadius(1, arrival, 1)` finds the player and `queryRadius(0, (102, 0, 100), 5)` does not. It
  walks in +x in space 1 (the owner's mover) with `stepUntil` the host-side distance to the return link < 2 m (at most
  300 steps), then interacts with the return link: back in space 0 at (103, 0, 102). `TransitionLog` lists both
  transitions with their links and spaces.
- **`objects_change_space`**: a `Wanderer` (host-owned) with a child marker entity is moved by `moveToSpace` between
  steps to space 1; a test-only object `Diver` moves itself in `update` (during a step) after 30 steps. Checks: spaces
  of roots and child (the child's pose follows the parent after phase 6), the objects keep ticking in space 1, in the
  network variants the replicas are in space 1 (`onClient`) and their `RenderPose::space` switches within
  `interpolationDelaySteps + delay + 3` steps; the variant's end comparison covers `Transform::space`.

`tests/test_engine_replication.cpp`: `SpaceReplication.SpawnInAnotherSpace` (an entity spawned in space 1 arrives in
space 1; the S16 apply dropped the space), `SpaceReplication.HostOwnedMoveSnaps` (the snapshot switches the replica's
space with a snap, the child follows at once and its pose in phase 6), `SpaceReplication.ClientOwnedMoveToSpace`
(`OwnerCommand` v2 applied in phase 1 on the owner; owner states with the old counter dropped; the host's interpolation
buffer breaks at the move), `SpaceReplication.OwnerStateSpaceIgnored` (a forged state with the current counter and
another space: rejected input, the host's space unchanged, no owner record created or updated and no interpolation
sample pushed; a correct state after it is applied; the samples of correct states carry the host's space),
`SpaceReplication.OwnerCommandWithUnknownSpace` (a forged `OwnerCommand` with `space = spaceCount()`: rejected input on
the owner, the pose not applied, the counter taken, so the owner's next states are not dropped as stale),
`SpaceReplication.InterpolationNeverCrossesSpaces` (samples of two spaces: `RenderPose` holds the old space's pose until
the new sample's step, then the new space with `kTeleported`), `SpaceReplication.AbortedStepUndoesTheCommand` (the S16
journal: an aborted step after `moveToSpace` of a client-owned root sends nothing and restores the owner record),
`SpaceReplication.LateJoinerSeesTheSubtree` (a parent with two children in space 1, a client joining later: every
replica in space 1, the initial `RenderPose` in space 1), `SpaceReplication.ChildBeforeParent` (the child's spawn
delivered before its parent's: the child ends in space 1 when the parent arrives),
`SpaceReplication.ResyncWhileOwnerInSpace1` (the owner's hero in space 1, a `Resync`: respawned in space 1, the owner's
root space correct without snapshots).

`tests/test_engine_extract.cpp`: `ExtractSpaces.MoveIsATeleportInTheNewSpace` (the recording sink's dynamic instance
gets `chunk.space == 1` with prev = curr), `ExtractSpaces.ClientUsesTheRenderPoseSpace`,
`ExtractSpaces.SpaceSettingsForwarded` (one `setSpaceSettings` per `SpaceId` at `setRenderSink`, counted per space
id in the recording, also for a sink set later; enclosed with its background).

`tests/test_engine_terrain.cpp` (S20's file; suite `TerrainSpaces`, not S20's `TerrainStreaming`), with a
`RecordingRenderSink` and a four-space setup: the three spaces of `test_engine_spaces.cpp` plus space 3, a hook-less
(flat) open-world template. The tests call `updateView` / `completeView` themselves (only the app does otherwise, S20).
Only the multi-space delta; S20's `Readiness` test covers readiness for one space and is not duplicated:

- `TerrainSpaces.SwitchDropsTheOldSpace`: `completeView` in space 0, then `updateView` in space 2: the recording sink
  holds no tile or mask of space 0 (dropped on space 2's pyramid, E1), the store holds space 0's pyramid only, space 2's
  pyramid is sent before its tiles.
- `TerrainSpaces.PyramidLazyAndCached`: `terrain().pyramid(2)` is null and `Stats::pyramidsGenerated` is 1 (space 0's)
  before space 2's first view, 2 after it; back in space 0 `pyramid(0)` is the same pointer as before,
  `pyramidsGenerated` stays 2 (the cached pyramid is re-sent: a `setTerrainPyramid` of space 0 in the recording), its
  tiles are streamed again (`Stats::generated` grows by tiles and masks only) and `World::spaceReadiness(0, ...)` is
  below 1 on the switch frame (the consequence stated in Scope 8).
- `TerrainSpaces.SpacesWithoutTerrain`: the view sequence 0 (terrain, `completeView`) -> 1 (enclosed) -> 3 (flat) -> 0
  (`completeView`) -> 3, one `updateView` per switch, counts `clearTerrain()` calls in the recording: one at 0 -> 1,
  none at 1 -> 3 (nothing was sent since), one at 0 -> 3; after each terrain-less switch the recording holds nothing,
  nothing is requested (`Stats::pending` 0) and `World::spaceReadiness` is 1.
- `TerrainSpaces.NewSinkGetsTheStreamedSpaceOnly`.

`tests/test_vulkan_terrain.cpp` (S20's file, `VulkanTest`, `RenderScene` driven with test tiles): S20's
`VulkanTerrain.NewSetOnOtherSpaceOrLayout` gets S21's multi-space case (merged, no separate test): a 1 m layout pyramid
and tiles for space 0, frames, a 2 m layout pyramid and tiles for space 2, frames, the 1 m pyramid of space 0 again,
frames, then `clearTerrain()` (an enclosed camera space), frames; validation clean with frames in flight, the terrain
nodes per level match each layout, no terrain nodes after `clearTerrain()`.

`tests/test_vulkan_scene.cpp`: `VulkanSceneSpaces.EnclosedSpace` (camera in an enclosed space: no terrain nodes, no
grass, water or sky in the statistics; the background pixels equal the space's `background` mapped by the file's CPU
reference of the tone map and output transfer (the ACES / clay helpers of S17) within 2/255; a dynamic instance in that
space drawn and one in space 0 not; **one lit instance pixel** (the centre of a flat-coloured box face towards the
camera) equals, within 1/255, the same pixel of a reference frame that renders the same box in an open-world space
without terrain with `setPrototypeEnvironment(kEnclosedLighting)` (fog off, sky off) and the clear colour =
`background`, while the game's environment is set to other values (sun elevation 10°, fog on): this pins that the
enclosed lighting is `kEnclosedLighting`, not the game's), `VulkanSceneSpaces.SwitchBack` (an open-world frame, a frame
in the enclosed space, the open-world frame again: the first and the last image pass the golden metrics of §17.6
against each other; no committed golden is referenced), `VulkanSceneSpaces.NoSettingsMeansOpenWorld`. No new golden:
the enclosed frame uses no new shading (it disables passes, changes the clear value and uses a fixed lighting block);
pixel and statistics checks pin it.

`tests/test_engine_world_api.cpp` (through `handleMessage`, R-MCP-1): `move_to_space` (ok with and without rotation;
`previous` restores the entity when sent back; a child, a state entity, an unknown space and a client edit are errors;
the `api` log line), `spawn(space: 1)` and `spawn(space: 3)` (an error), `set_transform` / `set_component` with `space`
name `move_to_space`, `describe_world.spaces` (incl. `has_terrain` and `site_origin`), `world_status.spaces`,
`list_entities(space)`.

`tests/test_engine_app.cpp`: `set_camera(space)` (an unknown space is an error; the hold).
`LoadingRules.EndsOnlyWithoutPendingUploads` (no device): a `World` with a `RecordingRenderSink` whose
`uploadsPendingForTests` is 4096: `loadEnds(loading, 1.0, world.terrain().terrainUploadsPending())` is false; at 0 it
is true; at `r` 0.99 it is false; at `frames == loadingMaxFrames` it is true (end rule 1 and 3 through S20's
forwarding). With a device (headless, lavapipe in CI; skipped without a driver as the existing device tests), on a
setup with five spaces: space 0 open world with a cheap integer test hook (spacing 2, 16 chunks), space 1
`SpaceTemplate::enclosed("cellar")`, space 2 open world with S20's `latchedHook` (`terrain_test_hooks.h`; spacing 2,
32 chunks = 8 km), space 3 open world with another unlatched integer hook (spacing 2, 32 chunks), space 4 a hook-less
(flat) open world.

- **The latch cannot hang the main thread.** `latchedHook` is file-static state (S20: a mutex, a condition variable,
  `open`, and the thread recorded by `armLatch()`); the test arms it on the main thread right after the app's first
  frame (a complete point in space 0, which never calls the latched hook) and it blocks only when called **off** that
  thread (the streaming pool's workers); on the main thread it returns at once. This covers S20's synchronous paths
  (`terrainHeight` / `TerrainSystem::sample` on a miss, `characterMoveSystem` for every character, idle ones too, and
  the calling thread's share of `completeView`). In addition the test switches spaces only with `set_camera(space)` (the
  app's view camera), never by moving a character into space 2, and runs no complete point while the latch is closed (no
  `screenshot`, no benchmark). While the latch is closed both pool workers may be blocked in space 2 jobs, so the test
  only visits spaces that need no pool work then (space 2 itself and the enclosed space 1). `openLatch()` runs in
  `TearDown` before the world is destroyed, so the pool can join.
- The sequence (one app per test, each step a few frames):
  1. to the enclosed space 1 and to the flat space 4: no load (`spaceLoading()` stays null, the game hook is never
     called);
  2. to space 2 with the latch closed: a load starts with `progress == 0` and stays at 0 for 10 frames (the pyramid job
     is blocked), `app_status.space_loading` reports it, a game `loadingScreen` hook is called once per frame (and not
     the default), `showLoadingScreen = false` draws nothing while `spaceLoading()` still reports;
  3. **moving camera** (still space 2, latch closed, `loadingMaxFrames = 30` in this test's setup): `set_camera` moves
     the camera by 100 m every frame from (1000, 300, 1000) along +x (3 km in 30 frames, inside the 8 km space);
     `progress` never decreases, and the load ends after 30 frames with the log line. The view is still in space 2;
  4. **hop out through the enclosed space 1** (no load, no pool work; not to spaces 0 or 3, whose streaming would wait
     behind the blocked workers), then a new switch to space 2 at a fixed camera: a new load starts; the latch is
     opened: `progress` is non-decreasing and the load ends within 600 frames at end rule 1 (`progress` reaches 1 and
     `world.terrain().terrainUploadsPending() == 0` at the end frame);
  5. **the complete-point cut-off** in the unlatched space 3 (the latch is open now): the switch starts a load (`r < 1`
     on the switch frame: the pyramid is generated on the pool), then an MCP `screenshot` (a complete point) ends it in
     that frame (`spaceLoading()` null afterwards, `progress` not required to have reached 1 before);
  6. **back to space 0** (hooked, pyramid cached): a load starts (its tiles were dropped at the first switch, D5)
     without a pyramid generation (`terrain().pyramid(0)` is the same pointer as before the first switch and
     `Stats::pyramidsGenerated` is unchanged) and ends within a bounded number of frames; "no load" holds only for the
     enclosed and the flat targets (step 1).
- with `--no-render` `spaceLoading()` stays null while the view space changes; with `--coop` the client's camera
  decides.

Death tests are inside `#ifdef SAIGA_ASSERTS`; configuration errors use `SAIGA_EXIT_ERROR` and are tested in every
build.

## Done when

- The §22 row: `uses_a_space_link` passes in every variant incl. SaveLoad and the three host + client variants in CI
  (`test_engine_scenarios`); the player ends in space 1 at the arrival pose (and back in space 0).
- "Further space instances": several templates, `spawnIn`, per-space terrain, space-state entities and saves (tests
  above). "`moveToSpace`": single world and replicated (host-owned and client-owned, late joiners, resync).
  "Entrances": `SpaceLink`, `requestTransition`, `spaceLinkSystem`. "Loading-screen transitions (ImGui until S46)": the
  app's load state and the default overlay (CI with lavapipe), the platformer cellar and `game_world`'s second space
  reviewed in screenshots (local).
- R-TEST-2 numbers reported in the PR: `test_engine_net_benchmark`, `test_engine_ecs_benchmark` (the `Transform`
  change), `game_world --benchmark flyover` and `forest` (space 0 must stay within 5 % of the baseline; no second
  pyramid is resident in the benchmark).
- The three tech demos' `--scenarios` pass with all variants (local), reported in the PR.
- `docs/engine_design.md` amended (list below) and `AGENTS.md` updated.

Status (2026-11-21): all met locally (numbers in Results); the CI runs (lavapipe, GCC, Clang, MSVC) happen with the
PR. Locally the GPU and lavapipe ctest runs, a Clang 19 build with warnings as errors and the three demos' scenarios
pass; MSVC was not built.

## Verification

```shell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSAIGA_BUILD_SAMPLES=ON -DSAIGA_ARCHNATIVE=OFF \
  -DSAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON            # + the CMake 4 / GCC 15 flags of AGENTS.md where needed
cmake --build build --parallel -- -k 0
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest --test-dir build --output-on-failure --parallel 4
./build/tests/test_engine_net_benchmark      # before and after (R-TEST-2), also test_engine_ecs_benchmark
for g in flight platformer racing; do ./build/samples/games/game_$g --scenarios; done
SAIGA_SCREENSHOT=/tmp/platformer_cellar.png ./build/samples/games/game_platformer --demo --cellar
SAIGA_SCREENSHOT=/tmp/platformer_coop.png ./build/samples/games/game_platformer --coop --demo --cellar
```

`game_world` (build-bench recipe of AGENTS.md for the benchmarks): `--benchmark flyover` / `forest` before and after;
interactively (the "Space" combo) or over MCP `set_camera {space: 1, position: [...]}`, `app_status` until
`space_loading` is null, `screenshot` during and after the load, the `terrain` block of `app_status` after the switch
(old tiles freed), and back to space 0.

## Files

- New: `src/saiga/engine/components/SpaceLink.h`, `PendingTransition.h`;
  `src/saiga/engine/world/SpaceTransitionSystem.{h,cpp}`; `src/saiga/engine/gameplay/SpaceLinkSystem.cpp` (declared
  in `gameplay/Systems.h`); `src/saiga/engine/app/LoadingScreen.{h,cpp}`; `tests/test_engine_spaces.cpp`.
- Changed: `components/{Transform.h, RenderPose.h, EngineIds.h, EngineComponents.cpp}`; `world/{SpaceTemplate.h,
  World.h, World.cpp, Events.h, TransformSystem.cpp}`; `save/{SaveFormat.h, SaveReader.h, SaveReader.cpp,
  WorldSave.cpp, WorldLoad.cpp, SaveDiff.cpp}`; `net/{Replication.h, Replication.cpp, EngineMessages.h,
  InterpolationSystem.h, InterpolationSystem.cpp}`; `sync/{RenderSink.h, VulkanRenderSink.*, RecordingRenderSink.*,
  ExtractSystem.cpp}` (`terrain/TerrainSystem.*` is not changed: the `clearTerrain()` trigger, the set rebuild, the
  store per space and the lazy pyramids are S20's; S21 only tests them); `api/WorldApi.*`; `tools/WorldTools.cpp`;
  `app/{EngineApp.*, EngineAppSetup.h, AppTools.cpp}`; `src/saiga/vulkan/scene/{RenderScene.*, RenderSceneTypes.h}`
  (space settings, the enclosed passes and `kEnclosedLighting`; no change to S20's `TerrainAtlas`, `TerrainPass`,
  `GrassPass`).
- Tests: `tests/CMakeLists.txt`, `tests/test_engine_{scenarios,replication,extract,world_api,app,save,terrain}.cpp`,
  `tests/test_vulkan_scene.cpp`, `tests/test_vulkan_terrain.cpp` (one case added to S20's
  `NewSetOnOtherSpaceOrLayout`); `data/tests/engine/schema.json` (regenerated).
- Games: `samples/games/platformer/{Level.*, PlatformerWorld.*, PlatformerRender.*, main.cpp}` (cellar level, doors,
  interact system, the route and `--cellar`, scenarios), `samples/games/common/{RenderAssets.*}` (`LevelGeometry` with a
  space), `samples/games/{flight,racing}/*Render.cpp` (camera space), `samples/games/world/main.cpp` (`islands`, the
  space combo), `data/platformer/input.json` (`interact`).
- Docs: `docs/engine_design.md`, `AGENTS.md`, this file.

## Design-doc amendments (made in step 10, R-DOC-2)

- **A1** (dropped in review round 1: the space seed is S20's amendment A3; D11.)
- **A2** §12.2, §12.4: `terrainHeight` of an enclosed space is 0 (the floor plane) until S23 geometry and S28 colliders;
  the kinematic mover walks on it.
- **A3** (dropped in review round 1: S20 draws the sea only for a space with terrain (`hasTerrain()`, E6), so enclosed
  spaces have none; `SpaceTemplate::enclosed` sets `seaLevel = std::nullopt`, and the `seaLevel` and `baseSpacing` of
  an enclosed template are ignored, not validated.)
- **A4** §10.3, §13.4: `RenderPose::space` and per-sample spaces in the interpolation buffer; no interpolation across a
  space change; the extract takes the space from `RenderPose`.
- **A5** §13.3/§13.4: clients apply the root's `Transform::space` from spawns and snapshots (S16 dropped it); a space
  change forces a snap; children take the space when attached; the owner gets its root's space only from the spawn, a
  resync and the `OwnerCommand`.
- **A6** §7.4: `Transform::space` is changed by `moveToSpace` and the engine's hierarchy and replication paths (one
  World-internal primitive); a child is always in its parent's space (immediately on `moveToSpace` and on attach);
  `moveToSpace` applies at once, also during a step (not a command).
- **A7** §7.10: `requestTransition(Entity e, Entity link)` for any root (vehicles, driver entities), not only players;
  the arrival pose is copied at the request; a target out of range is a logged refusal.
- **A8** §12.4, §7.10, §8.1, §8.2: `PendingTransition` (component, saved), `spaceTransitionSystem` (phase 1, host,
  engine-registered, revalidates at completion), `SpaceTransitionEvent`, `spaceLinkSystem` registered by the game,
  cross-space interactions ignored.
- **A9** §12.4 step 2: before S24 the loading screen starts when the view camera's space changes and the target is not
  ready (local); S24 adds a request-time signal for the player's peer. `World::spaceReadiness(space, camera, radius)` is
  the one readiness the app reads (E8; terrain in S21, content sets from S23b); progress is its running maximum; a load
  ends at readiness 1 with no pending terrain uploads, at a complete point, or after `loadingMaxFrames`.
  `EngineAppSetup::{loadingRadius, loadingMaxFrames, showLoadingScreen, loadingScreen}`, the engine default overlay.
- **A10** §7.8: one space-state entity per space, identified by its `Transform::space`; before S32 every client
  receives all of them.
- **A11** §15.1: optional blob sections in a fixed order (`HEAD, SCHM, SPCE?, GENR?, TERR?, CREC?, ENTS, COMP, EVNT,
  ACTN, STAT`), each with its own format version, added by the recipe of Scope 5 (parsed or as a raw range); the `SPCE`
  space table (the template name and kind per `SpaceId`, names may repeat; written with more than one space; the prefix
  rule: a save with fewer spaces loads); templates are append-only until S23 (then the plan's pinned instance list).
- **A12** §10.1: `RenderSink::setSpaceSettings` / `RenderScene::setSpaceSettings` (`SpaceRenderSettings {openWorld,
  background}`), forwarded by the World; an enclosed camera space draws no terrain, grass, water or sky, clears to
  `background` and lights instances with `kEnclosedLighting` (the `PrototypeSettings` defaults, fog off); one GPU
  terrain set for the camera's space (S20), cleared by `clearTerrain()` for a terrain-less space (E1); superseded by
  `setEnvironment` in S36 for ambient and fog.
- **A13** §13.3: `OwnerCommand` version 2 carries the space (kind `kTeleport` only); the host never takes a space from
  an owner state (a mismatch at the current counter is rejected input).
- **A14** §16.2: `move_to_space`, `spawn(space)` for every space, `describe_world.spaces`, `world_status.spaces`,
  `list_entities(space)`, `set_camera(space)`, `app_status.space_loading`.
- **A15** §12.4 "Players can be in different spaces ... the host simulates every space that contains a player": before
  S24 every space is simulated (every entity is `Simulated`); stated explicitly.
- **A16** §22: the S21 row's dependencies also name S19b (extract, `RenderPose`, view camera) and S16b (the platformer's
  client-owned hero used by the demo).
- **A17** §7.9, §12.4 (D12, E4): an enclosed space's chunk range is `[0, sizeChunks)^2` (`SpaceTemplate::enclosed(name,
  sizeChunks = 4)`), its authored site's origin at the centre of the middle chunk, `(sizeChunks / 2) * 256 + 128` m on
  both axes (`SpaceTemplate::enclosedOrigin`); `SpaceTemplate::background` until S36; an open-world template without a
  hook is flat (D1, with S20 A7).

## Open questions and risks

- **Enclosed spaces before S23 are empty**: only entities and game-rendered geometry (the platformer's level sets). The
  floor plane y = 0 (A2) is a stopgap; a game that places an enclosed level elsewhere uses its own collision (as the
  platformer does) until S28.
- **First view of another open-world space**: the lazy pyramid takes about 0.7 s with the prototype hook, plus the tiles
  around the camera; the loading screen covers it. Measured in `game_world` and reported.
- **Set rebuild cost**: a switch allocates a new atlas (64 MB at 1 m) and frees the old one after the frames in flight;
  peak GPU memory is two sets for a few frames. Measured with `game_world` and reported; within the 768 MB criterion
  of S20.
- **`requestTransition` of a vehicle and its driver**: two requests that complete in the same phase-1 pass (entity
  order). Fine for S21; S24's arrival must hold both until both islands are ready (noted for S24).
- **Space table and S23**: when the plan replaces templates, `SPCE` lists instance names derived from site keys; the
  rename check then compares those. Noted for S23.
- **Asynchronous variant timing**: the owner applies the command `1 + delay` (+ async offset) steps after the host;
  `uses_a_space_link` checks the host's pose at once and the owner's with `stepUntil` on the client copy (bounded by
  `2 * delay + 5` steps).
- **Pinned pre-simulation of `--demo --cellar`**: a change of the level, the controller or the co-op timing moves the
  entry step; the two scenarios log entry and stay per mode, and the two N constants are updated from them. The
  capture's step drifts with the frame rate (frames are not steps); the ≥ 240-step stay with N in its middle absorbs
  ±120 steps.
- **Exits into a terrain overworld re-stream** (D5, E1): the overworld's tiles around the camera are dropped on the way
  in and regenerated on the way out, with a short loading screen (the pyramid is cached). The platformer's overworld is
  flat (no load); `game_world`'s space combo shows the cost, measured and reported. Keeping the previous space's tiles
  for a quick return is S25's residency question.

## Review log

(Plan review rounds and their outcomes are recorded here before the implementation starts.)

- **Review round 1** (2026-11-21; Opus 6/10, Fable 6/10; 22 merged findings; cross-plan decisions D1–D14 shared with
  S20, S22, S23). Resolutions:
  1. "OpenWorld needs a hook" contradicted S20's flat default: dropped; hook-less open world is flat (D1), validation
     with `SAIGA_EXIT_ERROR`; death-test case removed, `HooklessOpenWorldIsFlat` added.
  2. Placeholder `TerrainStreaming`: replaced by S20's `TerrainSystem` / `TerrainStore` / `updateView` / `completeView`
     / `readiness`; space seed, `spaceTemplate`, `terrainHeight(space, ...)` and the empty-list default marked as S20's;
     A1 and the "if S20 ..." conditionals dropped; S21's terrain tests use the suite `TerrainSpaces`.
  3. Readiness on the wrong layer: S20's engine-side `readiness` (D6), monotone only for a fixed camera; the app test
     uses a test-controlled latch; `completeView` ends a load.
  4. One GPU terrain set: rebuilt on a switch with `deferDelete` (D5); lazy pyramids; a new sink gets the streamed space
     only; `SpaceSwitchRebuildsTheSet` (1 m and 2 m layouts); the benchmark generates no second pyramid.
  5. Old-space CPU tiles: dropped on a switch, only pyramids cached; S22 keeps its dirty boxes per space (D3/D5).
  6. `SpaceRenderSettings` without data source: reduced to `{openWorld, background}` + `SpaceTemplate::background`;
     fog/sky only from `PrototypeSettings` in open-world camera spaces; A3 dropped.
  7. `SPCE` vs the fixed section order: the optional-section mechanism (D7) with per-section versions; `SPCE` only with
     more than one space; kinds compared (reason given).
  8. `setSpace` callers: one primitive (`setSubtreeSpace` / `teleportTo`), the complete writer list, a single
     `teleports` increment, the subtree from the `Parent` pool (not `hierarchyOrder`), phase 6 only asserts.
  9. Client child paths: attach rule covers all parent paths; initial `RenderPose`; owner's root space sources stated;
     tests for a late joiner, child before parent, resync with the owner in space 1.
  10. Transitions: revalidation at completion (log, no assert); a target out of range is a logged refusal; the load
      checks `PendingTransition` / `SpaceLink` targets; a dangling `link` stays a dead handle (save/load/diff explained).
  11. S20 tests to update: `Terrain.TemplateValidation` cases, the constructor message, the `spawn` and load messages
      named in step 1/3; `test_engine_terrain.cpp` listed.
  12. One-frame flash: a load starts only if readiness < 1 (D6).
  13. No command record kinds: `makeCommand`, `Commands.h` removed from Files; the system's line in
      `registerEngineSystems`; IDs and `schema.json` per D10.
  14. API claims: only the error texts change; `move_to_space` returns an undo payload; `set_camera(space)` validated
      against the view world (client world in `--coop`).
  15. Hook wording: `showLoadingScreen` flag; `loadingRadius` in `EngineAppSetup`; the enclosed case asserts "no load"
      instead of `frames == 1`.
  16. Background vs tone map: compared through the S17 CPU reference within 2/255; `SwitchBack` compares against its
      own first frame, no golden.
  17. Scenario: `w.onClient(w.defaultPeer(), ...)`, positive coordinates (D12), a dedicated `spacesScenarioSetup()`,
      the save/load claim dropped, `stepUntil` by host-side distance, network bound `2 * delay + 5`.
  18. Load checks on staged `Transform`s (`checkSpaces`), `restoreStateEntities` per space incl. freeing.
  19. Client pilot: the route as shared setup data (no `DemoRoute` component), interact repeats, waypoint by space,
      `RestartRun` before the command, `--cellar` in the flag list, pinned pre-simulation, `pilot_visits_the_cellar`.
      Rejected: dropping the cellar from `--demo` — D14 keeps the demo, and the scenarios carry the automated check.
  20. `game_world` camera space: a "Space" combo and the walk mode per space; `game_set_camera` keeps the current space.
  21. Scope trim: `ownerStatesWrongSpace` and `kMoveToSpace` dropped (one kind with the space). Rejected in part: the
      suggested debug assert on a mismatching owner state — it is network input, so it is `rejectInput`, never an
      assert.
  22. `AGENTS.md`: test count from the configure, layout lines for the new components, systems, save sections and
      `app/LoadingScreen`.

- **Review round 2** (2026-11-21; Opus 7/10, Fable 7/10; 16 merged findings; cross-plan decisions E1–E14 shared with
  S20, S22, S23a/b/c). Resolutions:
  1. Switch contract vs S20 (E1): Scope 8 reduced to the delta (terrain-less target via `clearTerrain()`, 1 m ↔ 2 m
     switch and back, pyramid cache reuse); S20's set/atlas/pass files dropped from Files;
     `SpaceSwitchRebuildsTheSet` merged into S20's `NewSetOnOtherSpaceOrLayout`; `TerrainSpaces.ReadinessFixedCamera`
     and `VulkanTerrainDeathTest.TileOfAnotherSpace` dropped (S20 owns the set's contract).
  2. Enclosed site origin (E4): `(sizeChunks / 2) * 256 + 128`, `SpaceTemplate::enclosedOrigin`; cellar, scenario
     links and arrivals at (640, 0, 640), A17.
  3. Loading test hang: the latched hook blocks only off the main thread (main thread id recorded in `SetUp`), spaces
     are switched with `set_camera` (no character in the latched space), no complete point while the latch is closed;
     the complete-point cut-off is tested in a third, unlatched space.
  4. Back to space 0: space 0 of the app test has a hook; a quick load without pyramid generation is expected; "no load"
     only for enclosed and flat targets; the re-stream consequence stated in Scope 8, Design notes and risks.
  5. Readiness (E8): `World::spaceReadiness` is the app's only call; progress is the running maximum; end at readiness
     1 with no pending uploads, at a complete point (explained), or after `loadingMaxFrames`; moving-camera test.
  6. Validation: the stale "switches to `SAIGA_EXIT_ERROR`" removed; enclosed via `hasTerrain()` (E6), `seaLevel` and
     `baseSpacing` ignored for enclosed (S20's default `seaLevel = 0` would otherwise trap hand-written templates); the
     hook error names `SpaceTemplate::enclosed`; `HandWrittenEnclosedIsValid` added.
  7. Pre-simulation: a ≥ 240-step stay, N = entry + stay / 2 − 60 per mode (solo from the Plain variant, co-op from
     HostClientDelay = `--coop`'s delay 2), the view space in the capture log line.
  8. Event latency: the chain action → `ActionEvent` → `InteractEvent` → request → move stated step by step (move 2
     steps after the `ActionEvent` is readable), checked in `uses_a_space_link`; the "registered after" wording dropped.
  9. `MoveToSpace.RootAndDescendants`: spaces at once, the children's teleport flags after the next phase 6.
  10. `TransitionLog`: a world-state test component like `ActionLog` (per world, saved), no process global.
  11. `restoreStateEntities`: staged defaults per (state entity, component), built before the commit; the shared
      `World::makeStateEntity` (the loader is a friend; `reg.create()` + log record + `setSpaceInternal`).
  12. Step order: the target checks of `checkSpaces`, `PendingTransitionSurvives` and the target cases of
      `CorruptStates` moved to step 5; step 3 uses nothing from step 5.
  13. The golden case is `SaveFormat.CommittedSaveLoadsWithAppendedSpaces` in `test_engine_save.cpp` (golden setup +
      two appended templates).
  14. Enclosed shading: `kEnclosedLighting` (the `PrototypeSettings` defaults, fog off, background instead of sky),
      pinned by one lit instance pixel against a reference frame.
  15. Replication: samples and the first `RenderPose` use the host's space; the mismatch is rejected before the owner
      record and buffer are touched; a rejected `OwnerCommand` still takes the counter (reason given); `RenderPose`'s
      `space` is the last field and the one aggregate site is updated; tests added.
  16. D7 recipe (tag, table entry with `maxVersion`, `ParsedSave` member or raw range, writer position, load check,
      diff hook); `SPCE` rows are template names per `SpaceId` (may repeat); the prefix rule shared with `GENR`.
  Rejected: none.

- **Review round 3** (2026-11-21; Opus 8/10, Fable 8.5/10; 11 findings; cross-plan decisions F1–F4, none of which
  changes S21). Resolutions:
  1. Schema regeneration: the command is filtered to `SaveFormat.EngineSchemaIsCommitted` (unfiltered,
     `CommittedSaveLoads` would rewrite `save_v1.bin` / `.json`); "byte-identical" reworded: a single-space save has no
     `SPCE`, `save_v1.bin` is not regenerated and still loads, new saves differ in `SCHM`.
  2. Pyramid reuse: checked by `terrain().pyramid(space)` (null before the first view, the same pointer on return) and
     S20's new `Stats::pyramidsGenerated` (`Stats::generated` counts tiles and masks too), in `PyramidLazyAndCached` and
     the app test.
  3. Repeated `SPCE` names: moved to the reader-level `SaveSections.SpaceTableNamesMayRepeat` on a hand-built blob
     (S21's setups cannot produce them before S23a).
  4. Load end rule 1: `world.terrain().terrainUploadsPending() == 0` (S20's forwarding, 0 without a sink), the
     `RenderScene` special case dropped; `loadEnds` pinned by `LoadingRules.EndsOnlyWithoutPendingUploads` through
     `RecordingRenderSink::uploadsPendingForTests`.
  5. `clearTerrain` trigger: S20's code, S21 only tests it (Files and step 7 corrected); one `clearTerrain()` when
     leaving a terrain space, none between two terrain-less spaces; `TerrainSpaces` uses a four-space setup with a
     hook-less space 3.
  6. Scenario wiring: `registerSpacesScenarios` and a `Case` with `setup` / `registerScenarios` function pointers;
     `TransitionLog.step = stepIndex() - 1` (the event is read one step after the move).
  7. Save recipe: one parse signature `parseX(BinaryReader&, uint32_t version, X&, std::string& error)`; `SpaceTable`
     rows `{name, SpaceKind}`; the merge order with S22 (if S22 is first it adds the mechanism with these names, S21
     then adds only `SPCE`). Citing these names in S22's plan is S22's change.
  8. Loading test: the sequence written out; after the moving-camera load the view hops out through the enclosed
     space 1 (no pool work while the workers block); spaces 2 and 3 have 32 chunks, the camera moves 100 m per frame;
     the latch is S20's `latchedHook`, armed after the first frame.
  9. `setSpaceSettings` is forwarded once per `SpaceId` (counted per space in the test).
  10. Client-owned `moveToSpace` before `OwnerCommand` v2: asserts in steps 2–3 for a change of space; step 4 removes
      the assert.
  11. `SpaceKind` origin: S20's `SpaceTemplate.h` (S21 never claimed otherwise); the wording is S23a's to fix.
  Rejected: none.

- **Implementation** (2026-11-21, Claude Opus 5.5; against the implemented S20, whose names the plan already used).
  Deviations, each the smallest sound fix:
  1. Step 1: `World::createIn(space)` for the world API's `spawn` without a class (the API had no other way to create
     a plain entity in a space).
  2. Step 2: `Replication::ownerTeleport` became `ownerMove` already in step 2 (with the plan's assert for a
     client-owned root changing space, removed in step 4); writer 5 is the private `World::followParentSpace`; the
     space primitive notes its writes in the S16 write detector (`noteWritten`; `World` is a friend of `Replication`).
     `MoveToSpace.SameSpaceIsATeleport` reads the child's flag as `wasTeleportedLastStep` (the flags shift at the end
     of a step).
  3. Step 3: the loader keeps the staged defaults as `std::vector<World::StateDefaults>` (index 0 the world-state
     entity, 1 + s the space-state entity of space s) and adds missing components with `World::addStateComponent`.
     `diffSaves` reports `section_only_in_a/b` and `space` differences ("space 1: 'cellar' (enclosed) vs 'cave'
     (enclosed)"). `SaveSections.SpaceTableNamesMayRepeat` checks the parse and `diffSaves` only: `World::dumpJson`
     dumps a world's entities, and a blob with repeated names cannot load into an S21 setup (unique template names).
  4. Step 4: an unknown space in a decoded `Transform` is rejected input; a replica under a client-local parent keeps
     its parent's space (applyTransform, onOwnerCommand); `OwnerCommand`'s `space` is visited after `counter`. The
     interpolation also snaps when the sampled space differs from the `RenderPose`'s (the first sample after a buffer
     reset); the paused extract's settle path snaps on a space change too.
  5. Step 5: the target cases of `SpaceSave.CorruptStates` are the separate `SpaceSave.CorruptTransitionTargets`.
     The scenario `Case` holds `WorldSetup (*setup)(const std::string&)` (S20's `setupFor(name)` selects the walker's
     terrain setup); `ScenarioWorld::networkConfig()` gives the variant's delay; the asynchronous variant gets 6 more
     steps in the latency bounds.
  6. Step 6: the `space` key of `set_component`'s `Transform` values is checked by S20's field rule ("only position,
     rotation and scale"), whose text now names `move_to_space`; `add_component` of `Transform` names it too.
  7. Step 7: an enclosed camera space calls `TerrainPass::skipFrame` instead of its transfer; its background is
     sRGB-encoded in clay mode (the scene writes display values there); `kEnclosedLighting` is an inline constant in
     `PrototypeWorld.h`. The S20 test `CompleteViewFirstView` skips the space settings in the call log.
  8. Step 8: `loadEnds` takes `maxFrames`. The app test is three tests instead of one: the sequence (steps 1, 2, 4, 5,
     6 with the default frame limit: with `loadingMaxFrames = 30` step 4's load would end by rule 3), the moving
     camera (steps 2-3 with 30 frames and `showLoadingScreen = false`) and `--coop`; the spaces are switched with
     `setViewCameraByTool` (what `set_camera` calls), `set_camera(space)` is tested separately
     (`EngineAppNoRender.SetCameraSpace`). The camera JSON of the app tools has `space`.
  9. Step 9: the route of a hero is `RunState::route` (see Design changes); the host's demo runner has no `Player`, so
     at a door it emits the `InteractEvent` itself (`interactWithNearestDoor`, the game's rule) instead of queueing the
     action, the pilot (a player) queues it; the pilot system runs `RunOn::Both` (the single-world variants of
     `pilot_visits_the_cellar` pilot a host-owned hero); the goal is checked only in space 0. Doors: the entrance at
     (-2.5, 0, 4.5) on the start island (away from the wall and the step), the cellar's arrival 3 m behind its site
     origin, the exit at origin + (-2, 0, -21), back next to the entrance at (-1.5, 0, 3). `cellar_round_trip` checks
     the arrival back within 0.05 m (in the single-world variants the controller runs in the arrival step after phase
     1). The interact key is F.
  10. Step 10: `AGENTS.md` names 72 tests with the CI options (S20's 71 + `test_engine_spaces`).

## Design changes

The design-doc amendments A2, A4-A17 are made in `docs/engine_design.md` (§3 row, §7.4, §7.8, §7.9, §7.10, §8.1, §8.2,
§10.1, §10.3, §12.2, §12.4, §13.3, §13.4, §15.1, §16.2, §17.2, §18.4, §22). Changes against the plan made during the
implementation (each the smallest sound fix; details in the review log below):

- `World::createIn(space)` (between steps) for the world API's `spawn` without a class; `World::addStateComponent`
  next to `makeStateEntity` (the loader adds missing state components with it).
- `loadEnds(loading, r, pending, maxFrames)` takes the frame limit as a fourth argument.
- The platformer's route of a hero is `RunState::route` (version 2, a migration); the context holds both routes
  (`PlatformerContext::routes`) instead of one `demoRoute` chosen by `--cellar`, so the cellar scenarios run without
  the flag. The interact key is F (E turns the camera).

## Results

Measured on the RTX 5080 Laptop GPU and on lavapipe (Mesa with LLVM 21.1.8), build `build/` (RelWithDebInfo, samples
ON, warnings as errors).

- **Tests.** ctest 74/74 on the GPU (local build with samples; with the CI options 72 tests: S20's 71 + the new
  `test_engine_spaces`); every `test_vulkan_*` / `test_engine_*` on lavapipe 34/34. A full Clang 19 build with samples
  and `SAIGA_WARNINGS_AS_ERRORS=ON` (the CI Clang job's compiler family) builds without a warning. MSVC was not built
  locally.
- **Done when.** `uses_a_space_link` passes in all six variants (Plain, Repeat, SaveLoad, HostClient,
  HostClientDelay, HostClientAsync): the player ends in space 1 at the arrival pose (position within 1e-3 m, rotation
  within 1e-4 rad) and back in space 0; the move is exactly 2 steps after the interact step in every variant.
  `objects_change_space` passes in all six variants. Further spaces, `spawnIn`, per-space terrain, space-state entities
  and saves: `test_engine_spaces` (26 tests), `SaveSections.*`, `SaveFormat.CommittedSaveLoadsWithAppendedSpaces`;
  `moveToSpace` replicated: `SpaceReplication.*` (10 tests); entrances: `Transition.*`; the loading screen:
  `LoadingRules.*`, `EngineAppRender.SpaceLoading*` (incl. the default overlay's pixels, on lavapipe in CI);
  `VulkanSceneSpaces.*` (the enclosed frame: background through the ACES reference within 2/255, the lit pixel equal to
  the `kEnclosedLighting` reference frame within 1/255).
- **R-TEST-2.** `test_engine_net_benchmark` (2 runs each; before = after step 3, after = the final branch): snapshots
  of 10 000 moving entities host step 4.42-4.48 -> 4.17-4.20 ms, client step 2.67-3.02 -> 2.72-2.76 ms, small
  messages 1.80-1.83 -> 1.67-1.80 ms (unchanged within noise). `test_engine_ecs_benchmark` (after step 3 -> final):
  `World::step` idle 0.618 -> 0.562 ms, moving 2.650 -> 2.315 ms, extract 0.650 -> 0.656 ms, grid step 2.660 ->
  2.747 ms, saves of 100 000 entities save 39.7 -> 39.8 ms, load 115.0 -> 113.1 ms (the `Transform` change adds only a
  private method; unchanged within noise). `game_world --benchmark` (build-bench, 3 runs each, medians; before = after
  step 6, after = step 7, final = step 9 with the `islands` space configured but never viewed):

  | | before | after step 7 | final |
  |---|---|---|---|
  | flyover `cpu_ms` | 3.78 | 3.77 | 3.89 |
  | flyover GPU / `scene` | 2.47 / 2.40 | 2.50 / 2.43 | 2.53 / 2.45 |
  | flyover trees / terrain + water / sky | 1.400 / 0.190 / 0.013 | 1.399 / 0.189 / 0.013 | 1.412 / 0.191 / 0.013 |
  | forest `cpu_ms` | 2.18 | 2.18 | 2.19 |
  | forest GPU / `scene` | 2.19 / 2.14 | 2.19 / 2.14 | 2.19 / 2.14 |
  | forest trees / terrain + water / grass | 0.555 / 0.114 / 0.681 | 0.555 / 0.114 / 0.679 | 0.556 / 0.114 / 0.680 |

  All within 3 % (space 0 within the 5 % of the plan); the flyover's interval streams in most frames (S20), its GPU
  sections are the comparison. `sample_vulkan_scene_benchmark`: cull 0.057-0.058 ms, instance draw 1.26-1.30 ms
  (S20: 0.058 / 1.27).
- **Scenarios.** `--scenarios`: flight 18 passed, racing 18 passed, platformer 36 passed (the six S20 runs of each of
  the three old scenarios plus `cellar_round_trip`, `demo_runner_visits_the_cellar`, `pilot_visits_the_cellar` in all
  variants). `demo_reaches_goal` keeps 7.25 s. The cellar stays: the host's demo runner enters at step 33 and stays
  270 steps (goal after the detour at 12.43 s); the pilot enters at step 34 (Plain) / 67 (HostClientDelay) and stays
  270 steps. The pinned pre-simulation: solo `--demo --cellar` 108 steps (33 + 135 - 60), co-op 142 (67 + 135 - 60).
- **Screenshots** (`SAIGA_SCREENSHOT`, RTX; reviewed): `game_platformer --demo --cellar` and `--coop --demo --cellar`
  (both: "screenshot of frame 60 (view space 1)", the hero in the cellar with the exit door, the dark background, no
  sky), `game_platformer --demo` (unchanged, on the bridge), flight and racing `--demo` and `game_world` (unchanged).
  `game_world` headless over MCP: `set_camera {space: 1}` (the `islands` space): `app_status.space_loading` from 0
  (frames 1-22: the pyramid on the pool) through 0.33 and 0.6 to the end (an MCP screenshot is a complete point and
  ends the load), the terrain block `resident` 0 -> 62; back to space 0 a short load (progress 0.06 -> 0.5 -> done in
  about 15 frames) without a pyramid generation. The islands screenshot shows the lowered terrain with a lake and no
  trees (the prototype's forest is space 0's). A screenshot of the default loading screen itself cannot be taken
  through the app's capture paths (a capture is a complete point, which ends the load); the overlay is pinned by
  `EngineAppRender.SpaceLoadingCoopClientCamera` through `RenderScene::requestScreenshot`.
