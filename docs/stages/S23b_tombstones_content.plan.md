# S23b: Tombstones and content rendering

Status: planned.

Sections follow the stage plan template of S05; "Implementation steps", "Risks and open questions", "Design-doc
amendments" and "Review log" are added for this stage. Second of the three S23 stages (D13 of review round 1; the §22
rows are amended in `S23a_generation_pipeline.plan.md`, amendment 1).

## Goal

Generated tier-1 content becomes visible and modifiable on every peer (`docs/engine_design.md` §12.1, §12.2, §12.5,
§13.1, §13.2):

- chunk records on every peer carry the chunk's placement version and its tombstones; every tombstone advances the
  chunk's one modification sequence, which stays in S22's `TerrainEditStore` (E10);
- `World::tombstone` (host, validated, idempotent) with live replication (`TombstoneModification`) and a record
  snapshot for joining peers and host loads (`ChunkRecordState`), both on `Channel::State` with the S16 epoch;
- content render sets: one instance set per chunk with its tier-1 instances and the static parts of the plan's sites,
  tombstoned keys hidden, kept equal to the chunk record on every change; grass masks from the generator;
- terrain edits re-snap `ReSnapOnEdit` instances (derived, every peer with content sets, driven by S22's applied changed
  boxes) and tombstone `TombstoneOnEdit` instances (state, host, computed inside S22's edit before its commit);
- the MCP tools `chunk_info`, `list_sites`, `tombstone` and the generator and seed fields of `describe_world` /
  `world_status`;
- the CI golden `generated_site` (moved here from S23c, E12): a generated chunk with a site, tier-1 instances, a static
  part and a tombstoned instance through the real render path.

Acceptance: a tombstoned instance is hidden on the client in the host + client variants; the `generated_site` golden
with its count check.

## Depends on

- S23a (`S23a_generation_pipeline.plan.md`): `GenerationPipeline` with its accessors (synchronous
  `chunkBase(ChunkId)` and `chunkPlacement(ChunkId, placementVersion)` = tier-1 instances and grass; the pool form
  `contentSource(SpaceId)` with `ContentSource::requestPlacement(pool, chunk, version, priority, onMainThread)`;
  `stampList` / `staticList` from the plan; `height`), `ContentCache` (D8 in-flight rule), `ContentConvert`,
  `ChunkRecords` (`placementVersion`; records only for stamped chunks; host-only `stamped`), `CREC` format 1, `GENR`
  and the predicate `writesGenerationState()`, `GenerationState` with `generationMismatch()` / `generationChecked()`,
  `GeneratorSlot`
  (flags, mesh table), `World::spaceHasTerrain`, the shared test setup `tests/content_test_setup.h` (E12: the test
  generator with slots `test_tree` TombstoneOnEdit (16384 candidates per chunk) and `test_rock` ReSnapOnEdit (1024),
  sites 0 and 1, the classes and the
  test meshes `test/tree_a`, `test/tree_b`, `test/rock`, `test/pillar`, `test/chest` registered on every world) and
  the content scenarios.
- S20: `TerrainSystem::updateView` / `completeView`, the single send point with `sendGrassMask` (E3),
  `prototypeGrassMask(level0, layout, seaLevel, minNormalY, minAboveSea)`, `StreamingPool` (D8), `GrassMaskData`, and
  the engine-linked render test target `tests/test_engine_terrain_render.cpp` with `VulkanRenderSink` and goldens (a
  hard dependency of the `generated_site` golden, E12; no fallback).
- S21: the camera's space (D5: one GPU terrain set for the camera's space), `World::spaceReadiness(space, camera,
  radius)` (E8: the app's only readiness call; S23b extends it).
- S22: `TerrainEditStore` with `sequence(ChunkId)`, `bumpSequence(ChunkId)` (creates an entry without deltas) and
  `setSequence` (E10), and the terrain-only revision `TerrainChunkEdits::revision` (F1 of review round 3: bumped only when the
  chunk's deltas change, never by a tombstone; S22's edited-mask results are keyed on it); `applyTerrainEdit`'s compute
  steps 1-3, its reserve step 4 and its commit step 5 with the extension point S22 names for this stage (F2: the
  tombstone-on-edit records, key capacity and send-list entries are reserved in step 4 and written in step 5),
  `TerrainEditResult`,
  `World::terrainSample`, `onTerrainChanged`, `TerrainModification` / `TerrainChunkState` (a state without deltas for
  a chunk with only a sequence), the host send list (an ordered list of messages, `Replication::hostStateList()`),
  `Replication::sendChunkSnapshot(peer)` (E5), the client `Resync` clear, `NetStats::modificationGaps`, the host +
  client end condition and `compareReplicas`, `TERR` (entries without deltas), the grass path of edited chunks (E3).
- S19b / S18: `RenderSink` instance sets (`addInstanceSet(RegionId, instances, capacity)`, `updateInstances`,
  `removeInstance` (scale 0), `removeInstanceSet`), `StaticInstance` (slot in the upper 8 key bits), `World::findMesh`,
  `RecordingRenderSink`, `RenderScene::trackMeshStatistics` (at most 16 meshes, drawn instances per LOD).
- S14/S16: `registerWorldTools`, `handleMessage` tests, the read-only rule on clients.

## Rules that apply

R-STATE-1/2 (tombstones are chunk state; re-snapped y is derived), R-NET-1/2 (tombstones are host state replicated to
every client; clients never tombstone; untrusted messages are validated before they apply), R-SYS-1/2 (tombstones
during a step are commands), R-DET-1, R-MECH-1 (slot flags are game policy, the engine applies them), R-PAR-1 (set
generation on the streaming pool), R-MCP-1, R-SAVE-1 (`CREC` format 2), R-TEST-1/2, R-DOC-1/2.

## Scope

### 1. Records on every peer (`engine/content/ChunkRecords.*`, `engine/save/`)

- `struct ChunkRecord { uint16_t placementVersion; std::vector<InstanceKey> tombstones; };` (tombstones sorted,
  unique). `stamped` stays the host-only set of S23a. A record means "this chunk's tier-1 content is fixed at this
  placement version" on every peer.
- **No sequence in the record** (E10): the per-chunk modification sequence stays in S22's `TerrainEditStore` for all
  spaces. A tombstone advances it with `TerrainEditStore::bumpSequence(chunk)` (a tombstone-only chunk gets S22's
  entry without deltas; S22's `TERR` saves it, S22's `TerrainChunkState` sends it). "Terrain record" keeps meaning a
  chunk with deltas (S22); "chunk record" is this record.
- **Which chunks have a record** (E10): S23a's stamped chunks (host only); every chunk with a tombstone (host and
  clients); a record received or loaded with a placement version other than the current one. A terrain edit creates
  no record. Spaces whose generator has no slots (`FlatGenerator`, every `HookGenerator`, the prototype fixture) never
  get a tombstone, so they have no records on clients and only S23a's stamped records on the host; their edits live
  in S22's store alone (S22's hashes, saves and messages unchanged).
- **Creation**: host: S23a's stamping, and a tombstone (with the current placement version if the chunk has none).
  Client: only from `TombstoneModification` and `ChunkRecordState`, which carry the host's placement version. S22's
  terrain messages are unchanged (no placement version in them: a record created on the host by a tombstone has the
  current version, which is every peer's effective version of a chunk without a record).
- **Effective record** (comparison and rendering): a missing record is `{current placement version, no tombstones}`
  (the current version is equal on host and clients: S23a's `GenerationState` refuses a client with other versions).
- **`onRecordChanged(ChunkId)`**: the one notification every record writer calls after it changed a record (created,
  removed, version changed, keys added or removed): host tombstone apply and tombstone-on-edit commit, S23a's
  stamping record, client `TombstoneModification` / `ChunkRecordState` apply, the client `Resync` clear (per cleared
  record), a host load's commit (per chunk with a record before or after the load). It forwards to the content render
  system (Scope 4) and the grass path (Scope 5); without them it does nothing.
- S22's host + client end condition and `compareReplicas`: per space, over the union of chunks with a record on either
  side, the effective records must be equal (placement version, tombstone set); `stamped` is not compared. The end
  condition waits until they are (within S22's bound). Sequences are compared by S22 already (tombstone-only chunks
  are in its store). Only `TombstoneModification` and S22's messages set sequences; `ChunkRecordState` carries none, so
  no record message can raise a sequence ahead of a pending terrain state.
- `CREC` format 2 (D7 order unchanged; full layout): section format (u32 = 2); space count (u32); per space with records
  in `SpaceId` order: space (u16), record count (u32); per record in `ChunkId` order: x, z (i32), placement version
  (u16), stamped (u8, host-only meaning), tombstone count (u32), per key slot (u8) and counter (u32). Format 1 (S23a)
  reads as records without tombstones. No sequence and no `TERR` / `CREC` equality check (E10). Load checks (load
  errors naming the chunk): keys sorted and unique; a record with tombstones lies in a space whose generator has slots,
  every slot below the slot count and every counter below `candidatesPerChunk(slot)`; S23a's version and range checks.
- `world.hash()`: S23a's byte table (one table; S22's terrain block first) gains per record, after the stamped flag,
  `u32` tombstone count and per key `u32` slot, `u32` counter. S23b adds nothing for worlds without records (the bytes
  exist only per record, inside S23a's part, which `writesGenerationState()` gates), so S22's committed result hash and
  every flat or hook world's hash are unchanged; S23a's content scenarios get a hash change (no committed value exists
  for them).
- `diffSaves` reports placement version and tombstone differences per chunk (S22 reports sequences).

### 2. Tombstones (`engine/world/World.*`, `engine/content/Tombstones.{h,cpp}`)

- `void World::tombstone(ChunkId chunk, InstanceKey key)` (§12.5):
  - host only; on a client it stops with `SAIGA_EXIT_ERROR` in every build (R-NET-2: clients request through game
    actions), as S22's `editTerrain`;
  - **validated at the call** (between steps and during a step; `SAIGA_EXIT_ERROR` in every build, as S22's brushes;
    the world API checks first and returns errors): the space exists; the chunk lies in the space's chunk range (D12);
    the space's generator has the slot; `counter < candidatesPerChunk(slot)`; the key names an accepted, not excluded
    instance of the chunk at its effective placement version (this generates the chunk's `ChunkPlacement`
    synchronously through the cache; measured). The placement version cannot change between the call and the flush
    (only tombstones and loads change records; a load is not possible inside a step);
  - between steps it applies at once; during a step it is a command applied at the flush of the requesting phase
    (like `editTerrain`, §8.1); inside `parallelEach` it is refused like every command;
  - idempotent, checked when it applies: tombstoning a tombstoned key changes nothing (no sequence increment, no
    message, no notification);
  - applying it creates the record if needed (current version), inserts the key, calls `bumpSequence(chunk)`,
    `onRecordChanged(chunk)` and appends the entry to the host send list (Scope 3).
- `bool World::isTombstoned(ChunkId, InstanceKey) const`, `const ChunkRecord* World::chunkRecord(ChunkId) const`.

### 3. Messages (`engine/net/EngineMessages.*`, `engine/content/ContentMessages.{h,cpp}`)

Two raw engine messages after S23a's `GenerationState` in `EngineMessage`, host -> clients, `Channel::State`,
reliable, with the S16 epoch (ordered with `Resync`, the spawn set and S22's terrain messages; another epoch is
dropped). Applied in phase 1 (and by `stepNetworkOnly`).

| Message | Content |
|---|---|
| `TombstoneModification` | epoch (u32), space (u16), entry count (u16, 1..4096); per entry: chunk x, z (i32), placement version (u16), sequence after the change (u32), slot (u8), counter (u32) |
| `ChunkRecordState` | epoch (u32), space (u16), part count (u16, 1..1024), key total (u32, <= 8192); per part: chunk x, z (i32), placement version (u16), key count (u32), keys (slot u8, counter u32) |

- Sizes: a `TombstoneModification` is at most 19 bytes x 4096 + an 8-byte header (about 78 KB); a `ChunkRecordState` at
  most 1024 parts x 14 bytes + 8192 keys x 5 bytes + a 12-byte header (about 55 KB). A record whose tombstones exceed
  the key budget (up to `candidatesPerChunk` <= 2^24 keys) is split into several parts, each repeating the chunk and
  version, in consecutive messages; the client applies parts by union, so no "first part" flag is needed (records are
  only replaced wholesale after a `Resync`, which clears them first).
- **Host send list** (S22's ordered list of host state messages): a tombstone appends an entry to the last list entry
  if that is a `TombstoneModification` of the same space with fewer than 4096 entries, else a new message. A terrain
  edit's `TerrainModification` is never split (S22). The list order is application order, so per chunk the sequence
  order of terrain parts and tombstones is preserved on the reliable channel; S22's rules for aborted steps, edits
  between steps and `stepNetworkOnly` apply unchanged. Within an edit, tombstone-on-edit entries follow the edit's
  `TerrainModification`, in `ChunkId` then key order (Scope 6).
- **Snapshot** (E5): S22's `Replication::sendChunkSnapshot(peer)` sends this stage's `ChunkRecordState` messages
  before its `TerrainChunkState` messages. Order per join: `Resync` (epoch != 0), `GenerationState` (S23a), spawn
  set, `ChunkRecordState`s, `TerrainChunkState`s; per host load (inside `onHostLoad`): `Resync`, `GenerationState`,
  `ChunkRecordState`s, `TerrainChunkState`s, and the loaded entities as spawns in the next phase 8. Only records with
  tombstones or a placement version other than the current one are sent (stamped-only records equal the client's
  effective record), per space, split by the part and key bounds. As in S22, the snapshot is built from the state at
  the join's phase 8, which includes the step's applied tombstones; the send list's entries of that pass are not sent
  to the joining peer (no duplicate, no gap).
- **Client receive**, validate-then-apply per message (untrusted input; the whole message is decoded and validated
  before any entry applies; any failure counts in `NetStats::rejected`, logs with `SAIGA_LOG_E` category `content`, and
  applies nothing):
  - the epoch (another epoch: dropped, not counted); `generationMismatch()` not set (a client that refused the host's
    generation state rejects every record message);
  - the space exists; the chunk lies inside the space's range (D12); the counts within the bounds;
  - placement version between the client's pinned and **current** version (equal to the host's, S23a), and equal to
    the chunk's effective version when the chunk already has a record (a `TombstoneModification` for a chunk without a
    record must carry the current version). **Before S26 a client never receives another version than the current
    one**: S23a's `GenerationState` refuses a client whose pins differ from the host's, a fresh client pins the current
    versions, and a host's records lie between its pins and the current version. The older-version branch (a record
    created with that version, the set regenerated) is validated and kept, but dormant until S26's handshake adopts the
    host's pins; the set regeneration it triggers is tested on a host world (Tests);
  - for entries and parts **with keys** (E10): the space's generator has slots, slot below the slot count, counter
    below `candidatesPerChunk(slot)`, keys sorted and unique within the part. A part without keys (a record with a
    non-current version only) needs no slots;
  - `TombstoneModification` sequences, against S22's `sequence(chunk)` and **cumulatively** for several entries of one
    chunk in one message (each expects the previous entry's sequence + 1): `<= current` dropped (duplicate);
    `== current + 1` applied; larger: a gap, handled exactly as S22's (logged, counted in `modificationGaps`, applied;
    the host + client variants fail on any gap). Applying: record created if needed, key inserted, `setSequence`,
    `onRecordChanged`;
  - `ChunkRecordState`: per part the record is created with the part's version (or its version checked) and the keys
    are inserted (union); `onRecordChanged` once per chunk after the message.
  - The client does not re-validate that a key is an accepted candidate (that would generate chunks on message apply;
    a wrong key only hides nothing). It is checked on the host.
- **Client `Resync`**: one function `clearReplicatedChunkState()` clears S22's store (entries and blocks) and the chunk
  records together, then calls S22's `onTerrainChanged` per cleared chunk with deltas and `onRecordChanged` per cleared
  record (instances reappear until the snapshot's `ChunkRecordState` hides them again).
- Every peer hides the instance through its render set (Scope 4).

### 4. Content render sets (`engine/content/ContentRenderSystem.{h,cpp}`, `engine/sync/RecordingRenderSink.*`)

- **Only when needed** (the gate is who drives the content, not whether a sink exists; S20 finding 10 of round 3):
  `ContentRenderSettings::centres` is `None` by default, and then nothing happens: setting a render sink alone requests
  no set and submits no pool job (every scenario world has a `RecordingRenderSink`, S19b). Content sets exist only
  where a centre mode is set: `View` by `EngineApp` on its rendered world (through `updateView` / `completeView`,
  which only `EngineApp` calls; `--coop`'s hidden host world has neither a sink nor a mode) and `Players` by
  `ScenarioWorld` on every peer's world (asserted: a mode other than `None` needs a sink). Even then only spaces
  whose generator has slots or whose plan has static parts get sets, so S19b's and S20's instance-set counts and every
  existing scenario without such spaces are unchanged.
- **Centres** (`ContentRenderSettings::centres`, an enum `None | View | Players`):
  - `View`, set by `EngineApp` on its rendered world: `TerrainSystem::updateView(space, camera)` (S20, every rendered
    frame) also updates the content sets: the chunks of the camera's space whose square intersects the `fullRadius`
    disc around the camera (D5: sets of other spaces are dropped on a switch). The content part runs whether or not
    the space has terrain (`spaceHasTerrain`): in an enclosed or flat space S20's terrain part requests nothing and
    sends `clearTerrain`, and the content part still builds the static-geometry sets (a dungeon's rooms);
  - `Players`, set by `ScenarioWorld` on every peer's world: a content pass that runs in phase 8 after the replication
    pass and S19b's render extract (`World::step` calls it after `extractor->run()`), with the chunks within
    `fullRadius` of every `Player` entity of this world (on a client: every replicated `Player`), each in its space. It
    creates and removes sets; it does not run in `stepNetworkOnly`.
  - A set is removed when its chunk is more than `fullRadius + 256 m` from every centre (one chunk of hysteresis; S24
    replaces this with its radii and margins).
- **One instance set per chunk** (`RegionId{space, 0, x, z}`): the chunk's tier-1 instances in (slot, counter) order,
  then the static parts of the plan's sites in this chunk (`staticList`, key slot 255, engine reserved, counter = the
  part's index in the chunk's static list). A chunk with neither gets no set. One set per region keeps
  `instanceSetAt(RegionId)` unambiguous; each set keeps a sorted key -> index table (binary search) for tombstones and
  re-snaps, the applied tombstone list and the original (untombstoned, unre-snapped) instance of every index.
- `StaticInstance` from the integers (`ContentConvert`): chunk-relative float position (x / 256, y / 1024, z / 256;
  exact in float: below 2^24 units), rotation from the normalized yaw quaternion as S18's snorm pairs, `scale / 1024`,
  mesh through the slot's mesh table and `World::findMesh`, key `slot << 24 | counter`. Tombstoned keys get scale 0
  (`removeInstance` semantics). An unknown mesh name is logged once per (generator, slot, mesh) (`SAIGA_LOG_E`,
  category `content`), and those instances get scale 0 (not drawn; indices stay stable). For a chunk with terrain
  deltas, `ReSnapOnEdit` instances get y from the current surface (Scope 6) at creation.
- **Generation**: with `View` on S20's `StreamingPool` through S23a's pool form
  (`pipeline.contentSource(space).requestPlacement(pool, chunk, placementVersion, priority, completion)`: the job holds
  a `ContentSource` copy, never the `World`), drained in `updateView` and budgeted
  (`TerrainViewSettings::contentSetsPerFrame`, default 4); synchronous (`pipeline.chunkPlacement(chunk, version)`)
  with `Players` and in `completeView`, which also waits for the content sets, so screenshots and goldens see them. A
  result applies "as the state is at that moment": the chunk's current tombstones and re-snaps are applied to the array
  on the main thread before the upload.
- **Record changes** (`onRecordChanged(chunk)`, Scope 1) reach the sink **immediately**, in both modes and also inside
  `stepNetworkOnly` (one rule: the pass only creates and removes sets), for a chunk with a live or requested set:
  - the effective placement version changed (a load with an older version, a `Resync` clear of such a record; from a
    client message only after S26, Scope 3): the live set is removed and requested again at the new version (its
    grass mask too, Scope 5); with `Players` the next pass builds it synchronously;
  - otherwise the tombstones are diffed against the set's applied list: a new key -> `removeInstance(set, index)`; a
    key no longer tombstoned (a `Resync` clear, a host load) -> `updateInstances(set, index, {original instance})`
    (the sink's `updateInstances(InstanceSetHandle, uint32_t first, const std::vector<StaticInstance>&)`, re-snapped
    if the chunk has deltas), so the instance reappears.
  - Pool results: every chunk has a request counter, incremented when its set is requested or removed; a pool job
    carries the placement version and counter it started with, and its result is dropped (and the set requested again)
    when either differs at drain time. Tombstones and re-snaps never drop a result: the drain applies the chunk's
    current tombstones and re-snapped y to the array before the upload (above).
- `RecordingRenderSink` keeps each live set's region and full instance array and applies `updateInstances` and
  `removeInstance` to it (today it stores only set ID -> capacity, `RecordingRenderSink.h:92`): `instanceSet(handle)`,
  `instanceSetAt(RegionId)` (asserts at most one live set per region). Content sets use `RegionId{space, 0, x, z}`;
  `PrototypeForest`'s per-chunk tree sets (S19a, level 0) are added on `RenderScene` directly, not through the sink, and
  never share a space with content sets (`game_world` and racing have no slots; S23c drops flight's prototype forest),
  so no region collides.
- **Readiness** (E8): `World::spaceReadiness(space, camera, radius)` (S21) also counts the desired content sets of the
  space within the radius (an enclosed space: its static-geometry set); the loading screen of S21 waits for them.
- **Scenario worlds** have a `RecordingRenderSink` on every peer (§17.2, S19b) and use `Players`, so content sets exist
  on host and clients in scenarios with a content setup (amendment 7: §17.2's "requests no visual tiles or far
  content" gains "but builds content render sets around players").

### 5. Grass masks from the generator (E3; `engine/terrain/TerrainSystem.*`)

- **Generator worlds** (a space whose generator has `legacyTerrain() == false`): S20's `TerrainSystem` takes each
  chunk's base mask from `ChunkPlacement::grass` at the chunk's effective placement version instead of S20's rule,
  through S20's single send point `sendGrassMask` (pool results, a new sink, a rebuild). A placement-version change of
  a chunk (`onRecordChanged`) re-sends its mask. A tombstone does not touch the mask.
- **Legacy worlds** (`HookGenerator`, the prototype fixture; round 3): no `ChunkPlacement` is generated for masks
  (it would evaluate the hook a second time for every level-0 sample, next to S20's tile generation). The mask is S20's
  rule over S20's generated level-0 tile, as today, with the thresholds of the generator's `grassEditRule()` (below;
  `HookGenerator` returns its constructor thresholds); without a rule the mask is all zero. `HookGenerator::grassMask`
  (S23a) computes the same bytes from the hook and stays the hashed definition; the
  `GrassFromGenerator.*` test checks that both paths agree.
- **Edited chunks** (S22's path: on the pool with a copy of the merged heights, capped per frame; a result is dropped
  only if the chunk's **terrain-only revision** changed since submission, F1: S22's counter `TerrainChunkEdits::revision`,
  bumped only when deltas change; a tombstone bumps the sequence but never the revision, so it never drops or
  re-queues a mask):
  - generator worlds: mask = the generator mask AND S20's slope and sea rule over the merged heights
    (`prototypeGrassMask(level0 merged, layout, seaLevel, minNormalY, minAboveSea)`), with the thresholds from the
    generator: `virtual GrassEditRule Generator::grassEditRule() const` (new, default `{enabled = false}`; integers:
    `minNormalYQ16`, `minAboveSeaUnits`; converted to S20's float parameters outside `gen/`). Without a rule an edited
    chunk keeps its generator mask;
  - `legacyTerrain()` generators: S22's rule over the merged heights alone, as in S22 (its crater-wall test unchanged).
- The prototype's thresholds are the `HookGenerator`'s constructor arguments (content, fixed at registration), so
  `game_world`'s and racing's grass and S20's engine goldens are unchanged (checked). S20's run-time change of
  `grassMinNormalY` / `grassMinAboveSea` through `setViewSettings` no longer regenerates masks; S20's test of that
  behaviour is replaced by a test that two setups with different thresholds give different masks.

### 6. Terrain edits: tombstone-on-edit and re-snap (S22's edit and apply paths)

- **Tombstone-on-edit (host, state)**, inside S22's `applyTerrainEdit` through the extension point S22 names in its
  steps 4 and 5 (F2 of review round 3), so an edit is atomic:
  - in S22's **reserve step 4** (before anything is committed; a throw leaves the world unchanged): for every chunk
    the edit changes in a space whose generator has `TombstoneOnEdit` slots, the chunk's `ChunkPlacement` at its
    **effective placement version** (its record's, else the current one; synchronous, cache; measured); the
    non-tombstoned instances of those slots whose position lies in the changed box grown by one base spacing; their
    surface y recomputed with the integer surface rule from the **post-edit** samples (the kernel's output `h'` where
    changed, the current height elsewhere); every key whose y differs from its generated y **at that version** is
    collected, in `ChunkId` then key order. Still in step 4, everything the commit will write is reserved: the records
    of chunks without one, the capacity of each record's key vector, and the send-list entries (the
    `TombstoneModification` entries appended to S22's reserved list behind the edit's `TerrainModification`);
  - in S22's **commit step 5**, which only writes (no allocation, no throw), after S22's deltas, sequences and the
    `TerrainModification` entry: per collected key the record is created (reserved) if needed, the key inserted,
    `bumpSequence`, the reserved send-list entry filled (Scope 3), then `onRecordChanged` per chunk and S22's
    `onTerrainChanged`. The re-snap of `ReSnapOnEdit` instances uses only the `onTerrainChanged` queue (below);
  - `TerrainEditResult` gains `std::vector<std::pair<ChunkId, InstanceKey>> tombstoned` (that order); the world API's
    `edit_terrain` reply gains `tombstoned: [{chunk: [x, z], slot, counter}]`. Independent of a render sink.
- **Re-snap (every peer with content sets, derived)**: S22's `onTerrainChanged(space, box)` (every applied change on
  every peer: host edit, client `TerrainModification`, `TerrainChunkState`, `Resync` clear, host load) hands the box to
  a queue owned by `ContentRenderSystem` (only while it has sets; split per chunk, one merged box per chunk). The queue
  is drained in `updateView` (`View`) or in the phase-8 content pass (`Players`): for each live set intersecting a box,
  y is recomputed for `ReSnapOnEdit` instances in the grown box with the surface rule over `terrainSample`, and
  `updateInstances` is called for the changed index range. Equal heights on all peers (S22) give equal y. S22's visual
  dirty queue is not used (it exists only for the visual tile path).
- `std::optional<Vec3> World::tierOnePosition(ChunkId, InstanceKey)`: the exact conversion of the instance's position
  (generates `ChunkPlacement` synchronously); `ReSnapOnEdit` instances at the current surface, `TombstoneOnEdit`
  instances at their generated y; empty for a key that is not an accepted, non-excluded candidate or out of range; a
  tombstoned key still returns its position (callers check `isTombstoned`). For game code and tests; nothing in the
  simulation collides with tier 1 before S28.
- Pieces are handed to the game in S30 (§12.2).

### 7. World API and MCP tools (`engine/api/WorldApi.*`, `engine/tools/WorldTools.*`)

| Operation / tool | Contract |
|---|---|
| `chunk_info(space?, chunk?: [x, z], position?: [x, z])` | reads: the chunk, `placement_version` (effective), `recorded`, `stamped` (host), `sequence` (S22's store), `tombstones` (slot name, counter; at most 200, `truncated`), instance counts per slot, the site keys of the chunk, `content_hash` (generated on demand); errors for a chunk outside the space |
| `list_sites(space?, near?: {position, radius}, limit? (100, <= 1000), after?)` | reads: per site the key, prefab, version, position, quarter turns, base height (metres), footprint, linked space IDs, the stamped state of its chunks; paged with `next` |
| `tombstone(space?, chunk: [x, z], slot, counter)` | changes: tombstones one tier-1 instance (host only; on a client the read-only error of S16); slot by name; errors for an unknown slot, a counter out of range, a rejected or excluded candidate, a chunk outside the space; idempotent (`changed: false`); returns the new sequence |

- `edit_terrain` (S22) reports `tombstoned` (Scope 6).
- `describe_world` adds `generators: [{name, version, pinned, current, slots: [{name, flags}], integer_only}]`;
  `world_status` adds `generation: {seed (string), mismatch, checked}`.
- Each description states what it reads and changes (R-MCP-1); each has a `handleMessage` test.

### 8. Golden `generated_site` (`tests/test_engine_terrain_render.cpp`, S20's engine-linked render target; E12)

- Two headless worlds from `contentTestSetup(2)` with the overworld template's `fullRadius` lowered to 300 m (a setup
  override in the test), `centres = View` and the same fixed camera: world A with a `VulkanRenderSink` over a
  `RenderScene` (the image and the drawn counts), world B with a `RecordingRenderSink` (the exact set arrays).
- Camera: a fixed pose at site 0's `baseHeight` + 70 m, 90 m south of its origin (1100, 1300), looking at the origin
  at `baseHeight`, so the
  site's 64 m footprint, its patch, the static pillar and the instances around it fill the view; the test meshes'
  LOD end distances (4 km) exceed every camera distance.
- The tombstoned key: the `test_tree` instance of chunk (4, 5) nearest to the site origin outside its exclusion
  (computed from `ChunkPlacement`, so it is in view).
- Count check (the delta, not absolute counts: S18's culling and the sets of neighbour chunks within `fullRadius` make
  absolute counts camera dependent): `trackMeshStatistics` on the five test meshes; `completeView`, render, read the
  drawn instances per mesh summed over the LODs; tombstone the key; `completeView`, render again: the count of the
  tombstoned instance's mesh drops by exactly 1, every other mesh's count is unchanged. Statistics lag the frames in
  flight, so each read takes the statistics of the rendered frame itself: `scene->takeStatistics(out)` until
  `out.frame` equals the `scene->frames()` value right after that render (rendering further frames until it
  arrives), never `statistics()` of an older frame.
- Set check: world B's `instanceSetAt(RegionId{0, 0, 4, 5})` equals, element by element, the conversion of the
  chunk's `ChunkPlacement` instances followed by its `staticList` parts, with the tombstoned index at scale 0.
- Image: the frame after the tombstone is compared with the new golden `data/tests/golden/generated_site.png`
  (lavapipe gating, §17.6; other devices print the metrics). It shows the terrain with the site's patch, the tier-1
  test meshes, the pillar and the gap of the tombstoned tree. Created on lavapipe in a dedicated commit (AGENTS.md
  procedure, filter `EngineTerrainRenderGolden.GeneratedSite`).

## Out of scope

- Streaming radii, prefetch, simulation-ready results, arrivals (S24). The render chunk set is the stopgap named in
  §12.1's "S23 scope".
- Per-chunk snapshots on `ChunkData`, re-requests on gaps, the handshake (S26). A runtime content-hash comparison
  between peers (§20).
- Records of visited but unmodified chunks (S25's chunk store).
- Far content, region summaries of tombstones (S27). Tier-1 and static colliders (S28). Pieces (S30). Tombstones from
  interaction and promotion (S31).
- The flight demo's generator (S23c).

## Design notes

- **Re-snap is derived, tombstone-on-edit is state.** A re-snapped y follows from the edited heights on every peer, so
  nothing is sent or saved. A tombstone must survive a later edit back to the old height and is sent like any
  tombstone.
- **Tombstone-on-edit inside the edit's compute step**: the chunk generation and the y comparison can throw (cache,
  generator asserts); doing them before S22's commit keeps an edit atomic (terrain and its tombstones together or
  nothing).
- **Changed boxes, not the visual queue**: the re-snap must happen on every peer with content sets, including
  scenario clients that never stream tiles; the applied boxes exist on every peer in exactly those places.
- **The sequence stays in S22's store** (E10): one counter per chunk for all modification kinds, in one place, saved by
  `TERR` and sent by S22's messages; records only add what tombstones need. No second copy, so no equality check.
- **Records only where state needs them** (E10): tombstoned chunks, stamped chunks (host) and non-current versions.
  The terrain messages need no placement version: a record that a client does not know of has the current version,
  which the client uses anyway.
- **One notification for record changes**: tombstones are not the only writer; a `Resync`, a host load, a snapshot or a
  version change can also add, remove or replace keys. Funnelling all of them through `onRecordChanged` keeps the sets
  equal to the records, including instances that must reappear.
- **One set per chunk** with statics appended under the engine-reserved slot 255: a single lookup per chunk, no
  ambiguity in `instanceSetAt`, and one draw registration per chunk; empty chunks cost nothing.
- **Clients do not validate candidate acceptance**: it would generate chunks inside message apply; the host already
  validated it, and a wrong key on a client can only hide an instance of a host bug, which the host + client
  comparison of tombstone sets catches.
- **Snapshot size**: only records with tombstones or an older version are sent, in messages bounded by parts and keys,
  so a host with thousands of tombstoned chunks or a chunk with millions of tombstones builds no unbounded message.
- **The golden checks a count delta**: absolute drawn counts depend on S18's culling, the LODs and the neighbour
  chunks' sets; the delta of one tombstone and the exact set arrays of a recording world test what this stage adds.

## Implementation steps

Each step builds (warnings as errors, as CI) and passes ctest with the tests named in it (round 3).

1. **Reconcile** with S23a's and S22's final code (record type, accessors, `bumpSequence`, the terrain-only revision,
   the reserve/commit extension point, send list, `sendChunkSnapshot`, changed boxes, end condition, the edited-chunk
   grass path) and S20's render test target.
2. **Records**: `ChunkRecord` with tombstones, `onRecordChanged` (no listeners yet), effective records,
   `compareReplicas` and the end condition, `CREC` format 2 (reading 1), `world.hash`, `diffSaves`, the test-only
   `ChunkRecords::injectForTests(ChunkId, placementVersion, keys)` (later steps' bound and regeneration tests use it);
   `ChunkRecords.*`.
3. **Tombstones**: `World::tombstone` / `isTombstoned` / `chunkRecord`, validation at the call, commands,
   `bumpSequence`; `Tombstone.*`.
4. **Messages**: `TombstoneModification`, `ChunkRecordState` (part and key splitting), the send-list entries,
   `sendChunkSnapshot`'s record part, client validation (cumulative sequences, keys only with slots), the shared gap
   counter, `clearReplicatedChunkState`; `TombstoneNet.*` checking records, sequences and messages (no render sets
   yet: the host-load test's recording-sink assertion is added in step 5).
5. **Content render sets**: `ContentRenderSystem` (centres `None | View | Players`, sets, immediate record changes,
   request counters), `RecordingRenderSink` set contents, readiness, `ScenarioWorld` (`Players`) and `EngineApp`
   (`View`); `ContentRender.*` except the re-snap at creation (step 7), the sink assertion of `TombstoneNet`'s host-load
   test, and the scenario `tombstone_hides_instance`.
6. **Grass**: masks from the generator through `sendGrassMask`, the legacy path from S20's tile, `grassEditRule`, the
   edited-chunk combination keyed on the terrain-only revision; `GrassFromGenerator.*`.
7. **Edits**: tombstone-on-edit in S22's reserve and commit steps, `TerrainEditResult::tombstoned`, the re-snap queue,
   `tierOnePosition`; `ContentEdit.*`, the `ContentRender` test of a set created after an edit (re-snapped y), the
   scenario `terrain_edit_resnaps_and_tombstones`.
8. **World API and MCP tools** with their `handleMessage` tests.
9. **Golden** `generated_site` (S20's render target exists, E12) and its count check; the image in a dedicated commit.
10. **Docs** (`docs/engine_design.md` amendments, `AGENTS.md` test list, count and golden recipe).

## Tests

`tests/test_engine_content.cpp` (S23a's file, extended; all worlds from `tests/content_test_setup.h`):

- `ChunkRecords.*`: a terrain edit on a `FlatGenerator` space and on a test-generator space creates no chunk record
  (only S22's sequence); a tombstone after an edit on one chunk continues S22's sequence (edit 1, tombstone 2), and a
  tombstone on an unedited chunk creates S22's entry without deltas (sequence 1) that `TERR` saves; `CREC` format 2
  round trip; format 1 (`save_s23a.bin`) loads with no tombstones; unsorted keys, a slot out of range, a counter out
  of range and tombstones in a slot-less space are load errors; `world.hash` changes with a tombstone, is equal after
  save/load and unchanged for a flat world with edits; `save_s23b.bin` (records, tombstones, an edited chunk, a
  stamped chunk, a tombstone-only chunk) loads.
- `Tombstone.*`: an invalid key (rejected candidate, excluded candidate, counter out of range, chunk outside the
  space) stops on the host at the call, also inside a step (death tests, every build) and the API returns errors; a
  client call stops; idempotence (sequence unchanged, nothing sent); inside a step a valid key applies at the phase
  flush.
- `TombstoneNet.*` (loopback host and client):
  - an old sequence is dropped; a gap is logged and counted in `modificationGaps`; another epoch is dropped; two
    entries of one chunk in one message apply with consecutive sequences, and a message whose second entry skips is a
    gap only for that entry;
  - rejected, counted, nothing applied: a chunk outside the space, a version above the client's current one, a
    version differing from the chunk's effective one, a slot beyond the table, unsorted keys, keys in a space without
    slots, a message with one invalid entry among valid ones (no entry applies), any record message after a refused
    `GenerationState`;
  - a late joiner after a terrain edit on a `FlatGenerator` space: no rejections, equal sequences (E10);
  - a late joiner gets the records of chunks tombstoned before it joined, `ChunkRecordState` before `TerrainChunkState`
    (E5 order in the loopback transport's log); a joiner in a step with pending tombstones receives them in the
    snapshot and not again live (no duplicate, no gap);
  - host load (`Resync`): the client clears records and S22's store in one call; an instance tombstoned only after the
    save reappears in the client's recording sink (scale restored), one tombstoned in the save stays hidden after the
    `ChunkRecordState`; equal sequences afterwards;
  - bounds: 4096 tombstones in one step arrive in one message, 4097 in two; 3000 records (test-only injection
    `ChunkRecords::injectForTests`, one key each, chunks of the 128 x 128 overworld) in three `ChunkRecordState`
    messages; one injected record with 10000 `test_tree` keys (counters 0-9999; the slot has 16384 candidates per
    chunk, S23a) in two messages, which union to the full set.
- `ContentRender.*` (`RecordingRenderSink`, `Players` unless noted):
  - one set per chunk around players with tombstoned instances at scale 0 and the static parts after the tier-1
    instances; `removeInstance` on a later tombstone; no set for an empty chunk; no sets and no pool job with
    `centres == None` although a `RecordingRenderSink` is set (the content pass counter stays 0), in a setup without
    slots and static parts (S19b's and S20's counts unchanged), and on the sink-less host of a loopback pair;
    `centres != None` without a sink asserts;
  - `View` with the camera in `test_dungeon` (no terrain: S20 sends `clearTerrain` and requests nothing): the
    dungeon's static-geometry set is built and `spaceReadiness` reaches 1;
  - a client record change applied in `stepNetworkOnly` reaches the client's recording sink at once (no full step);
  - an instance reappears after a client `Resync` clear (`updateInstances` with the original instance) and is hidden
    again by the following `ChunkRecordState`;
  - older placement versions, on a single host world with a recording sink and `Players` (the client branch is dormant
    until S26, Scope 3): a save of a `TestTerrainGenerator(1)` world (pins v1, chunk (4, 5) recorded at v1 by a
    tombstone) loaded into `contentTestSetup(2)`: the set of (4, 5) equals the v1 placement with the tombstone, an
    unrecorded chunk's set the v2 placement; then `injectForTests` gives the unrecorded chunk a v1 record and calls
    `onRecordChanged`: its set is removed and rebuilt at v1;
  - a pool result whose request counter changed before the drain (the set was removed and requested again) is dropped
    and the set requested again (`View`, held job); a tombstone during a held job does not drop the result, which
    arrives with the tombstoned instance at scale 0;
  - the recording sink's arrays follow `updateInstances` / `removeInstance`; a second live set for one region asserts;
  - sets removed beyond the hysteresis; an unknown mesh name logged once and at scale 0; a set created after an edit
    has re-snapped y (step 7); `completeView` waits for the sets; `spaceReadiness` below 1 until the sets of the
    radius exist.
- `ContentEdit.*`: an `Add` brush over a `test_rock` and a `test_tree` of chunk (4, 5): the rock's y in the set and
  `tierOnePosition` equal the edited surface; the tree is tombstoned, its sequence after the terrain part's, listed in
  `TerrainEditResult::tombstoned` and in the `edit_terrain` reply; a test hook that makes the chunk generation throw
  inside the edit leaves the terrain unchanged (no deltas, no sequence, no message); an edit in a space without slots
  does no content work (no `ChunkPlacement` generated: the cache counter stays 0); a client re-snaps from an applied
  `TerrainModification`; `tierOnePosition` of a rejected candidate is empty.
- `GrassFromGenerator.*`: the test generator's masks reach the sink through `sendGrassMask`; a prototype setup's masks
  equal S20's rule byte for byte, computed from S20's level-0 tile without a `ChunkPlacement` (the cache's generation
  counter stays 0) and equal to `HookGenerator::grassMask`; an edited chunk of the test generator gets its mask AND
  the rule over the merged heights; an edited prototype chunk gets the rule alone (S22's crater-wall test unchanged);
  two setups with different thresholds give different masks; F1: a tombstone in an edited chunk while its mask job is
  held on a latch does not drop the result (the mask arrives, `maskResultsDropped` unchanged), while a second edit
  does drop it.

`tests/test_engine_world_api.cpp`: `chunk_info`, `list_sites` (paging), `tombstone` (incl. the client error, an
unknown slot, idempotence), `edit_terrain`'s `tombstoned` and `describe_world`'s generators / `world_status`'s
generation block, through `handleMessage`.

`tests/test_engine_scenarios.cpp` (`registerContentScenarios` on `contentTestSetup(2)`, every variant; the setup
registers the test meshes on every peer):

- `tombstone_hides_instance` (the S23b acceptance):
  - the scenario spawns a player (owned by the last peer in the host + client variants) at (1150, 1400), inside chunk
    (4, 5) next to site 0, so the chunk is within `fullRadius` (128 m in scenarios) on every peer, and steps (bounded:
    `stepUntil`, at most 120 steps) until `instanceSetAt(RegionId{0, 0, 4, 5})` exists in every peer's recording sink
    (a client builds it once the replicated `Player` has arrived);
  - the host tombstones the first accepted `test_tree` key of chunk (4, 5) through `call("tombstone")`;
  - the scenario steps until every client's effective record equals the host's;
  - host and every client report `isTombstoned`, and every peer's recording sink has that instance at scale 0 in
    `instanceSetAt(RegionId{0, 0, 4, 5})`, while the other instances keep their scale;
  - only in the `SaveLoad` variant (`ScenarioWorld::saveLoad` aborts in the host + client variants): after
    `saveLoad()` and one step (the fresh world's set is built by its first content pass), the host's sink still has
    the instance at scale 0. The `Resync` case is `TombstoneNet.*`'s host-load test.
- `terrain_edit_resnaps_and_tombstones`: an `Add` brush over a rock and a tree of chunk (4, 5); on host and clients the
  rock's rendered y equals the edited surface and the tree is tombstoned.

`tests/test_engine_terrain_render.cpp` (S20's engine-linked render target, Vulkan guard):
`EngineTerrainRenderGolden.GeneratedSite` (Scope 8; lavapipe gating, the count delta and the set check on every
device).

Benchmarks (R-TEST-2; local, reported): the host cost of `tombstone` and of the tombstone-on-edit step on a cold and a
warm cache; content set creation per chunk (generation and conversion) for the test generator; `game_world --benchmark
flyover` and `forest` unchanged within 5 % (no slots: no content work; grass from the prototype `HookGenerator`).

## Done when

The S23b row of §22 (S23a amendment 1):

- **A tombstoned instance is hidden on the client in the host + client variants**: `tombstone_hides_instance` passes
  in `HostClient`, `HostClientDelay` and `HostClientAsync`, checking every client's recording sink (CI).
- **`generated_site` golden** passes on lavapipe with its count delta and set check (CI).

Also: the remaining tests above (CI); S22's terrain tests unchanged (CI); `game_world`'s grass and S20's engine
goldens unchanged (CI and local); the benchmark numbers (local); `docs/engine_design.md` and `AGENTS.md` updated.

## Verification

- The CI recipe of AGENTS.md (samples ON for the games' compile check).
- `ctest -R "test_engine_(content|scenarios|world_api|save|terrain.*|replication)" --output-on-failure`.
- `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R test_engine_terrain_render --output-on-failure`.
- `game_world --demo` and `game_racing --demo` screenshots unchanged (grass from the generator).

## Files

New: `src/saiga/engine/content/{ContentRenderSystem,Tombstones,ContentMessages}.{h,cpp}`,
`data/tests/engine/save_s23b.bin`, `data/tests/golden/generated_site.png`.

Changed:

- engine: `engine/content/{ChunkRecords,GenerationPipeline,ContentConvert,Generator}.*` (`grassEditRule`),
  `engine/world/World.{h,cpp}` (`tombstone`, `isTombstoned`, `chunkRecord`, `tierOnePosition`, the phase-8 content
  pass), `engine/terrain/{TerrainEdit,TerrainSystem}.*` (tombstone-on-edit in the reserve and commit steps,
  changed-box hand-off, grass from the generator, content sets in `updateView` / `completeView`, readiness),
  `engine/save/{SaveFormat.h,WorldSave.cpp,WorldLoad.cpp,SaveDiff.cpp}`, `engine/net/{EngineMessages,Replication}.*`
  (`sendChunkSnapshot`'s record part, the send-list entries, `clearReplicatedChunkState`),
  `engine/sync/RecordingRenderSink.{h,cpp}`, `engine/api/WorldApi.{h,cpp}`, `engine/tools/{WorldTools,ScenarioWorld}.*`
  (end condition, comparison, `centres = Players`), `engine/app/EngineApp.cpp` (`centres = View`).
- tests: `tests/test_engine_{content,scenarios,world_api,terrain_render}.cpp`, `tests/content_test_setup.h` (the golden
  helpers), `tests/CMakeLists.txt` (no new target).
- docs: `docs/engine_design.md`, `AGENTS.md`, this file.

## Risks and open questions

1. **Synchronous chunk generation on the host** for tombstone validation and tombstone-on-edit: one `ChunkPlacement`
   per affected chunk (an edit touches at most 2 x 2 chunks, S22). Measured; S24 moves simulation generation to the
   pool and can pre-generate around edits.
2. **Render chunk centres**: apps use the camera, scenario worlds the players. A client app renders tier 1 only around
   its camera; the hidden-on-client check uses the scenario rule (players).
3. **Unbounded record growth** on clients is limited to records with tombstones or older versions; S25's chunk store
   and S26's per-chunk snapshots replace the stopgap.
4. **The golden depends on S20's engine-linked render target** (E12, hard dependency): step 9 starts only when it is
   merged; there is no recording-sink fallback for the image.

## Design-doc amendments

1. §12.1 chunk records: `ChunkRecord {placementVersion, tombstones}` on every peer; the modification sequence stays in
   S22's store for every modification kind (E10); records only for stamped (host), tombstoned and older-version
   chunks; `stamped` host-only; the effective record of a chunk without one; `onRecordChanged`; `CREC` format 2.
2. §12.5 `World::tombstone`: host only, validated at the call (chunk range, slot, counter, accepted candidate),
   idempotent, applied at the requesting phase's flush; `isTombstoned`, `chunkRecord`, `tierOnePosition` (optional).
3. §13.1 / §13.2: `TombstoneModification` and `ChunkRecordState` on `Channel::State` with the epoch, the join and
   host-load order of E5 (`GenerationState`, spawn set, record states, terrain states), the size bounds (entries,
   parts, keys; record parts united), client validation (cumulative sequences, keys only with slots); tombstone gaps
   count in S22's `modificationGaps`; S22's terrain messages unchanged.
4. §12.2 terraforming: re-snapped y is derived (not saved or sent) and driven by applied changed boxes on every peer
   with content sets; tombstone-on-edit keys are computed in the edit's reserve step and committed with it, after its
   terrain part, and replicated; `edit_terrain` reports them.
5. §10.1 / §12.1: one instance set per chunk (tier 1, then static parts under the engine-reserved slot 255); no set
   for an empty chunk; sets follow record changes at once (instances reappear); content render centres `None | View |
   Players` (`None` by default: a sink alone builds nothing); grass
   masks from the generator from S23b (legacy generators keep S20's tile path), edited chunks combined with the
   generator's `grassEditRule` and re-queued only on S22's terrain-only revision (E3, F1; the prototype's grass
   thresholds are fixed at registration, no longer view settings).
6. §16.2: the tools `chunk_info`, `list_sites`, `tombstone`; `edit_terrain`'s `tombstoned`; `describe_world`'s
   generators; `world_status`'s generation block.
7. §17.2: scenario worlds build content render sets around players (they still request no visual tiles or far
   content); §17.3 / S22's end condition and comparison cover effective chunk records.
8. §12.3 / E8: `World::spaceReadiness` counts the content sets of the space.
9. §17.6: the `generated_site` golden (engine test generator, count delta, set check).

## Review log

Round 1 (Opus, Fable; `review_r1_S23.md`; decisions D1-D14). This plan was split off the S23 plan (D13); the S23a
plan logs the findings it owns.

- 0 Split: this is S23b (D13).
- 4 (tombstone part) Chunk generation for tombstone validation and tombstone-on-edit stays synchronous on the host
  (at most 2 x 2 chunks per edit, one per tombstone) and is measured (Risks 1).
- 15 Records and S22: the sequence moves into the record; the effective record (missing = current version, sequence
  0, no tombstones) is what `compareReplicas` and the end condition compare, including tombstone sets and placement
  versions; `stamped` stays host-only and out of `ChunkRecordState`; clients create records with the placement
  version carried by S22's terrain messages (Scope 1).
- 16 Re-snap trigger: driven by the applied changed boxes (host scatter, client message apply) into a queue owned by
  `ContentRenderSystem`, not S22's visual queue; scenario worlds have a recording sink on every peer; §17.2 amended
  (amendment 7).
- 17 Ambiguous sets: one set per chunk with statics under slot 255; no set for empty chunks; no content work for
  spaces without slots and static parts (existing counts unchanged).
- 19 Client validation and message size: chunk-range check on every message; `ChunkRecordState` split per space and
  per 1024 records and limited to non-default records; rejections (version above the client's or below the pin,
  version differing from a record, slot beyond the table) counted. Slot-table mismatches are refused earlier by S23a's
  `GenerationState`.
- 1, 2, 3, 5-14, 18, 20, 21: S23a (pipeline, seed, hashes, cache, saves, stamping, locale).
- Rejected: none.

Round 2 (Opus 6, Fable 6; merged findings in `review_r2_S23.md`; cross-plan decisions E1-E14, which override D1-D14
where they differ). Findings 1-18 of the S23b part:

- 1 Slot-less records rejected: E10/E11; `ChunkRecordState` and `TombstoneModification` need slots only for entries
  with keys; test: a late joiner after a terrain edit on a `FlatGenerator` space, no rejections (Scope 3, Tests).
- 2 Host-load order: E5 (Scope 3 Snapshot).
- 3 Sets never invalidated: one `onRecordChanged(chunk)` from every record writer; a version change regenerates set and
  mask, otherwise tombstones are diffed (`removeInstance`, or `updateInstances` with the original instance to restore);
  per-chunk generation counters drop stale pool results; tests for the reappearing instance, the older version and the
  stale result (Scope 1, Scope 4). The S23a re-pin case is a load error now (E11).
- 4 Grass of edited chunks: E3; generator worlds combine the generator mask with S20's rule over the merged heights
  (`grassEditRule`), legacy generators keep S22's rule alone (Scope 5); no S22 text change needed.
- 5 `CREC` / `TERR` sequences: E10; no sequence in records, `CREC` or `ChunkRecordState`; no equality check.
- 6 Message sizes: `ChunkRecordState` bounded by 1024 parts and 8192 keys, a large record split into parts that union;
  `TombstoneModification` at 4096 entries, one message per space and run, list order kept; the joiner rule (snapshot
  includes the step's tombstones, its live entries are not sent to it); the "sequence below current" case is gone with
  E10; tests at both bounds.
- 7 Send list: S22's ordered list of messages, tombstones appended to a running `TombstoneModification`; an edit's
  `TerrainModification` never split; cumulative client validation of several entries per chunk, the whole message
  validated before any entry applies; tombstone-on-edit in `ChunkId` then key order.
- 8 Client version check: between pinned and the current version (equal to the host's by S23a's refusal), equal to the
  chunk's effective version; the test's "version above" means above current.
- 9 Acceptance scenario: meshes registered on every peer by `content_test_setup.h`; a player inside chunk (4, 5) on
  every peer; save/load checked only in the `SaveLoad` variant through the host's sink after one step; the `Resync`
  case moved to `TombstoneNet.*` with a sink check.
- 10 Centres: `ContentRenderSettings::centres = View | Players`, set by `EngineApp` and `ScenarioWorld`; a client's
  players are all replicated `Player` entities; the pass runs after replication and the extract in phase 8, not in
  `stepNetworkOnly`; a world without a sink does no work.
- 11 Records in slot-less spaces: E10; none on clients, only S23a's stamped ones on the host; S22's hashes unchanged;
  no placement version in S22's messages.
- 12 The `first` flag is deleted; `clearReplicatedChunkState` clears S22's store and the records in one function on
  `Resync`, then `onTerrainChanged` / `onRecordChanged` per cleared chunk.
- 13 End condition: moot with E10 (sequences come only from S22's messages and `TombstoneModification`; stated in
  Scope 1).
- 14 Edit atomicity: generation and y comparison in S22's reserve step 4 (against the kernel output), tombstones
  committed in step 5; a throwing generation leaves the terrain unchanged (test); `World::tombstone` validated at the
  call; `tombstoned` in `edit_terrain`'s reply and its test.
- 15 Hash table: one byte table in S23a (S22's block, then generation state and records), S23b appends the tombstones;
  hash changes stated; "terrain record" (S22, deltas) vs "chunk record" (this stage) defined in Scope 1.
- 16 `CREC` slot u8 like the messages; full byte layout incl. per-space counts.
- 17 Tests: 3000 records by test-only injection in the 128 x 128 overworld; `tierOnePosition` returns an optional
  (empty for rejected keys, a position for tombstoned ones); `RecordingRenderSink` stores full arrays and applies
  `updateInstances` / `removeInstance`; region collisions with `PrototypeForest`'s sets excluded (they bypass the sink
  and never share a space with content sets), one live set per region asserted.
- 18 `generated_site` golden moved here (E12, Scope 8): a count delta of one tombstone instead of absolute counts,
  `trackMeshStatistics` on the five test meshes summed over LODs, LOD end distances beyond the camera, a camera framing
  the 64 m site, exact set arrays through a parallel recording world, the test meshes named in `content_test_setup.h`.
- Rejected: none.

Round 3 (Opus 8, Fable 8; findings 1-13 of the S23b part of `review_r3.md`, plus the cross-plan decisions F1-F4 of
that file):

- 1 Mask staleness (F1): edited-chunk mask results are dropped only on S22's terrain-only revision (bumped only when
  deltas change); a tombstone never drops or re-queues a mask; test with a tombstone during a held mask job (Scope 5,
  Tests).
- 2 Older-version client test: the client branch is dormant until S26 (`GenerationState` refuses differing pins), stated
  in Scope 3; the regeneration is tested on a single host world (a v1 save loaded into `contentTestSetup(2)`, then
  `injectForTests` + `onRecordChanged`).
- 3 F2: records, key capacity and the `TombstoneModification` send-list entries are reserved in S22's reserve step 4;
  commit step 5 only writes; re-snap uses only the `onTerrainChanged` queue (Scope 6).
- 4 Tombstone-on-edit uses the chunk's effective placement version and compares y against that version's generated y
  (Scope 6).
- 5 The content part of `updateView` / `completeView` runs whether or not the space has terrain; test in
  `test_dungeon` with `View` (static set built, readiness 1) (Scope 4, Tests).
- 6 Golden: statistics taken for the rendered frame (`takeStatistics` until `frame` matches); the camera at site 0's
  `baseHeight` + 70 m (Scope 8).
- 7 Bound test with 10000 keys: S23a pins `candidatesPerChunk(test_tree) = 16384` (Depends on, Tests).
- 8 Hook worlds: legacy generators keep S20's mask path from the generated level-0 tile with the generator's
  thresholds (no `ChunkPlacement`, no second hook evaluation); `HookGenerator::grassMask` stays the hashed definition
  and is compared with it (Scope 5, Tests).
- 9 Hash sentence: "S23b adds nothing for worlds without records" (Scope 1).
- 10 Per-step ctest: `injectForTests` in step 2; the host-load sink assertion in step 5; the re-snap-at-creation test
  and the edit scenario in step 7; the scenario steps folded into 5 and 7 (Implementation steps).
- 11 Record changes reach the sink immediately in `onRecordChanged` (both modes, also in `stepNetworkOnly`); the pass
  only creates and removes sets; tombstones and re-snaps never drop a pool result; `tombstone_hides_instance` waits
  (bounded) until every peer has the set of chunk (4, 5) (Scope 4, Tests).
- 12 `TombstoneModification` header is 8 bytes; `updateInstances(set, first, const std::vector<StaticInstance>&)` used
  with its real signature (Scope 3, 4).
- 13 Gate wording (S20 finding 10): content work is gated on `ContentRenderSettings::centres` (`None` by default; `View`
  set by `EngineApp`, `Players` by `ScenarioWorld`), not on the presence of a sink; a sink alone requests nothing
  (Scope 4, Tests).
- Rejected: none.

## Design changes

(Filled when the stage is done.)

## Results

(Filled when the stage is done.)
