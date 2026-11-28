# S26: World over the network

Status: planned.

Sections follow the stage plan template of S05; "S24 interfaces used", "Conflicts with S24", "S25 touchpoints",
"Implementation steps", "Risks", "Open questions" and "Design-doc amendments" are added for this stage. Reconciled with
`docs/stages/S24_streaming.plan.md` (decisions G1-G23, its "Interfaces for S25 and S26"); every S24 name used here is
listed in "S24 interfaces used", the points where S26 deviates from S24's interface section in "Conflicts with S24".

## Goal

A client joins a host's world through a handshake and then holds exactly the modification state of the chunks it has
loaded (`docs/engine_design.md` §13.2, §12.3, §13.1, §13.6, §15.3):

- **Handshake**: the client announces its protocol version, network schema hash, render name tables, environment flag,
  seed and generator capabilities (`JoinRequest`); the host rejects a mismatch with a reason both sides can read
  (`JoinReject`) or admits the peer (`JoinAccept`: the epoch and the generation info, i.e. seed, pinned and current
  generator versions, slot tables, space table). The client adopts the host's pins (E11 of S23a). S23a's
  `GenerationState` and the join-time `Resync` are folded into it.
- **Chunk interest**: in a streaming world (`WorldConfig::streaming`, S24 G1) the client subscribes the chunks its
  own residency has requested (S24's `ChunkResidency` around the players it owns and their pending arrivals, G4, plus
  explicit holds) and unsubscribes the ones it unloads (`ChunkInterest`); in a world without streaming it subscribes
  every chunk of every space at once (`ChunkInterest` with the "all" flag, answered by a `ChunkIndex` of the chunks with
  state, D19). The host answers every chunk subscription with one `ChunkSnapshot` on
  `Channel::ChunkData` (placement version, tombstones, terrain deltas, the chunk's modification sequence) and sends live
  modifications (S22's `TerrainModification`, S23b's `TombstoneModification`) only to the clients that have the chunk
  subscribed. S22's `TerrainChunkState`, S23b's `ChunkRecordState` and `Replication::sendChunkSnapshot` are removed.
- **Client queue**: live modifications of a chunk are queued until the host's snapshot is applied, those with a
  sequence `<=` the snapshot's are dropped, a gap is an error that re-requests the snapshot. A client drops a chunk's
  state when it unloads it. In a streaming world a chunk counts as resident on a client only after its snapshot is
  applied (§12.3; S24's modification gate), so the owner's mover holds its character until then (S24 G11).

Acceptance (§22): the host + client variants pass with streaming; a join with a different schema hash is rejected; a
client that reloads a chunk while an edit is in flight ends with the host's heights (CI).

## Depends on

- **S24** (`S24_streaming.plan.md`): `WorldConfig::streaming` (G1), `ChunkResidency` with its client centres (G4),
  the phase-7 apply (G5), `setModificationGate`, `forEachChunk` and the listeners, the mover hold on `chunkReady` (G11),
  arrivals through the replicated `PendingTransition` (G12, G13), the synchronous `ScenarioWorld` and
  `streamingScenarioSetup()` with its scenarios. Every name is listed in "S24 interfaces used".
- **S16** (replication; exists): `Replication` (`engine/net/Replication.{h,cpp}`): the epoch (`epoch()`,
  `onHostLoad`, `ResyncMessage`, `onResync`), the joined/regular peer split of `hostReplicate`, the staged bookkeeping
  (`extract(networkOnly)`, `commit()`, `abort(Phase)`), `rejectInput`, the host + client variants in `ScenarioWorld`
  (`HostClient`, `HostClientDelay`, `HostClientAsync`) and `stepNetworkOnly`.
- **S15** (exists): `Transport`, `LoopbackTransport::createNetwork(n, LoopbackOptions{dropRate, delayRounds, seed,
  bytesPerRound})`, the three channels (`Channel::Control`, `State`, `ChunkData`; reliable messages ordered per
  channel only), `Fragmenter` (reliable packets up to `kMaxReliablePacket` = 64 MB), `MessageList` (`add`, `addRaw`,
  `Entry`), `World::send` / `sendRaw`, `World::NetStats`, phase-1 receive (`World::receiveMessages`, connect events
  into `peers` / `joined` / `left`).
- **S22** (exists): `TerrainEditStore` (`sequence`, `setSequence`, `bumpSequence`, `denseDelta`, `writeDelta`,
  `reserveFor`, `makeEntry` / `insertEntry`, `chunks(space)`, `chunkSamples`, `sizeChunks`, `clear`),
  `TerrainDeltaCodec` (`kMaxTerrainDelta` = 2^24), `TerrainModification` and the host send list
  (`Replication::HostStateEntry`, `appendTerrainModification`, `sendHostStates`), `World::onTerrainChanged(space, box)`,
  `NetStats::modificationGaps`.
- **S23a** (exists): `GenerationPipeline` (`build(setup, pins, cache)`, `validateEntities`, `replacePlans`,
  `GenerationPins{pinned, current}`, `GenerationPlans::combinedHash`), `World::writesGenerationState`,
  `generationMismatch` / `generationChecked`, `net/GenerationStateMessage` (`generationStateOf`,
  `encodeGenerationState`, `decodeGenerationState`, `generationStateDifference`), the load's
  `WorldLoader::commitGeneration` (plans replaced, `TerrainSystem::replaceSources`, `ContentRenderSystem::dropAll`).
- **S23b** (exists): `ChunkRecords` (`find`, `reserve`, `insert`, `replace`, `clear`, `sortedUniqueKeys`),
  `TombstoneModification` (`ContentMessages`), `Replication::appendTombstone`, `clearReplicatedChunkState`,
  `World::onRecordChanged(chunk)`, the content render sets' record diff.
- **S19b** (exists): `World::renderTableHash()` (names, IDs, aliases; "what S26's handshake compares").
- **S36a1a** (exists): `EnvironmentSetup::enabled` (`WorldSetup::environment`; "host and clients must agree", no join
  check before S26).
- **S11** (exists): `SchemaTable::fromWorld`, `SchemaType`, `FieldInfo` (with `quantize` and `enumValues`),
  `PoolBase::describe` / `describeNested`, the event queues' `describe`.

S25 (world save directory) has no dependency in either direction (§22); where the two stages touch is listed in
"S25 touchpoints".

## Rules that apply

- R-NET-1 (every scenario passes the host + client variants, now with streaming and chunk subscriptions), R-NET-2
  (clients write chunk state only from host messages; they never edit or tombstone).
- R-DET-1 (the host + client variants stay reproducible: seeded loopback, chunk interest derived from committed state,
  no wall clock).
- R-REG-1 (the new messages are registered in `registerEngineMessages`; IDs from the `EngineMessage` enum),
  R-REG-2 (no string lookups in C++: names only in the schema, the reject text and the tools).
- R-FILE-1 (one class per file: `HostChunkSubscriptions`, `ClientChunkSync`, `NetSchema`; message formats in their
  message headers).
- R-PAR-1 (snapshot encoding on the main thread in phase 8; no new threads).
- R-MECH-1 (the snapshot budget and the per-peer chunk limit are `WorldConfig` parameters; which chunks a client
  wants is S24's residency policy plus game holds).
- R-CONTENT-1 (clients regenerate the base; heights are equal across machines because generation is integer only and
  the deltas are integers).
- R-MCP-1 (`world_status` and `chunk_info` gain fields: their `handleMessage` tests are extended).
- R-TEST-1, R-TEST-2 (`engine/net/` is performance relevant: `test_engine_net_benchmark` before and after, the new
  snapshot benchmark case), R-DOC-1 (every existing API above was grepped; new ones are specified here), R-DOC-2 (the
  amendments at the end).
- AGENTS.md "Test speed": every new test binary about a second; no fixed sleeps; small spaces (16 chunks, 2 m spacing)
  unless size is what is tested.

## Decisions

- **D1 Client-driven chunk interest.** The client tells the host which chunks it has loaded (`ChunkInterest`
  subscribe / unsubscribe) instead of the host deriving it from the client's players. Reasons: (a) the client's
  residency (S24) is computed from its own, newer player poses and its own margins; a host-side copy would lag by the
  owner-pose latency and disagree at the edges, and the client could not know which chunks it will receive; (b) the
  re-request after a gap and the re-subscription after a `Resync` need a client -> host message anyway; (c) clients are
  trusted (§2), so the host only bounds the requests (range, count, epoch). The host keeps one subscription table per
  peer. The alternative (the host derives each peer's chunks from its own copy of the client's centres) is not used.
- **D2 Handshake of two messages plus a rejection.** `JoinRequest` (client -> host) carries everything the host checks;
  the host answers `JoinReject` (reason code, text naming the differing schema section) or admits the peer and sends
  `JoinAccept` (epoch, generation info) before the spawn set, both on the reliable channels. The host checks what it
  can (protocol, environment, schema, render tables, seed, generator capabilities); the client re-validates
  `JoinAccept` (untrusted input) and adopts the pins; a failure of the adoption (a different plan hash after the
  adoption: a determinism bug or other prefab data) is a local failure on the client (`JoinState::Rejected` with the
  reason, `generationMismatch()` set), which the host does not learn before S33's disconnect.
- **D3 Frozen handshake framing.** `JoinRequest`, `JoinReject` and `JoinAccept` get the message IDs 0, 1, 2; the first
  field of `JoinRequest` is the protocol version and `JoinReject` starts with its reason and text. These IDs and
  prefixes, together with S15's packet and fragment framing, never change, so a peer of another protocol version always
  receives a readable rejection. Everything else is covered by `kProtocolVersion` (engine framing constants, the
  handshake's later fields) and by the network schema hash (message IDs, names, versions, channels).
- **D4 A network schema, not the save schema.** The save `SchemaTable` lists only saved types; the wire needs every
  registered component (also `Saved::No`), its replicate mode, its kind (plain, global, space state: this is where
  `EnvironmentSetup::enabled` shows), the fields with their quantization, the events with their replicate mode, the
  classes, the messages with version, reliability, channel and raw flag, the player actions and held buttons, and the
  space templates' layout and the world flags that change the protocol (`WorldConfig::streaming`, D19); not the
  streaming radii and margins (each client uses its own, D1) and not `Saved` (it does not change the wire).
  `NetSchema::fromWorld(world)` builds it; the canonical encoding of each section (types, messages, classes, actions,
  templates and flags) is hashed with FNV-1a 64 (`core/Hash.h`), and the schema hash is the hash of the section
  hashes. `JoinRequest` carries the section hashes, so the host's rejection names the differing section without a
  schema transfer or an untrusted schema decoder. Data files (tunable values) and prefabs are not in it (prefabs
  are in the plan hash).
- **D5 Explicit fields for the specific reasons.** The environment flag, the render table hash, the seed and the
  generators are separate `JoinRequest` fields checked **before** the schema hash (the environment flag also changes
  the schema), so the rejection names the cause: "environment: the host's world has WorldTime and Weather enabled,
  this client's not".
- **D6 Pins are adopted (E11).** A client builds the plans at the host's pinned and current versions
  (`GenerationPipeline::build(setup, pins, cache)`, phase B validated) when they differ from its own, requires the
  host's space table (space count and combined plan hash) and commits them with the same steps as a host load's
  `commitGeneration` (factored into `World::replaceGeneration`). The seed is compared, not adopted (D11 of S20: the
  seed is setup; S33's session creates the client world from the host's announcement). A client whose generator is
  older than the host's current version (`version() < current`) is rejected by the host. `Resync` (a host load) carries
  the generation info too and the client adopts the loaded pins the same way.
- **D7 One snapshot message per chunk, on `ChunkData`.** `ChunkSnapshot` carries the whole modification state of one
  chunk: sequence, record (placement version and tombstones) and terrain deltas, so a chunk's state arrives atomically
  and the S23b record/terrain order (E5) disappears. Every subscription is answered, also for a chunk without state (a
  few bytes): the client needs the host's sequence to know its chunk is complete. A per-chunk `ticket` (the client's
  request number) is echoed, so a snapshot of an earlier subscription of the same chunk is dropped.
- **D8 Live modifications stay on `Channel::State`, filtered per peer.** A multi-megabyte snapshot transfer must not
  delay small live edits (head-of-line on one channel), and live edits stay ordered with `Resync` and the spawn set.
  Snapshots and live modifications therefore race across channels; the sequence protocol (D9) makes the result
  independent of the arrival order. The host sends each `TerrainModification` / `TombstoneModification` per peer with
  only the parts and entries of chunks that peer has subscribed (no message when nothing remains).
- **D9 Client queue.** Per subscribed chunk the client is `Pending` (a ticket, a queue) or `Synced`. Live parts of a
  `Pending` chunk are queued (after the whole message validated); when the snapshot with the current ticket applies,
  queued parts with `sequence <= snapshot.sequence` are dropped and the rest applied in order, each `current + 1`. A
  gap (`sequence > current + 1`) on a `Synced` chunk or in the queue applies nothing for that chunk, counts
  `modificationGaps`, logs, and **re-requests**: the chunk becomes `Pending` with a new ticket, keeps its present state
  (no flicker) until the snapshot replaces it, and the subscribe is sent again. A queue above its bound
  (`kMaxQueuedPartsPerChunk` = 1024 parts) is dropped and re-requested the same way.
- **D10 Unload drops the client's state.** When a chunk leaves the client's interest it is unsubscribed and its state
  dropped: its record erased, its `TerrainEditStore` entry removed, and the samples of its closed range zeroed except
  those that another subscribed chunk's closed range contains (`Pending` or `Synced`: shared borders stay; the
  neighbour's snapshot and live parts keep them correct). Before S26 clients kept every edited sample forever (§12.2); that rule ends here.
- **D11 Epochs.** Every chunk message carries the epoch. A `Resync` (host load) voids every subscription on both
  sides: the host clears its tables and its snapshot queue in `onHostLoad`, the client clears its chunk state
  (`clearReplicatedChunkState`), adopts the generation info and re-subscribes every chunk it wants with new tickets in
  the new epoch. The host drops `ChunkInterest` messages of another epoch (in flight across the load; counted, not
  rejected). Because the client subscribes only after it applied the `Resync`, no snapshot of a newer epoch can reach
  it before the `Resync`; older ones are dropped.
- **D12 Snapshot budget.** The host encodes snapshots when it sends them (the current state, so a snapshot includes
  every edit up to its encoding), in phase 8 after the host state list, FIFO over all peers in subscription order,
  until `WorldConfig::network.chunkSnapshotBytesPerStep` (default 1 MB of encoded bytes per step, all peers together)
  is reached; at least one snapshot per step. A resubscription of a chunk whose snapshot is still queued replaces the
  queued ticket (one snapshot, the newest ticket). Live parts are sent to a peer from the step its subscription was
  received, also while its snapshot waits in the queue (the client queues them).
- **D13 Admission gates the peer lists.** `World::connectedPeers()`, `peersJoined()` and `peersLeft()` list admitted
  peers (handshake accepted), not transport connections: the replication, the join policy hooks of `--coop`
  (`EngineAppSetup::onPeerJoined`) and game code see a peer once it may receive the world. Before admission the host
  accepts only `JoinRequest` from a peer (anything else: rejected input) and sends nothing but `JoinReject` /
  `JoinAccept`; `World::send` to a peer that is not admitted (host) and every game message a client sends before it is
  `Joined` (client) are dropped and counted in that world's `NetStats::droppedNotJoined`.
- **D14 A tombstone cap per chunk.** A chunk record holds at most `kMaxChunkTombstones` = 2^20 keys (a snapshot's
  keys then fit a few MB; tier-1 content has at most thousands of candidates per chunk; grass edits are S31's masks,
  not tombstones). `World::tombstone` and tombstone-on-edit refuse a key beyond it (a new check next to S23b's
  exhausted-sequence refusal; a log line, the world API's error).
- **D15 No base snapshots.** They exist only for retired generator versions (§15.3), which S25 defers (its G17: the
  stage that first retires a version, proposed S27); that stage adds a flag bit (unknown bits are rejected today).
- **D16 Steps and aborts.** Host: `ChunkInterest` applies to the subscription table in its phase-1 handler (a
  consumed message keeps its changes, S15); the snapshot queue's pops in phase 8 are staged and committed with the step
  (an aborted phase 8 keeps the queue; the outbox drops the encoded messages). Client: the interest diff is computed in
  phase 8 (`Replication::extract`, also in `stepNetworkOnly`) from the residency's chunk table as phase 7 left it
  (S24 G5: results applied in phase 7 stay applied when a later phase aborts; S26 diffs against them in the next
  phase 8 either way) and the holds, and staged; `commit()` applies it to the chunk table and drops the unloaded
  chunks' state, `abort()` discards it with the outbox entries (the next phase 8 computes the same diff again).
- **D17 Every space is synchronized the same way.** Open-world and enclosed spaces, terrain or flat: every chunk a
  client wants is subscribed and answered (an enclosed space's snapshot has `N = 0` and only a record). No special case
  for spaces that "cannot have state" (every open-world space can be edited, D1 of S20; enclosed spaces can carry
  records).
- **D18 Resident on a client means "base applied and snapshot applied".** `World::chunkSynced(chunk)` is true on the
  host and, on a client, when the chunk is `Synced`. A streaming client installs it as S24's modification gate
  (`ChunkResidency::setModificationGate`): a chunk whose parts are ready stays `Requested` until the gate holds, so
  `World::chunkReady` is false there and S24's mover hold (G11, the owner's `characterMoveSystem`) holds the character,
  S24's stall counters count it, and S24's loading screen (G19: "on a client it also waits until the player's chunk is
  resident") waits for the snapshot. No mover or app change in S26. `spaceReadiness` on a client takes the synced
  fraction of the subscribed chunks inside its radius into its minimum (the camera-switch loads of S21, which G19 does
  not cover). Worlds without streaming have no hold (S24 G1): their clients behave as before S26 (the snapshots arrive
  after the spawn set and apply when they arrive).
- **D19 Worlds without streaming subscribe everything.** S24 makes streaming a world policy (G1, default off; on only
  for `game_flight`, G22); a client of such a world has no residency to subscribe from. It sends one `ChunkInterest`
  per space with the flag `all` (identified by its epoch) when it is `Joined` and after every `Resync`, and never unsubscribes.
  The host answers with one `ChunkIndex` per space (on `ChunkData`, not budgeted, 8 bytes per chunk with state:
  the chunks whose snapshot carries something: a sequence > 0, or a record with tombstones or another placement version
  than the current one, S23b's rule), then queues a `ChunkSnapshot` per listed chunk
  (budgeted) and sends the space's live parts to that peer from then on. Until the index arrives every live part of
  the space is queued per chunk; on the index the listed chunks become `Pending` with ticket 0, every other
  chunk is `Synced` at sequence 0 (its queued parts apply from sequence 1). This keeps today's behaviour (every client
  holds every edited chunk) for every non-streaming world, with the same queue, gap and epoch rules; the host + client
  variants of all existing scenarios (non-streaming) exercise it. `WorldConfig::streaming` is in the network schema,
  so host and client always agree on the mode.

## S24 interfaces used

From `S24_streaming.plan.md` (its "Interfaces for S25 and S26" and G1-G23); a deviation is listed in "Conflicts with
S24".

1. **`WorldConfig::streaming`** (G1, default `false`): S26 subscribes from the residency only in streaming worlds;
   other worlds use D19. In the network schema (D4).
2. **Client residency** (G4, G10): `ChunkResidency` (`World::residency()`, engine-internal) runs on clients in phase 7
   (`streamingSystem`, `RunOn::Both`) with the centres "the player roots and `StreamingAnchor` entities it owns (`ownedBy(localPeer)`) and their
   replicated `PendingTransition` targets" (a target kept until the owner command is applied, S24 G4); the view camera is never a centre. Chunks of spaces that want no part
   (flat, enclosed, slot-less without terrain) become resident in the requesting update without a job.
3. **The chunks S26 subscribes**: every chunk the client's residency has in state `Requested` or `Resident`
   (`ChunkResidency::state(ChunkId)`: from G3's requested distance `R + prefetchMargin` to the kept distance `R +
   prefetchMargin + unloadMargin`, `R = residencyRadius`, 0: `fullRadius`). So the subscription starts at the request (prefetch), not when the
   chunk becomes resident, and ends with S24's unload. S26 enumerates them in phase 8 with
   `ChunkResidency::forEachChunk(space, fn)` (S24 Scope 2; `Requested` or `Resident`, `ChunkId` order).
4. **Phase-7 apply** (G5): pool results and the residency update run in phase 7; S26's phase-8 diff sees the state of
   this step's phase 7. A snapshot applied in phase 1 lets the gate pass in phase 7 of the same step.
5. **The modification gate**: `ChunkResidency::setModificationGate(std::function<bool(const ChunkId&)>)` ("a chunk
   without it is not `Resident`"); S26 sets `[&](const ChunkId& c) { return chunkSync.state(c) == Synced; }` on
   streaming clients at construction. The gate and `forEachChunk` are agreed in S24's "Interfaces for S25 and S26"
   and added to `ChunkResidency` by S26; the gate is re-checked in every `update()` for a `Requested` chunk whose parts
   are ready, and `complete()` does not wait for it (S24 Scope 2: complete points never block on the network).
6. **`World::chunkReady(SpaceId, double x, double z)`** (G7: requested and resident) behind the mover hold (G11) and
   the stall counters (`streamingStalls`, `playerStallSteps`: on a client its own players); with the gate they count a
   chunk without its snapshot as missing.
7. **No load hook** (S24 G20; S25 G2 keeps all modification state in memory): the client apply paths of
   `TerrainModification`, `TombstoneModification` and the new `ChunkSnapshot` write directly, with S26's per-chunk
   queue in front (D9); the host's snapshot encoder reads `TerrainEditStore` / `ChunkRecords` directly.
8. **Re-patch by construction** (G6): residency results are base only; deltas are read from `TerrainEditStore` and
   tombstones from `ChunkRecords` by every reader, and a changed effective placement version (S24 names "a client
   record message") drops the result and requests again. S26's snapshot changes the record through `onRecordChanged`,
   which S24's rule already covers; no new S24 hook.
9. **Modification state stays in memory** (G2): the host never unloads it, so it can encode any chunk; S24 drops
   nothing on clients either, S26 owns the client-side drop (D10).
10. **Arrivals** (G12, G13, G19): `requestArrival` is host-only and adds `PendingTransition`, now `Replicate::Host`;
    the owning client's residency makes the target a centre at once, so S26 subscribes the target island's chunks
    while the host is still loading it. The host completes the arrival on its own island (G12) and moves the player
    with `moveToSpace`; the client's mover holds until the gate passes; G19's loading screen waits for it. Joins in
    S26 need no arrival of their own (the game's join policy may call `requestArrival` for the joining player).
11. **Synchronous `ScenarioWorld`** (G5, Scope 1 of S24): every world of a scenario builds requested chunks inline in
    phase 7; the scenario distances are S24 Scope 1's (requested 128 m, kept 192 m). In the host + client variants the only latency of a client's
    chunk is the subscribe / snapshot round trip.
12. **`streamingScenarioSetup()` and its scenarios** (S24 Tests): `streaming_follows_player`, `far_entities_freeze`,
    `arrival_far_teleport`, `edit_far_chunk_then_visit`, `space_link_with_streaming` run in every variant; in S26 their
    host + client variants run over chunk subscriptions (`edit_far_chunk_then_visit` is the "edit before the client
    subscribed" case and replaces the scenario this plan planned for it).
13. **Space switches** (G4): a client player's move to another space moves its centre; the old space's chunks are
    unloaded by the residency and unsubscribed.
14. **S24 adds no message** (G15) and no host-side per-peer chunk knowledge; S24 Scope 10's "end condition unchanged"
    is replaced by S26's (Scope 8).

## Conflicts with S24

None open. The first draft's proposals were applied to `S24_streaming.plan.md`: subscription at request through
`forEachChunk` (S26 uses no listener), the gate re-checked in every `update()`, no host-derived
per-peer chunks, no channel named for the subscribe messages, no `ensureModificationState` hook (S26's queue lives in
its own apply paths), and S26's §13.2 amendment replacing S24's.

## S25 touchpoints

S25 and S26 do not depend on each other (§22); they touch here:

- **Base snapshots** (D15): neither stage implements them (S25 G17 defers retiring).
- **Chunk store**: S25 keeps every chunk's modification state in memory (its G2), so the snapshot encoder reads
  `TerrainEditStore` / `ChunkRecords` unchanged. S25's every-visited-chunk records carry the current placement version
  and do not set the snapshot's "record present" flag.
- **`Despawn`**: S25 (implemented before S26) adds `EngineMessage::Despawn` (per entity, not for bound NetIds); S26's
  renumbering includes it (Scope 1). Despawns go to every joined peer like `Destroy`; S26 filters only chunk messages.
- **Records**: S25 records every generated chunk's placement version (§15.3). The snapshot's "record present" flag
  follows S23b's rule (bit 2 set when the record has tombstones or a placement version other than the current one), so
  S25's tombstone-free current-version records are not sent; the client's rule (no record: current version) is
  unchanged.
- **Space unloading** (S25): a space without players is unloaded on the host; clients have no players there, so they
  have unsubscribed its chunks already (S24 interface 13).
- Clients never save (`canSave()` is false), so nothing of the client's chunk table reaches a save.

## Scope

### 1. Messages (`engine/net/NetIds.h`, `EngineMessages.*`, `JoinMessages.{h,cpp}`, `ChunkMessages.{h,cpp}`)

`EngineMessage` is renumbered (backward compatibility is not a goal; the protocol version and the schema hash
change):

| ID | Message | Direction, channel, reliability | Status |
|---|---|---|---|
| 0 | `JoinRequest` (raw) | client -> host, Control, reliable | new, frozen prefix (D3) |
| 1 | `JoinReject` (raw) | host -> client, Control, reliable | new, frozen prefix (D3) |
| 2 | `JoinAccept` (raw) | host -> client, State, reliable | new (replaces the join's `Resync` and `GenerationState`) |
| 3..12 | `Spawn`, `Destroy`, `Hierarchy`, `ComponentChange`, `Snapshot`, `OwnerState`, `OwnerCommand`, `PlayerAction`, `Ping`, `Pong` | unchanged | renumbered |
| 13 | `Resync` (raw, version 2) | host -> client, State, reliable | epoch plus the generation info (was typed, epoch only) |
| 14 | `Despawn` | host -> client, State, reliable | S25's, renumbered |
| 15 | `TerrainModification` | host -> client, State, reliable | unchanged layout, filtered per peer (D8) |
| 16 | `TombstoneModification` | host -> client, State, reliable | unchanged layout, filtered per peer (D8) |
| 17 | `ChunkInterest` (raw) | client -> host, Control, reliable | new |
| 18 | `ChunkSnapshot` (raw) | host -> client, ChunkData, reliable | new |
| 19 | `ChunkIndex` (raw) | host -> client, ChunkData, reliable | new (worlds without streaming, D19) |

Removed: `TerrainChunkState`, `GenerationState`, `ChunkRecordState` (and their encoders, decoders, handlers and
`Replication::Stats` counters).

`MessageList::Entry` gains `uint32_t version` (typed messages: `M::version`; `addRaw(id, name, version, reliability,
channel, handler)` takes it), so message versions enter the schema (S15 plan: "S26 adds messages to the handshake's
schema table").

Layouts (little-endian fixed-width integers; varints of `Varint.h`; strings as varint length + bytes; every decoder
is untrusted-input safe: bounds, no bytes left over, `false` with an error):

| Message | Layout | Bounds |
|---|---|---|
| `JoinRequest` | protocol version (u32); schema hash (u64); the section hashes (count u8, u64 each, in the order types, messages, classes, actions, templates and flags); render table hash (u64); seed (u64); flags (u8: bit 0 environment enabled, bit 1 writes generation state); generator count (u16); per generator in name order: name (string), `version()` (u16), slot count (u16), per slot its name (string) | names <= 256 bytes, <= 1024 generators, <= 255 slots, payload <= 256 KB; unknown flag bits rejected |
| `JoinReject` | reason (u8, `JoinRejectReason`); text (string; for `Schema` it names the differing sections) | text <= 4096 bytes |
| `JoinAccept` | epoch (u32); flags (u8: bit 0 generation info present); generation info | as `GenerationInfo` |
| `Resync` v2 | epoch (u32); flags (u8: bit 0 generation info present); generation info | as `GenerationInfo` |
| generation info | seed (u64); generator count (u16); per generator in name order: name (string), pinned (u16), current (u16), slot count (u16), per slot name (string) and flags (u32); space count (u32); combined plan hash (u64) | S23a's `GenerationState` bounds without its epoch |
| `ChunkInterest` | flags (u8: bit 0 `all`, D19); epoch (u32); space (u16); without `all`: remove count (u16); add count (u16); removes: chunk x, z (i32); adds: chunk x, z (i32), ticket (u32) | `all`: no entries, only in a world without streaming, once per space and epoch (the epoch identifies it); else 1..4096 entries in total, removes applied before adds, only in a streaming world (or a re-request in an `all` space) |
| `ChunkIndex` | epoch (u32); space (u16); count (varint); per chunk in `ChunkId` order: x, z (i32) | count <= the space's chunk count (`sizeChunks^2`, <= 262144: at most ~2 MB); ascending, unique, in range |
| `ChunkSnapshot` | flags (u8: bit 0 zstd body, bit 1 deltas present, bit 2 record present: the record has tombstones or a placement version other than the current one, S23b's rule; other bits rejected); epoch (u32); ticket (u32, 0 for a snapshot of an `all` space's index); space (u16); chunk x, z (i32); sequence (u32, 0 allowed); N (u16; 0 for a space without a sample grid); body: [record: placement version (u16), key count (varint), keys ascending as varint differences of `slot << 24 \| counter` (the first absolute)] [deltas: encoded length (varint), the dense N x N deltas in `TerrainDeltaCodec`] | keys <= `kMaxChunkTombstones` (2^20), strictly ascending, slot <= 255, counter < 2^24; N <= 257; codec bounds of S22; decompressed body <= 8 MB |

- `JoinRejectReason`: `ProtocolVersion`, `Environment`, `Schema`, `RenderTables`, `Seed`, `Generators`, `Malformed`.
- The generation info is S23a's `GenerationStateMessage` without the epoch; `net/GenerationStateMessage.{h,cpp}` is
  renamed `net/GenerationInfo.{h,cpp}` (`GenerationInfo`, `generationInfoOf(world)`, `encodeGenerationInfo`,
  `decodeGenerationInfo(ByteReader&, ...)`, `generationInfoDifference`).
- Sizes: a `ChunkSnapshot` of a fully edited 1 m chunk is at most 66049 x 4 bytes of codec varints plus headers
  (~260 KB raw before zstd, S22); with the key cap at most ~5 MB raw; an empty snapshot is 24 bytes. A `ChunkInterest`
  is at most ~49 KB.
- `kProtocolVersion = 1` in `JoinMessages.h`, with the bump rule in its comment (engine constants of the framing,
  the handshake fields after the frozen prefix, the chunk sync semantics).

### 2. Network schema (`engine/net/NetSchema.{h,cpp}`)

```cpp
// The registration a peer's wire format depends on (D4): built once per World (constant after construction)
struct SAIGA_ENGINE_API NetSchema
{
    struct Type      // a component, event or nested struct
    {
        SchemaKind kind = SchemaKind::Struct;
        std::string name;
        uint32_t id = 0, version = 0;
        uint8_t replicate = 0;   // Replicate (components), the event mode (events)
        uint8_t stateKind = 0;   // 0 plain, 1 global, 2 space state (components; EnvironmentSetup::enabled shows here)
        std::vector<FieldInfo> fields;  // name, type, enum values, quantize
    };
    struct Message { MessageId id; std::string name; uint32_t version; uint8_t reliability, channel; bool raw; };
    struct Template { std::string name; uint8_t kind; int32_t baseSpacing, sizeChunks; GeneratorId generator; };
    std::vector<Type> types;                                  // components by ID, events by ID, nested structs by name
    std::vector<std::pair<uint32_t, std::string>> classes;    // game object classes (ID, name)
    std::vector<Message> messages;                            // by ID
    std::vector<std::pair<uint32_t, std::string>> actions;    // ActionList actions (ID, name)
    std::vector<std::pair<int, std::string>> held;            // ActionList held buttons (bit, name)
    std::vector<Template> templates;                          // WorldConfig::spaceTemplates (the layout fields)
    uint32_t worldFlags = 0;                                  // bit 0: WorldConfig::streaming (D19)

    static NetSchema fromWorld(const World& world);
    void encode(std::vector<uint8_t>& out) const;             // canonical
    uint64_t hash() const;                                    // fnv1a64Bytes over the section hashes
};
// (member of NetSchema) the section hashes: types, messages, classes, actions, templates and flags
std::array<uint64_t, 6> sectionHashes() const;
```

- `World` builds it at the end of construction (after `renderAssets`) and keeps it with its hash:
  `const NetSchema& World::netSchema() const`, `uint64_t World::netSchemaHash() const`.
- Not included: data files, prefabs (plan hash), meshes and materials (render table hash), the generators (checked by
  their own fields), the seed. Every included item is something whose difference makes a peer misread a message or a
  component.
- `world_status` shows the hash (hex); the engine's committed `data/tests/engine/schema.json` (the save schema) is
  unchanged.

### 3. Handshake (`engine/net/JoinCheck.{h,cpp}`, `Replication`, `World`)

```cpp
enum class JoinState : uint8_t { Local, Joining, Joined, Rejected };  // Local: host or no transport
JoinState World::joinState() const;
const std::string& World::joinRejection() const;   // the reason text (Rejected; for Schema the differing sections)

// Host: the first failing check of a request, in the order protocol, environment, schema, render tables, seed,
// generators (D5); nullopt: admit
SAIGA_ENGINE_API std::optional<std::pair<JoinRejectReason, std::string>> checkJoinRequest(const World& host,
                                                                                         const JoinRequest& r);
// Client: validates the host's generation info against this build and returns the pins to adopt, or an error
SAIGA_ENGINE_API bool adoptablePins(const World& client, const GenerationInfo& host, GenerationPins& pins,
                                    std::string& error);
```

Sequence:

1. Client: on its transport's `Connected(0)` (phase 1 receive) it sends `JoinRequest` (built from `netSchemaHash`,
   `renderTableHash`, `setup.seed`, `environmentSetup().enabled`, `writesGenerationState`, the registered generators
   with `version()` and slot names) and enters `Joining`. It sends nothing else before `Joined`.
2. Host: the transport's `Connected(p)` puts `p` into an internal joining list (not `connectedPeers()`, D13). The
   `JoinRequest` handler (phase 1) first reads only the frozen u32 protocol version: another version is rejected with
   `JoinReject{ProtocolVersion}` before the rest is decoded (D3: a later version may change the layout); then it decodes
   (malformed: `JoinReject{Malformed}` and `NetStats::rejected`) and runs
   `checkJoinRequest`:
   - protocol version equal;
   - environment flag equal ("environment: ... enabled on the host, disabled here");
   - schema hash equal (the reject names every section whose hash differs: "schema: types, messages differ");
   - render table hash equal ("render name tables differ (meshes, materials, aliases)");
   - seed equal;
   - generators: the same names, the same `writesGenerationState`, per generator `version() >=` the host's current
     version (the client can run every version the host uses: pinned <= current), the host's slot table a prefix of
     the client's (S23a's load rule).
   A failure sends `JoinReject`, counts `NetStats::joinsRejected`, logs (category `net`, naming the peer and the
   reason) and keeps the peer out of the lists; a later `JoinRequest` of that peer is checked again (a client of
   another build cannot change, but the rule is simple). Success adds the peer to `connectedPeers()` and
   `peersJoined()`.
3. Host phase 8 (`hostReplicate`, the joined branch): `JoinAccept` (epoch, generation info if
   `writesGenerationState`) instead of S16's "Resync if the epoch is not 0" and S23a's `GenerationState`; the
   non-integer generator warning (once per world, unchanged); then S16's spawn set. No chunk state: the client
   subscribes (Scope 4).
4. Client `JoinAccept` handler: decode; `adoptablePins` (generator names, pinned <= current <= `version()`, slot
   prefix, seed equal); if the pins differ from its own: `GenerationPipeline::build` with them, `validateEntities`, the
   space count and combined plan hash equal to the host's, then `World::replaceGeneration(plans)` (the body of
   `WorldLoader::commitGeneration`'s plan branch, moved: `replacePlans`, `TerrainSystem::replaceSources`,
   `ContentRenderSystem::dropAll`). A failure: `JoinState::Rejected` with the reason, `generationMismatch()` set, a
   `content` log line; every later world message is rejected input. Success: epoch taken from the message,
   `generationChecked()` set, `JoinState::Joined`, peer 0 in `connectedPeers()` / `peersJoined()`.
   Adoption at join requires the adopted plan to have the client's space table (space count and the template per
   `SpaceId`; the client's space-state entities exist per space): otherwise the failure names "the host's world has N
   spaces, this client's plan M" (lifted by S33, which creates the client world from the host's info).
5. Client `JoinReject` handler: `JoinState::Rejected`, `joinRejection()` = the text; a `net` log line. Nothing else changes; the world keeps running locally.
6. While `Joining` a client handles `JoinAccept` / `JoinReject` and **defers** every other message in arrival order
   (at most 1024; more is rejected input): channels are ordered only per channel, so a Control message of the
   admission step (an `OwnerCommand` of a join policy that moves the new player) can overtake the State channel's
   `JoinAccept`. The deferred messages are handled right after `Joined`, before the step's other messages. After
   `Rejected` every message is rejected input. A `JoinRequest` from a peer that is already admitted is rejected input
   (ignored).

Removed with it: `Replication::onGenerationState`, `sendGenerationState`, `terrainRefused` and the "Spawn before a
GenerationState" check of `onSpawn` (the handshake orders `JoinAccept` before every world message), `ResyncMessage`
(typed). `ScenarioWorld`'s `renderTableHash` comparison is removed (the handshake checks it).

`Resync` v2 (`onHostLoad`): to each connected peer `Resync` with the loaded generation info. The client: drops
messages of an older epoch as before, clears its chunk state (`clearReplicatedChunkState`) and its chunk table,
adopts the generation info (step 4's adoption; a failure as in step 4), destroys the replicated entities (S16), and
re-subscribes its wanted chunks in the new epoch (Scope 4). §15.3's "connected clients are refused until S26" ends.

### 4. Client chunk sync (`engine/net/ClientChunkSync.{h,cpp}`, owned by `Replication`)

```cpp
class SAIGA_ENGINE_API ClientChunkSync
{
   public:
    enum class State : uint8_t { None, Pending, Synced };
    State state(const ChunkId& chunk) const;
    uint32_t ticket(const ChunkId& chunk) const;          // 0 for None
    size_t queuedParts(const ChunkId& chunk) const;
    size_t pendingCount() const, syncedCount() const;
    const std::map<ChunkId, Entry>& chunks() const;       // ChunkId order (tools, ScenarioWorld)
    struct Stats { uint64_t subscribed = 0, unsubscribed = 0, snapshotsApplied = 0, snapshotsStale = 0,
                   partsQueued = 0, partsDroppedNotSubscribed = 0, partsDroppedOld = 0, rerequests = 0,
                   queueOverflows = 0; };
    const Stats& stats() const;
    // ...
};
// ClientChunkSync::hold / release: explicit interest on top of S24's residency, reference-counted per chunk
// (engine-internal: tests through ScenarioWorld; a no-op in a world without streaming). A game API follows when a game
// needs one.
// Host: true; client: the chunk is Synced (D18)
bool World::chunkSynced(const ChunkId& chunk) const;
```

- **Wanted set** (streaming worlds) = the chunks of the client's `ChunkResidency` in state `Requested` or `Resident`,
  every space (`forEachChunk`, added to S24's class by S26) union the holds (`ClientChunkSync::hold` / `release`,
  engine-internal; `ScenarioWorld` exposes them for tests), read in phase 8 of every step and
  `stepNetworkOnly` (D16), only while `Joined`. The diff against the table gives removes and adds per space,
  encoded as `ChunkInterest` messages (removes before adds, adds nearest-first by distance to the nearest centre so the
  host's FIFO serves the chunk under a held player first; at most 4096 entries per message, more messages if needed)
  and staged; `commit()` applies them: a removed chunk is dropped (D10, `onTerrainChanged` with its box when it had
  deltas, `onRecordChanged` when it had a record), an added chunk becomes `Pending` with the next ticket (a per-world
  u32 counter, never 0). The diff of a chunk released and held again in the same step is empty (nothing dropped).
- **`ChunkSnapshot` handler** (phase 1): decode; the epoch (another: dropped, not counted); validate against the world
  in scratch (space exists; chunk inside the space's range; `N` equals `chunkSamples + 1` for an open-world space and
  0 otherwise; deltas only with `N > 0`; record: placement version between the pinned and the current version, keys
  only for a generator with slots, slot below the slot count, counter below `candidatesPerChunk`); a failure is
  rejected input (`NetStats::rejected`, log `net`/`content`) and applies nothing. Then: a chunk that is not `Pending`
  or a ticket other than its current one is stale (`snapshotsStale`, dropped). Apply: reserve (blocks, entry, record
  capacity), then write the chunk's closed range (the dense delta, or zero without deltas; S22's `onTerrainChunkState`
  body moves here), `setSequence` (an entry without deltas only for sequence > 0; sequence 0 removes the entry),
  replace the record (erase without one), `onTerrainChanged(space, chunkBox)` when the chunk had or has deltas,
  `onRecordChanged(chunk)` when the effective record changed; then the queue (D9): drop `<=` sequence, apply the rest
  each at `current + 1`, a gap re-requests; then `Synced`.
- **Live messages** (`TerrainModification`, `TombstoneModification`, phase 1): the whole message is decoded and
  validated as today (S22, S23b; minus the generation-refusal checks of Scope 3), then routed per part / entry by the
  chunk's state: `None` -> dropped (`partsDroppedNotSubscribed`; a race with an unsubscribe), `Pending` -> queued (a
  copy of the part or entry with its sequence; overflow -> D9), `Synced` -> S22's / S23b's per-sequence rule, except
  that a gap re-requests instead of applying. Validation that needs the chunk's current record (S23b: the placement
  version equals the chunk's effective version) runs when a queued entry is applied; a failure there is rejected input
  and re-requests the chunk. A `TerrainModification` whose parts land in several states applies its `Synced` parts in
  one `onTerrainChanged` as before.
- **Re-request** (D9): `modificationGaps++` (S22's counter: the host + client variant still fails on any, a gap is a
  bug on a lossless transport), `rerequests++`, the chunk becomes `Pending` with a new ticket and an empty queue (the
  parts after the gap are queued anew), its state is kept, and the chunk joins the **unsent adds** (per chunk the
  ticket; per space a pending `all`): every phase 8 stages an add for every unsent entry and only `commit()` clears it,
  so an aborted phase 8 re-sends it next step (the wanted-set diff alone would be empty for a re-requested chunk).
  The same holds for the `all` subscription staged on `Joined` and after a `Resync`. The residency keeps the chunk `Resident` meanwhile
  (deliberately: its present state is valid up to the gap, and holding the mover for a re-request would stall on every
  gap).
- **Generation replaced** (the join's pin adoption, `Resync`): `World::replaceGeneration` calls
  `residency().dropAll()` (S24 interface 5: `replaceSources` dropped the pinned tiles, and the held placements belong to
  the old plans), so the next update requests every chunk again; with the cleared chunk table no chunk is `Resident`
  before its new snapshot (the gate).
- **`Resync`** (D11): the residency is dropped (above), the table is cleared without staging (`clearReplicatedChunkState` already notifies), every wanted
  chunk is added again in the next phase 8 (a world without streaming: one `all` subscription per space again).
- **Worlds without streaming** (D19): on `Joined` and after a `Resync` the client stages one `ChunkInterest{all}` per
  space; the space is `AllPending` (every live part queued per chunk, the queue bound applies per chunk)
  until its `ChunkIndex` of the current epoch arrives (another epoch: stale, dropped); then the listed chunks are
  `Pending` with ticket 0 (their snapshots follow) and every other chunk of the space is `Synced` at sequence 0, its
  queued parts applied from sequence 1 (a gap re-requests that chunk with a per-chunk add, which the host accepts in
  an `all` space too). A `ChunkIndex` is validated as a whole (ascending, unique, in range) before anything changes. `chunkSynced` is true for a chunk of an indexed space that is not
  `Pending`.
- **The gate** (D18): a streaming client sets `residency().setModificationGate(...)` in the World constructor (after
  the residency exists) to `chunkSync.state(c) == Synced`.
- `World::spaceReadiness` on a client (D18): the minimum also includes the synced fraction of the subscribed chunks
  inside the radius (unsubscribed chunks count as ready, so a free camera far from the players does not block).
- `Replication::clearReplicatedChunkState` (S23b's, `net/Replication.h:260`) stays the one clear of both stores.

### 5. Host subscriptions and snapshots (`engine/net/HostChunkSubscriptions.{h,cpp}`, owned by `Replication`)

```cpp
class SAIGA_ENGINE_API HostChunkSubscriptions
{
   public:
    bool subscribed(uint8_t peer, const ChunkId& chunk) const;
    size_t subscribedCount(uint8_t peer) const;
    size_t queuedSnapshots(uint8_t peer) const;
    struct Stats { uint64_t interestMessages = 0, adds = 0, removes = 0, droppedEpoch = 0, snapshotsSent = 0,
                   snapshotBytes = 0, partsFiltered = 0, entriesFiltered = 0; };
    const Stats& stats() const;
    // ...
};
```

- **`ChunkInterest` handler** (phase 1, host): only from an admitted peer (else rejected input); decode; another epoch
  -> dropped (`droppedEpoch`); validate (space exists; every chunk inside its range; no chunk twice in one message;
  `all` only in a world without streaming, entries only with streaming or as a re-request inside an `all` space);
  an `all` subscription marks the space subscribed for the peer and queues a `ChunkIndex` for this phase 8 (not
  budgeted), whose listed chunks are then queued as snapshots with the `all` ticket;
  removes erase the subscription and a queued snapshot; adds set the subscription's ticket and enqueue a snapshot (or
  replace the queued ticket, D12). A peer above `WorldConfig::network.maxChunksPerPeer` (default 16384) subscriptions
  is rejected input for the whole message (applies nothing).
- **Phase 8** (`hostReplicate`, both extract paths): after the spawns and component changes, `sendHostStates` sends
  each list entry **per peer** with the parts / entries of the peer's subscribed chunks (every chunk of an `all`
  space) (an entry with nothing left is
  not sent; `partsFiltered`, `entriesFiltered` count the removed ones); then the snapshot queue is served FIFO under
  the budget (D12), after this phase's `ChunkIndex` messages: `encodeChunkSnapshot(epoch, ticket, chunk, store,
  records, out)` from the current `TerrainEditStore` and `ChunkRecords` state (in memory, S25 G2), sent with `sendRaw` on `ChunkData`. Pops are staged; `commit()`
  removes them, `abort(Phase::...)` keeps them.
- `onHostLoad`: every table and queue is cleared (D11) before the `Resync`. A peer in `peersLeft()` loses its table.
- S22's "the joining step's send-list entries are not sent to the joiner" rule is removed: a joiner subscribes later,
  and a duplicate would be dropped by its sequence anyway.

### 6. Chunk state stores (`engine/terrain/TerrainEditStore.*`, `engine/content/ChunkRecords.*`)

```cpp
// TerrainEditStore, client unload (D10): removes the chunk's entry and writes 0 to every sample of its closed range
// that no chunk with keep(c) == true contains (reserve-free: zero writes never allocate); neighbours' nonZeroSamples
// follow writeDelta's rule
void dropChunk(const ChunkId& chunk, const std::function<bool(const ChunkId&)>& keep);
// ChunkRecords: the client's snapshot apply and unload
void erase(const ChunkId& chunk);
ChunkRecord& set(const ChunkId& chunk, uint16_t placementVersion, std::vector<InstanceKey> keys);  // after reserve
```

- `kMaxChunkTombstones = 1 << 20` in `ChunkRecords.h`; `World::tombstone` / `canTombstone` and the tombstone-on-edit
  reserve step refuse a key beyond it (D14; the edit is refused before it changes anything, like S23b's exhausted
  sequence).

### 7. World, config and tools

- `WorldConfig::network` (`NetworkConfig`, new): `size_t chunkSnapshotBytesPerStep = 1 << 20;` (host, D12),
  `size_t maxChunksPerPeer = 16384;` (host). `kMaxQueuedPartsPerChunk = 1024` is a constant of `ClientChunkSync`.
- `World::NetStats` gains `joinsRejected` (host), `droppedNotJoined`, `chunkRerequests` (client);
  `modificationGaps` keeps its name with the new meaning (counted, not applied, re-requested).
- `Replication::Stats`: the removed messages' counters go (`generationStatesSent`, `generationStatesDroppedEpoch`,
  `recordStatesSent`); the chunk counters live in `ClientChunkSync::Stats` and `HostChunkSubscriptions::Stats`.
- Profiler counters `net/chunk_snapshot_bytes` and `net/chunk_snapshots` (host, per step).
- World API / MCP (`api/WorldApi.cpp`, `tools/WorldTools.cpp`): `world_status.net` gains `join` (`state`,
  `rejection`), `schema_hash`, on a client `chunks` (`subscribed`, `pending`, `synced`, `queued_parts`,
  `rerequests`), on the host `peers` (per admitted peer `subscribed`, `queued_snapshots`); `chunk_info` on a client
  gains `sync` (`state`, `ticket`, `queued_parts`). Read-only, both roles.
- `EngineApp` `--coop`: a rejected client world logs the reason and the app exits with an error (it cannot happen with
  one binary; the check is for a broken setup). No other app change: the client world's S24 residency drives its
  subscriptions.

### 8. Scenario framework (`engine/tools/ScenarioWorld.{h,cpp}`, `ScenarioRunner.cpp`)

- **Join before the scenario starts**: `ScenarioWorld`'s constructor pumps `stepNetworkOnly` on host and clients
  (no host step: the host's step count and hash are unchanged) until every client is `Joined` and, in a world without
  streaming, every space indexed and no chunk `Pending` (the initial snapshots applied, so a client-owned character's
  first steps stand on the edited surface); in a streaming world the initial hold of up to `2 * delaySteps + 3` steps
  of a client-owned character is expected; a client that is not `Joined` within `4 * delaySteps + 8` pumps, or that becomes
  `Rejected`, fails the variant with its `joinRejection()`. So the handshake's extra round trips do not eat into
  short scenarios or the final sync's bound. Streaming checks on a client that count steps (stalls, "resident at most
  `2 * delaySteps + 3` steps after it was requested") count from the later of the request step and the client's first
  synced step.
- The final sync (§17.3) waits until every client has no `Pending` chunk and no queued part, the host has no queued
  snapshot, and every `Synced` chunk's sequence equals the host's (replaces `terrainSynced` / `recordsSynced`); the
  bound stays 600 steps and `2 * (delaySteps + maxReorderSteps) + 3` without loss.
- The comparison (replaces S22's and S23b's union rule): per client and space over its `Synced` chunks the sequence,
  the dense delta of the closed range and the effective record equal the host's; a chunk that is not subscribed has no
  entry and no record on the client (its interior samples read 0); `modificationGaps`, `chunkRerequests` and
  `queueOverflows` are 0. A difference names the peer, the chunk, the sequence and the first differing sample or key.
- `ScenarioWorld::holdChunk(peer, chunk)` / `releaseChunk(peer, chunk)` (the client's `ClientChunkSync::hold` / `release`), for
  scenarios that drive interest without moving a player (streaming worlds).
- In a world without streaming every client is subscribed to everything (D19), so the comparison covers every chunk
  with state on either side, as S22's and S23b's union rule did; the end condition adds "every space indexed".
- S24's Scope 10 ("the end condition is unchanged") is superseded by this section.

### 9. Engine scenarios (`tests/test_engine_streaming_scenarios.cpp`, S24's binary)

S24's five streaming scenarios (`registerStreamingScenarios` on `streamingScenarioSetup()`, S24's Tests; in S24's `test_engine_streaming_scenarios` binary) run
their host + client variants over chunk subscriptions; `edit_far_chunk_then_visit` is the "edited before the client
subscribed" case (the client never receives the live edit; the snapshot after the arrival carries it). Their checks on
the client ("every chunk within `fullRadius` is resident", `streamingStalls` 0 in `streaming_follows_player`) now
include the gate. In the scenario worlds the prefetch margin is 0, so a chunk becomes required and requested in the same
step and is resident on a client only one round trip later (the subscribe, the snapshot: at most `2 * delaySteps + 3`
steps). The player itself is not held (it stands on chunks requested ~240 steps earlier at 32 m/s and 128 m), so the
stall check stays; the client-side residency check of `streaming_follows_player` becomes "every chunk within
`fullRadius` is resident at most `2 * delaySteps + 3` steps after it was requested" (a change to S24's scenario,
made by S26).

One new scenario, added to `registerStreamingScenarios` and run in every variant:

- `streaming_reload_during_edit`: a player standing on chunk C (the setup's terrain); the host edits C every 3 steps
  for 30 steps while the player is moved (`requestArrival`, host) 600 m away (beyond the scenario worlds' kept distance
  of 192 m, S24 Scope 1, so C is unloaded and, on the client, unsubscribed and dropped) and back, dwelling 16 steps at each end
  (more than twice the `2 * delaySteps + 3` sync bound at delay 2), so edits are in flight during unsubscribe,
  resubscribe and pending snapshots at delay 2 and in the asynchronous variant; checks after the final sync: on host
  and client `terrainHeight` at the player's x/z equals the edited surface, and the end comparison checks the heights
  and sequences of C and its neighbours on the client.

## Out of scope

- Region levels and summaries in chunk messages (§13.2 "region level: the summary's version list"; S27). Until then a
  client sees edits only in subscribed chunks; coarse terrain levels and the pyramid beyond its interest show the base
  (S22's "clients keep every edited sample" no longer holds).
- The view camera as an interest source on clients (S24 G4: the camera is never a centre; a free camera far from a
  streaming client's players sees base content there; holds can add chunks); S27 / S32 decide. Clients of worlds
  without streaming subscribe everything (D19) and are not affected.
- Base snapshots of retired versions (D15; the retiring stage).
- Entity interest (`interestRadius`; `Despawn` exists from S25), snapshot rates, loss and reordering of the state channel (S32); the
  chunk protocol is correct under reordering of reliable messages across channels, which S26 tests, and under loss
  of nothing (reliable).
- Disconnect, rejoin, character files, seed adoption, a client world created from the host's info (S33); a
  `JoinAbort` from a client whose adoption failed (S33's disconnect covers it).
- Encoding snapshots on the streaming pool (main thread in S26; the benchmark reports the cost).
- Data-file hashes in the handshake (open question 3).
- Pieces in snapshots (S30 adds a section to `ChunkSnapshot`, version bump).

## Design notes

- **Why the queue is needed although each channel is ordered.** Snapshots (`ChunkData`) and live parts (`State`) are
  independent; a live part sent after a snapshot may arrive first (queued: its sequence is above the snapshot's), and
  one sent before may arrive after it (dropped: at most the snapshot's). Unsubscribe / resubscribe while an edit is in
  flight produces both orders, which is the acceptance test.
- **Gaps cannot occur on a lossless transport.** The host registers a subscription in phase 1 of step t and filters
  live parts from that step on; the snapshot is encoded at or after step t's phase 8 and includes every edit up to its
  encoding; every edit after the registration is sent to the peer. So the client receives every sequence above its
  snapshot. A gap is a bug or (from S33) a broken link; the re-request is the recovery, and the variants fail on it.
- **Shared border samples.** A snapshot writes its chunk's whole closed range, including samples shared with a synced
  neighbour that may already hold a newer value. The regression is transient: an edit that changed the shared sample
  after the snapshot's encoding also produced a part for this chunk (a part per chunk whose closed range contains a
  changed sample, S22), which is queued behind the snapshot and rewrites it; an edit before the encoding is in the
  snapshot. The host is authoritative; no neighbour check (S22's rule).
- **Interest diff in phase 8, not per event.** Reading the residency's chunk table in phase 8 keeps aborts trivial
  (D16), merges an unload and reload within one step into nothing, and needs one S24 enumeration (`forEachChunk`)
  instead of event plumbing.
- **Why an index for worlds without streaming.** Without residency the client must learn which chunks have state;
  sending a snapshot for every chunk of a 32 km space would be 16384 messages, while the index is 8 bytes per edited
  chunk, and every unlisted chunk is known to be at sequence 0 from the moment the index was encoded (later edits reach
  the client as live parts from sequence 1, because the host filters by the `all` subscription from the same step).
- **Tickets instead of epochs per chunk.** A ticket identifies one subscription of one chunk; it is cheaper than a
  per-chunk epoch on the host (the host only echoes it) and makes the stale-snapshot rule local to the client.
- **Admission instead of transport connection in the peer lists.** All existing consumers (replication, the join
  policy, `world_status`) want "may receive the world"; a transport connection without a handshake is not that.
- **Live parts per peer are encoded per peer.** Edits are rare (S22's budget: one per step at most in the terraform
  benchmark); the encoder runs once per peer with a non-empty subset. Peers with an identical subset could share the
  encoding; not needed at 8 players.

## Implementation steps (each builds with the CI recipe and passes `ctest`)

1. `MessageList::Entry::version`, `addRaw` with a version; `NetSchema` with its canonical section encoding and hashes;
   `World::netSchema` / `netSchemaHash`; unit tests.
2. `GenerationInfo` (rename and split of `GenerationStateMessage`); `World::replaceGeneration` factored out of
   `WorldLoader::commitGeneration`; `adoptablePins`.
3. Handshake: `JoinRequest`, `JoinReject`, `JoinAccept`, `Resync` v2, the renumbered `EngineMessage`, admission in
   `World::receiveMessages` (joining list, D13), `checkJoinRequest`, the client's `JoinState`; remove
   `GenerationState`, `onGenerationState`, `terrainRefused`, the `onSpawn` check; update the S16/S22/S23a/S23b tests
   that relied on the old join order (Tests). At this step the join still calls `sendChunkSnapshot` after the spawn
   set, so terrain and content tests keep passing.
4. `ChunkSnapshot`, `ChunkInterest`, `ChunkIndex`, `ClientChunkSync`, `HostChunkSubscriptions`,
   `TerrainEditStore::dropChunk`, `ChunkRecords::erase` / `set`, the tombstone cap; the `all` subscription of worlds
   without streaming (D19: every existing host + client scenario runs on it from here) and holds; remove
   `sendChunkSnapshot`, `TerrainChunkState`, `ChunkRecordState`; per-peer filtering of the host state list.
5. Streaming clients: `forEachChunk` as the wanted set, the modification gate, the per-chunk queue in the new apply
   path, `spaceReadiness`.
6. `ScenarioWorld` final sync and comparison, `streaming_reload_during_edit`, the client check of
   `streaming_follows_player`, `world_status` / `chunk_info`, `--coop` rejection.
7. Benchmarks before/after, documentation (amendments), AGENTS.md.

## Tests

New binary `tests/test_engine_world_net.cpp` (`saiga_test(test_engine_world_net.cpp "saiga_engine")` inside
`if (MODULE_ENGINE)`; budget about 1.5 s; worlds of `tests/world_net_test_setup.h` (new: `worldNetTestSetup(streaming)`,
S24's `streamingTestSetup` reduced to a 16 x 16-chunk overworld at 2 m with its slots and `test_dungeon`, streaming on or
off), two
or three worlds over `LoopbackTransport::createNetwork`, stepped host then clients; every timing loop is a `stepUntil`
with a bound, no sleeps):

- `NetSchema.EqualSetupsHashEqual`: two worlds of one setup: equal hash and section hashes.
- `NetSchema.HashCoversRegistration`: the hash and the named section change for: a component field added, a component
  version, a replicate mode, `EnvironmentSetup::enabled` (WorldTime's kind), an event, a class, a message's version and
  channel, an action, a held button, a space template's spacing, `WorldConfig::streaming`; not for `Saved`, a streaming
  radius or a data file.
- `JoinMessages.RoundTripsAndBounds`: `JoinRequest`, `JoinReject`, `JoinAccept`, `Resync`:
  round trip; truncation, an unknown flag bit, oversized text, bytes left over are errors; the frozen prefixes (D3) are
  asserted byte by byte against a literal.
- `ChunkMessages.RoundTripsAndBounds`: `ChunkInterest` (0 and 4097 entries rejected; `all` with entries rejected),
  `ChunkIndex` (empty, 1000 entries; unsorted, duplicate, out of range, more than `sizeChunks^2`: errors) and
  `ChunkSnapshot` (empty, record only, deltas only, both, zstd and raw; keys not ascending, a counter >= 2^24, more
  than 2^20 keys, N > 257, an unknown flag bit, a codec value above `kMaxTerrainDelta`, bytes left over: errors).
- `Handshake.JoinOrderAndAdmission`: delay 2: the client's first message is `JoinRequest`; the host's `connectedPeers()`
  and `peersJoined()` contain the peer only from the step that admitted it; the client receives `JoinAccept` before
  any spawn; a game message the client sends before `Joined` is dropped (`droppedNotJoined`); `joinState()` goes
  `Joining` -> `Joined`; a second `JoinRequest` of an admitted peer is ignored.
- `Handshake.ControlBeforeAcceptIsDeferred`: `LoopbackOptions::bytesPerRound` small enough that the spawn set and
  `JoinAccept` take several rounds, and a join policy that teleports the new player in the admission step: the
  `OwnerCommand` arrives before `JoinAccept`, is deferred and applied after `Joined` (the player ends at the target).
- `ChunkSync.PinAdoptionDropsResidency` (streaming): a joined client whose residency built chunks around its player;
  the host loads a save with other pins (`Resync` v2): after the adoption every chunk is requested again with the new
  plans and none is `Resident` before its new snapshot. (Before `Joined` a client owns no centre, so the join-time
  adoption has no residency to drop; `ClientAdoptsOlderPins` covers its plans.)
- `ChunkSync.RerequestKeepsResident`: a gap on a `Resident` chunk re-requests it; the chunk stays `Resident` (the mover
  does not hold) and ends equal to the host after the new snapshot.
- `Handshake.SchemaHashMismatchIsRejected` (the acceptance test): the client registers one extra test component; host:
  `joinsRejected == 1`, no admitted peer, nothing sent but `JoinReject`; client: `Rejected`, `joinRejection()` names
  the schema and its differing section ("types"), no entity arrived, `connectedPeers()` empty.
- `Handshake.RejectionReasons`: one case each for environment flag, render tables (a mesh registered only on the
  client), seed, a generator missing, a client generator older than the host's current version (`TestTerrainGenerator`
  1 vs 2), a slot table that is not a prefix; each `joinRejection()` names its reason and the environment case is
  reported as environment, not as schema (D5).
- `Handshake.ProtocolVersionMismatchIsRejected`: a `JoinRequest` with version 2 and a different layout after the
  version (trailing bytes of another shape), sent from a raw loopback end (the test frames the packet with S15's
  framing): the reply is `JoinReject{ProtocolVersion}` (not `Malformed`), decodable with the frozen prefix.
- `Handshake.MalformedRequestAndEarlyMessages`: garbage `JoinRequest` -> `JoinReject{Malformed}`, `rejected` counted;
  a `ChunkInterest` or `PlayerAction` before admission -> rejected input, no state change.
- `Handshake.ClientAdoptsOlderPins` (E11): a host world loaded from a save of a `TestTerrainGenerator(1)` world but
  running version 2 (pins v1, a chunk record at v1 with a tombstone); a fresh client of the v2 build joins: its
  `generation().space(0).pinned == 1`, `terrainHeight` equals the host's at 64 seeded points, the chunk's snapshot
  creates the v1 record (S23b's dormant older-version branch: the content set is regenerated at v1, checked through the
  recording sink).
- `Handshake.HostLoadResyncAdoptsNewPins`: a joined client; the host loads a save with other pins: the client adopts
  them, clears its chunk state, re-subscribes every held chunk in the new epoch (`ChunkInterest` epoch == new epoch) and
  ends equal to the host; an old-epoch `ChunkInterest` in flight is dropped (`droppedEpoch`).
- `Handshake.AdoptionFailureIsLocal`: a client whose plan hash after adoption differs (a test generator whose plan
  depends on a per-world flag) ends `Rejected` with `generationMismatch()`, and rejects later world messages.
- `ChunkSync.SubscribeSnapshotAndLiveEdits`: hold chunk C, step until `chunkSynced(C)`; host edits C and tombstones a
  key in C: the client applies both live; sequences and dense deltas equal; `snapshotsApplied == 1`.
- `ChunkSync.LiveEditsOnlyToSubscribers`: two clients, only client 1 holds C; an edit across C and its neighbour D
  (client 2 holds D): client 1 receives only C's part, client 2 only D's; `partsFiltered` counts; client 2 has no
  entry for C.
- `ChunkSync.ReloadDuringEditConverges` (the acceptance test; parameterized over the release step offset k in -1..3
  relative to the edit, delay 2, and over one or two edits): client holds C and D (neighbours), host edits across their
  border; at offset k the client releases C and holds it again one step later; step until no chunk is pending: C's and
  D's dense deltas and sequences equal the host's, `modificationGaps == 0`; for the offsets where a live part arrives
  while C is pending, `partsQueued > 0` and the snapshot dropped it or applied it after (both paths asserted across
  the parameter set); a stale snapshot of the first subscription is dropped (`snapshotsStale`) in the offsets that
  produce one.
- `ChunkSync.UnloadDropsStateKeepsSharedBorder`: client holds C and D, host edits both; release C: C has no entry or
  record, its interior samples read 0, the shared border with D keeps the host's values, `onTerrainChanged` and
  `onRecordChanged` fired once for C.
- `ChunkSync.GapRerequests`: a crafted `TerrainModification` with `sequence = current + 2` fed to the client's handler:
  nothing applies, `modificationGaps == 1`, `chunkRerequests == 1`, C is `Pending` with a new ticket, the next phase 8
  sends an add with it; after the real messages the client equals the host. The same for a `TombstoneModification`.
- `ChunkSync.QueueOverflowRerequests`: the client holds C at delay 2; before C's snapshot arrives, 1025 crafted
  `TerrainModification`s for C (sequences 1..1025) are fed to the client's handler: the queue overflows at 1025, is
  dropped (`queueOverflows == 1`), C is re-requested with a new ticket, the first snapshot is stale, and the client
  ends equal to the host (which never edited C).
- `ChunkSync.InvalidSnapshotsApplyNothing`: crafted snapshots: unknown space, chunk out of range, wrong N, deltas for
  an enclosed space, placement version above current, keys for a slotless generator, slot or counter out of range:
  rejected, nothing applied; another epoch: dropped, not counted; an unknown ticket: stale.
- `ChunkSync.HostValidatesInterest`: out-of-range chunk, a chunk twice, more than `maxChunksPerPeer`, from a client
  role: rejected input, the table unchanged.
- `ChunkSync.SnapshotBudgetPerStep`: 8 edited chunks, budget 1 byte: one snapshot per step, FIFO; a resubscription of a
  queued chunk replaces its ticket (one snapshot sent).
- `ChunkSync.JoinDataFor50EditedChunks`: 50 chunks of a 1 m space edited with a seeded noise brush covering each
  chunk; a client holding all 50: the total `ChunkSnapshot` bytes are printed and asserted <= 20 MB (§17.10 join data;
  deterministic byte count, so it gates). About 0.3 s (the reason is in the test's comment).
- `ChunkSync.ResidentAfterSnapshot` (a streaming world, S24's gate and hold): a client-owned player on chunk C at
  delay 2, the snapshot held back by the budget for a few steps: the client's `residency().state(C)` stays `Requested`
  with every part ready, `chunkReady` is false, the owner's `characterMoveSystem` holds the player (`moverHolds` grows,
  `streamingStalls` counts), and the step after `chunkSynced(C)` the chunk is `Resident` and the player moves.
- `ChunkSync.SubscribeAtRequest`: a streaming client with `prefetchMargin` 256: the `ChunkInterest` add of a chunk is
  sent in the step the residency requests it (before its base is generated, asynchronous pool with a held job via
  `jobHookForTests`), and the snapshot is applied before the chunk becomes resident.
- `ChunkSync.ArrivalSubscribesTarget`: the host calls `requestArrival` for a client-owned player 3 km away: the
  replicated `PendingTransition` makes the target a client centre (S24 G4) and the client subscribes the target island
  before the host moves the player; after the move the player is held until its chunk is synced, then free.
- `ChunkSync.AllSubscriptionWithoutStreaming`: a world without streaming, 3 edited chunks and one with a record before
  the join: the client's `ChunkIndex` lists exactly those 4; an edit of an unlisted chunk between
  the index's encoding and its arrival is queued and applied from sequence 1; the client ends equal to the host over
  every chunk; a hold is a no-op; a remove in a world without streaming, and an add in a space the peer has not
  `all`-subscribed, are rejected input (an add in an `all` space is the re-request form); an `all` from a streaming world's client too.
- `ChunkSync.AbortsKeepTheProtocol` (D16): a throwing phase-8 test system on the host after the snapshot queue was
  served: the queue keeps the snapshots and they are sent next step; on a client after the interest diff: the staged
  `ChunkInterest` is dropped and the next phase 8 computes the same diff with new tickets; a gap's re-request and the
  `all` subscription whose phase 8 aborts are re-sent next step (the unsent adds); all end synced.
- `ChunkSync.CompletePointDoesNotBlock`: `completeResidency` on a streaming client whose snapshot is held back by the
  host's budget returns within the parts' time (the chunk stays `Requested`).
- `ChunkSync.ReadinessWaitsForSnapshots`: `spaceReadiness` on the client is below 1 while a subscribed chunk inside the
  radius is pending and 1 after.
- `TombstoneCap.RefusesBeyondTheCap`: `ChunkRecords::injectForTests` with 2^20 keys; `canTombstone` refuses one more
  with the error.

Changed existing tests:

- `test_engine_net.cpp`: the peer-list tests (`peersJoined` / `connectedPeers` on the first receive) take the
  handshake's round trip into account (admission one round later; the assertions stay, the step at which they hold
  moves).
- `test_engine_terrain_net.cpp`: `LateJoinerGetsEveryChunkState` and `HostLoadSendsResyncThenTheStates` are replaced by
  `ChunkSync.*` / `Handshake.HostLoadResyncAdoptsNewPins`; `DuplicatesGapsAndEpochs` asserts the re-request instead of
  the applied gap; the other tests are non-streaming worlds, which subscribe everything (D19), so they keep their
  assertions with the handshake's extra round trip.
- `test_engine_content.cpp`: `GenerationState.*` move to `Handshake.*` (equal setups pass, mismatches are rejected by
  the host, the legacy case); `TombstoneNet.LateJoinerSnapshotOrder`, `HostLoadResyncs` are replaced by the snapshot
  tests; `LiveAndFlatJoiner`, `SequencesGapsAndEpochs`, `RejectionsApplyNothing`, `ContentRender.ClientRecordChanges`,
  `ContentEdit.ClientResnaps` keep their assertions (non-streaming, D19); `TombstoneNet.MessageBounds` drops
  `ChunkRecordState`.
- `test_engine_app.cpp`: the `--coop` join test sees `peersJoined()` on the admission step; `world_status` /
  `chunk_info` `handleMessage` tests get the new fields (R-MCP-1).
- `test_engine_streaming_scenarios.cpp` (S24's binary): `streaming_reload_during_edit`; S24's streaming scenarios pass their host + client
  variants over subscriptions (the client check of `streaming_follows_player`, Scope 9). `test_engine_scenarios.cpp`:
  every other scenario (worlds without streaming) passes its host + client variants over the `all` subscription (D19).
- `test_engine_streaming.cpp` (S24's `StreamingNet.*`): "no message type is added" becomes "the client's messages
  are `JoinRequest` and `ChunkInterest` only, the host's add `JoinAccept`, `ChunkSnapshot` and `Despawn` (S25)"; the hold test is unchanged (the gate adds to it).
- `test_engine_net_benchmark.cpp` (non-gating): `NetBenchmark.ChunkSnapshotEncode` prints the encode and apply time
  of a fully edited 1 m chunk's snapshot and of 100 empty ones, and the per-peer filtering cost of 8 peers.

Speed: `test_engine_world_net` <= 1.5 s alone on lavapipe (no rendering), `test_engine_streaming_scenarios` grows by at
most 0.3 s (one scenario) and `test_engine_scenarios` by at most 0.2 s (the handshake pumps); the binaries above stay within their AGENTS.md budgets.

## Done when

- The host + client variants (delay 0, 2, asynchronous) of every engine scenario pass: S24's streaming scenarios and
  `streaming_reload_during_edit` over subscriptions, every other scenario over the `all` subscription (CI).
- `Handshake.SchemaHashMismatchIsRejected` passes: a join with a different schema hash is rejected with a message on
  both sides (CI).
- `ChunkSync.ReloadDuringEditConverges` passes for every offset: the client ends with the host's heights (CI).
- Every test above passes on GCC, Clang (with `SAIGA_CONTENT_UBSAN=ON`) and MSVC (CI).
- The tech demos' scenarios pass their host + client variants (`--scenarios`) and `--coop --demo` screenshots of the
  three demos show the client's world as before (local); `game_flight --coop --demo` (the streaming demo, S24 G22)
  shows the ruin and the terrain around the client's plane, and the Engine tab's Streaming section shows no stall.
- `test_engine_net_benchmark` before/after reported; `ChunkSync.JoinDataFor50EditedChunks` <= 20 MB.
- `docs/engine_design.md` and AGENTS.md amended (below); the S26 row of §22 marked done.

## Verification

```shell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSAIGA_BUILD_SAMPLES=OFF -DSAIGA_ARCHNATIVE=OFF \
  -DSAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON
cmake --build build --parallel -- -k 0
(cd build && ctest --output-on-failure --parallel 4)
./build/tests/test_engine_world_net --gtest_print_time=1
./build/tests/test_engine_streaming_scenarios --gtest_print_time=1
./build/tests/test_engine_scenarios --gtest_print_time=1
./build/tests/test_engine_net_benchmark            # before and after (R-TEST-2)
# Clang job equivalent: -DCMAKE_CXX_COMPILER=clang++ -DSAIGA_CONTENT_UBSAN=ON, then test_engine_world_net and
# test_engine_content
# Tech demos (samples ON, local, headless screenshots only):
./build/samples/games/game_flight --scenarios && ./build/samples/games/game_racing --scenarios && \
  ./build/samples/games/game_platformer --scenarios
SAIGA_SCREENSHOT=/tmp/racing_coop.png ./build/samples/games/game_racing --headless --coop --demo
```

## Files

New:

- `src/saiga/engine/net/NetSchema.{h,cpp}`
- `tests/world_net_test_setup.h`
- `src/saiga/engine/net/JoinMessages.{h,cpp}` (`JoinRequest`, `JoinReject`, `JoinAccept`, `Resync` v2 formats,
  `kProtocolVersion`, `JoinRejectReason`)
- `src/saiga/engine/net/JoinCheck.{h,cpp}` (`checkJoinRequest`, `adoptablePins`)
- `src/saiga/engine/net/ChunkMessages.{h,cpp}` (`ChunkInterest`, `ChunkSnapshot`, `ChunkIndex`)
- `src/saiga/engine/net/ClientChunkSync.{h,cpp}`, `src/saiga/engine/net/HostChunkSubscriptions.{h,cpp}`
- `tests/test_engine_world_net.cpp`

Renamed: `src/saiga/engine/net/GenerationStateMessage.{h,cpp}` -> `GenerationInfo.{h,cpp}`.

Changed:

- `src/saiga/engine/net/NetIds.h`, `EngineMessages.{h,cpp}`, `MessageList.{h,cpp}`, `Replication.{h,cpp}`
- `src/saiga/engine/terrain/TerrainMessages.{h,cpp}` (remove `TerrainChunkState`), `TerrainEditStore.{h,cpp}`
- `src/saiga/engine/content/ContentMessages.{h,cpp}` (remove `ChunkRecordState`), `ChunkRecords.{h,cpp}`,
  `Tombstones.cpp`
- `src/saiga/engine/world/World.{h,cpp}` (join state, admission, holds, `chunkSynced`, `replaceGeneration`,
  `netSchema`, `NetworkConfig`, `spaceReadiness`)
- `src/saiga/engine/save/WorldLoad.cpp` (`commitGeneration` uses `replaceGeneration`)
- `src/saiga/engine/tools/ScenarioWorld.{h,cpp}`, `ScenarioRunner.cpp`
- `src/saiga/engine/api/WorldApi.cpp`, `src/saiga/engine/tools/WorldTools.cpp`, `src/saiga/engine/app/EngineApp.cpp`
- `tests/CMakeLists.txt`, `tests/test_engine_net.cpp`, `tests/test_engine_terrain_net.cpp`,
  `tests/test_engine_content.cpp`, `tests/test_engine_app.cpp`, `tests/test_engine_scenarios.cpp`,
  `tests/test_engine_streaming_scenarios.cpp`,
  `tests/test_engine_streaming.cpp` (S24's `StreamingNet.*`), `tests/test_engine_net_benchmark.cpp`
- `docs/engine_design.md`, `AGENTS.md`

## Risks

- **Scenario timing in the streaming host + client variants.** In a streaming world a client-owned character is held
  until its chunk's snapshot arrives (one round trip after the request). Only S24's streaming scenarios are affected
  (worlds without streaming have no hold, S24 G1); the known change is the client check of `streaming_follows_player`
  (Scope 9); any other change is listed with its reason in the stage's results.
- **The index of a large space.** A world without streaming whose space has very many edited chunks sends a large
  `ChunkIndex` (8 bytes per chunk, at most ~2 MB for 512 x 512 chunks) and then every edited chunk's snapshot, as
  S22's join snapshot did; the join-data test measures the 50-chunk case.
- **Snapshot cost on the host step.** Encoding a fully edited 1 m chunk costs about a millisecond (dense extraction
  plus zstd); the 1 MB budget bounds a step to roughly four of them. The benchmark measures it against §17.10's 4 ms
  host step; moving the encoding to the streaming pool is the fallback.
- **Adoption at join changes a running client world.** Plans, terrain sources and content sets are replaced while
  the client may already have rendered its own pins' base; the same path as a host load (tested since S23a), but on a
  client for the first time.

## Open questions (with the chosen defaults)

1. Client-driven or host-derived interest? Default: client-driven (D1).
2. Adopt the host's seed? Default: no, compare and reject; S33 creates the client world from the host's info (D6).
3. Data files (tunable values) in the handshake? Default: no; the host simulates, clients use data for presentation.
   A `dataHash` field can be added with a protocol version bump if a game needs it.
4. The view camera as a client interest source? Default: no (S24 G4; holds exist; S27 / S32 decide).
5. Gaps still fail the host + client variants although they now recover? Default: yes (lossless loopback: a gap is a
   bug).
6. `chunkSnapshotBytesPerStep` default 1 MB, `maxChunksPerPeer` 16384, the queue bound 1024 parts, the tombstone cap
   2^20 keys: defaults as listed; the benchmark and the join-data test may move the first.
7. A client whose adoption fails tells the host? Default: no (`JoinAbort` waits for S33's disconnect, D2).
8. Base snapshots: none (D15); the retiring stage adds a flag bit.
9. Worlds without streaming: subscribe everything (D19) or make streaming mandatory for networked worlds? Default:
   subscribe everything (S24 G1 keeps streaming a policy; the racing and platformer demos and most tests are
   networked without it).

## Design-doc amendments (R-DOC-2, with the implementation)

- §13.2: an "S26 as implemented" paragraph (handshake, interest, snapshot, queue, re-request, epochs, the removed
  messages); the S22, S23a and S23b "before S26" paragraphs shortened to what still holds (live message layouts and
  their validation) with a pointer.
- §13.1: the "with the features that need them" bullet: S26 exists; tombstones and terrain edits are sent to the
  clients that subscribed the chunk.
- §13.6: the message list and IDs (renumbered), `Entry::version`, admission in the peer lists (D13),
  `droppedNotJoined`, `joinsRejected`, `chunkRerequests`, the new meaning of `modificationGaps`.
- §12.2: "clients keep every edited sample" replaced by D10; the "Before S26" sentence.
- §12.3: the client bullets (queue, resident after the snapshot through the modification gate, `spaceReadiness`) as
  implemented; the subscribed set (requested or resident, `forEachChunk`), holds, the `all` subscription of worlds without
  streaming (D19); S24's "S24 (before S26)" sentence of §13.2 replaced.
- §15.3: "connected clients ... are refused until S26" -> adopted (D6); base snapshots with the retiring stage (D15).
- §7.9: "no join check before S26" -> the environment flag is a handshake field.
- §10.1 (name tables): the handshake compares `renderTableHash`.
- §17.3: the end condition and comparison of Scope 8.
- §22: the S26 row's status.
- AGENTS.md: `engine/net/` in the layout (handshake, `NetSchema`, chunk sync), `test_engine_world_net` and its budget.
