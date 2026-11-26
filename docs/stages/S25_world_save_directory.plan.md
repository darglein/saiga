# S25: World save directory

Status: done (implemented on `stage-s25` on top of `stage-s24` at 2c75103a2; reconciled with the merged S24 code in step
1; the deviations are listed in "Deviations during the implementation", the measurements in "Results").

Sections follow the stage plan template of S05; "S24 interfaces used", "Conflicts with S24", "Interfaces for S26",
"Implementation steps", "Risks and open questions" and "Design-doc amendments" are added for this stage. Decisions
are numbered G1-G19; S24's plan also uses G numbers, so its decisions are cited as "S24 G<n>".

## Goal

A world lives in a **save directory** that is written incrementally and survives a kill at any point, and entities far
from players leave the simulation (`docs/engine_design.md` §9.6, §12.3, §12.4, §15.1-§15.3):

- a **chunk store**: per chunk with state one self-contained record (terrain modification, chunk record, entity
  groups) in append-only region files, zstd compressed, with a schema table number per entity group and lazy
  migration on reload;
- **commits**: between steps the main thread freezes the world-state snapshot and copy-on-write references of the dirty
  chunk states; the streaming pool writes the records, syncs them, then writes the world-state file (temporary file,
  sync, rename) naming commit c; the previous commit is kept as the backup; compaction runs inside a commit;
- **entity unload / reload** (worlds with `WorldConfig::streaming`, S24): when a chunk is beyond the kept distance of
  every centre (S24's `kept`, a pure predicate of the current centres), its root entities are serialized with their
  descendants into entity groups (group-local `Entity` remapping, references out of the group stored as null) and
  removed by the engine-internal `unload` command (no `onDestroy`, structural log records flagged `unloaded`,
  replication sends `Despawn`); roots that end up on such a chunk (spawned, flung or teleported there) are unloaded by
  the same phase-7 pass; a chunk that becomes ready (requested and resident) reloads its groups before the next phase 1
  (the step-start hook, or phase 7) with `configure`, `onLoad` and `onCatchUp(elapsed)`; players are never unloaded and stay in the world-state file (until
  S33);
- **space unloading**: a space without streaming centres (players, arrival targets) stores its space-state entity and is
  marked unloaded;
- **every generated chunk gets a chunk record** (§15.3), so a newer placement version never changes a visited chunk;
- `world.save()` / `load()` blobs stay self-contained (they carry the stored entity groups), the save/load scenario
  variant also round-trips through a directory, and the world API, MCP and the engine app can commit.

Acceptance (§22): the unload/reload scenario in every variant; the save directory survives a kill at any point of a
commit, also with an unload between two commits (CI).

## Depends on

- **S24** (streaming; the names in "S24 interfaces used"): `WorldConfig::streaming`, `ChunkResidency` (requested,
  ready, kept, centres, listeners, parts), synchronous mode, arrivals through
  `PendingTransition`, `Simulated` in phase 7 with `evaluateCreated` and `refreshSimulated`, the modification state kept
  in memory, the step-start refresh, the load's residency rebuild, the streaming benchmark.
- **S11** (save, schema, migration): `World::save` / `tryLoad` (`engine/save/WorldSave.cpp`, `WorldLoad.cpp`: the
  `WorldLoader` with validate-then-commit and the three object passes), `SaveFormat.h` (`kSaveSections`,
  `SectionWriter`, optional sections with their own format version), `SchemaTable` (`fromWorld`, `write`, `read`,
  `compareTypes`), `SaveDecoder` / `GenericDecode` (binary or by field list, migrations), `MigrationList` (versions,
  `Removed` policies), `LoadResult`, `diffSaves`, the committed golden saves.
- S21: several spaces, one space-state entity per space mapped by `Transform::space` (A10 of S21: `makeStateEntity`),
  `SPCE` with the prefix rule, `requestTransition` / `moveToSpace`, `spaceStateEntity(space)` (a dead or null handle is
  already skipped by `worldTimeSystem` and `EnvironmentExtract`).
- S22: `TerrainEditStore` (blocks keyed by sample, one copy per sample, entries per chunk with sequence, the cached
  content hash, reserve-then-commit writers), `TerrainDeltaCodec`, `World::onTerrainChanged` (the single point of every
  applied terrain change), `TERR`.
- S23a/b: `ChunkRecords` (`reserve`, `markStamped`, `replace`), `World::onRecordChanged` (the single point of every
  record change), `writesGenerationState()`, `GENR` / `CREC` format 2, `chunkStampSystem` (phase 7, the stamped set
  keeps a reloaded chunk from being stamped again), `ContentCache`, `tests/content_test_setup.h` (the test generator
  with versions 1 and 2).
- S16: the replication's structural log consumer, `DestroyMessage`, `onReplicatedDestroy`, the host + client variants
  and `compareReplicas`. S10: `GameObject::onCatchUp(double)` exists as a virtual with an empty body (never called
  before S25), `Behaviour::alwaysSimulate`, `Persistent` / `makePersistent` / `resolve` with the World's
  `persistentIndex`. S20: `StreamingPool` (priority queue, completions in `drain()` on the main thread). S12/S14:
  `ScenarioWorld::saveLoad`, the runner's variants, `WorldApi`, `registerWorldTools`, `handleMessage` tests. S13:
  `engineMain`'s command line, `EngineAppSetup`.
- Not used: `ZipStreamWriter` (`core/util/zlib/zip_stream_writer.h`, master f92a2a42d). It writes forward-only, stored
  (uncompressed) zip archives; the chunk store needs append across sessions, random reads of single records, its own
  checksums and per-file sync and rename for the commit point. A later `export_save` tool that packs one commit into a
  single file for bug reports could use it (out of scope).

## S24 interfaces used

The S24 names this plan builds on (`S24_streaming.plan.md`: decisions G1-G22 of S24, cited as "S24 G<n>", and its
"Interfaces for S25 and S26"). S25's own decisions keep their numbers G1-G19 (cited without a prefix).

- **B1 Streaming is opt-in** (S24 G1): `WorldConfig::streaming` (default `false`). **Without streaming S25 unloads
  nothing** (G18): no entity unloading, no space unloading, no auto records (Scope 12); the three
  S25 systems are registered only with `streaming` (as S24 registers `streaming` and `simulation`). The chunk store,
  commits, the save directory, `EGRP` and the hash part exist for every world but stay empty of groups, so every
  existing test, scenario, hash and save is unchanged. `game_flight` is the only game with streaming (S24 G22).
- **B2 Residency** (S24 Scope 2): `World::residency()` -> `ChunkResidency` with `state(ChunkId)` (`None`, `Requested`,
  `Resident`), `requested(ChunkId)` (within the requested distance of a centre: a pure function of the current
  centres), `ready(ChunkId)` (requested and resident, S24 G7), `centres()` (per space: `StreamingCentre {space,
  position, entity, arrival}`), `kept(ChunkId)` (pure: within the kept distance of a current centre),
  `addListener(ResidencyListener {onUnload, onReadyChanged})` (called inside `update()` / `complete()`, in `ChunkId`
  order; listeners never write World state). S25 adds the agreed extensions (S24 "Interfaces for S25"): the listener
  members `onResident` and `onKeptChanged`, `addPart`, `World::addStepStartHook`. S25 accumulates the flips it needs
  until its pass consumes them (a second `update()` between steps loses nothing); it never changes residency.
- **B3 Distances** (S24 G3, Scope 1): per `SpaceTemplate` the required distance `R = residencyRadius` (0: `fullRadius`),
  `R + prefetchMargin` (requested, default 256), `+ unloadMargin` (kept, default 128); `simulationRadius` (384). The
  scenario distances are stated once in S24 Scope 1 (requested 128 m, kept 192 m).
- **B4 Phase order** (S24 Scope 2, G8; §8.1): before phase 1 S24's step-start refresh when centres changed between
  steps (S24 G5), then S25's step-start hook (reloads of chunks that became ready, Scope 9); phase 7 S24's `streaming` (`residency().update()`: centres, requests, unloads, the pool drained
  through `TerrainSystem::drainStreamingPool()`, results applied) and `simulation` (the `Simulated` flags), then S25's
  `spaceLoadSystem`, `chunk_stamp`, S25's `chunkEntitySystem` and `spaceUnloadSystem`.
- **B5 Arrivals** (S24 G12, G13): `World::requestArrival(e, space, position, rotation?)` adds `PendingTransition` (held:
  frozen, hidden) and the target is a streaming centre (S24 G4); `spaceTransitionSystem` completes it in phase 1 when
  `islandProgress == 1`. S25: a group that contains an entity with `PendingTransition` is never unloaded; a space with
  a centre (players or arrival targets) is loaded.
- **B6 The pool** (S24 G16): `TerrainSystem`'s `StreamingPool`, one per World, drained in phase 7; priorities: required
  chunks `-100000 + d`, prefetch chunks `50000 + d`. S25's IO lane submits its job at the priority of its head task
  (Scope 4).
- **B7 Synchronous mode** (S24 G5, Scope 1): `World::setStreamingSynchronous(true)` / `StreamingSettings::synchronous`,
  set by `ScenarioWorld` on every world it creates: phase 7 builds every requested chunk inline. S25's
  `SaveDirectoryOptions::synchronousIo` defaults to it, and the `PartStored` read runs inline too.
- **B8 Parts** (S24 Scope 2): `ChunkResidency::addPart(ResidencyPartInfo {bit, wanted(ChunkId), build(ChunkId)
  (synchronous), request(pool, ChunkId, priority, done) (asynchronous), drop(ChunkId)})`; S25 registers `PartStored =
  4` (Scope 9): wanted on the host for a chunk with a group whose bytes are only on disk, so such a chunk becomes `Resident` (and `ready`) only with its groups read.
- **B9 Modification state in memory** (S24 G2, G20): S22's deltas and S23b's records stay in memory for every chunk;
  edits of non-resident chunks are synchronous. S25 keeps all of it in memory (G2), so it needs no load hook (S24 adds
  none).
- **B10 Load** (S24 Scope 9): a committed load in a streaming world calls `residency().dropAll()` and
  `completeResidency` for the required chunks of the loaded centres before `tryLoad` returns. S25 clears its
  accumulated flips and sets `fullReload` and `fullUnload`: the next step-start hook reloads every ready chunk with
  groups (and writes the records of resident chunks without one, Scope 12), the next phase-7 pass unloads every
  unloadable root on a chunk that is not kept. The load itself changes no entity and no record (so `save()` right after
  a load equals the blob), and the plain world, whose step-start refresh (S24 G5) sees the same centres, does the same
  at the same points. `openSaveDirectory` uses the same path.
- **B11 Chunk of a root**: S24 has no per-entity chunk index; S25 computes `ChunkId{space, floor(x / 256), floor(z /
  256)}` of the root's `Transform` position, clamped to the space's chunk range (the stamp system's rule).
- **B12 `Simulated`** (S24 G8, G9, Scope 6): kept in the ENTS slot byte; reloaded entities are created entities, so
  S24's `evaluateCreated` sets their flags at the reload's flush (exact before the next phases 2-5);
  `World::refreshSimulated(Entity)` for single edits; un-freezing calls no `onCatchUp` (only a reload does).
- **B13 Statistics and benchmark** (S24 G11, G21): `StreamingStats` (`streamingStalls`), `streaming_status`, and the non-gating `tests/test_engine_streaming_benchmark.cpp`
  (`SAIGA_STREAMING_BENCHMARK=full`: 10^5 entities, a player at 30 m/s). S25 adds unloads, reloads and the freeze to the
  same run.
- **B14 Clients** (S24 G10): clients stream their own residency, every entity is `Simulated` except held ones, clients
  never stamp; S25 does nothing on clients (no store content, no unloading); the host's `Despawn` removes entities.
- **B15 S24 tests that change with S25**: S24 worlds with entities beyond the kept distance now unload them. Adapted
  in step 1 (S24 places its "far" test entities inside the requested distance and beyond the simulation radius, so
  `FarEntitiesFreeze` and `far_entities_freeze` are unaffected): `Arrival.NonPlayerArrivalUsesS21Rule` (the NPC
  landing 3 km from every player is unloaded in that step's phase 7, not frozen), `streaming_follows_player`'s
  stamped-entity check (site entities behind the player unload once their chunk is not kept),
  `Simulated.MatchesBruteForce` (2000 drifters and a seeded walk: the brute force skips stored groups and counts
  reloads), `Simulated.ChildrenFollowRoot` and `NewEntitiesEvaluatedAtCreation` (a far root or spawn beyond the kept
  distance is unloaded instead of frozen), `StreamingSave.BetweenStepNpcChangeMatchesLoad` (the far root of the
  re-parenting case), `StreamingNet.*` (far replicated entities are despawned on the client instead of simulated; the
  message list adds `Despawn`), `Simulated.MoverHolds` and the hold after a direct `moveToSpace` (the target is
  requested), and S24's streaming benchmark (its entities are spread inside the player's requested area and it
  reports the resident count and the unloads and reloads per second instead of assuming 10^5 resident).

## Conflicts with S24

None open. S24's plan carries the agreed interfaces (its Interfaces for S25: entities unload beyond the kept distance
of every current centre, a pure predicate; S24 adds no `ensureModificationState` hook, since no
stage evicts modification state) and with the listener and part interfaces this plan uses (B2, B8), the step-start
refresh (B4, B10) and `evaluateCreated` (B12).

## Rules that apply

- R-STATE-1/2: entity groups carry only components; game objects are recreated (`configure`, `onLoad`) and catch up
  through `onCatchUp` (what catches up is game policy, R-MECH-1); references that survive unloading use
  `PersistentId` (out-of-group `Entity` fields become null).
- R-SYS-1/2: the unload and reload are engine-internal commands at the phase-7 flush; the new systems
  are free functions registered with phase, order and `RunOn::Host`.
- R-PAR-1: all store IO (record writes, reads, compaction, deletes) runs on the streaming pool; the main thread
  serializes and freezes only (no step phase does IO).
- R-DET-1: reloads in `ChunkId` order, unloads in root entity-index order, group order is unload order; `ScenarioWorld` runs the IO
  synchronously; the bytes of a group are a function of the state (equal hashes in the repeat variant).
- R-NET-1/2: unloading on the host is replicated as `Despawn`, reloading as a new spawn; scenarios pass the host +
  client variants.
- R-SAVE-1, §15.2: every record section has a tag and format version, unknown sections are load errors; entity groups
  store their schema table number and are migrated lazily on reload; the blob's new optional sections follow the S21
  recipe.
- R-MECH-1: the commit interval, the catch-up and which games use a save directory are game policy.
- R-REG-1, R-FILE-1: systems in the engine's system list, `Despawn` in the engine message list, one class per file.
- R-MCP-1: `save_status` and `commit_save` with descriptions and `handleMessage` tests.
- R-TEST-1/2 (unloading and the step loop are performance relevant: S24's `test_engine_streaming_benchmark` is re-run),
  R-DOC-1/2.
- AGENTS.md test speed: every new binary about a second; the crash loops use the memory file system and small worlds.

## Scope

### 1. The save file system (`engine/save/SaveFileSystem.{h,cpp}`, `engine/save/MemorySaveFileSystem.{h,cpp}`)

All directory IO goes through one interface, so a test can stop it after any operation (G9):

```cpp
// Paths are relative to the save directory, '/' separated. Every call returns false with *error on failure.
class SAIGA_ENGINE_API SaveFileSystem
{
   public:
    virtual ~SaveFileSystem() = default;
    virtual bool exists(const std::string& path) = 0;
    virtual bool list(const std::string& dir, std::vector<std::string>& names, std::string* error) = 0;
    virtual bool fileSize(const std::string& path, uint64_t& size, std::string* error) = 0;
    virtual bool readAt(const std::string& path, uint64_t offset, uint64_t size, uint8_t* out, std::string* error) = 0;
    // Mutating operations (counted by the memory file system)
    virtual bool makeDirectory(const std::string& dir, std::string* error) = 0;
    // writeNew creates or truncates; append creates a missing file
    virtual bool writeNew(const std::string& path, const uint8_t* data, size_t size, std::string* error) = 0;
    virtual bool append(const std::string& path, const uint8_t* data, size_t size, std::string* error) = 0;
    virtual bool truncate(const std::string& path, uint64_t size, std::string* error) = 0;
    virtual bool syncFile(const std::string& path, std::string* error) = 0;
    virtual bool syncDirectory(const std::string& dir, std::string* error) = 0;
    virtual bool rename(const std::string& from, const std::string& to, std::string* error) = 0;  // atomic replace
    virtual bool remove(const std::string& path, std::string* error) = 0;
    // An exclusive lock of the directory (the file <dir>/lock, created if missing), released by unlock or when the
    // process ends
    virtual bool lock(std::string* error) = 0;
    virtual void unlock() = 0;
};
SAIGA_ENGINE_API std::shared_ptr<SaveFileSystem> diskFileSystem(const std::string& root);
```

- `DiskSaveFileSystem` (in `SaveFileSystem.cpp`): `std::filesystem` plus the platform calls: POSIX `open` / `pwrite` /
  `pread` / `fsync` (files and directories), `rename(2)`, `flock` for the lock; Windows `CreateFileW`,
  `FlushFileBuffers`, `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`, `LockFileEx`, directory sync
  a no-op. Paths through `string_to_path` (UTF-8, the lesson of #88). Files are opened per call (no handle cache in
  S25; measured, Scope 15).
- `MemorySaveFileSystem` (tests and `ScenarioWorld`): files as byte vectors with a **crash model** (G9): per file the
  current and the durable bytes (`syncFile` makes the current durable), per directory the current and the durable
  entries (create, rename and remove change the current entries; `syncDirectory` makes them durable). Every mutating
  call increments `operations()` and is appended to an operation log. `failAfter(n, bool tornWrite)`: calls after the
  n-th mutating one fail with "simulated crash" and change nothing; with `tornWrite` the first failing `append` /
  `writeNew` applies the first half of its bytes before it fails. `clone()` copies the whole state; `replay(log, n,
  torn)` applies the first n logged operations to a clone (the crash tests build every image from one run, G9).
  Images: `afterKill()` (the current state: a killed process leaves its completed writes in the OS),
  `afterPowerLoss()` (only durable bytes and durable entries), `afterEntriesOnly()` (the current directory entries
  with only the durable bytes of each file: the classic ordering failure where a rename reached the disk before the
  renamed file's data), each a new `MemorySaveFileSystem`. Reads and `lock` are not counted. Thread safe (one mutex): the IO queue and synchronous reads may use it concurrently.

### 2. On-disk layout (`engine/save/SaveDirectoryFormat.h`, `engine/save/RegionFile.{h,cpp}`)

```
<dir>/lock                               the lock file (Scope 1; ignored by the open's cleanup)
<dir>/header.sgh                         the header (G5)
<dir>/world.<c>.sgw                      the world-state file of commit c (decimal, no padding); c and its backup c-1
<dir>/regions/<space>/<rx>.<rz>.<g>.sgr  region file generation g of region (rx, rz) of a space (G6)
<dir>/*.tmp, regions/**/*.tmp            temporary files (removed at open)
<dir>/world.<c>.sgw.corrupt              a world-state file that failed its checksum (kept for diagnosis)
```

All integers little-endian; every file ends with an FNV-1a 64 (`core/Hash.h`) of all its preceding bytes except
region files, whose records carry their own checksum.

- **Header** `header.sgh` (format 1): magic `SGWH`, format (u32), the engine version that created the directory (string), the schema tables (count
  u32; per table `SchemaTable::write`), trailer. The seed and the generator pins are not repeated here: the
  world-state blob's `GENR` is their single authority. Tables are append-only and numbered from 0 by position (G7); a build uses
  the number of the stored table whose `write` bytes equal `SchemaTable::fromWorld(world)`'s, or appends a new one
  (written before the first record that uses it, in the same commit). Rewritten only when a table is appended:
  `header.sgh.tmp`, sync, rename, sync the directory.
- **World-state file** `world.<c>.sgw` (format 1): magic `SGWF`, format (u32), commit c (u64), the number of header
  tables the commit needs (u32), then two framed parts, each its uncompressed length (u64) and a zstd frame: the
  **world-state blob** and the **chunk index** (`CIDX`, Scope 11), then the trailer. The world-state blob is a plain
  `World::save` blob without the store's state (`HEAD, SCHM, SPCE?, GENR?, ENTS, COMP, EVNT, ACTN, STAT`: no `TERR`,
  `CREC` or `EGRP`; that state is in the records and the index), so `tryLoad`'s parser reads it unchanged. The index is
  built on the pool from an index snapshot taken at the freeze plus the locations the commit task produced (incl.
  compaction's moves), so no placeholder has to fit inside the blob. Players and every resident entity are in `ENTS` /
  `COMP` (until S33).
- **Region files** (G6): a region is 16 x 16 chunks (4096 m, the CDLOD root node; `kStoreRegionChunks = 16`); rx =
  floor(x / 16). File header (24 bytes): magic `SGRF`, format (u32 = 1), space (u16), padding (u16), rx, rz (i32),
  generation (u32). Then records, appended, never rewritten in place.
- **Record** (format 1): a 32-byte header — magic `SGCR` (u32), record format (u16 = 1), kind (u8: 0 chunk, 1 space),
  flags (u8 = 0), space (u16), reserved (u16 = 0), x, z (i32; 0, 0 for kind 1, stored in region (0, 0)), commit (u64),
  payload length (u32) — the payload (a zstd frame, `SaveDirectoryOptions::zstdLevel`, default 3) and an FNV-1a 64 over
  header and payload. The uncompressed payload is sections in a fixed order, each tag (u32), format version (u32),
  length (u64), bytes; an unknown tag, a section out of order or twice and a version above this build's are errors
  naming the chunk and the tag (§15.2):
  - `RTER` (format 1, chunk with terrain state): sequence (u32 > 0), flags (u8: bit 0 deltas present); with deltas N
    (u16), the encoded length (u64) and the dense N x N delta of the chunk's closed range in `TerrainDeltaCodec` form
    without its own zstd (the frame compresses). `TERR`'s per-chunk entry plus N and without the inner zstd.
  - `RREC` (format 1, chunk with a chunk record or in the stamped set): placement version (u16), stamped (u8), the
    tombstone count (u32) and per key slot (u8) and counter (u32). `CREC` format 2's per-chunk entry.
  - `RGRP` (format 1, chunk with stored entity groups): the group count (u32), then per group in their order (oldest
    first) its schema table number (u32, the header's numbering) and its bytes in the group form of Scope 6.
  - `RSPC` (format 1, kind 1 only): the space-state entity of an unloaded space as one group (Scope 10).
  A record is the complete state of its chunk at its commit: the chunk index names at most one record per chunk. A
  chunk whose whole state fits the index summary (a chunk record without tombstones and no deltas or groups: the
  every-visited-chunk records of Scope 12) has an index entry without a record (file index `none`). The payload's
  uncompressed size is bounded (256 MB, from the zstd frame header before decompressing; untrusted input).
- **Commit numbers** start at 1 and increase by one per successful commit; a failed commit's number is reused. Records
  carry the commit number they were written for.

### 3. The chunk store (`engine/save/ChunkStore.{h,cpp}`; World-owned, host only)

The World always has a `ChunkStore` (`World::chunkStore()`, engine-internal; `const ChunkStore& chunkStoreView() const`
for tests and tools). Without a save directory it is a memory store (stored groups as bytes in memory); with one it
also holds the index of the records on disk (G3).

```cpp
struct StoredGroup
{
    uint64_t id = 0;                                    // a store-wide serial (not saved): identifies it across commits
    std::shared_ptr<const std::vector<uint8_t>> bytes;  // the group form (Scope 6); null: only on disk
    uint32_t schema = 0;                                // table number (header numbering; memory store: current table)
    uint64_t hash   = 0;                                // FNV-1a 64 of bytes (kept when bytes are dropped)
    int32_t recordIndex = -1;                           // its position in the chunk's current record (-1: not in it)
};
struct RecordLocation { uint32_t regionFile = 0; uint64_t offset = 0; uint32_t length = 0; uint64_t commit = 0; };
struct ChunkEntry
{
    // The chunk's groups in stored order, the single list (G4): each in memory, on disk (in the current record), or both
    std::vector<StoredGroup> groups;
    std::optional<RecordLocation> record;  // the record of the newest successful commit (null: none)
    bool dirty = false;           // the state differs from the record (Scope 4)
    uint64_t readJob = 0;         // a running read of the record's groups (0: none)
};
class SAIGA_ENGINE_API ChunkStore
{
   public:
    struct Stats
    {
        uint64_t groupsUnloaded = 0, entitiesUnloaded = 0, farUnloads = 0, unloadCandidates = 0, groupsReloaded = 0,
                 entitiesReloaded = 0, droppedReferences = 0, failedGroups = 0, reloadHookErrors = 0,
                 spacesUnloaded = 0, spacesReloaded = 0, recordsRead = 0, syncReads = 0, recordsWritten = 0,
                 bytesWritten = 0, commits = 0, failedCommits = 0, compactions = 0;
        double lastFreezeMs = 0, maxFreezeMs = 0, lastCommitWriteMs = 0;
    };
    const std::map<ChunkId, ChunkEntry>& chunks() const;
    bool hasGroups(const ChunkId& c) const;                  // in memory or on disk
    bool needsRead(const ChunkId& c) const;                  // a group with bytes == null (PartStored wanted)
    size_t groupCount(const ChunkId& c) const;
    bool spaceUnloaded(SpaceId s) const;
    const Stats& stats() const;
    // ---- engine-internal writers (main thread)
    void addGroup(const ChunkId& c, StoredGroup g);           // unload: marks the chunk dirty
    // The groups of a chunk for its reload: false while a group's bytes are only on disk (S24 keeps such a chunk
    // Requested through PartStored, so this happens only after a failed read); else all groups, the entry keeps none
    // afterwards and is dirty (its record still lists them until the next commit: G4)
    bool takeGroups(const ChunkId& c, std::vector<StoredGroup>& out);
    void markDirty(const ChunkId& c);                         // modification state changed (Scope 4)
    // ...
};
```

- **Dirty marking** (host only: a client's store stays empty, also through S26's snapshot apply and drop) at the
  existing single points: `World::onTerrainChanged(space, box)` marks every chunk whose
  closed range intersects the box (the chunks whose sequence changed); `World::onRecordChanged(chunk)` marks the chunk;
  `bumpSequence` without a delta change is only reached through tombstones, which call `onRecordChanged`; stamping
  (`markStamped`) and the auto records of Scope 12 mark directly. Unload and reload mark through
  `addGroup` / `takeGroups`. A committed blob load marks every chunk with state or groups dirty (Scope 11); a
  directory open starts clean (the index matches the loaded commit).
- **When bytes leave memory** (one rule): a group's bytes are dropped only when the group is in the current record
  (`recordIndex >= 0`) **and** the chunk's S24 state is `None`: at a commit's completion for chunks that are `None`,
  and in S24's `onUnload` listener for chunks that leave. So a `Requested` or `Resident` chunk never has a disk-only
  group except right after the open (then `PartStored` reads them before the chunk becomes resident), and a chunk that
  re-enters the requested set while still kept (S24 G3 hysteresis) reloads without IO.
- Only the host has a chunk store with content; a client's store stays empty (its chunk state comes from the host).
- Space records (kind 1): `ChunkEntry`s keyed by `ChunkId{space, INT32_MIN, INT32_MIN}` (`spaceRecordKey(space)`;
  outside every chunk range: `validateSpaceTemplates` bounds sizes to 512 chunks).

### 4. Commits (`engine/save/SaveDirectory.{h,cpp}`, `engine/save/SaveIoQueue.{h,cpp}`)

```cpp
struct SaveDirectoryOptions
{
    std::shared_ptr<SaveFileSystem> fileSystem;  // null: diskFileSystem(path)
    // World time between automatic commits (game policy, §21); 0: only commit()
    double commitIntervalSeconds = 300;
    bool sync               = true;              // syncFile / syncDirectory (the crash tests count them either way;
                                                 // off only for tests on a real disk)
    int zstdLevel           = 3;                 // the crash tests use 1
    // The commit's IO runs inside commit(), reads inside the step (tests, ScenarioWorld); unset: follows
    // streamingSettings().synchronous (B7)
    std::optional<bool> synchronousIo;
};
struct SaveStatus
{
    bool open = false;
    std::string path;
    uint64_t committed = 0, inFlight = 0;  // the newest successful commit; the commit being written (0: none)
    uint64_t lastCommitStep = 0;
    double lastCommitTime = 0;             // world time of the last freeze
    std::string lastError;                 // the last failed commit's error (empty after a success)
    uint64_t dirtyChunks = 0, storedChunks = 0, storedGroups = 0, unloadedSpaces = 0;
    uint64_t regionFiles = 0, regionBytes = 0, liveBytes = 0;
};
// World (host, between steps, not in a game hook; asserts otherwise like save())
SaveOpenResult openSaveDirectory(const std::string& path, const SaveDirectoryOptions& options = {});
bool hasSaveDirectory() const;
uint64_t commit();       // freezes commit c now and hands it to the IO queue; returns c, or 0 if refused (see below)
bool waitForCommits();   // runs the IO queue on the caller until no commit is in flight (quit, tests); false if the
                         // last one failed
bool closeSaveDirectory(); // waitForCommits, reads every disk-only group back into memory (synchronous IO, measured),
                           // unlocks; the world keeps a memory store with every group's bytes
SaveStatus saveStatus() const;
```

Constants (not options): two kept commits (the newest and its backup), compaction of a region file when its live bytes
fall below half of its size and it is at least 1 MiB.

**Freeze** (main thread, between steps; G10): `commit()` with no commit in flight:

1. serializes the world-state blob into memory (the plain blob of Scope 11, uncompressed) and takes a snapshot of the
   store's index (every entry's location and summary, the region file table with each file's committed end);
2. for every dirty chunk: a `FrozenChunk` with copy-on-write references of its terrain blocks (`TerrainEditStore`
   blocks become `std::shared_ptr<TerrainEditBlock>`; the freeze takes references of the blocks that intersect the
   chunk's closed range plus a copy of its entry; a writer clones a block whose `use_count() > 1` in its reserve step,
   so the commit step still does not allocate, G11), a copy of its `ChunkRecord` and stamped flag (small), and its
   group list (ids, schema numbers, shared byte references, and for disk-only groups their `recordIndex` in the old
   record, read and copied on the pool);
3. the space records of unloaded spaces whose entry is dirty, likewise;
4. clears the dirty flags of the frozen chunks, records the commit as in flight, and submits it to the IO queue;
5. measures 1-4 (`Stats::lastFreezeMs`, `maxFreezeMs`; the §17.10 budget, Scope 15).

A `commit()` while one is in flight returns 0 and sets `commitRequested`; the World commits at the first step boundary
after the in-flight one completed. `commit()` on a client, inside a step, in a game hook or without a directory
returns 0 (logged in category `save`). **Automatic commits**: after a completed step (between steps, before the step
listeners) when `time() - lastCommitTime >= commitIntervalSeconds > 0`.

**The IO queue** (`SaveIoQueue`): a queue of IO tasks on the streaming pool (B6) of which at most one runs at a time
(one pool job at a time drains it; the World creates the pool if it has none). Reads (`PartStored`, Scope 9) run
before queued commit tasks and are submitted at their chunk's residency priority (required `-100000 + d`, prefetch
`50000 + d`); a commit task runs at priority `-50000` (after the required chunks, before the visual and prefetch work,
so movement cannot starve it); the lane's pool job carries the priority of its head task, and a task that arrives with
a higher priority while the lane job is queued (not running) cancels and resubmits it. A running task always finishes
first. `waitForCommits` runs the lane on the caller. With `synchronousIo` the task runs inside the call that submits
it. Commits, record reads for reloads (Scope 9), compaction and deletes all go through it, so a
read never sees a file that a later task deletes (a read queued before a commit task runs before it, and a file is
deleted only two commits after the index stopped naming it), and records are appended in commit order (G8).

**Write** (the commit task, pool thread):

0. every region file this task will touch is `truncate`d to its committed end from the index snapshot (a failed or
   torn earlier attempt leaves garbage behind the end; the offsets of this commit start at the committed end); a
   region file that exists but is not in the index (a crashed first commit, a failed compaction's generation `g + 1`)
   is recreated with `writeNew` (its header), never appended to;
1. if the header needs a new schema table (or is missing): write `header.sgh.tmp`, `syncFile`, `rename` to
   `header.sgh`, `syncDirectory(".")`;
2. compaction is decided first (G12): a region file below the thresholds gets a new generation `g + 1`; this commit's
   records of its chunks go there, and its live records that this commit does not rewrite are copied verbatim (offsets
   change, bytes do not); the old generation stays for the backup commit;
3. per frozen chunk build the record: `RTER` from the frozen blocks (dense delta, codec), `RREC`, `RGRP` (the frozen
   group list in order; a disk-only group's bytes are read from the old record by its `recordIndex`), compress, append
   to the chunk's region file generation (creating `regions/<space>/` and the file with its header when new; a new
   region file, also after its region dropped out of the index, takes a generation above every existing generation of
   that region in the directory listing, so it never overwrites a file the backup commit names); a chunk
   whose state is empty (no sequence, no record, not stamped, no groups) is dropped from the index; a chunk whose state
   fits the summary gets no record (Scope 2). **An unreadable old record** (checksum or decode failure): its groups
   cannot be separated, so the new record omits them, they are counted in `failedGroups` and logged with the chunk,
   and the commit succeeds; the backup commit still names the old record for one more commit (G14);
4. `syncFile` every region file appended to or created in this commit; `syncDirectory` of every directory that got a
   new entry (`regions`, `regions/<space>`, and `.` when `regions/` was created), all before step 5; on the directory's
   creation also its parent;
5. build the index from the snapshot and this task's locations (incl. compaction's moves), compress the
   world-state blob and the index, write `world.<c>.sgw.tmp`, `syncFile`, `rename` to `world.<c>.sgw`,
   `syncDirectory(".")` — **the commit point**;
6. remove `world.<c - 2>.sgw` and every region file generation that no kept commit references (garbage at open
   otherwise; no sync).

**Completion** (main thread, in the drain): success: the store takes the task's full location table (every frozen
chunk's new record, compaction's moved records, the region file table with the new committed ends); per frozen chunk,
the groups whose ids were frozen and are still in the entry get their new `recordIndex`, groups added since keep -1,
groups taken since are gone (the chunk is dirty, so the next commit writes it without them: G4); bytes are dropped by
the one rule of Scope 3; `committed = c`, `Stats::commits`; a pending `commitRequested` commits at the next step
boundary. Failure (any file system error): `lastError`, logged (category `save`), `failedCommits`; the frozen chunks
are marked dirty again (nothing was dropped), the commit number is reused, the next commit's step 0 cuts the garbage.
A load (blob or snapshot) resets `lastCommitTime` to the loaded `time()`.

**Loads, the IO lane and pool resets**: every lane task (commit or read) carries the store's **generation**, bumped by
every load and open; a completion of another generation applies nothing. `tryLoad` / `load_snapshot` /
`openSaveDirectory` on a world with a directory first run `waitForCommits()` (a commit's completion must not land in a
replaced store, and the record files must be consistent before the new index is used), then cancel the queued reads
(their chunks are dropped with the residency anyway). The lane keeps a "job outstanding" flag; `TerrainSystem` notifies
the lane before its pool reset (`replaceSources`, `TerrainSystem.cpp:1234-1235`: the pool destructor discards
completions without calling them), which clears the flag, and the lane resubmits its head task to the new pool, so
`waitForCommits` never waits for a discarded job.

**Errors after the commit point**: once the step-5 rename succeeded the commit is a success (`committed = c`); a failing
`syncDirectory(".")` after it is reported in `lastError` and logged, without rollback (the file is visible, its records
are synced), and step-6 removal errors are logged only (garbage removed at the next open). The open compares each read
record's header (kind, space, x, z, commit, payload length) with its index entry; a mismatch is corruption of that
candidate (Scope 5).

### 5. Open (`World::openSaveDirectory`)

```cpp
struct SaveOpenResult
{
    bool ok = false;
    std::string error;          // set when !ok: the world is unchanged
    bool created = false;       // the directory had no commit: the current world was adopted and committed
    uint64_t commit = 0;        // the loaded (or first) commit
    bool usedBackup = false;    // the newest world-state file failed its checksum
    LoadResult load;            // the world-state load (hook errors as for tryLoad)
    double openSeconds = 0;     // wall time of the open (reported, Scope 15)
    uint64_t recordsRead = 0;   // records read for the modification state
};
```

- Between steps on the host only, not in a hook; a second open of a world that has a directory is refused. The lock
  is taken first (another process or world holding it: an error naming the directory).
- **No commit, adopt**: only when the directory is missing or empty, or contains nothing but names of the known
  patterns other than world-state files (the lock file, `*.tmp`, `regions/**/*.sgr`, `header.sgh`) (the garbage of a crashed first commit, whose step 1 writes the header before
  the commit point; the header holds no world state): no `world.*.sgw*` of any kind, incl. `.corrupt`. Then the directory is created, the current world is adopted (every chunk with state and every stored
  group dirty) and commit 1 is made at once (`created`, `commit = 1`; with `synchronousIo` written before the call
  returns). A directory with world-state files but no valid one, or any name outside the known patterns, is an open
  error (never overwritten: a player's save must not be replaced by a fresh world).
- **Existing commits**: the candidates are the valid `world.<c>.sgw` files (trailer checksum and frames), newest first.
  Validate-then-commit like `tryLoad`, per candidate until one succeeds (`usedBackup` when it is not the newest):
  1. the parts are assembled into one `ParsedSave`: the world-state blob is parsed as today; the index is checked
     (Scope 11); every record whose summary has deltas or tombstones is read (checksum, sections) and converted into
     the `TerrainSection` / `ChunkRecordSection` values `TERR` / `CREC` would have produced; summary-only entries give
     their records directly; the groups and unloaded spaces from the index go into the staged store;
  2. the unchanged `WorldLoader` stages and checks it (every S11-S23 check, incl. the cross-chunk border rule and the
     plan checks of `stageGeneration`). Only **corruption** (a world-state file's or record's checksum or framing
     fails) makes the open try the next older candidate; every semantic or version failure (a section version above
     this build's, a missing generator version, a plan or load check) is an open error at once that changes nothing on
     disk (a newer build may open it; the newest commit must never be discarded for a build mismatch);
  3. commit: the world is replaced (the loader's commit and object passes), the store gets the index (groups without
     bytes, their hashes and `recordIndex` from the summaries), unloaded spaces from the kind-1 entries (their
     records read here and kept in memory, Scope 10);
  4. only now the file system is cleaned: newer invalid world-state files and the newer candidates that failed are
     renamed to `.corrupt` (logged), `*.tmp` removed, world-state files other than the loaded one and its predecessor
     removed, region generations that neither references removed, and every kept region file truncated to the largest
     end offset that a kept commit records for it (records of a crashed later commit disappear, G8); the header is read
     before step 1 (missing or invalid with world-state files present: an open error);
  5. in a streaming world S24's load path runs (B10: `dropAll`, `completeResidency`, `PartStored` read inline) and S25's
     `fullReload` and `fullUnload` are set (B10).
- The next commit is `c + 1` of the loaded commit.

### 6. Entity groups (`engine/save/EntityGroup.{h,cpp}`)

**Group form** (format 1; untrusted input, bounded like §15.1 saves):

- unload step (u64; the elapsed time at reload is `(step - unloadStep) * dt`, exact),
- entity count n (u32, 1..65536; a root with a larger subtree is not unloadable, logged once per root): local entity 0 is the root, then the descendants in hierarchy order (parents before
  children, then entity index),
- the `PersistentId`s of the group (count u32, then u64 each, ascending): checked at loads without decoding,
- the component blocks in component ID order (`Transform` first), each as `COMP`: schema index (u32), value count
  (u32), byte length (u64), per value the local index (u32) and the binary form. **`Entity` fields** inside the group
  are written as `Entity{local + 1, 0}`, every other `Entity` value (outside the group, dead, null) as `Entity{}`
  (`Stats::droppedReferences` counts the non-null ones; one `save` log line per unload step with the count). `Parent`
  is a component like the others (its target always inside the group).

The schema table number is not part of the bytes: it is stored next to them (`StoredGroup::schema`, the record's
`RGRP` framing, `EGRP`'s framing), so a renumbering never changes the bytes or their hash.

`GroupWriter` (main thread): per component value of the group a copy is remapped by one `EntityRemapVisitor` (a `visit`
over the copy that replaces every `Entity` field through a map; `NetVisitors.h` is the precedent) and written with the
pool's existing binary form; `PoolBase` gains `virtual void appendRemapped(uint32_t entityIndex,
std::vector<uint8_t>& out, const EntityRemap&) const` (copy, remap, then the value's bytes as `appendBytes` writes
them, `ecs/ComponentPool.h:78`). `BinaryWriter` and the `hash()` / `save()` path stay untouched (no
branch in the hot path; the ECS benchmark is run anyway, R-TEST-2). `GroupLoader` decodes with S11's `SaveDecoder`
against the group's schema table (the header's, or the blob's `EGRP` table list) and the current one: `Same` types straight from
the binary form, `Compatible` / `Migrate` through the field list and `MigrationList` (lazy migration, §15.2), `Removed`
policies (drop the entity: it and its descendants leave the group; drop the component; replace); then the same
`EntityRemapVisitor` over every decoded value replaces `Entity{i, 0}` by the new handle of local entity i - 1.
**One staging core** (step 4): the per-type decode and the checks that `WorldLoader` runs today keyed by save slot
(`WorldLoad.cpp:171-323`: Fast / Slow / Skip / DropEntity / Replace, `mapClasses`, `dropEntities`, `checkHierarchy`,
`checkPersistent`, `checkSpaces`, `checkRenderMeshes`) are extracted into a staging core over an index space, used by
`WorldLoader` (slot indices) and `GroupLoader` (local indices); no second copy of these rules.

### 7. Unload (`engine/save/Systems.h`, `engine/save/ChunkEntitySystem.cpp`, `World`, `StructuralLog`, replication)

- `chunkEntitySystem` (phase 7 Streaming, `RunOn::Host`, registered by the World only with `WorldConfig::streaming`,
  after `chunk_stamp`, B4) runs the reloads of chunks that became ready inside the step (Scope 9) and then **one unload
  pass**: every unloadable root whose chunk (B11) is not `residency().kept(chunk)` (beyond the kept distance of every
  current centre: a pure predicate, so the hysteresis band between the requested and the kept distance needs no
  history; whether a band chunk's groups are stored or resident is state, saved in `EGRP`) is unloaded, in root
  entity-index order, each with one command. The candidates (so the pass costs what changed, not 10^5 roots per step; `setOwner` changes and a subtree shrinking
  below the 65536 limit add their root too):
  the roots of chunks whose kept coverage flipped to 0 (accumulated from `onKeptChanged`, which S25 defines over the
  kept discs themselves, also for chunks that were never requested; found through the spatial grid's cells of the
  chunk), the roots that moved or were created since the last pass (`changedTransforms`, the structural
  log's created records, incl. the phase-7 flush of the previous step), and the roots whose exclusion ended (`Player`,
  `PendingTransition`, `NetOwner` or `Parent` removed: the structural log); every root while `fullUnload` (after a
  load or open, B10). This one pass replaces §9.6's separate phase-6 "pending" unload: a root spawned with `spawnIn`
  far from players, flung or teleported out of the kept area is unloaded by it in the same step (phase 7 instead of
  phase 6; recorded deviation), and a plain and a loaded world unload the same roots in the same phase.
- **Abort safety**: the system only selects; the accumulated flips, the candidate list and `fullUnload` are consumed
  when the step commits (an aborted phase 7 keeps them for the next pass), and the unload and reload commands take
  their groups from the store at the flush (`takeGroups` inside the command), so an aborted phase loses no group and
  creates no entity (`Unload.AbortedPhase7LosesNothing`).
- **Never unloaded** (the group stays; it is skipped every time): a group that contains a `Player`; a group whose root
  has `NetOwner` of another peer than the host (client-owned: its owner simulates it; G16); a group with an entity that
  has `PendingTransition` (held by an arrival, B5; its target is a centre and it completes in phase 1); state
  entities. `Behaviour::alwaysSimulate` does not prevent unloading (§9.6).
- **The `unload` command** (engine-internal, `World::unloadNow(Entity root, const ChunkId& into)` (`farUnloads` counts roots whose chunk was never ready); at the
  flush, or at once between steps): writes the group (Scope 6) from the state at the flush, adds it to the store
  (`addGroup`), then removes the root and its descendants like `destroy` but without `onDestroy`: the structural log
  records the removals with `Record::unloaded = true`; slots are freed (generation incremented); `Persistent` index
  entries removed (`resolve` returns `Entity{}`); the game objects are deleted without hooks.
- **Structural log**: `StructuralLog::Record` gains `bool unloaded = false` (`Prepared::append(e, component, added,
  unloaded = false)`); consumers that do not care treat it as a removal (spatial grid, extract: the render instance
  is removed; S28 bodies later).
- **Replication** (host): every replicated entity whose removal record is `unloaded` is sent as `DespawnMessage {NetId
  netId}` (`EngineMessage::Despawn`, version 1, `Channel::State`, reliable, in the spawn/destroy order of S16; one per
  entity, as `Destroy` is sent per entity, `Replication.cpp:1066`) instead of `DestroyMessage`; the client removes the
  entity without `onReplicatedDestroy` and frees the NetId mapping (§13.4's "no callbacks"). The host frees the NetIds
  as for a destroy. A reload is an ordinary spawn with new NetIds. **Bound NetIds** (state entities,
  `net/NetIds.h:43-49`): no `Despawn` is sent for them (a space unload, Scope 10); the client keeps its state entity
  (its `spaceLoaded` stays true, its values stale until the reload). The work is on the host: its structural log
  consumer suppresses `Destroy` / `Despawn` for the bound NetId of an unloaded state entity, and the reload re-binds
  that NetId to the new handle and sends its bound `Spawn`, which the client applies to its existing entity
  (`net/Replication.cpp:1873-1893`).
- **World API journal**: `changes_since` reports `kind: "unloaded"` for these records (reloaded entities appear as
  `created`).

### 8. Roots on chunks that are not kept

Folded into the one unload pass of Scope 7 (no phase-6 system, no `pending` flag): roots in a kept chunk stay (frozen
by S24's `Simulated` rule outside the simulation radius, held by its mover where not resident); roots on a chunk that
is not kept are unloaded by the phase-7 pass into that chunk's groups (memory; part of the next commit's record and of
every blob). A world without streaming has no such pass (B1, G18).

### 9. Reload (`chunkEntitySystem`, `engine/save/GroupLoader.{h,cpp}`)

- **Reads: the residency part `PartStored`** (B8; registered by the World with streaming): `wanted(chunk)` on the
  host iff `needsRead(chunk)`; `request` enqueues the record read
  on the IO queue at the chunk's priority, `build` reads inline (synchronous mode); the completion fills the bytes of
  the groups by `recordIndex` (structurally checked, checksum checked); `drop` (the chunk unloaded by S24 before it was
  ready) drops them again by the rule of Scope 3. So a chunk with stored groups becomes `Resident` only with them. A record that fails its checksum: logged with the chunk, `failedGroups` counts its
  groups, the part counts as applied without them (the chunk must not stay pending forever), the record stays on disk
  and in the index (G14).
- **Reload** wherever chunks become ready: in the **step-start hook** (`World::addStepStartHook`, B4: after S24's
  refresh, between steps, so commands apply at once; `fullReload` after a load or open is consumed here) for chunks
  made ready by the refresh or by `completeResidency`, and in `chunkEntitySystem` (before its unload pass, commands at
  the phase-7 flush) for chunks made ready in phase 7. Every chunk whose `ready` flipped to true since the last pass
  (accumulated from `onReadyChanged`) and that has groups, in `ChunkId` order, gets one reload command that takes its
  groups at its flush and recreates them in stored order. So the entities of an arrival's island exist before the
  phase 1 that completes the arrival, and a loaded and a plain world reload at the same point. A chunk that re-enters
  the requested set while still kept has its groups in memory (Scope 3), so no read is needed.
- **`GroupLoader`, validate-then-commit per group** (G14): the schema number exists; every type maps (an
  `Incompatible` type, a version gap without a migration, a newer version, an unknown removed type are errors); local
  indices in range; `Parent` only to an earlier local entity, the root without `Parent`; class IDs known; every
  `Transform::space` equal to the chunk's space; `PendingTransition` / `SpaceLink` targets below `spaceCount()`;
  `PersistentId`s not in use (`resolve` is `Entity{}`); `RenderInstance` mesh names through `checkRenderMeshes`' rule
  (unknown names kept). A group that fails is **kept** (put back into the entry, the chunk marked dirty only if other
  groups were consumed), logged in category `save` with the chunk, the group's position and the reason, and counted in
  `failedGroups`; it is retried at the next reload (a game patch may add the missing migration). Never silently
  dropped.
- **Commit of the groups of one reload command**: entities created at free slots (new handles), components added in
  block order with their log records, the `Persistent` index, `hierarchyDirty`, for every group of the command; then in
  passes over all of its game objects in group order, as `tryLoad` does: create all, `configure()` all (load rules: no
  structural change), `onLoad()` all (load rules: only `Saved::No` components of its own entity, no terrain edit), then
  `onCatchUp(elapsed)` all with `elapsed = (step() - unloadStep) * dt` (>= 0; no load rules: it runs like `onSpawn`
  inside the flush, so structural changes apply at once). So `resolve` works between groups of one chunk in every
  hook. `onSpawn` is not called. A hook that throws removes its
  entity with its descendants without hooks (as a load), logged and counted in `reloadHookErrors`; the rest of the
  group stays. The reloaded entities are created entities, so S24's `evaluateCreated` sets their flags at the flush
  (B12), exact in the next phases 2-5; tick evaluation is rebuilt in the next phase 6.
- The reload consumes the groups: the entry keeps none and is dirty, so the next commit writes the chunk's record
  without them (G4: an entity is in the world-state file or in one record of a commit, never both).
- A chunk that is stamped (S23a) is not stamped again (the stamped set is state), so a reloaded chunk gets its
  stamped entities from its groups only.

### 10. Space unloading (`engine/save/SpaceUnloadSystem.cpp`, `World`)

- `spaceLoadSystem` (phase 7, `RunOn::Host`, only with streaming, after S24's `simulation` and before `chunk_stamp` and
  `chunkEntitySystem`, B4) recreates the state entities of spaces that became loaded (below), so they exist before any
  chunk group of that space reloads (also run by the step-start hook before its reloads). `spaceUnloadSystem` (phase 7,
  after `chunkEntitySystem`; both in `engine/save/SpaceLoading.cpp`, R-FILE-1's `Systems.h` rule): a space is **loaded**
  while it has a streaming centre (`residency().centres()`: a player root, a `StreamingAnchor` or a pending player
  arrival's target, B2, B5). Level-triggered: every phase 7 (and after a load) a loaded space without centres is
  unloaded once it has no unloadable root left (the World keeps a per-space count of roots other than state entities,
  updated by `moveToSpace`, creation, removal and `Parent` add / remove, so no scan); the unload pass's commands apply
  at the phase-7 flush after `spaceUnloadSystem` ran, so the space unloads in the phase 7 after its last root left (one
  step later, deterministic in both worlds; chunk groups go first; a root that cannot be unloaded, Scope 7,
  keeps the space loaded): its space-state entity (if the world has one) is written
  as a group into the space record (`RSPC`) and removed with the `unload` command; `World::spaceLoaded(SpaceId)`
  becomes false and `spaceStateEntity(space)` returns `Entity{}`.
- A `moveToSpace` inside a step (phases 1-6) into an unloaded space recreates its state entity at once from the
  in-memory space group (the same path as `spaceLoadSystem`), so phases after it never run in a space without state;
  `spaceState<T>` asserts only for a space that is really unloaded.
- A space becomes loaded again when it gets a centre (a player moved in, a player's arrival targets it): its space
  record's group is always in memory (a single small group per space: read at the open, never dropped), so
  `spaceLoadSystem` recreates the state entity through `makeStateEntity`'s path at a free slot without IO.
- A non-player root that enters an unloaded space (an NPC arrival, `spawnIn`, `moveToSpace`) is not kept there: the
  space has no centre, so every chunk is beyond the kept distance and the root is unloaded by the same step's phase-7
  pass into its chunk's groups (it never runs a phase 2-5 in a space without state; it reloads when a player brings the
  space back).
- **Replication of a space unload**: no `Despawn` for the state entity (a bound NetId, Scope 7); the client keeps it
  with stale values until the reload; `compareReplicas` skips the state entities of spaces the host has unloaded.
- The world-state entity is never unloaded. A world without space-state components only flips the flag.
- Readers of space state of an unloaded space: `spaceState<T>` / `readSpaceState<T>` gain the assert "the space is not
  loaded" (today `World.h:977` checks only the index); the loader creates no default state entity for an unloaded space
  (`buildStateDefaults`); `ScenarioWorld::count<Transform>()` loops over `spaceCount()` instead of stopping at the first
  invalid `spaceStateEntity` (`ScenarioWorld.h:171`); `worldTimeSystem` and `EnvironmentExtract` already skip a dead entity; `set_weather`
  (`EnvironmentWrites`), `get_global` / `set_global` with a space and `get_lighting`'s weather (`LightingTools.cpp`)
  return an error naming the space instead of reading it.

### 11. Blobs, world-state form, hash, diff (`engine/save/{SaveFormat.h,WorldSave.cpp,WorldLoad.cpp,SaveDiff.cpp}`)

- **`EGRP`** (new optional section, format 1; written by `save()` when the store has groups or unloaded spaces): the
  count of extra schema tables (u32) and the tables (`SchemaTable::write`) that stored groups use other than the
  blob's `SCHM` (number 0 is `SCHM`, extra tables 1..); the count of chunks with groups (u32); per chunk in
  (`SpaceId`, `ChunkId`) order: space (u16), x, z (i32), group count (u32), per group its table number in this
  numbering (u32) and its bytes (copied unchanged); the count of unloaded spaces (u32); per space in order: space (u16),
  a flag (u8: 1 with a state entity group) and the group. `save()` (`const`) reads disk-only groups through the file
  system's read path only (`ChunkStore::readGroupsForSave`: it waits for a running IO task, reads the records, and
  mutates nothing: no completion is drained, no byte is cached; `syncReads`): blobs stay self-contained (tests,
  scenarios, API snapshots, `World::diff`, scenario failure reports).
- **The chunk index** (`CIDX`, format 1; the second framed part of a world-state file, not a blob section): the region
  file table (count u32; per file space (u16), rx, rz (i32), generation (u32), committed end offset (u64)), the entries
  (count u32; per entry in (kind, `SpaceId`, `ChunkId`) order: kind (u8), space (u16), x, z (i32), file index (u32,
  `0xFFFFFFFF`: summary only), offset (u64), length (u32), commit (u64), and the summary: sequence (u32), flags (u8:
  bit 0 deltas, bit 1 chunk record, bit 2 stamped), delta content hash (u64, with deltas), placement version (u16,
  with a record), tombstone count (u32), group count (u32), per group its schema number (u32) and hash (u64)).
- Section order (D7): `HEAD, SCHM, SPCE?, GENR?, TERR?, CREC?, EGRP?, ENTS, COMP, EVNT, ACTN, STAT`; `kSaveSections`
  gains `EGRP`; `kSaveFormatVersion` stays 1 (older builds reject a blob with it).
- **Load checks** (before the commit, errors with the world unchanged): `EGRP` chunks inside their space, sorted and
  unique; group forms structurally valid (counts and lengths within bounds, schema numbers within the table list);
  `PersistentId`s unique across `COMP` and every group (a blob load decodes the groups' ID lists; a directory open
  reads only the records it needs, so there the check is the reload's "not in use" rule of Scope 9); an unloaded space has no space-state entity in `ENTS` and
  every space-state entity's space is loaded. The index (open only): entries sorted, unique, inside their spaces, files
  and offsets within the table and below each file's committed end. Components inside groups are decoded at reload, not at load (lazy migration; a group that fails
  then is kept, Scope 9).
- **Commit of a blob load**: the store is replaced by the blob's groups and unloaded spaces (memory), every chunk with
  state and every chunk with groups is dirty (a world with a directory writes them at its next commit; chunks of the
  old index that have no state now are dropped from the next `CIDX`).
- **`world.hash()`** (§17.4): after S23b's block, only when the store has groups or unloaded spaces (worlds without
  unloading hash as before): per chunk with groups in (`SpaceId`, `ChunkId`) order the chunk, the group count and each
  group's hash in stored order folded sequentially (the hash of the bytes, kept when the bytes leave memory, so disk
  groups are never read for a hash and a directory world hashes like a blob world); per unloaded space the space and
  its group's hash. Schema numbers and record locations are not hashed.
- **`diffSaves`**: kind `stored_group`: chunks with groups on one side, group count differences, and per differing
  group its position, root class and entity count; unloaded space differences. `dumpJson` lists per chunk its groups
  (count, root class, entities), not their components.

### 12. Records of every generated chunk (§15.3)

- Host, streaming worlds, spaces whose generator has slots or whose plan has sites (the condition of the existing load
  check, `WorldLoad.cpp:978-980`): when a chunk becomes resident (S24's
  `onResident` listener, B2) and has no chunk record, it
  gets `{current placement version, no tombstones}` (`ChunkRecords::reserve`) and is marked dirty. The listener only
  accumulates the chunk (listeners never write World state, S24); the records are written by the step-start hook and
  by `chunkEntitySystem` (the same points as the reloads; `fullReload` covers every resident chunk without one), so a
  load never writes a record and a plain and a loaded world write the same records at the same point. The effective record
  does not change, so this writer does **not** call `onRecordChanged` (an exception to S23b's rule, recorded in
  §12.1: nothing derived changes). Clients get nothing new (S23b sends only records with tombstones or another version).
- Consequences: a newer placement version reaches only chunks that were never resident; `CREC` and the hash grow with
  the visited chunks of such spaces (streaming worlds only: S23a's and S23b's content scenarios, which do not stream,
  are unchanged). **Worlds without streaming** keep §15.3's accepted placement drift in visited but unmodified chunks
  (there is no residency that says which chunks were visited; G18). In the directory such records are
  summary-only index entries (Scope 2), so they cost no record IO.
- **Retiring a version is deferred** (G17): no shipped game has a retired generator version; storing generated bases
  touches every base consumer (`TerrainSystem::sample`, placements, stamping, grass, content render sets, S27's far
  content) and the base snapshots to clients go through S26's `ChunkSnapshot` (a new flag bit then). The record format
  takes it as a new section without a format break. §15.3 and §22 are amended (the stage
  that first retires a version implements it; proposed: with S27, which reads stored bases for far content).

### 13. Scenarios (`engine/tools/ScenarioWorld.*`, `ScenarioRunner.*`)

- Scenario worlds have a memory store (no directory) and run IO synchronously (B7: `ScenarioWorld` sets
  `setStreamingSynchronous(true)`). Unloading happens only in scenarios whose setup turns `WorldConfig::streaming` on
  (the S24 streaming scenarios and S25's unload scenarios); with S24's scenario distances (S24 Scope 1: requested
  128 m, kept 192 m) a group unloads once its chunk is 192 m from the last player.
- **The save/load variant also round-trips through a directory**: at the swap point, after the blob round trip as
  today, the old world opens a save directory on a fresh `MemorySaveFileSystem` (`created`: commit 1 of the current
  state, synchronous IO) and closes it (`closeSaveDirectory`: the lock is released), a second fresh `World` of the same
  setup opens it, and its `save()` must equal the blob
  (`diffSaves` explains a mismatch: "the directory round trip differs"). The scenario continues with the blob world as
  before. This puts every scenario's state through the world-state file, `CIDX`, records and the group form.
- `ScenarioWorld::storedGroups(ChunkId)` (a test helper: group count) and `entitiesNamed(PersistentId)` (resident or
  stored: 0 or 1; the duplicate check of the crash tests).

### 14. API, World API, MCP tools, app

- `World`: `openSaveDirectory`, `closeSaveDirectory`, `hasSaveDirectory`, `commit`, `waitForCommits`, `saveStatus`,
  `spaceLoaded`,
  `chunkStoreView`; `~World` waits for an in-flight commit (it does not commit).
- World API (`WorldApi`, read-only rule unchanged): `save_status` -> the `SaveStatus` fields and the store statistics
  (reads); `commit_save` -> `{commit}` (changes: freezes a commit; errors: no directory, a client, a commit in flight
  (`deferred: true` with the in-flight number instead of an error)). MCP world tools of the same names
  (`registerWorldTools`), descriptions per R-MCP-1; `world_status` gains `save: {committed, in_flight}` when a
  directory is open.
- App: `EngineAppSetup::saveDirectory` (empty: none), `saveCommitIntervalSeconds` (300) and `saveSync` (true; the app
  test turns it off); `engineMain`'s `--save
  <dir>` (not with `--scenarios` or `--benchmark`; with `--coop` the host world). The app opens the directory after the
  game's `init` hook (an existing directory replaces the world through the load path, as `load_snapshot` does; a new
  one adopts the world), commits at quit (`quit` tool, window close) and waits (`waitForCommits`, the result logged).
  The Engine tab (F3) shows one line: commit, in flight, last error. The tech demos and `game_world` do not save by
  default (`--save` works for every engine game).

### 15. Budgets and measurements (§17.10)

- **Freeze** (world-state serialization plus the copy-on-write references): <= 10 ms (§17.10's frame budget; a commit
  every 300 s may cost one such frame) with
  10^5 resident entities and 50 dirty edited chunks, release, desktop reference (Scope 15; reported, the trigger for an
  incremental world-state file if exceeded).
- **The first step after a large load or open**: `fullReload` decodes and runs the hooks of every ready chunk's groups
  in one step-start hook; reported with 1000 stored groups around the players (spread over several steps if above
  the frame budget).
- **Unload and reload**: the group write and removal, and the decode, creation and hooks, per entity and for a chunk
  of 100 entities (target <= 1 ms each on the main thread); reported.
- **Synchronous loads** (§12.3 "S25 measures it"): `editTerrain` and `tombstone` on a non-resident chunk (S24's base
  build; the state is in memory, G2, so no record is read), the synchronous group read of `ScenarioWorld` / `save()`,
  and the open (`openSeconds`, `recordsRead`) for 1000 edited chunks; reported.
- **Write throughput** of a commit (bytes and ms on the pool) and the region file growth / compaction ratio over the
  terraform benchmark; reported.
- S24's streaming benchmark (`tests/test_engine_streaming_benchmark.cpp`, `SAIGA_STREAMING_BENCHMARK=full`, B13) with
  unloading (streaming on, a memory-store world and a directory world committing every 10 s of world time): step p95
  within §17.10's 4 ms, unloads and reloads per second, the freeze, 0 streaming stalls (local; its short CI form prints
  the same keys, non-gating).

## Out of scope

- Retiring generator versions and stored bases (G17); base snapshots to clients (S26).
- Evicting the modification state of non-resident chunks from memory (G2; revisit with S27's region summaries).
- Character files, players outside the world-state file, rejoin rules (S33, §15.4).
- Cross-component migrations (split, merge; §15.2: when a game needs them).
- Moving NPCs abstractly far from players, an index of where an unloaded `PersistentId` is (game policy / later).
- A game hook to keep an entity loaded besides players (R-MECH-1: a game keeps chunks resident with its own player-like
  entities if needed; revisit when a game asks).
- Read-only inspection of a directory without a world, `export_save` (zip), a directory upgrade tool.
- Client-side stores; S26's chunk messages and handshake.

## Design notes

- **G1 One self-contained record per chunk.** A chunk's terrain modification, chunk record and entity groups are
  written together, so the index names one record per chunk and loading never combines records of different commits.
  The cost is rewriting a chunk's whole state when one part changes; records are small except edited terrain, which
  changes as a whole anyway.
- **G2 Modification state stays in memory** (amends §15.1 "unloaded chunks live on disk, not in memory"):
  `terrainHeight` / `terrainSample` are exact, lock-free and thread safe for any point (R-DET-1, §12.2), edits of
  non-resident chunks need their state at once (§12.3), and the visual path's coarse levels read the deltas of far
  chunks; evicting them needs S27's region summaries. Its size is bounded by what players edited (S22: blocks only
  where deltas are non-zero; §17.10's join budget assumes 50 edited chunks). Entity groups, the large and unbounded
  part, live on disk. The store still holds the modification state durably, so a commit writes only dirty chunks.
- **G3 The memory store and the directory are one store.** Without a directory the groups simply stay in memory; the
  unload, reload, blob and hash code is the same, so scenarios test the real path.
- **G4 One place per entity per commit**: a chunk's group list is one list (Scope 3); a reload consumes the groups and
  dirties the chunk, an unload adds them and dirties it; both are frozen together with the world-state snapshot of the
  same step boundary, and a completion maps frozen groups by id, so a change during an in-flight commit is never
  counted twice. So every commit has
  each entity either in its world-state file or in exactly one of its records, and a crash returns to one such commit.
  The crash tests check this with `PersistentId`s.
- **G5 A header for the schema tables every commit shares**, rewritten rarely and before the
  records that need it; the world-state file still carries its own `SCHM`, so the S11 loader reads it unchanged.
- **G6 Regions of 16 x 16 chunks**: few files (a 32 km space at 256 m chunks: at most 64 regions), appends that stay
  local to where players are, compaction per file.
- **G7 Schema numbers per group** (§15.2): only entity groups contain registered types; the record sections are
  engine formats with their own versions.
- **G8 Append-only plus an index in the world-state file**: the rename of the world-state file is the only commit
  point; records of later or failed commits are invisible (not indexed) and cut off at the next open by truncating to
  the ends that the kept commits record. The IO queue keeps all appends in commit order, so the cut is a suffix.
- **G9 Kill at any point, deterministically**: the memory file system logs mutating operations; replaying a prefix onto
  a clone builds the image after any of them from one run. Its images model a killed process (completed writes
  survive), a power loss (only synced data and synced directory entries survive), entries without data (a rename
  durable before the renamed file's bytes) and a torn last write. A test can thus enumerate every point of a commit in
  CI without processes or timing, and checks durability (n = K loads the new commit) as well as atomicity.
- **G10 Freeze between steps, write on the pool**: the world-state snapshot is serialized on the main thread (it must
  be consistent with the frozen chunk states), compression and IO are not.
- **G11 Copy-on-write terrain blocks**: freezing 50 edited 1 m chunks by copying dense deltas costs ~13 MB of copies
  in the step; shared block references cost pointer copies, and the reserve step (which may allocate) clones a block
  only when a frozen commit still holds it.
- **G12 Compaction inside the commit task**: a new generation is written and referenced by the new world-state file;
  the old generation stays until no kept commit names it. No separate protocol, the same crash tests cover it.
- **G13 Group-local entities as `Entity{local + 1, 0}`**: the group form is the `COMP` form with remapped handles, so
  S11's decoder, migrations and `Removed` policies work unchanged; one remap visitor serves both directions, and the
  loader's staging core is shared with `WorldLoader`.
- **G14 Failed groups are kept**: a group that cannot be decoded (a missing migration) stays stored and logged; a
  record that cannot be read is logged and counted (its groups are lost with the next commit; the backup keeps it one
  commit longer); dropping data that can still be decoded later would be unrecoverable, keeping it lets a patch fix it.
- **G15 Despawn, not destroy**: clients must not run destruction logic for an entity that still exists in the world.
- **G16 Client-owned groups stay**: their owner simulates them (R-NET-2); unloading them under the owner would break
  ownership. Players already keep their chunks resident, so this rarely matters.
- **G17 Retiring later**: see Scope 12.
- **G18 Unloading needs streaming** (B1): without `WorldConfig::streaming` there are no centres and no kept set, so
  nothing decides what is far; such worlds (small levels, tests without players, `game_world`, `game_racing`,
  `game_platformer`) keep every entity and space, while their saves still use the directory, the chunk records and
  commits. Unloading follows S24's kept predicate of the current centres, so the repeat and save/load variants stay
  exact.
- **G19 One unload pass in phase 7** (deviation from §9.6's phase-6 pending list): the chunk unloads and the roots
  that end up on a chunk that is not kept are the same rule (not kept), so one pass with one order (entity
  index) serves both; a loaded world (full pass) and a plain world unload the same roots in the same phase. Records
  of every visited chunk exist only in streaming worlds.

## Interfaces for S26

S26 (`S26_world_network.plan.md`: the handshake, `ChunkInterest` / `ChunkSnapshot` on `ChunkData` replacing
`GenerationState`, `ChunkRecordState`, `TerrainChunkState` and `sendChunkSnapshot`, per-peer filtering) and S25 do not
depend on each other; S26 is implemented after S25 and integrates. Where they touch (S26's "S25 touchpoints"):

- **Chunk snapshots read memory, not the store**: the modification state of every chunk stays in memory (G2), so S26's
  `encodeChunkSnapshot` reads `TerrainEditStore` / `ChunkRecords` unchanged and never reads region files; S26's
  touchpoint "S25 changes the source, not the message" needs no change at all. Sequences stay S22's.
- **Records of every visited chunk** (Scope 12): they carry the current placement version, so the snapshot's "record
  present" flag (only for another version) is unaffected, as S26 notes.
- **Base snapshots**: S26 reserves `ChunkSnapshot` flag bit 3 (S26 D15). S25 defers retiring (G17), so neither stage
  implements base snapshots; the retiring stage fills bit 3 from a record's `RBAS` section.
- **`Despawn`**: S25 lands before S26 and appends `EngineMessage::Despawn` (a typed message `DespawnMessage {NetId}`,
  State, reliable, one per entity, none for bound NetIds) after today's last ID; S26's renumbering includes it, and
  S32 reuses it for interest. Despawns go to every joined peer like `Destroy` (S26 filters only its chunk messages).
- **Entity groups never leave the host**: clients get entities only through replication (`Despawn` on unload, a new
  spawn on reload); S26's join sends the spawn set of resident entities only.
- **Host load / open**: `openSaveDirectory` on a host with connected clients is a host load (S26's `Resync` v2 with the
  generation info, then the per-chunk snapshots for the clients' subscriptions), exactly like `tryLoad`.
- **Space unloading**: clients have no player in an unloaded space, so they have unsubscribed its chunks (S26's
  client residency); the space-state entity gets no `Despawn` (a bound NetId, Scope 10).
- **Hash**: stored groups are host-only and not part of S26's replica comparison.
- **Record encoders**: `RTER` / `RREC` are byte-equal to `TERR` / `CREC` per-chunk entries; S26 may reuse the
  per-chunk encoders of `ChunkRecordCodec` (`encodeChunkTerrain`, `encodeChunkRecord`) if useful.

## Implementation steps

Each step builds with warnings as errors and passes ctest with the tests named in it.

1. **Reconcile** with the merged S24 code (the names of "S24 interfaces used": `ChunkResidency`, `ResidencyPartInfo`,
   the phase order, `setStreamingSynchronous`, the load path, `refreshSimulated`); update this plan (R-DOC-1). S26 is
   implemented after S25.
2. **File system layer**: `SaveFileSystem`, `DiskSaveFileSystem`, `MemorySaveFileSystem` with the crash model;
   `SaveFileSystem.*` tests.
3. **Formats**: `SaveDirectoryFormat.h`, `RegionFile`, `ChunkRecordCodec` (record header, sections, `RTER` / `RREC` /
   `RGRP` / `RSPC`), header and world-state file framing with checksums; format tests without a World.
4. **Groups and the memory store**: the staging core extracted from `WorldLoad.cpp` (both loaders on it, the existing
   save tests unchanged), `ChunkStore`, `EntityRemapVisitor`, `PoolBase::appendRemapped`, `GroupWriter`,
   `GroupLoader`, `unloadNow`, the structural log flag, `Despawn`, `chunkEntitySystem` (reloads and the one unload
   pass), `onCatchUp`; `EntityGroup.*`, `Unload.*`.
5. **Blobs**: `EGRP` (chunk groups), the hash, `diffSaves`, `dumpJson`; `Unload.BlobRoundTrip` without an unloaded
   space; the scenarios `chunk_unload_reload` and `far_spawn_reload` in every variant.
6. **Space unloading** and the space-state readers, `EGRP`'s unloaded spaces; `SpaceUnload.*`, the unloaded-space part
   of `Unload.BlobRoundTrip`, scenario `space_unload_reload`.
7. **Records of every generated chunk**; `SaveDirectory.EveryGeneratedChunkHasRecord` (memory part).
8. **Save directory, synchronous IO**: header, the index, open (incl. candidates, cleanup, truncation), commit,
   `closeSaveDirectory`, retention,
   copy-on-write blocks in `TerrainEditStore`; `SaveDirectory.*`.
9. **IO queue on the pool**, asynchronous commits and reads, failed commits; compaction.
10. **Crash tests** `SaveCrash.*` (`test_engine_save_crash`, the operation log and replay).
11. **ScenarioWorld** directory round trip in the save/load variant.
12. **World API, MCP tools, app** (`--save`, quit commit, Engine tab line); their tests.
13. **Benchmarks** (`test_engine_save_benchmark.cpp`, `test_engine_streaming_benchmark.cpp` with unloading), the
    committed directory `data/tests/engine/save_dir_s25/`, docs (`docs/engine_design.md` amendments, `AGENTS.md`).

## Tests

`tests/test_engine_save_directory.cpp` (new, `"saiga_engine"`; memory file system unless named; small worlds of a flat
open-world space with edits and a test game object; budget about 1.5 s):

- `SaveFileSystem.MemoryCrashModel`: a write without `syncFile` is gone after `afterPowerLoss` and present after
  `afterKill`; a rename is atomic in both images and durable only after `syncDirectory`; `failAfter(n)` fails the
  (n+1)-th mutating call and every later one without a change; a torn `append` leaves exactly its first half;
  `operations()` counts only mutating calls.
- `SaveFileSystem.DiskRoundTrip`: in a fresh temporary directory: append, `readAt`, `truncate`, `rename` over an
  existing file, `remove`, `list`, `syncFile` / `syncDirectory` (a handful of syncs on small files: measured, a few ms
  on the CI runner), a UTF-8 file name; a second `lock` on the same directory fails.
- `SaveDirectory.CreateAndOpen`: a new directory adopts the world (`created`, commit 1; files: lock, header,
  `world.1.sgw`, one region file per region with state); `closeSaveDirectory`; a fresh world opens it: `save()` and `hash()` equal the original's,
  `created == false`, `recordsRead` equals the chunks with deltas or tombstones.
- `SaveDirectory.CommitsAndRetention`: three commits leave `world.2` and `world.3`; an unchanged world
  commits no record (`recordsWritten` 0); an edit of one chunk writes exactly one record; opening gives the newest.
- `SaveDirectory.CorruptNewestUsesBackup`: one flipped byte in `world.3.sgw`: open loads commit 2 (`usedBackup`,
  `world.3.sgw.corrupt` exists after the open, the state equals commit 2's blob); both corrupt: an error, the world
  unchanged and no file renamed or removed.
- `SaveDirectory.NeverAdoptsOverASave`: a directory whose world-state files are all invalid, one with only `.corrupt`
  files, and a directory with an unknown file name: open errors, nothing written; a directory with only a lock file,
  `*.tmp`, region files and a header (a crashed first commit) is adopted, and its stale region files are recreated.
- `SaveDirectory.LoadWhileCommitInFlight` (a latch holds the IO task): `tryLoad` of a blob with the same pins and one
  with other pins wait for the commit; afterwards `inFlight` is 0, the store is the loaded one and the next commit
  succeeds; the same with a held `PartStored` read instead: its completion fills nothing in the loaded store, and
  after the pool reset of the other-pins load `waitForCommits` returns.
- `SaveDirectory.RegionGenerationNeverReused`: a region whose chunks all became empty drops out of the newest index
  while the backup names generation g; a new chunk there creates generation g + 1 (above every existing generation of
  that region in the listing), and opening the backup still works.
- `SaveDirectory.RecordErrors`: a flipped payload byte in a record with groups: open succeeds, the reload logs the
  chunk, `failedGroups` counts them; an edit of that chunk then commits: the new record omits them, the old location
  is still named by the backup commit, the commit succeeds; in a record
  with deltas of the newest commit while the backup's record of that chunk is intact: the open uses the backup
  (`usedBackup`); with no intact candidate: an open error naming the chunk; an unknown section tag and a section
  version above 1: errors naming the chunk and the tag; a frame header claiming more than 256 MB: refused before
  decompressing.
- `SaveDirectory.SchemaTablesAndLazyMigration`: groups unloaded with a test component `Crate` v1 (table 0); a world
  whose setup registers `Crate` v2 with a migration opens the directory: the next commit appends table 1 to the header
  (before any record of the commit), the unloaded group keeps number 0 until it reloads with the migrated value, and
  its rewritten record uses table 1; a `Removed {Crate, drop entity}` setup drops that entity and its descendants from
  the group on reload.
- `SaveDirectory.Compaction`: twenty commits that each edit the same chunk cross the threshold: a new generation is
  written, the old one survives one more commit (the backup names it) and is removed after the next; the region file
  shrinks; open equals the live world.
- `SaveDirectory.AsyncEqualsSynchronous`: the same steps with pool IO (`waitForCommits`) and with `synchronousIo` give
  byte-equal files.
- `SaveDirectory.CommitWhileInFlight` (a latch holds the IO task): `commit()` returns 0 and the request runs after the
  first completes; a chunk edited and one unloaded while the commit is in flight are in the next commit; a reload
  while its record is being written reads the frozen groups from memory; a chunk with disk-only groups (from an
  earlier commit) reloaded while commit c + 1 is in flight: after the completion the entry has no groups, the next
  commit's record has none and the entities exist once (opening commit c + 2 and counting `PersistentId`s).
- `SaveDirectory.FailedCommitRetries`: a file system that fails each mutating operation of a commit in turn; before
  the rename: the commit fails and is retried as below; after the rename (the directory sync, a step-6 remove): the
  commit counts as done, the error is reported, and the next open loads it. A failing append (clean, and with
  `tornWrite`, which leaves half a record behind the committed end): `lastError` set, the world continues, the dirty chunks stay dirty;
  after the file system recovers the next commit truncates the garbage, writes them and opens equal (the index
  offsets point at the records).
- `SaveDirectory.EveryGeneratedChunkHasRecord` (`tests/content_test_setup.h`, test generator v1): a player's resident
  chunks get records at version 1 (no `onRecordChanged`: the content render set request counter is unchanged); a world
  of `contentTestSetup(2)` opening the directory keeps version 1 for the visited chunks and uses 2 for an unvisited one.
- `SaveDirectory.LockedAndRefusals`: a second world opening the same directory fails naming it, and succeeds after
  `closeSaveDirectory`; `openSaveDirectory`
  inside a step and on a client assert; `commit()` without a directory returns 0.
- `SaveDirectory.CommittedDirectoryOpens`: `data/tests/engine/save_dir_s25/` (made by this test with
  `SAIGA_UPDATE_GOLDEN=1`, only for a new format version: an unloaded group, an edited chunk, a tombstone, an unloaded
  space) is copied into a `MemorySaveFileSystem` first (the open cleans and locks; the source tree stays untouched) and
  opens with the expected `PersistentId`s, heights and records (R-SAVE-1 guard of the on-disk format).
`tests/test_engine_save_crash.cpp` (new binary, the crash enumeration of G9; budget about 1 s alone): each case runs
its world once: to a reference state, commit c, change the world, record the reference blobs `save()` at c and at the
freeze of c + 1, then commit c + 1 on a `MemorySaveFileSystem` that logs its K mutating operations. Every image is
built from that one run by `replay` onto a clone of the pre-commit state (n = 0..K, four kinds: kill, kill with a
torn write, entries without data, power loss), opened in a fresh world, and asserted:

- the open succeeds; the loaded commit is c or c + 1, and it is **monotone in n** (once an image of a kind loads
  c + 1, every later n of that kind does); at n = K every kind loads c + 1 (the protocol really syncs before the
  commit point);
- the loaded world's `save()` equals that commit's reference blob;
- every test `PersistentId` exists exactly once, counted over the resident entities and the stored groups **read** from
  the records (a failed read fails the test);
- no `.tmp` and no unreferenced file remain; one more commit and open succeed (checked at n = 0, K / 2 and K only: the
  open of every image is the expensive part; equal images are opened once, deduplicated by a hash of the file system
  state, and the cases share the setup's `ContentCache`; the budget is measured in step 10).

Cases:

- `SaveCrash.KillDuringCommit`: commit c + 1 with an edited chunk, a tombstone and resident entities;
- `SaveCrash.UnloadBetweenCommits` (the acceptance): after commit c the player leaves, a group is unloaded (in memory
  only); commit c has the group resident in `world.c.sgw` and not in a record, commit c + 1 has it in the record and
  not in the world-state file;
- `SaveCrash.ReloadBetweenCommits`: unloaded before commit c (in its record, bytes dropped), reloaded after it: never
  duplicated;
- `SaveCrash.FirstCommit`: the directory creation at every point: the open either adopts (`created`) or finds commit
  1;
- `SaveCrash.CompactingCommit` and `SaveCrash.HeaderRewrite` (a commit that appends a schema table);
- `SaveCrash.KillDuringOpenCleanup`: the open's cleanup (truncate, remove, `.corrupt` rename) killed at every point:
  the next open loads the same commit.

`tests/test_engine_unload.cpp` (new, `"saiga_engine"`; worlds with `WorldConfig::streaming` and synchronous streaming
(B7), a flat open-world space and S24's scenario radii (128 m, prefetch 0), a test game object `Grower` with
`GrowerState {double grown; Entity buddy; Entity outside; PersistentId friendId}` and `onCatchUp` adding elapsed to
`grown`):

- `EntityGroup.RoundTripRemapsEntities`: a root with two children and a grandchild; `Entity` fields inside the group
  point to the new handles after reload, the field to an outside entity is null (`droppedReferences` 1), `Parent`
  chains are restored, `PersistentId`s kept: `resolve` is `Entity{}` while unloaded and the new handle after.
- `EntityGroup.FormBounds`: entity count 0 and 65537, a local index out of range, `Parent` to a later entity, a root
  with `Parent`, a duplicate `PersistentId`, a schema number out of range: refused at load (blob) or kept with
  `failedGroups` (reload), never a crash.
- `Unload.NoOnDestroyAndLogFlag`: `onDestroy` is not called; the structural log records are `unloaded`; the recording
  render sink's instance is removed; `queryRadius` no longer finds the entity; `changes_since` reports `unloaded`.
- `Unload.Exclusions`: a player, a group with a `Player` child, a client-owned root, a state entity and an entity with
  a pending `PendingTransition` (`requestArrival`) stay; an `alwaysSimulate` object unloads.
- `Unload.StreamingOffUnloadsNothing` (G18): the same world with `streaming = false`: a group 5 km from the player and
  a `spawnIn` far away stay resident, no space unloads, the store has no groups and `save()` has no `EGRP`.
- `Unload.NeverRequestedBandRoot`: a root spawned in a chunk between the requested and the kept distance (never
  requested) is unloaded when the player leaves so that the chunk is not kept, in the plain world and in a copy loaded
  at that moment, in the same step.
- `Unload.FollowsKeptSet`: a group in a chunk between the requested and the kept distance stays resident (frozen);
  beyond the kept distance it is unloaded; when the player turns back, the group reloads once its chunk is ready (from
  memory while S24 still holds the chunk: `recordsRead` unchanged); a player oscillating 20 m across the requested
  distance 50 times causes no unload or reload (the hysteresis); asynchronous mode with a held read job: a chunk with disk groups is not `Resident` until
  `PartStored` applied, and its entities exist in the step it becomes ready.
- `Unload.SimulatedAfterReload`: a reloaded group outside the simulation radius is frozen in the next phases 2-5
  (`evaluateCreated`), one inside it is simulated.
- `Unload.CatchUpElapsed`: elapsed equals (reload step - unload step) * dt exactly; order: all configure, all onLoad,
  all onCatchUp over the groups of a chunk (an `onLoad` resolves a `PersistentId` of another group of the chunk);
  `onSpawn` not called again; an `onCatchUp` that spawns an entity: the spawn applies in the flush; an `onLoad` that
  adds a saved component is refused as in loads.
- `Unload.AbortedPhase7LosesNothing`: a test system after `chunkEntitySystem` throws in the step that would unload
  one group and reload another: no group is lost, no entity is created, the next step does both.
- `Unload.ReloadBeforeArrivalPhase1`: a between-step `requestArrival` onto a chunk with stored groups (synchronous): the
  entities exist before the phase 1 that completes the arrival; a loaded copy saved right after the request does the
  same at the same point.
- `Unload.ReloadHookThrows`: a throwing `onCatchUp` removes that object with its descendants (no hooks),
  `reloadHookErrors` 1, the other groups of the chunk reload.
- `Unload.FarSpawnUnloadsInPhase7`: `spawnIn` 2 km from the player: the root and its child are unloaded in phase 7 of
  the spawning step (`onSpawn` ran once); when the player arrives the group reloads with `onCatchUp`.
- `Unload.CandidatesNotFullScan`: 10^4 roots inside the kept set and a player standing still: the pass examines
  only changed roots (a statistic `unloadCandidates`), and a root whose `Player` is removed is unloaded in that step;
  after a load the full pass runs once.
- `Unload.LoadEqualsPlain`: a player teleported 1 km between steps right before a save: the plain world and the
  loaded world unload and reload the same groups in the same phase 7 (equal `save()` after the step).
- `Unload.DeterministicHash`: two runs with unloads and reloads give equal per-step hashes; the hash changes when a
  stored group differs.
- `Unload.BlobRoundTrip`: `save()` while groups and an unloaded space are stored contains `EGRP`; a fresh world loads it
  and saves equal bytes; `diffSaves` reports a changed group (`stored_group`); a directory world and its blob hash
  equal.
- `Unload.ReplicationDespawn` (loopback host and client): the client receives `Despawn`, the entity is gone without
  `onReplicatedDestroy`; the reload arrives as a new spawn; `compareReplicas` passes; a space unload sends no `Despawn`
  for the space-state entity, the client keeps it, and the reload's bound spawn updates it.
- `SpaceUnload.StateEntityStoredAndRestored`: two spaces with `Weather`; the player transitions into space 1: space 0
  unloads (`spaceLoaded(0)` false, `spaceStateEntity(0)` null, `worldTimeSystem` runs), `set_weather` on space 0 is an
  error; on return the `Weather` values are restored; the world-state entity stays.

`tests/test_engine_streaming_scenarios.cpp` (S24's binary; S25's three scenarios and the directory round trip of the
save/load variant add at most 0.4 s, measured in step 11; `registerUnloadScenarios`, every variant incl. host + client, `strictContinuation`;
the setup of S24's streaming scenarios with `WorldConfig::streaming`):

- `chunk_unload_reload` (the acceptance): a player and a `Grower` group (root, child, an in-group reference, an outside
  reference, a `PersistentId`) 100 m away; the player is moved 1 km away with `requestArrival` (B5; it completes in
  phase 1 after the island is resident), so the group's chunk is no longer kept: the group is stored (`storedGroups` 1), `resolve` empty, on
  clients the entity is gone; the player returns: the group is resident again with remapped references, the outside
  reference null, `grown` advanced by the elapsed time, `resolve` gives the new root. In the save/load variant the swap
  falls into the unloaded phase (the blob and directory round trips carry the stored group).
- `far_spawn_reload`: a `Grower` spawned 2 km from the player is unloaded in its spawning step and reloads when the player
  arrives (`requestArrival` next to it).
- `space_unload_reload`: the S21 `SpaceLink` setup: the player enters the dungeon, space 0 unloads with its entities
  and state entity, the player returns and finds them with caught-up `grown`.

`tests/test_engine_world_api.cpp`: `save_status` (without and with a directory), `commit_save` (the commit number, a
deferred commit, the errors on a client and without a directory) through `handleMessage`.

`tests/test_engine_app.cpp`: `--save` parsing and its refusals with `--scenarios` / `--benchmark`; a headless app with
`--save <temp dir>` and `EngineAppSetup::saveSync = false` (no fsync on the CI disk; the protocol is covered by the
memory file system): an edit through `edit_terrain`, `quit` commits; a second app on the directory has the edit.

`tests/test_engine_save.cpp`: the committed blobs `save_v1.bin`, `save_s23a.bin`, `save_s23b.bin` still load and save
equal (no `EGRP` written for them); the engine schema (`schema.json`) is regenerated only if `Despawn` or another
engine type changes it.

`tests/test_engine_save_benchmark.cpp` (new, non-gating, prints): the freeze with 10^5 resident entities and 50 edited
1 m chunks, unload and reload of 100-entity chunks, `save()` of a world with 1000 stored groups, open of 1000 edited
chunks.

## Done when

- **Unload/reload scenario**: `chunk_unload_reload` passes in every variant (plain, repeat, save/load with the directory
  round trip, host + client with delay 0, 2 and asynchronous stepping) (CI).
- **Kill at any point**: `SaveCrash.KillDuringCommit` and `SaveCrash.UnloadBetweenCommits` pass for every operation
  and image kind; the other `SaveCrash.*` cases pass (CI).
- The other tests above (CI); every existing engine test, scenario, golden save and hash unchanged for worlds without
  unloading (CI); the measurements of Scope 15 reported, S24's streaming benchmark (full form) with unloading within
  §17.10 (local); `game_world --save <dir>`: a terrain edit survives a restart (local, headless); `game_flight
  --scenarios` passes in every variant and `game_flight --demo` runs with unloading (the only streaming game; its
  driver and aircraft are separate roots linked by `Entity` fields, `samples/games/common/Driving.h:52-68`: both stay in
  the player's kept chunks, and a scenario checks the link after flying far from the start) (local);
  `docs/engine_design.md` and `AGENTS.md` updated.

## Verification

- The CI recipe of AGENTS.md; `ctest -R "test_engine_(save|save_directory|save_crash|unload|scenarios|streaming|
  streaming_scenarios|world_api|app|content|replication|spaces)" --output-on-failure`.
- `./build/tests/test_engine_save_directory`, `test_engine_save_crash` and `test_engine_unload` with
  `--gtest_print_time=1` within the AGENTS.md budget.
- The scenario binaries' times before and after (the extra directory round trip per save/load variant: at most 0.3 s
  for `test_engine_scenarios`, included in the 0.4 s of `test_engine_streaming_scenarios`). With S26's 0.2 s
  `test_engine_scenarios` goes from 1.07 s to about 1.6 s: AGENTS.md records it next to its budget line. The round trip runs for the
  engine's scenario lists; the games' runners keep the blob round trip (`ScenarioRunner` option, off for the games).
- `./build/samples/games/game_flight --scenarios` (local).
- Local, release build of AGENTS.md's benchmark recipe: `test_engine_save_benchmark`, `SAIGA_STREAMING_BENCHMARK=full
  test_engine_streaming_benchmark` with unloading; `game_world --headless --save /tmp/s25_world` with an `edit_terrain`
  over MCP, `quit`, restart, the edit is present (screenshot through `SAIGA_SCREENSHOT`, headless).

## Files

New:

- `src/saiga/engine/save/{SaveFileSystem,MemorySaveFileSystem,RegionFile,ChunkRecordCodec,ChunkStore,SaveDirectory,
  SaveIoQueue,EntityGroup,GroupLoader}.{h,cpp}`, `SaveDirectoryFormat.h`, `Systems.h`, `ChunkEntitySystem.cpp`,
  `SpaceLoading.cpp`.
- `tests/test_engine_save_directory.cpp`, `tests/test_engine_save_crash.cpp`, `tests/test_engine_unload.cpp`,
  `tests/test_engine_save_benchmark.cpp`,
  `tests/save_crash_harness.h`, `data/tests/engine/save_dir_s25/`.

Changed:

- `engine/save/{SaveFormat.h,WorldSave.cpp,WorldLoad.cpp,SaveDiff.cpp,SaveReader.*}` (`EGRP`, `CIDX`, world-state form),
  `engine/ecs/ComponentPool.h` (`appendRemapped`), `engine/world/{World,StructuralLog,SystemList}.*` (store, unload,
  reload, space loading, auto commits, `spaceLoaded`), `engine/terrain/TerrainEditStore.*` (shared blocks, clone in the
  reserve step, freeze), `engine/content/ChunkRecords.*` (auto records), `engine/net/{EngineMessages,NetIds,
  MessageList,Replication}.*` (`Despawn`), `engine/lighting/EnvironmentWrites.cpp`, `engine/app/{LightingTools,
  EngineApp,EngineMain,EngineAppDebugUi}.*`, `EngineAppSetup.h`, `engine/api/WorldApi.*`, `engine/tools/{WorldTools,
  ScenarioWorld,ScenarioRunner}.*`.
- `tests/CMakeLists.txt` (four targets), `tests/test_engine_{streaming_scenarios,streaming,streaming_benchmark,world_api,app,
  save}.cpp` (S24's tests adapted per B15).
- `docs/engine_design.md`, `AGENTS.md` (layout, test list and count, the committed directory's update recipe), this
  file.

## Risks and open questions

Each with the default this plan uses.

1. **Modification state in memory (G2)** contradicts §15.1's sentence. Default: keep it in memory, amend §15.1, report
   `TerrainEditStore::memoryBytes()` in `save_status`; revisit with S27.
2. **Commit interval clock**: world time (deterministic, pauses with the game) or wall time. Default: world time, 300
   s (`commitIntervalSeconds`); the app uses the same.
3. **Retiring generator versions deferred (G17)** although §22's S25 row implies §15.3's "planned S25". Default: defer,
   amend §15.3 / §22 to name S27 (stored bases for far content) or the first stage that needs it.
4. **Reads behind a long commit**: reloads queue behind a running commit on the single IO lane. Default: accept
   (prefetch hides it; reported as the reload latency in the step benchmark); split reads into their own lane only if
   stalls show up.
5. **`save()` of a large directory world reads every stored group** (blobs are self-contained). Default: accept; API
   snapshots are a development tool.
6. **Client-owned groups never unload (G16)**. Default: yes; S33's sessions may revisit for disconnected owners.
7. **Space-state entity on unloaded spaces**: `Weather` does not drift while its space is unloaded (no catch-up for
   state entities). Default: accept (policy, R-MECH-1).
8. **Region size and zstd level**: 16 x 16 chunks, level 3. Default as stated; measured in Scope 15.
9. **Kept commits**: 2, a constant (no option).
10. **Directory lock**: OS advisory locks (released when the process dies, so a killed game reopens its directory).
    Default as stated.
11. **Games and `--save`**: games keep entity handles from `init`; after an open that replaces the world they must
    re-find them as after `load_snapshot` (load listeners). Default: `--save` is available to every engine game; only
    `game_world`'s headless check is part of this stage.
12. **§12.4's open question** ("keeping them is S25's residency question": the GPU terrain of a space the camera
    left): answered no; the visual set stays S20's (re-streamed on return); §12.4 is amended to close the question.

## Design-doc amendments

1. §9.6: as implemented (Scopes 6-9): the group form, `Entity{local + 1, 0}`, the exclusions (players, client-owned
   groups, held arrivals), the one unload pass in phase 7 instead of the phase-6 pending list (G19), reload order and
   hook rules (passes, `onCatchUp` like `onSpawn`, elapsed from steps), failed groups kept, `Despawn` (not for bound
   NetIds).
2. §7.6: `StructuralLog::Record::unloaded`. §7.8: a space-state entity of an unloaded space is stored in the space
   record; `spaceStateEntity` is `Entity{}`, `spaceLoaded`.
3. §12.1: chunk records for every chunk that became resident in a non-legacy space (no `onRecordChanged`); the
   modification state stays in memory (G2).
4. §12.3: entity unloading only in streaming worlds (G18) and beyond the kept distance of every centre; synchronous loads: none for
   the modification state (it stays in memory); stored groups are the residency part
   `PartStored` and reload when the chunk is ready; the load's residency rebuild reloads groups.
5. §12.4: space unloading as implemented (Scope 10: a space without streaming centres); the GPU-terrain residency
   question closed (open question 12).
6. §13.1 / §13.4: `Despawn` exists from S25 (unloading), S32 adds interest; S26 renumbers it with the other messages.
7. §15.1: the directory layout, header, world-state file (world-state form, `CIDX`), region files, records, commit
   protocol (fsync order, commit point, retention, compaction, cleanup at open), the memory store, `EGRP`, blob
   self-containment, G2's sentence replacing "unloaded chunks live on disk, not in memory".
8. §15.2: schema numbers per entity group, header tables append-only; record sections and versions.
9. §15.3: every generated chunk (as implemented); retiring a version moved to the stage named in open question 3.
10. §16.2: `save_status`, `commit_save`; `world_status`'s save block. §8.3 / §18.4: `--save`, the quit commit.
11. §17.2 / §17.4: the save/load variant's directory round trip; the hash part of stored groups and unloaded spaces.
12. §17.10: the freeze budget and the reported measurements.
13. §22: S25's row (retiring moved), S27's row if it takes retiring.

## Deviations during the implementation

Recorded per R-DOC-2; the plan's intent is kept in each case.

- **Reconcile (step 1)**: the S24 names exist as planned. The S24 tests adapted (B15): `Residency.ResidentVersionChange`
  (the resident chunk now has an auto record at version 2: the test rewrites its version through `ChunkRecords::replace`
  instead of `injectForTests`), `Simulated.MatchesBruteForce` (the edits skip unloaded drifters; it asserts that groups
  were unloaded), `Simulated.AlwaysSimulateWhileRequestedAndResident` (the player steps back 100 m instead of 250 m:
  S24's position was beyond the kept distance, so the ticker now unloads), `Simulated.PlayersAndStateEntities` (the
  spaces without a centre are unloaded), `Arrival.NonPlayerArrivalUsesS21Rule` (the NPC unloads in the arriving step's
  phase 7), `StreamingNet.ClientStreamsAroundItsOwnPlayer` (the frozen "far" entity lies in the requested band; an
  entity 20 km away unloads in its first step, before it is ever sent) and the scenario `far_entities_freeze` (by
  `PersistentId`: in the delayed host + client variants the host extrapolates an owner-side teleport for up to 6 steps,
  S16, and the transient pose unloads the tickers, which reload when the player arrives). The other B15 cases pass
  unchanged (their far entities lie inside the kept distance).
- **The kept coverage** (`onKeptChanged`) is the difference of the kept chunk set that `requestPart` computes anyway
  (every chunk within the kept distance of a centre, also never-requested ones), not per-chunk counts.
- **Parts**: `ResidencyPartInfo {bit, wanted, build, request(chunk, priority, done), drop}`; the requests of added parts
  are counted separately (`partsInFlight`: they must not throttle the base jobs). `PartStored` registers no `drop`: the
  unloading's `onUnload` listener applies the bytes rule to every leaving chunk.
- **The unload pass**: candidates as planned; after a load the full pass runs, and the phase-6 changed list (every
  transform is marked changed by the load) names every root once more in the next step. `farUnloads` counts roots
  whose chunk has residency state `None` at the pass. The per-space root count is not kept: a scan runs only while a
  loaded space has no centre (transitional: its roots unload in that step). `droppedReferences` is logged per unloaded
  root (not one line per step). Roots above 65536 entities are re-examined every pass (logged once).
- **Group form**: per value the slot `local index + 1` (the handles' numbering, so the staging core reads it unchanged)
  and a block count before the blocks. `EntityFieldVisitor` maps the `Entity` fields of the live value in place and
  restores them (no copy: components need not be copyable); `BinaryWriter` is unchanged.
- **Staging core**: `StagingCore` (`save/StagingCore.h`) holds the staged values and the checks; `WorldLoader` and
  `GroupLoader` derive from it (the methods kept in `WorldLoad.cpp`). A directory open passes `DirectoryLoad` (the
  terrain and records assembled from the records, the staged store) to the unchanged loader.
- **Files**: the World's members are in `save/ChunkEntitySystem.cpp` (unload, reload), `save/SpaceLoading.cpp` (space
  unload / load), `save/WorldSaveDirectory.cpp` (open, commit, status); the bookkeeping class is
  `save/EntityUnloading.{h,cpp}`; the options and results in `save/SaveTypes.h`. The crash harness is in
  `tests/test_engine_save_crash.cpp` (no `save_crash_harness.h`); the shared worlds in `tests/unload_test_setup.h`.
- **Formats**: a framed part of the world-state file carries its compressed length besides the uncompressed one; the
  index's region files are keyed by (space, rx, rz, generation) in memory (`RecordLocation::file`). The directory sync
  of the parent at the directory's creation is not done (the file system interface has no parent; the lock creates the
  directory).
- **Dirty marking**: a no-op without a directory; `onTerrainChanged` / `onRecordChanged` do not mark during a load (a
  blob load marks every chunk with state explicitly, an open starts clean). Groups get their store-wide ids at the
  open too (a completion maps frozen groups by id).
- **Unreadable old records**: the groups the new record cannot carry are removed from the entry at the completion (they
  are lost with that commit, G14; the backup commit still names the old record).
- **save()** does not wait for a running IO task (appends never change a committed record); it reads each chunk record
  once for all of its disk-only groups (`SaveDirectory::readRecordGroups`).
- **`moveToSpace` into an unloaded space** loads it at once between steps and at a flush, as a command during a step.
  `get_lighting` reads the extract (the defaults for an unloaded space) instead of returning an error.
- **Tests**: `test_engine_unload` and the unload scenarios use flat spaces (`unloadTestSetup`, S24's scenario radii)
  instead of S24's content setup (no generation cost); `space_unload_reload` enters through a `SpaceLink` and returns by
  `requestArrival`; `ScenarioWorld::storedGroups` / `entitiesNamed` are not added (the tests read `chunkStoreView()` and
  `resolve`). Not covered by a test: the asynchronous held `PartStored` read of `Unload.FollowsKeptSet` and the held read
  / other-pins load of `SaveDirectory.LoadWhileCommitInFlight` (the commit case is tested); the committed directory has
  no tombstone (the flat setup has no slots: RREC is covered by `EveryGeneratedChunkHasRecord` and the codec cases);
  `SaveCrash.KillDuringCommit` has no tombstone; `SaveCrash.HeaderRewrite` interns a foreign table to force the header
  rewrite. `Unload.DeterministicHash` compares run hashes; the open death tests of `LockedAndRefusals` are not written.
- **App**: `EngineApp::finishSave` (run's end or the destructor) waits for an in-flight commit first (a commit requested
  meanwhile would only be deferred). The Engine tab line is below the Streaming section.
- **Performance fix**: `World::collectSubtree` caches the sorted (parent, child) links and verifies them by the child
  counts at every use (rebuilt when stale): the full unload of 1000 groups of 100 entities took 7.2 s with the
  per-call pool scan, 0.1 s with the cache.
- **Flight**: the scenario `driver_link_after_far_flight` (the driver and the plane stay linked and resident after 1 km).

## Results

Measured on the development machine (24 threads; tests in RelWithDebInfo on lavapipe, alone; benchmarks in the
release bench build without asserts unless named).

- **Tests**: `test_engine_unload` 19 tests in 0.03 s, `test_engine_save_directory` 17 tests in 0.30 s,
  `test_engine_save_crash` 7 cases in 0.52 s (every operation of each commit in four image kinds, deduplicated),
  `test_engine_save_benchmark` 0.28 s (CI form), `test_engine_streaming_scenarios` 48 cases (S24's 30 plus the three
  unload scenarios in six variants) in 0.35 s (before: 0.33 s), `test_engine_scenarios` 1.08 s (before: 1.07 s; the
  directory round trip of the save/load variant), `test_engine_streaming` 1.21 s, `test_engine_save` 0.45 s,
  `test_engine_world_api` 2.0 s, `test_engine_app` 7.9 s (the S25 tests 4 ms; the binary varies 7-8 s on this machine).
  The whole `ctest --parallel 8` on lavapipe: 95 of 95 passed. `game_flight --scenarios`: 36 passed (incl. the new
  `driver_link_after_far_flight` in six variants).
- **Freeze** (`SAIGA_SAVE_BENCHMARK=full`, 10^5 resident entities, 50 dirty edited 1 m chunks): 9.9-16 ms (release;
  13-26 ms RelWithDebInfo), nearly all of it the world-state serialization (a plain `save()` of that world takes 25 ms
  with `TERR`); the copy-on-write references and the dense deltas moved to the pool cost nothing measurable (before the
  delta hash moved to the pool: +12 ms). At the 10 ms budget: the trigger for an incremental world-state file stands.
  The commit's write on the pool: 38-41 ms for 2.3-2.5 MB (51 records, zstd 3).
- **Unload and reload** (1000 groups of 100 entities in one step): unload 97 ms (0.1 ms per 100-entity group, 1 us per
  entity), reload with the record reads, decode, creation and hooks 84 ms (0.08 ms per group); both within the 1 ms
  target per chunk of 100 entities. (Before the `collectSubtree` link cache: unload 7.2 s, reload 1.3 s.)
- **save() of a directory world with 1000 stored groups on disk**: 17 ms (8 record reads, 11 MB blob); before reading
  each record once per chunk: 556 ms (RelWithDebInfo).
- **Open of 1000 edited chunks**: 215 ms, 1000 records read (the first commit of them: 79 ms, 0.8 MB).
- **S24's streaming benchmark with unloading** (`SAIGA_STREAMING_BENCHMARK=full`, release, 10^5 entities, player at
  30 m/s, 3600 steps paced at 60 Hz): **0 streaming stalls**; step median 3.63 ms, p95 4.95 ms, max 25.7 ms (S24 without
  unloading: 3.55 / 6.51 / 10.4 ms); 987 unloads/s and 494 reloads/s (the entities behind the player unload, the ones
  ahead reload), 70407 resident entities at the end; the `chunk_entities` system median 0.29 ms, p95 0.47 ms, max 12 ms
  (a full pass after the start's load). The median is within §17.10's 4 ms; the p95 too (S24's was not).
- **game_world --headless --save /tmp/...**: an `edit_terrain` over MCP, `quit` (commit 2 written), a restart opened
  commit 2 (1 record read) with the edited height (terrain_height 45.99 m instead of 40.99 m). No screenshot taken.
- **Not done**: the release `game_world --benchmark flyover|forest` (it opens a window); `game_flight --demo` was not run
  (no visible windows: the scenarios cover the streaming game, incl. the far flight); the MSVC build is not compiled
  here (the Windows file API branch of `SaveFileSystem.cpp` is written to the documented calls).
