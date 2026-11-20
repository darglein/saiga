# S11: Save, data and migration

Status: done.

Sections follow the stage plan template of S05.

## Goal

A world can be saved to a byte blob and loaded back into an equivalent world, also by a later build whose components,
classes and events changed (schema tables, migrations, removed types; docs/engine_design.md §15.1, §15.2). Game
objects come back through their class factory (`configure`, `onLoad`); the `Persistent` counter and the world-state
entities are part of the save. Saves can be compared field by field (`diffSaves`, `World::diff`, `dumpJson`). Games
load tunable data from JSON files (§14, R-DATA-1). Agents get the same through the world API: in-memory snapshots to
save, restore and compare, and reading and reloading the data files. S12 (the save/load scenario variant) and S25 (the
save directory) build on it.

## Depends on

S10 (game objects, `Behaviour`, `Persistent`, world-state entities, `WorldApi`, `VisitJson.h`), S08 (structural log,
events, hash), S07 (visit, binary visitors, field lists).

## Rules that apply

R-SAVE-1 (version + migration for changed fields), R-DATA-1 (data file errors are fatal and name file and key),
R-STATE-1/2 (game objects rebuild `cache` in `onLoad`; entities keep their handles), R-REG-1/2 (registration lists;
strings only at the save, data and MCP boundaries), R-DET-1, R-FILE-1, R-MCP-1, R-DOC-1/2, R-TEST-1, R-TEST-2 (changes
under `engine/ecs/`: the ECS benchmark before and after).

## Scope

### 1. Schema tables (`engine/save/SchemaTable.h/.cpp`)

- A schema table lists every saved type by name: components (`Saved::Yes`, incl. `Transform`), the nested visit structs
  reachable from them and from events (they already have `name` and `version`), events, and the classes as (ID, name)
  pairs (class IDs are sparse: engine 0..63, game from 64). Per struct type: kind, name, ID, version and the field list
  (name, type string, enum value names).
- Built from the World's registrations (`SchemaTable::fromWorld`). Nested structs are collected by a recursive field
  walk at registration (new `PoolBase::describeTypes` and `EventQueueBase::describeTypes` virtuals; `describeFields`
  gives only the top level).
- Comparison per type: **same** (equal version and identical field lists, nested structs compared by name recursively:
  the fast path), **compatible** (same version, fields added or removed, enum values added or reordered: the slow path
  by name) or **incompatible** (a field whose type string changed without a version increment: a load error).

### 2. Save format (`engine/save/SaveFormat.h`, `engine/save/WorldSave.cpp`)

`std::vector<uint8_t> World::save() const` (between steps, not inside a game hook; asserts otherwise). Little-endian,
uncompressed (S25 compresses the files of the save directory). Sections with a tag and a byte length; an unknown tag is
a load error.

| Section | Content |
|---|---|
| header | magic `SGWS`, format version 1, engine version, steps per second, step index, time |
| schema | the schema table (section 1) |
| entities | the slot table: per slot its generation, state and flags (world state, space state); the free list in order |
| components | per saved component type, in schema order: the entity indices and the binary `visit` form (`VisitBinary.h`), in the pool's dense order |
| events | the pending events per event type, in emit order |
| actions | the queued player actions in queue order and the action sequence counter |
| state | the RNG state, the `Persistent` counter |

- Entity indices and generations are kept, so `Entity` and `PersistentId` fields stay valid without remapping (§15.1).
- Pools are restored in their saved dense order, so views over saved components iterate in the same order after a load.
- Not saved, reset or rebuilt by the load: `Saved::No` components, game objects (recreated from `Behaviour.classId`),
  the spatial grid, the hierarchy order and child counts, pending child teleports, the `Persistent` index and the
  pending `Persistent` IDs, the tick evaluation (player cells), the `Transform` changed flags, the readable events of
  the last step, retired game objects. Kept: the profiler, step/destroy listeners and structural log consumers (they
  belong to the observers, not the world). Not saved (app state): paused, time scale, accumulator.
- `save()` is deterministic: the same world gives the same bytes; `save(load(save(w))) == save(w)`.
- Component fields holding a component, event or class ID are not remapped by a load; such references are stored as
  names (documented in §15.2; `Behaviour.classId` is the one field the engine remaps).

### 3. Loading and migration (`engine/save/MigrationList.h/.cpp`, `engine/save/WorldLoad.cpp`)

```cpp
struct LoadResult
{
    bool committed = false;        // false: rejected, the world is unchanged; error says why
    std::string error;
    struct HookError { Entity entity; std::string className, message; };
    std::vector<HookError> hookErrors;   // committed, but these objects threw in configure/onLoad (removed)
    bool ok() const { return committed && hookErrors.empty(); }
};
LoadResult World::tryLoad(const std::vector<uint8_t>& data);   // between steps, not inside a game hook
void World::load(const std::vector<uint8_t>& data);            // tryLoad; stops the program unless ok()
struct FieldMap { json11::Json::object fields; /* get, set, rename, remove helpers */ };
enum class RemovedPolicy { DropEntity, DropComponent, Replace };
template <class T> void MigrationList::add(uint32_t fromVersion, void (*fn)(FieldMap& fields));      // v -> v + 1
void MigrationList::add(const char* typeName, uint32_t fromVersion, void (*fn)(FieldMap& fields));   // removed types
void MigrationList::addRemoved(const char* name, RemovedPolicy policy, const char* replacement = nullptr);
// WorldSetup::migrations (registered after the classes)
```

- **Validate, then commit.** The whole blob is decoded and checked into staged values (per pool a typed buffer) before
  the world changes; any error returns `committed = false` with a message naming the type, field and entity. Checks:
  truncated or malformed data, another format version, a type of a newer version than registered, a version gap
  without a migration, an incompatible field type, an unregistered component, class, event or nested struct without a
  `Removed` entry, a stored enum value whose name no longer exists (a removed name that no value uses is fine), the slot
  table (each slot alive or once in the free list; at most one world-state entity, and one exactly when the saved
  schema has globals; space-state entities match their flags), `Parent` naming a live entity and no cycles, unique
  `PersistentId`s with the counter above the largest, every `Behaviour.classId` registered. A different steps-per-second
  is accepted (the time is kept, see below).
- **Fast path**: a type whose saved schema is the same as the current one is read with `BinaryReader` straight into `T`.
- **Slow path** (compatible or migrated types): the saved bytes are decoded generically by the saved field list into the
  JSON form of `VisitJson.h` (exact: 64-bit beyond 2^53 as strings, non-finite doubles as strings, enums by name), the
  migrations run in version order on the `FieldMap`, fields that no longer exist are dropped, then `fromJson` merges it
  into a default `T` (new fields keep their defaults; non-finite values allowed). Nested structs migrate the same way,
  recursively. `Replace`: the old type's migrations run first (by name), then its fields are merged by name into the
  replacement.
- **IDs by name**: component, event and class IDs are remapped through the schema table; `Behaviour.classId` is
  rewritten to the current class ID. A component now registered `Saved::No` is dropped silently.
- **Removed types**: `DropComponent` drops the component (for a class: the entity stays without `Behaviour`; for a
  nested struct: not allowed, a removed nested struct needs a migration of its parent); `DropEntity` drops the entity
  and its descendants (their slots become free with an incremented generation, appended to the free list in index
  order, as if destroyed: handles to them are stale); `Replace` loads the fields into the replacement type or class.
  Pending events of event types that are no longer registered are dropped with a log line (they live one step).
- **Commit**: every current entity and game object is removed without hooks (world-state entities included; structural
  log records as for a destroy), the unsaved members of section 2 are reset, then the slot table, the components
  (records as for a create), the events, the actions, the RNG, the counters and the time are restored. Then the derived
  state is rebuilt: hierarchy, child counts, `Persistent` index, the state-entity handles, the spatial grid
  (`rebuildSpatialGrid`, queries work before the next step), all `Transform`s flagged changed, every `Behaviour`
  re-evaluated by the next `tickSystem`. Globals or space states registered since the save get a new state entity
  (the first free slot, deterministic) with their defaults; a state entity whose components were all removed is freed.
  The load counter (`World::loadCount()`) increments. The structural log range of the load (`appendedTotal()` before
  and after the commit, hook removals included) is passed to the load listeners, so a consumer can tell the load's
  records from earlier ones.
- **Game objects** in three passes over the loaded `Behaviour`s in index order: create every object from its class
  factory and bind it, then `configure()` on all, then `onLoad()` on all (`onSpawn` is not called). During a load,
  `configure` may make no structural changes and write no saved component, and `onLoad` may only add or remove
  `Saved::No` components of its own entity (asserted; e.g. re-adding a render component), so a load followed by
  `save()` gives the saved bytes. An exception from a hook removes that entity and its descendants without
  hooks and is reported in `hookErrors`; the other objects still load.
- **Load listeners**: `World::addLoadListener(fn(const LoadInfo&)) -> id`, `removeLoadListener(id)`: called after a
  committed load with the log range and the load count (the app's render caches, its local player and the world API
  react to it).
- **Cross-component migrations** (split or merge components) are not in S11: rename, drop, replace and per-type
  migrations cover field changes; a staged whole-world migration needs a game save version and comes with the save
  directory (S25) when a game needs it.
- `time()` becomes `timeBase + (steps - stepBase) * dt`, so a save from a build with another step rate keeps its time.

### 4. Comparing saves (`engine/save/SaveDiff.h/.cpp`)

```cpp
struct SaveDifference { std::string kind; Entity entity; std::string component, field; json11::Json a, b; };
// kinds: field, only_in_a, only_in_b, component_only_in_a, component_only_in_b, rng, step, events, actions, free_list,
// counter
struct SaveDiff { std::vector<SaveDifference> differences; size_t total; json11::Json summary; };
SaveDiff diffSaves(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const DiffFilter& filter = {});
static std::string World::diff(const World& a, const World& b, int maxLines = 20);   // diffSaves(a.save(), b.save())
std::string World::dumpJson(Entity e = {}) const;
```

- `diffSaves` works on blobs (no `World` needed, schema tables of both sides decoded by name, the slow-path decoder of
  section 3); bytes are compared first, the JSON form names the fields of the components that differ. Saved state only.
  `DiffFilter {entity, components, limit, after}` for paging. The summary counts differences per component and the
  entities only on one side (with class and position).
- `World::diff` returns an empty string for equal saved state, else one line per difference (`entity 12:3 Health.hp:
  10 != 7`) and `... and N more`.
- `dumpJson`: `{step, time, entities: [{entity, class?, flags, components: {Name: values}}]}` in index order, through
  `VisitJson.h` (exact numbers instead of the `%.17g` of §15.1).

### 5. Data files (`engine/data/DataStore.h/.cpp`, `engine/data/DataFile.h`, `engine/data/DataRef.h`)

```cpp
template <class T> void DataTypeList::add();   // T: name, version, visit; file <dataDirectory>/<snake_case(name)>.json
// WorldSetup::dataTypes, WorldSetup::dataDirectory
template <class T> const T& World::data(const std::string& name) const;   // fatal on a missing name (every build)
template <class T> bool World::hasData(const std::string& name) const;
template <class T> std::vector<std::string> World::dataNames() const;     // sorted
template <class T> DataRef<T> World::dataRef(const std::string& name) const;   // a stable handle (see below)
template <class T> bool readDataFile(const std::string& path, std::map<std::string, T>& out, std::string& error);
bool readJsonFile(const std::string& path, json11::Json& out, std::string& error);   // the checked reader of both
```

- One JSON file per type: an object of named entries (`{"wolf": {...}, "bear": {...}}`), loaded in the `World`
  constructor. Each entry is merged into a default `T` with `fromJson` (missing keys keep the struct defaults).
- Fatal in every build (not only with `SAIGA_ASSERTS`: a release game must not run on half-read data): a missing file,
  a file that is not a JSON object, an unknown key, a wrong type, a value out of range. The message names the file and
  the key path (`data/tests/engine/creature_stats.json: wolf.speedd: unknown field (fields: speed, health)`). The check
  is an always-on fatal error (`SAIGA_EXIT_ERROR`-style, not `SAIGA_ASSERT`). Duplicate keys: json11 keeps the last
  (documented).
- `data<T>` with an unknown name throws `DataError` (naming type and name): fatal when it escapes to the top (the
  `World` constructor, a step), but a load, `reload_data` or an API call reports it as an error (in `hookErrors` for
  hooks) instead of ending an agent's game.
- `DataRef<T>`: a field type for components (`v("model", model)`), saved and shown in JSON as the entry name,
  resolved to the entry once (load, `fromJson`, `dataRef`); `*ref` is a pointer read that stays valid across
  `reloadData` (entries are updated in place, new ones appended). Systems read tunables through it per step without a
  string lookup (R-REG-2), and a reload tunes running objects at once.
- `readDataFile` and `readJsonFile` are the world-independent readers with the same rules and messages (non-fatal; S13's
  input map uses `readJsonFile`).
- `World::reloadData(std::string& error)`: reads all files again and applies them only if all are valid and no entry
  was removed (else an error and no change). Entries are updated in place.
  Members set in `configure()` keep the old values until the object is configured again (`reload_data` can do that).
- Worlds of the same setup can share the parsed files (`WorldSetup::sharedData`, the scenario runner builds one
  per run instead of re-reading the files for every world).
- Data are not saved (they come from the files of the build that loads).

### 6. World API and MCP tools (`engine/api/WorldApi`, `engine/tools/WorldTools`)

| Operation / tool | Arguments -> result | Reads / changes |
|---|---|---|
| `save_snapshot` | `name`, `overwrite?` (false) -> name, bytes, step, hash | keeps `world.save()` in the API's memory (at most 16 snapshots and 256 MB; refused beyond); allowed read-only |
| `load_snapshot` | `name`, `paused?` (keep) -> step, hash, entities, paused, world_generation, hook_errors | replaces the live world (game objects get `configure` + `onLoad`); edit |
| `export_snapshot`, `import_snapshot` | `name`, `file` (a plain file name in the directory set by `WorldApi::setSnapshotDirectory`, no paths; refused when none is set) | writes / reads a snapshot file, so a save survives a rebuild (testing migrations); edits (they touch the disk) |
| `list_snapshots` | -> [{name, bytes, step, hash}] | reads |
| `delete_snapshot` | `name` | API state only; allowed read-only |
| `diff_snapshot` | `name`, `against?` (another snapshot; default the live world), `entity?`, `components?`, `limit?` (20), `after?` -> summary, differences [{kind, entity, class?, component?, field?, a, b}], next? | reads |
| `get_data` | `type`, `name?` -> one entry, or the entry names | reads the data files |
| `reload_data` | `reconfigure?` (true) -> types, entries, hook_errors, error? | re-reads the data files (nothing changes on an error; `DataRef`s see the new values at once); with `reconfigure` the world is saved and reloaded (`tryLoad`) so every object runs `configure()` again; edit |

- `describe_world` also lists the data types (file, fields, entry names) and the save format version; `world_status`
  reports `world_generation` (the load count).
- **Loads in the change journal**: the API drains its consumer up to the load's log range, skips the range and writes
  one entry `{kind: "world_loaded", step, from_step, world_generation}`; move tracking restarts; a cursor from before the load stays
  accepted and its answer carries `reloaded: true` (the agent re-lists). Entity handles obtained after the snapshot was
  taken may name other entities after the load (documented in the tool description). A pending `measure_performance`
  fails with "the world was loaded".
- Snapshots are for agents: save, try something, `diff_snapshot`, roll back with `load_snapshot`; or compare two
  experiments with `against`. They live as long as the API (export to keep one); the save directory is S25.
- **Rebinding**: `WorldApi::rebind(World&)` moves the API to another world (S12's save/load variant, S14's
  `run_scenario`) and keeps its epoch, snapshots and cursors, writing a `world_loaded` entry like a load. The MCP tools
  use the API, so they follow.

## Out of scope

Save directory, chunk store, commits and compression (S25), world id and generator versions (S23, S25), character
files (S33), message and piece-type schemas (S15, S30), an editor for data files, in-memory data overrides by agents
(`set_data`, a candidate for S14), saving `Saved::No` components (by definition), saving pending commands (there are
none between steps).

## Design notes

- The save is per type, not per entity: one schema decision per type and a tight loop per pool, and the dense order is
  kept for free. `dumpJson` gives the per-entity view for humans.
- Validate-then-commit makes `tryLoad` safe for the API: a bad snapshot or an old save never leaves half a world. After
  the commit only game hooks can fail (reported per entity) and `bad_alloc`.
- Comparing blobs (`diffSaves`) instead of `World`s: the API, the S12 runner and tests can compare snapshots without
  building a second world (which would need the `WorldSetup` and rerun hooks).
- The slow path through JSON reuses the checked conversions of `VisitJson.h` (2^53 boundary, enums by name, ranges)
  instead of a second converter; it runs only for changed types, so it can be slower.
- A loaded world equals the saved one (`save()` bytes and `hash()` equal). It continues identically for the engine's
  systems in the tests, but that is not a guarantee (§8.6: e.g. the unsorted order of `forEachWithin` after a grid
  rebuild, `Saved::No` pools re-added in index order).
- The engine's consumers see a load as removes and creates in the structural log (no special kind); the world API folds
  it into one `world_loaded` entry, so agents are not flooded and the journal does not overflow.
- Data files are fatal in every build (design change of R-DATA-1, which says `SAIGA_ASSERT`); a missing entry name at
  run time is a recoverable `DataError`, so an agent's reload or snapshot cannot end the game.
- `DataRef<T>` is a new visit field type (a design change of §7.2 and §14): the alternative, string lookups per step or
  copies in components, either breaks R-REG-2 or makes live tuning impossible.

## Tests

- `tests/test_engine_save.cpp`:
  - Format stability: a committed golden save (`data/tests/engine/save_v1.bin`, written once by this stage from a fixed
    world) loads in every later build and matches its committed `dumpJson`; a committed schema of the engine types
    (`data/tests/engine/schema.json`) fails the test when a field list changes without a version increment (R-SAVE-1
    in CI).
  - Round trip: entities with engine and game components, a hierarchy, world-state and space-state components, pending
    events, queued actions, `Persistent` IDs and stale handles in fields; equal `save()` bytes and `hash`, free list and
    future entity indices equal, `resolve` works, grid queries work at once.
  - Game objects: a class with a `cache` member, `configure` + `onLoad` called once each in three passes (not
    `onSpawn`), the cache rebuilt, `update` continues; class IDs renumbered between the two builds; a throwing
    `onLoad` (entity removed, `hookErrors`, others loaded); `onLoad` adding a `Saved::No` component; a structural change
    of a saved component in `onLoad` (death test).
  - Migration: a component v1 -> v2 (renamed field, changed meaning); a nested struct in a vector with its own
    migration; an added and a removed field; an added enum value (compatible); a type change without a version
    increment, a version gap without a migration, a save from a newer version, a removed enum value in use (errors).
  - Removed types: `DropComponent`, `DropEntity` (with children: their handles become stale), `Replace` for a component
    (with a migration of the removed type) and a class; an unregistered type without an entry (error naming it); a
    dropped event type; a component now `Saved::No`.
  - Robustness (a small world of a few KB): truncation at every length, flipped bytes (a seeded sample), an unknown
    section, a corrupt slot table, a `Parent` cycle, a duplicate `PersistentId`, an unknown class ID; the world is
    unchanged after each rejection (`save()` bytes equal before and after).
  - State entities: a save without globals into a build with globals, and the reverse; another step rate.
  - Load listeners with the log range (spawn, then load, then a consumer reads both correctly); `loadCount`;
    `WorldApi::rebind`; load from inside a hook and structural changes in `configure` during a load (death tests).
  - Continuation: a world saved, loaded and stepped 100 steps has the same per-step hashes as the original.
  - `diffSaves` (each kind, filter, paging, summary), `World::diff`, `dumpJson`.
- `tests/test_engine_data.cpp` with files in `data/tests/engine/`: entries load, defaults kept, `data<T>`/`hasData`/
  `dataNames`, `configure()` reading data, `DataRef` (save/load by name, JSON, valid across `reloadData`),
  `readDataFile`, `reloadData` (valid, invalid file, a used entry removed: unchanged), `DataError` from a hook during a
  load; death tests (every build): unknown key, wrong type, missing file, not an object, each naming file and key.
- `tests/test_engine_world_api.cpp`: every new operation and tool through `handleMessage` (snapshot limits and
  `overwrite`, unknown names, `load_snapshot` with game objects and `paused`, `hook_errors`, the `world_loaded` journal
  entry with a world larger than the journal, an old cursor after a load, `diff_snapshot` with `against`, filters and
  paging, `export_snapshot`/`import_snapshot` (names with paths refused), `get_data`, `reload_data` with and without
  `reconfigure`, `describe_world`/`world_status` additions, read-only mode).
- `tests/test_engine_ecs_benchmark.cpp`: save, load (fast path) and load with one migrated component for 10^5 entities
  (`Transform` + 3 components, 10^4 game objects), and `diffSaves` of two equal saves; numbers in the results.

## Done when

- The S11 row of §22: a save/load round trip incl. a migration, a removed type and an entity with a game object
  (`configure` + `onLoad` called, cache rebuilt); data file load incl. the unknown-key death test (CI).
- Every new API operation and tool has a `handleMessage` test; all tests pass in CI on GCC, Clang and MSVC.
- `docs/engine_design.md` (§3, §7.2 `DataRef`, §7.9, §7.10 (`tryLoad`, `diff(a, b, maxLines)`, listeners, data), §14,
  §15.1, §15.2, §16.2, §17.4 (compare `save()` bytes), R-DATA-1, the S11 row) and `AGENTS.md` updated.

## Verification

- The CI recipe of AGENTS.md ("Build (Linux)"), `ctest -R engine`, then the full `ctest`.
- AddressSanitizer + UBSan build of the engine tests (malformed blobs).
- The ECS benchmark before and after (R-TEST-2).

## Files

`src/saiga/engine/save/{SchemaTable.h,SchemaTable.cpp,SaveFormat.h,SaveReader.h,SaveReader.cpp,WorldSave.cpp,
WorldLoad.cpp,MigrationList.h,MigrationList.cpp,GenericDecode.h,GenericDecode.cpp,SaveDiff.h,SaveDiff.cpp,LoadResult.h}`,
`src/saiga/engine/data/{DataStore.h,DataStore.cpp,DataFile.h,DataRef.h}`, `src/saiga/engine/ecs/{ComponentPool.h,
Registry.h,Registry.cpp,TypeSchema.h,PagedArray.h,VisitBinary.h}`, `src/saiga/engine/world/{World.h,World.cpp,
EventQueues.h,StructuralLog.h}`, `src/saiga/engine/gameplay/GameObject.h`, `src/saiga/engine/api/WorldApi.{h,cpp}`,
`src/saiga/engine/tools/WorldTools.{h,cpp}`, `tests/test_engine_save.cpp`, `tests/test_engine_data.cpp`,
`tests/test_engine_world_api.cpp`, `tests/test_engine_ecs_benchmark.cpp`, `tests/CMakeLists.txt`,
`data/tests/engine/{creature_stats.json,save_v1.bin,save_v1.json,schema.json}`, `docs/engine_design.md`, `AGENTS.md`,
this file.

## Design changes

- `export_snapshot` and `import_snapshot` are edits (they touch the disk), not reading operations; the directory comes
  from `WorldToolsOptions::snapshotDirectory`.
- Types without saved values need no `Removed` entry (a save lists every registered type; only values need a
  counterpart). Classes need an entry only when a `Behaviour` uses them.
- `DataRef<T>` is a nested visit struct `{entry}` (no new field type in the visitors): its JSON form is
  `{"entry": name}`. It caches the resolved entry per data store (`world.data(ref)`); a reload never removes entries,
  so the cached pointers stay valid.
- `diffSaves` finds value boundaries with a skipping decoder and decodes JSON only for values whose bytes differ
  (10^5 entities: 688 ms with full decoding, 105 ms now).
- The fast path compares the saved and registered schema recursively (nested structs); a nested struct whose
  registered counterpart is gone is decoded as saved and left to its parent's migration.
- `World::load` keeps the `SAIGA_EXIT_ERROR` contract of §7.10 (fatal unless ok); tools use `tryLoad`.
- §15.1 `dumpJson` uses the exact JSON form of `VisitJson.h` instead of `%.17g`.
- R-DATA-1: fatal in every build (`SAIGA_EXIT_ERROR`), an unknown entry name at run time is a `DataError`.

- Review round 1 (two reviewers: save/load core; API, tools and data): saved schemas are validated (type strings of at
  most 256 characters, container nesting of at most 8, array lengths of at most 2^16, balanced names) and elements that
  take no bytes share a budget of 2^20 per decoder (a crafted nest of zero-byte arrays ran for minutes, a deep one
  overflowed the stack); duplicate checks of the schema and the value blocks are linear; an even RNG increment is
  rejected before the commit; a state entity that the load would free must have no children, `PersistentId` or game
  object, and a game object on a state entity is rejected; `time()` stays `steps * dt` after a load when the step rate
  and the saved time agree (it drifted in the last bit); R-SAVE-1 is checked for nested structs too; factory exceptions
  of any type are hook errors; a throwing load listener no longer hides the load from the others
  (`LoadResult::listenerErrors`); `diffSaves` sizes its tables by the indices present. API: the load's records
  `[logBegin, logEnd)` are folded, records of other load listeners after them are journaled; `rebind` resets the
  journal state; `world_generation` is the API's own counter (it only grows, also across `rebind`);
  `export_snapshot` takes `overwrite` and writes atomically (temporary file, rename), refuses Windows device names;
  `import_snapshot` checks the size before reading, its hash is `null`; the `diff_snapshot` entity filter compares the
  generation. Data: `DataRef` caches by store ID and entry name (a changed entry resolves again; `visit` no longer drops
  the cache); an entry that is not an object names the entry; reload errors name the full path. Not changed: a missing
  `DataRef` entry fails when it is read (`DataError`), not at load time; `rebuildHierarchyState` is linear in the depth
  of the chains; a shared data store is reloaded for all worlds that share it (scenario runs do not reload).

- Review round 2 (robustness, regressions, agent experience): the zero-byte budget also covers struct fields and
  whole values (nested empty structs multiplied the work, 10^depth field visits from a few KB); migrations that throw
  are load errors; the schema has a lookup index; the header's time and step rate must be finite, at most 2^24 slots;
  a reused slot is reported as two one-sided entities with class and position (the entity filter hid it); a failed
  object is removed at once with its descendants, which are listed in `LoadResult::removed` (`load_snapshot`:
  `removed`) and get no further hooks; refused structural changes in load hooks are refused in every build; queued
  actions of players that the load dropped are dropped; `World::load` is also fatal on listener errors; an imported
  snapshot learns its hash when it is loaded; a typo in the `diff_snapshot` components filter is an error;
  `reload_data` reports listener errors; records made before a load keep the step before it; exports write to a
  temporary name of their own; the `DataRef` cache is atomic (a first resolution in `parallelEach`).

- Review round 3 (portability, a build without asserts, docs): the save reader's action size was wrong (49 instead of
  45 bytes: saves with 3 or more queued actions were rejected); structural changes refused in load hooks throw (a hook
  error) also without asserts (a spawn in `configure` crashed); a load inside a step, a game hook or a view returns an
  error in every build (a "quick load" in `update` destroyed the running object); `DataTypeList` and `DataStore`
  delete their copies (MSVC instantiates the copy operations of exported classes); the tests remove their files and
  use unique temporary directories. Not changed: json11 parses numbers with the C locale functions (saiga sets no
  locale; a game that sets `LC_NUMERIC` must keep it "C"); a `save()` whose writer fails (more than 2^16 empty structs
  in a vector) asserts.

- CodeRabbit (PR #54): new slots after a load start at generation 0 (the paged array handed back an old one);
  `clearForLoad` clears the membership bits before the components; the values of dropped components and removed event
  types are checked before the commit (each entity once); queued actions stay as in the live world (also for a dropped player); the
  defaults of state components are built before the commit and moved in (no construction after the world was
  cleared); `load_snapshot` notices a load listener that rebinds the API, and an imported snapshot learns only its own
  hash (a revision per stored snapshot), taken before the load listeners ran (`LoadResult::hash`); snapshot saves
  inside a game hook are refused (`World::canSave`); a load is refused while another load's
  listeners run, and a listener removed during the dispatch is not called; `save()` throws instead of returning a
  block the reader rejects (an enum value without a name, more than 2^16 empty pending events); load-hook refusals
  are catchable in every build (no assert); array lengths up to 2^31 in saved schemas; the diff stores values sparsely,
  compares free-list generations, `Behaviour` classes by name and equal bytes only with the same schema (nested structs
  too); export names cannot start with '.' and the temporary file is created exclusively (never a file of an
  earlier process); a registered `std::array` has at most 2^31 - 1 elements (a `static_assert`, the schema's limit).

Updated: §3, R-DATA-1, §7.2, §7.9, §7.10, §14, §15 (header, §15.1, §15.2), §16.2, §17.4, the S11 row; AGENTS.md.

## Results

- `ctest`: 61 tests (new `test_engine_save` 32 tests, `test_engine_data` 8 tests; `test_engine_world_api` 52 tests).
- ECS benchmark (RelWithDebInfo, 10^5 entities with 3 components, 10^4 game objects, 15.5 MB): save 40 ms, load
  73 ms, load with one migrated component 117 ms, `diffSaves` with one difference 105 ms. The other ECS numbers are
  unchanged (create 14.4 ms, idle step 0.52 ms, moving step 2.27 ms, 10^4 game objects 0.095 ms per step).
