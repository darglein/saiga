# S22: Terraforming

Status: implemented (2026-11-21; steps 1-12 on the branch `stage-s22`, stacked on S20 and S21).

Sections follow the stage plan template of S05. The S20, S21 and S23a/b/c plans were revised in parallel with this one
under the shared cross-plan decisions of review rounds 1 to 3 (D1-D14, E1-E14 and F1-F4; later letters override earlier
ones where they differ; quoted where they bind S22); "S20 interfaces used" lists the S20 names this plan relies on.
Before step 1 the plan is checked against the merged S20 and S21 code (a plan update in the S22 branch, R-DOC-2).

## Goal

The host changes the height-map terrain at run time with brushes (`World::editTerrain`, §12.2). The edit computes an
integer result synchronously in the requesting step (between steps: at once; during a step: at the phase flush), so
`terrainHeight`, the kinematic mover and every later system of the step see the new surface. The result is chunk
modification state (per chunk with state: a sequence number, §12.1, and the per-sample difference from the generated
base, stored once per space sample; the base itself is never stored, it is the pure base function), saved in the
optional `TERR` section, covered by `world.hash()`, and replicated to every client as integer difference rectangles
(never the brush, §13.2); clients keep every edited sample in memory. The base terrain data of S20 (`TerrainStore`,
tiles and pyramid) is never mutated: the deltas are merged when data passes S20's single send point, and sent tiles and
the pyramid are patched through S20's `patchTerrainTile` / `patchTerrainPyramid` within a per-frame budget, grass masks
of edited chunks recomputed on the streaming pool (visual only; S22 is only the caller of S20's patch calls, E2). A
terraforming benchmark measures the integer result, its encoding and the visual patching.

## Depends on

S20 (`SpaceTemplate` and `hasTerrain()`, `kHeightUnit`, `kTerrainHeightLimit` and `clampTerrainHeight`, the terrain
surface rule, `TerrainSystem` / `TerrainStore` / `TerrainGeneration` and the height source, `TerrainTileData` /
`TerrainPyramidData` / `GrassMaskData`, the single send point incl. `sendGrassMask`, the sink calls
`patchTerrainTile` / `patchTerrainPyramid` implemented and tested by S20, `TerrainPass`'s probe, the streaming pool,
`EngineApp::setCompleteViewEachFrame`, `World::terrainHeight`, `characterMoveSystem` on the terrain), S16 (roles, the replication pass in phase 8
with its staged bookkeeping, epochs and `Resync`, late joiners, the host + client scenario variant), S15 (raw messages,
fragmentation of reliable messages), S11 (save sections, `tryLoad`'s validate-then-commit, `diffSaves`), S12/S14
(scenarios, world API, MCP tools), S19b (`RenderSink`, `RecordingRenderSink`, `VulkanRenderSink`). S21 is not required:
state is keyed by `ChunkId` / `SpaceId`, so the same code serves S21's spaces; where S21 is merged first, S22 uses its
optional-section mechanism (D7) and appends its IDs after S21's (D10); otherwise S22 adds that mechanism itself in the
form D7 fixes and S21 reuses it.

## Rules that apply

- R-MECH-1: the engine offers brushes; whether, where and how strongly a game terraforms is policy (limits and
  protected areas are decided in the S44 plan, §21). The engine bounds (radius, strength, center, height range) are
  safety bounds of the mechanism, not gameplay limits.
- R-DET-1, §8.6: edits are applied in command order at the phase flush; the brush is evaluated with integer arithmetic
  only, so the repeat variant (also with `varyThreads`) and the committed result hash agree on every compiler.
- R-CONTENT-1 (applies from S23; S22 follows it already for the brush): inputs are bounded first, then quantized with
  exact operations (multiplication by powers of two, `floor`), no libm, no floating point after quantization.
- R-NET-1/R-NET-2: brushes exist only on the host; clients only apply received integer results (they write no
  component; the modification state is not a component). Every terraforming scenario passes the host + client
  variants.
- R-STATE-1/R-SAVE-1: the modification state lives in the `World` (chunk state, §12.1), not in game objects; it is saved
  in its own optional section with a section format version (§15.2, D7). No component changes, so no migration.
- R-SYS-2: an edit during a step is a command; the `TerrainEditedEvent` is a broadcast.
- R-PAR-1: the integer result runs serially on the main thread (no `parallelEach`, no pool jobs). The edit state is
  read lock-free and read-only from `terrainHeight`, which may run inside `parallelEach`; it is mutated only at phase
  flushes, in phase-1 message handlers and between steps. The only pool work is visual: grass masks of edited chunks
  (on S20's streaming pool, with copies of the merged heights, E3).
- R-REG-1/2, R-FILE-1: messages in `registerEngineMessages`; the event in `world/Events.h` (`EngineEvent::TerrainEdited`,
  appended in merge order, D10) and registered in `registerEngineEvents`; one class per file under `engine/terrain/`.
- §5.2 / D2: `saiga_vulkan` never includes engine headers. S22 changes nothing in `saiga_vulkan` except the benchmark
  pose; tests that need the engine path live in the engine-linked test target.
- R-MCP-1: the new world API operations become tools with descriptions and `handleMessage` tests.
- R-TEST-1/R-TEST-2: engine scenarios, unit tests, one engine-path render check and a golden for patched terrain;
  terrain is performance relevant: the terraforming benchmark plus `flyover`/`forest` before and after.
- R-MATH-1/2: the brush API is double (`Vec3 center = Vec3::Zero()`), the state is integer; the GPU data are S20's int32
  heights (E2), so no float appears after quantization.
- R-DOC-1/2: `TerrainBrush` and `editTerrain` as in §12.2; every new signature below is a design-doc amendment listed at
  the end and applied in the stage's change.
- Error policy: role, space and brush violations of `editTerrain` stop the program in every build (`SAIGA_EXIT_ERROR`
  with the message; the world API checks first and returns errors), so their death tests run without `SAIGA_ASSERTS`.

## S20 interfaces used

Names from the S20 plan (revised to D1-D9 and E1-E9). S22 changes the parts marked "S22:".

1. **Sample grid** (`TerrainLayout`): base spacing `sp = SpaceTemplate::baseSpacing` (1 or 2 m), `S = 256 / sp`,
   `N = TerrainLayout::samplesPerSide() = S + 1`; space sample `(i, k)` lies at `(i * sp, k * sp)` metres; chunk
   `(cx, cz)` covers samples `[cx * S, cx * S + S]` (closed: borders are shared); the space covers samples
   `[0, sizeChunks * S]` (D12: content and edits live in chunks `[0, sizeChunks)^2`).
2. **Base heights**: `int32_t TerrainSystem::sample(SpaceId, int64_t sx, int64_t sz) const` (base grid sample: the
   resident level-0 tile in `TerrainStore`, else S20's height source (`TerrainHeightSource`, the wrapped hook, E7) at
   `(sx * sp, sz * sp)` integer metres, clamped with `clampTerrainHeight`, else 0 for a space without terrain,
   `SpaceTemplate::hasTerrain()` false, D1/E6). Every base sample therefore lies within `+-kTerrainHeightLimit` (S20,
   E7). It stays the base: S22 never writes the `TerrainStore` and never caches base values (section 2). S22 relies on
   `sample` staying a cheap point query that is safe to call concurrently from `parallelEach` bodies (it runs inside
   `terrainHeight`); S23a's pipeline source keeps that contract. S22 adds a relaxed atomic counter of the samples
   `sample` reads from the height source instead of a resident level-0 tile (`TerrainSystem::baseHookSamples()`,
   section 3).
3. **Levels and pyramid**: a level-`L` tile (`RegionId{space, L, x, z}`, `L < streamedLevels()`) covers `2^L x 2^L`
   chunks with `N x N` samples at spacing `sp * 2^L`; the pyramid samples the whole space at 32 m. Both are point
   samples of the same function (S20 settled: no filter), so level-`L` sample `(a, b)` of region `(x, z)` is base sample
   `(x * 2^L * S + a * 2^L, z * 2^L * S + b * 2^L)`, and pyramid sample `(a, b)` is base sample `(a * 32 / sp,
   b * 32 / sp)`.
4. **Simulation query**: `World::terrainHeight(space, x, z)` -> `TerrainSystem::height`, three base samples of the
   containing triangle interpolated with `terrainSurfaceDouble`. S22: the three samples come from
   `World::terrainSample` (base + delta, section 2); S20's early return for a space without terrain
   (`!hasTerrain()`) stays only while that space has no chunk with deltas (D1). The mover (`characterMoveSystem`)
   follows `terrainHeight`, so it sees edits.
5. **Visual streaming and the single send point**: `TerrainSystem::updateView(space, camera)` / `completeView(...)`
   (main thread, between steps) drain the streaming pool and send to the sink only through the private send point
   `sendTile(RegionId, const TerrainTileData&)`, `sendPyramid(SpaceId, const TerrainPyramidData&)` and
   `sendGrassMask(ChunkId, const GrassMaskData&)` (E3: pool-generated masks, a new sink and a GPU-set rebuild all re-send
   through it), plus `removeTerrainTile` / `removeGrassMask`; the sink drops its tiles and masks itself on a pyramid of
   another space or layout, and `clearTerrain()` when the camera's space has no terrain (E1). S22: the three send
   functions pass the base data through the merge of section 6 (D3), and `updateView` / `completeView` run the patch
   step after their sends. `EngineApp::setCompleteViewEachFrame(bool)` (E9) switches S20's benchmark mode (a
   `completeView` per frame, its time a separate wait part); the terraform benchmark turns it off (section 9).
6. **Settings and stats**: `TerrainViewSettings` (S22 adds `editSamplesPerFrame`, `editMasksPerFrame`),
   `TerrainSystem::Stats` (S22 adds the patch counters).
7. **Renderer and patches (S20, E2)**: the GPU set is S20's int32 storage-buffer atlas and pyramid (page-table entry =
   layer only, no per-tile reference; values are the integer heights). S20 owns, implements and tests
   `RenderSink::patchTerrainTile(RegionId, int x0, int z0, int w, int h, const int32_t* heights)` and
   `patchTerrainPyramid(SpaceId, ...)` in `RenderScene`, `VulkanRenderSink` and `RecordingRenderSink`: heights are copied
   at the call; a patch enters the same upload queue behind a still-queued full copy of that tile and counts against
   the per-frame byte cap; node bounds widen at every affected level; a patch of a region without a layer (or another
   space) is ignored and counted; `TerrainAtlas::readLayerForTests` reads a layer back. S22 is only the caller.
8. **Grass mask**: `GrassMaskData prototypeGrassMask(const TerrainTileData& level0, const TerrainLayout&,
   std::optional<int32_t> seaLevel, double minNormalY, double minAboveSea)` (slope and sea-level rule; the thresholds
   from `TerrainViewSettings`). S22: masks of edited chunks are computed from merged heights on the pool (section 6,
   E3).

## Scope

### 1. Brush and integer kernel (`engine/terrain/TerrainBrush.h`, `engine/terrain/BrushKernel.{h,cpp}`)

```cpp
struct TerrainBrush                                // §12.2; a nested visit struct (name, version, visit)
{
    enum Op : uint8_t { Add, Subtract, Flatten, Smooth, Count };   // Count: visit's EnumNames convention
    Op op = Add;
    Vec3 center = Vec3::Zero();   // x, z: brush center; y: the Flatten target (finite for every op, used only by Flatten)
    double radius = 1;            // metres, (0, kMaxBrushRadius]
    double strength = 0;          // Add/Subtract: metres at the center, [0, kMaxBrushStep];
                                  // Flatten/Smooth: blend factor at the center, [0, 1]
};
constexpr double kMaxBrushRadius = 128;            // metres: a brush touches at most 2 x 2 chunks
constexpr double kMaxBrushStep   = 64;             // metres per edit (Add/Subtract)
// All checks run on the doubles before anything is quantized: finite x, y, z, radius, strength; op < Count; radius and
// strength in range; x, z in [-kMaxBrushRadius, sizeChunks * 256 + kMaxBrushRadius] of the space (D12); for Flatten
// |y| <= kTerrainHeightLimit * kHeightUnit (8192 m). The error names the field and the bound.
bool validateBrush(const TerrainBrush& b, const SpaceTemplate& space, std::string* error);
```

Quantization (exact: power-of-two scaling and `floor`; only after `validateBrush` passed, so every value fits):

| Value | Integer | Unit |
|---|---|---|
| `cx`, `cz` | `floor(center.x * 256)`, `floor(center.z * 256)` (`int64_t`, `|c| < 2^26`) | 1/256 m |
| `cy` (Flatten target only) | `floor(center.y * 1024)` (`|cy| <= 2^23`) | height unit |
| `r` | `max(1, floor(radius * 256))` (`<= 2^15`) | 1/256 m |
| `s` (Add/Subtract) | `floor(strength * 1024)` (`<= 2^16`) | height unit |
| `f` (Flatten/Smooth) | `floor(strength * 65536)` (`<= 2^16`) | 1/65536 |

Kernel (`BrushKernel`, pure functions on arrays, no `World`):

- Sample position in 1/256 m: `px = i * sp * 256`; `dx = px - cx`, `dz = pz - cz` (`int64_t`).
- Affected rectangle: `i in [ceilDiv(cx - r, sp * 256), floorDiv(cx + r, sp * 256)]`, same for `k`, clipped to
  `[0, sizeChunks * S]`; inside it `|dx|, |dz| <= r <= 2^15`, so `d2 = dx^2 + dz^2 <= 2^31` and `d2 << 16 < 2^47` (no
  overflow; `d2` is never formed outside the rectangle). A sample is affected iff `d2 < r2` (strictly inside,
  `r2 = r^2 <= 2^30`). Empty rectangle: no-op.
- Weight, the quartic falloff `(1 - t^2)^2` without square roots: `t2 = floorDiv(d2 << 16, r2)` in `[0, 65535]`,
  `u = 65536 - t2` in `[1, 65536]`, `w = (u * u) >> 16` in `[0, 65536]` (`u < 256` gives `w = 0`: an affected sample
  near the rim may stay unchanged); `w = 65536` at `d2 = 0`.
- Rounding helpers (portable, no right shift of negative numbers): `floorDiv(a, b)` for `b > 0`;
  `roundShift16(v) = floorDiv(v + 32768, 65536)` (nearest, ties toward positive infinity).
- Ops on the pre-edit height `h` (all reads from the pre-edit copy, all writes to a separate output, so the result does
  not depend on the iteration order):
  - Add: `h' = h + ((s * w) >> 16)`; Subtract: `h' = h - ((s * w) >> 16)` (both products non-negative).
  - Flatten: `g = (f * w) >> 16`; `h' = h + roundShift16((cy - h) * g)` (`|exact step| <= |cy - h|`, and rounding to
    the nearest integer cannot leave `[0, cy - h]`: never overshoots; at `f = 65536` the center sample gets `cy`
    exactly).
  - Smooth: `sum` = the sum of the 3 x 3 pre-edit samples around `(i, k)` (indices clamped to the space);
    `g = (f * w) >> 16`; `h' = h + floorDiv(2 * (sum - 9 * h) * g + 9 * 65536, 18 * 65536)`: the exact step
    `(sum / 9 - h) * g / 65536` rounded once to the nearest integer, ties toward positive infinity (one rounding, so
    the per-sample error against the exact rational value is at most 0.5 unit per pass with no systematic sign; a
    rounded mean followed by a rounded blend would err up to 1 unit, `floorDiv(sum, 9)` would sink terrain by about
    half a unit per pass). A constant field and the interior of a linear ramp (`sum = 9 * h`) are unchanged. The
    pre-edit copy covers the rectangle grown by one sample (clipped).
- Height range: `kTerrainHeightLimit` (2^23 units, 8192 m) and `clampTerrainHeight` are S20's (E7). The kernel computes
  `v` (int64) by the op, then `h' = (v == h) ? h : clamp(v, -kTerrainHeightLimit, kTerrainHeightLimit)`, and a sample
  is **changed iff `h' != h`**, decided after the clamp: an Add on a sample at the limit clamps back to `h` and is not
  changed (no sequence, no part, no event); a sample outside the limit at zero weight (`v == h`, only possible in pure
  kernel tests, every engine base is clamped by S20) stays unchanged. In the engine every pre-edit height and every
  result lies within the limit, so every delta satisfies `|delta| <= 2^24`.
- `int64_t` throughout; the largest product is `2 * (sum - 9 * h) * g < 2 * 9 * 2^24 * 2^16 < 2^45`.

### 2. Modification state (`engine/terrain/TerrainEditStore.{h,cpp}`, `TerrainEditBlock.h`, `TerrainChunkEdits.h`)

- **Deltas, once per space** (one copy per sample, no duplicated borders): per space a hash map from block key
  `(k >> 4) << 32 | (i >> 4)` (sample indices are non-negative) to a `TerrainEditBlock {int32_t delta[256];}` of
  16 x 16 samples (1 KB). `S` is a multiple of 16 at both spacings, so chunk borders fall on block borders. An
  unallocated sample has delta 0. **On every path** (host edit, client part, `TerrainChunkState`, load) a block is
  allocated only if at least one of its samples gets a non-zero delta that differs from the current one, and all
  allocation happens after validation and before the first write. A block whose deltas return to zero stays allocated
  (invisible: it reads 0). Memory grows with the edited area (a 16 m crater at 1 m: 9 to 16 blocks, at most 16 KB).
- **No base cache (lazy base)**: the base is never stored; it is the pure base function `TerrainSystem::sample`
  (resident level-0 tile, else the clamped height source). So applying received or loaded deltas never calls the hook
  (a full-chunk `TerrainChunkState` or a `TERR` load costs the decode only), and the base cannot go stale. The cost moves
  to reads of edited samples without a resident level-0 tile (one hook call per sample, as S20's miss path; risk below).
- **Per-chunk state** (`TerrainChunkEdits.h`): `std::map<ChunkId, TerrainChunkEdits {uint32_t sequence;
  uint32_t nonZeroSamples; uint64_t revision; uint64_t contentHash; bool hashDirty;}>` per space. An entry exists iff
  `sequence > 0`. The sequence (§12.1) is incremented by every applied terrain modification that changes at least one
  sample of the chunk's closed `N x N` range (a changed border sample increments every chunk that contains it), and by
  S23b's tombstones through the accessor. `nonZeroSamples` counts the non-zero deltas in the closed range: maintained
  at each delta write for every chunk with an entry whose range contains the sample, and recomputed from the blocks (a
  dense scan of the closed range) whenever an entry is created (any path) or replaced (`TerrainChunkState`, load), so
  border samples written through a neighbour's message before the chunk's own entry exists are counted. A chunk **has
  deltas** iff it is > 0, and a "terrain record" is a chunk with deltas (E10). Entries are never removed except by a
  load or a client `Resync`: a chunk whose deltas returned to zero keeps its sequence (state).
- **Terrain revision (F1)**: `revision` is the value of the store's monotone `uint64_t` delta-change counter (never
  reset, also not by a load or `Resync`) at the last write that changed a delta of the chunk's closed range (host
  edit, client part, `TerrainChunkState`, load, `Resync` clear; also for a neighbour whose shared border sample
  changed). `bumpSequence` alone (S23b's tombstones) does not change it. It is local visual bookkeeping: not saved,
  hashed, replicated or compared; the edited grass masks key their staleness on it (section 6), so a tombstone never
  drops or recomputes a mask and a cached mask can never match a later state by accident.
- **Sequence accessor (E10)**: `uint32_t TerrainEditStore::sequence(ChunkId) const` (0 without an entry),
  `bumpSequence(ChunkId)` (creates the entry; host edits and S23b's tombstones), `setSequence(ChunkId, uint32_t)`
  (client messages, loads); `uint64_t revision(ChunkId) const` (0 without an entry). The sequence stays in
  `TerrainEditStore` for all spaces; S23b's `ChunkRecord` holds only its placement version and tombstones and bumps this
  sequence. Iteration sets: "chunks with `sequence > 0`" for `TERR`, `world.hash()`, the snapshot and the replica
  comparison; "terrain records" for deltas.
- **Dense form** (derived, never stored): `denseDelta(ChunkId, std::vector<int32_t>&)` writes the chunk's `N x N` deltas
  (row-major, row = z) from the blocks; used by saves, `TerrainChunkState`, `diffSaves`, the content hash and the
  replica comparison, for terrain records only. Iteration is in `ChunkId` order.
- Never evicted, on the host and on clients (§12.2 "clients keep all edited tiles"); S25 moves unloaded chunk states to
  the chunk store.
- Reads: `int32_t World::terrainSample(SpaceId, int64_t i, int64_t k) const` (new public read:
  `clampTerrainHeight(TerrainSystem::sample(...) + delta)`, the delta from the block when allocated, else 0; never
  allocates; the clamp is the identity for every state the host produces and bounds received or loaded sums, which are
  not checked against the base because that would need the hook). `TerrainSystem::height` takes its three samples
  through it (S20 interface 4), so `terrainHeight` is exact and sees edits from the requesting step on.
  `const TerrainEditStore& World::terrainEdits() const` (tests, tools, the visual path; read-only). The maps and blocks
  are not modified while systems run, so concurrent const lookups from `parallelEach` bodies are safe.
- The `TerrainEditStore` is owned by the `World`; `TerrainSystem` holds a const pointer to it (for `height` and the
  merge of section 6).
- **Statistics**: `TerrainEditStore::Stats {uint64_t edits, noopEdits, changedSamples, chunksEdited;}` (host edits;
  `World::terrainEdits().stats()`), and `size_t TerrainEditStore::memoryBytes() const` (blocks and entries; the
  benchmark's `edit_store_bytes`). Hook reads are counted by S20's `TerrainSystem`, not here (section 3).

### 3. `World::editTerrain` (`engine/world/World.{h,cpp}`, `engine/terrain/TerrainEdit.{h,cpp}`)

- `void World::editTerrain(SpaceId space, const TerrainBrush& brush)` (§12.2): host only; the space must exist and be
  `SpaceKind::OpenWorld` (with or without a height hook: a hook-less space has base 0, D1); the brush must pass
  `validateBrush` for that space. All three are checked **at the call** (also during a step, before the command is
  recorded), never at the flush, so the error points at the caller. A violation stops the program with
  `SAIGA_EXIT_ERROR` and the message in every build (the world API checks first and returns errors).
- During a step it records a command (the brush copied) applied at the phase flush in recording order (§8.1); between
  steps it applies at once. Inside a `parallelEach` body it is refused like every command (R-PAR-1).
- Engine-internal `TerrainEditResult applyTerrainEdit(SpaceId, const TerrainBrush&)` (used by the command, by
  `editTerrain` between steps and by the world API, which reports the result). Compute first, then commit; every
  allocation happens before the first sample is written, so an exception leaves the state unchanged:
  1. quantize, compute the affected rectangle and the read rectangle (Smooth: +1 sample), clipped; empty: return;
  2. gather the base and the pre-edit heights of the read rectangle into scratch arrays with `gatherSamples(space, i0,
     k0, w, h, base, height)`: the base per level-0 tile as a row copy from the resident tile, else
     `TerrainSystem::sample` (the clamped hook; counted in `baseHookSamples`, below); `height =
     clampTerrainHeight(base + delta)` with the delta of the allocated block, else 0. The base lies within the limit
     (S20 clamps every hook result, E7; a `SAIGA_ASSERT`, no runtime error path). Nothing is allocated;
  3. evaluate the kernel into the output rectangle; collect the changed samples (`h' != h` after the clamp rule,
     section 1) and their tight global bounding box; none changed: return an empty result (no sequence, no part, no
     event);
  4. reserve: the blocks that contain changed samples (a changed sample always has a non-zero new delta or lies in an
     allocated block), the entries of new chunks, and the host send-list entry with its parts (section 5). **S23b's
     extension point (F2)**: S23b computes its tombstone-on-edit here from the kernel output (the post-edit samples)
     and reserves its chunk records, key capacity and send-list entries in this step too; nothing is written yet;
  5. commit (writes only, no allocation): write `delta = h' - base` of the changed samples (updating `nonZeroSamples`
     and `revision`); per chunk whose closed range contains a changed sample (1, 2 or 4 chunks): `bumpSequence`,
     `hashDirty`, and a part `{chunk, sequence, chunk-local box = the changed box clipped to the chunk's range, the
     delta values of that box}` in the reserved send-list entry (section 5); then S23b writes the tombstones it
     reserved in step 4 (keys, `bumpSequence`, its send-list entries after the `TerrainModification` entry);
  6. one `onTerrainChanged(space, box)` with the global changed box (below);
  7. count statistics (`TerrainEditStore::Stats`); inside a step the application is part of the phase flush's profile
     (scope `terrain/edit` under it); between steps callers time it themselves (the benchmark, section 9).
  An edit that changes nothing counts in `noopEdits` and stops after step 3.
  The bounding box of the parts equals the changed box (each extreme of the box is attained by a changed sample, and
  that sample lies in the part of a chunk that contains it), so a client computes the same box from a message.
- **`baseHookSamples`** (owner: S20's `TerrainSystem`; S22 adds it): `uint64_t TerrainSystem::baseHookSamples() const`,
  a relaxed `std::atomic<uint64_t>` incremented by `TerrainSystem::sample` for every base sample it reads from the
  height source instead of a resident level-0 tile, from every caller (the gather, `terrainHeight` /
  `terrainSample`, tools); S20's pool generation does not go through `sample` and is not counted. Tests and the
  benchmark read differences of it (`world.terrain().baseHookSamples()`) to prove that a path calls no hook.
- `onTerrainChanged(SpaceId, box)` is the single function every applied change goes through (host edit: one call per
  edit; client `TerrainModification`: one call per message with the bounding box of its parts; `TerrainChunkState`: one
  call with the whole-chunk box; client `Resync` clear: one per cleared chunk with deltas; host load: one per chunk
  with deltas before or after the load): it records the change for the event (below; not for a load), from step 8
  pushes the box to the visual dirty queue of section 6 (only while a render sink is attached), and feeds S23b's
  re-snap queue of tier-1 instances (D13: driven by applied changed boxes; re-snap only, F2: S23b's tombstones are
  computed in `applyTerrainEdit`'s steps 4 and 5, not here).
- `TerrainEditedEvent {SpaceId space; double x0, z0, x1, z1;}` (metres, inclusive bounds of the changed samples;
  `EngineEvent::TerrainEdited`, appended in merge order, D10; `Replicate::Never`; host and clients). **One emission path
  per change, never two**:
  - a change applied between steps (world API, tools, a load listener on the host) is emitted at once (pending, readable
    in the next step) and does not enter the change list;
  - a change applied inside a step (a command at a phase flush; a client's phase-1 message handler, also inside
    `stepNetworkOnly`) is appended to the change list `{space, box}` (no event yet); when the step or `stepNetworkOnly`
    commits, one event per entry is emitted in application order and the list is cleared (readable in the next step,
    as if emitted at application time);
  - an aborted step keeps the list (`World::abortStep` drops the aborted step's events while the edits of its finished
    phases stay applied, §7.6; keeping the list allocates nothing). The next `runStep` emits the kept entries at its
    start, before `events.beginStep()`, so they are readable in the retried step (if that one aborts too,
    `events.abortStep` turns them back into pending events, so nothing is lost or duplicated); a `stepNetworkOnly`
    that commits first emits them at its commit;
  - a host load emits nothing and clears the list (a load restores the saved pending events as they were, so `hash()`
    after a load equals the saved world's). A save taken between an aborted step and the next step does not contain
    the kept entries' events (the list is not saved; accepted: an aborted step is an error path, and the terrain state
    itself is saved).
  So a 4-chunk edit is one event on the host and one on every client, with the same box. `World::pendingTerrainChanges()`
  exposes the list read-only (`std::vector<TerrainChange>`, global sample boxes, application order; the in-step changes
  whose events are not emitted yet: those of the running step plus the kept ones of aborted steps; empty between
  steps unless a step aborted). The event is the broadcast for game reactions and later engine consumers (nav patch
  S45, region summaries S27, collider patch bookkeeping S28). The "affected tier-1 instances and pieces" report of
  §12.2 has nothing to report before S23b/S30 (S23b reports its tombstones in `TerrainEditResult::tombstoned` and
  re-snaps from the `onTerrainChanged` queue, §22).
- `TerrainEditResult` (engine-internal; the API returns it as JSON; S23b appends its `tombstoned` list, computed in
  step 4 and written in step 5):
  ```cpp
  struct TerrainEditResult
  {
      struct Chunk { ChunkId chunk; uint32_t sequence; int x0, z0, w, h; };  // chunk-local box of the part (samples)
      SpaceId space;
      std::vector<Chunk> chunks;              // ChunkId order
      uint64_t changedSamples = 0;
      int64_t i0, k0, i1, k1;                 // the global changed box (inclusive)
  };
  ```

### 4. Save, load, hash, diff (`engine/save/`)

- Optional section `TERR` (D7: `HEAD, SCHM, SPCE?, GENR?, TERR?, CREC?, ENTS, COMP, EVNT, ACTN, STAT`), written only
  when at least one chunk has `sequence > 0` (a save without edits has no `TERR`; the committed golden `save_v1.bin` is
  not regenerated and still loads; the blob format version stays 1, the section has its own; new saves differ from
  pre-S22 ones only by the event type in `SCHM`, see the hash paragraph). Per chunk: the sequence plus optional deltas
  (E10: a chunk with a sequence and no deltas, e.g. S23b's tombstone-only chunk or deltas returned to zero, has an entry
  without deltas):
  ```
  TERR  section format version (u32 = 1); space count (u32); per space in SpaceId order: SpaceId (u16), base
        spacing (u8), N (u16), chunk count (u32); per chunk with sequence > 0 in ChunkId order: x (i32), z (i32),
        sequence (u32, > 0), flags (u8: bit 0 = deltas present, other bits 0); if deltas are present: encoded length
        (u64), the dense N x N delta encoded by the delta codec (below)
  ```
  Added with S21's extension recipe (D7; if S22 merges first it adds the mechanism in that form and S21 then adds only
  `SPCE`): `kTagTerrain = saveTag("TERR")` in `SaveFormat.h`, its `{tag, optional, maxVersion}` entry (`optional =
  true`, `maxVersion = 1`) at the D7 position, a raw range `std::optional<ParsedSave::Range> terrain` (version and length
  checked by `parseSave`, decoded by the loader's own pass before the commit), the writer
  `s.beginOptional(kTagTerrain, 1)` ... `s.end()` in `World::save` between `SCHM` and `ENTS` in table order, the load
  check in `WorldLoader` before the commit, and the `TERR` case in `SaveDiff.cpp`'s optional-section comparison.
- **Delta codec** (`engine/terrain/TerrainDeltaCodec.{h,cpp}`, step 4; shared by `TERR` and the messages of section
  5): a rectangle of deltas in row-major order (row = z) as zigzag varints of the difference to the previous value;
  the predictor starts at 0 for every rectangle (each `TERR` chunk, each message part, each `TerrainChunkState`), so
  every rectangle decodes on its own; then optionally zstd at a fixed level (always for `TERR`, flag-selected in
  messages). The decoder bounds the decompressed size (`w * h * 4` bytes: a zigzag difference of two deltas with
  `|delta| <= 2^24` is below `2^26`, at most 4 varint bytes), the value count and `|delta| <= 2^24`, and leaves no
  bytes over.
- `tryLoad` validates the section before the world changes (§15.1), decoding into staged scratch (no store write, no
  block allocation before the whole section passed): known `OpenWorld` space, spacing and `N` equal to the loading
  world's template (a mismatch is a load error naming the space: changed spacing needs a migration that S22 does not
  offer), chunk inside `[0, sizeChunks)^2` (D12), ascending unique chunk order, `sequence > 0`, flags known, the codec's
  bounds (decompressed size `N * N * 4` bytes, value count `N * N`, `|delta| <= 2^24`), deltas present only with at
  least one non-zero value, and the border rule (load only: a save is one complete state, unlike the per-chunk
  messages of section 5): a sample
  shared by two chunks with deltas has equal values in both, and a non-zero sample on a chunk border is present
  (non-zero, equal) in every chunk of the space that contains it (a chunk without deltas counts as all zero). The sum
  `base + delta` is not checked (it would need the hook for every sample; `terrainSample` clamps it, section 2). Then,
  in the commit with the rest of the state (before `startObjects`): the staged blocks (only blocks with a non-zero
  delta are allocated) and entries replace the store, and `onTerrainChanged` receives the whole-chunk box of every
  chunk with deltas before or after the load (the dirty queue re-patches sent tiles, the pyramid and masks to the
  loaded state; nothing is evicted; no event, section 3), the host send list is cleared, and the change list is
  cleared. No hook is called by a load.
- Replication of the load (E5): `Replication::onHostLoad` (called after `discardOutboxAfterLoad`, with `loadHookPhase`
  0) sends, to every connected peer, S16's `Resync`, then (S23a) `GenerationState`, then `sendChunkSnapshot(peer)`
  (section 5) at once (same epoch, same `State` channel, so they arrive after the `Resync` and before the next step's
  messages); the loaded entities follow as spawns in the next phase 8.
- `world.hash()` covers the modification state from S22 (§17.4 said "from S25 the chunk states"): when at least one
  chunk has `sequence > 0`, after the existing content, per such chunk in `ChunkId` order the `ChunkId`, the sequence,
  the deltas-present flag and, with deltas, the chunk's content hash (FNV-1a 64 of the dense delta, cached, recomputed
  lazily when `hashDirty`, so stepping without `hash()` calls costs nothing). Without such chunks nothing is appended.
  This does not keep existing hash values: registering the `TerrainEdited` event queue changes every `world.hash()`
  (`EventQueues::hashPending` hashes each registered queue's id and pending count) and the schema (`SCHM`), so a
  world without edits hashes and saves equal to a world **of the same build** without edits, not to a pre-S22 world
  (no test commits a world hash; `data/tests/engine/schema.json` is regenerated, `save_v1.bin` still loads).
- `diffSaves` gains a `terrain` kind: one-sided chunks, sequence differences and per differing chunk the number of
  differing samples and the first one (needed to explain a repeat-variant mismatch).

### 5. Replication (`engine/net/`, `engine/terrain/TerrainMessages.{h,cpp}`)

Two raw engine messages (IDs after `Resync`, before `Count` in `EngineMessage`, appended in merge order), host ->
clients, **`Channel::State`, reliable**:

| Message | Content |
|---|---|
| `TerrainModification` | flags (0 raw, 1 zstd), epoch (u32), space (u16), part count (u16); per part: chunk x, z (i32), sequence after the change (u32), box `x0, z0, w, h` (u16, chunk-local sample coordinates, `x0 + w <= N`, `z0 + h <= N`, `w, h >= 1`), `w * h` new delta values (new height minus base) in the delta codec of section 4 (row-major zigzag varint differences, the predictor reset to 0 for each part) |
| `TerrainChunkState` | flags (bit 0 zstd, bit 1 deltas present), epoch, space, chunk x, z, sequence (> 0), `N`; with deltas: the dense `N x N` delta in the same codec (predictor reset to 0 for the chunk) (one message per chunk with `sequence > 0`) |

- Values are absolute (the new delta, "new heights minus base", §13.2), never increments and never the brush; one
  `TerrainModification` per applied edit, with one part per changed chunk, applied atomically by the client.
- **State channel, not `ChunkData`**: the messages must be ordered with `Resync` and the spawn set (epoch changes on a
  host load); `ChunkData` would need its own epoch buffering. The sizes in S22 are small (a 16 m brush: about 1-3 KB
  raw); a 128 m brush at 1 m is at most about 260 KB raw, fragmented by the transport (S15, reliable up to 64 MB). S26
  moves chunk snapshots to `ChunkData` together with its re-request protocol (amendment of §13.1/§13.2).
- **Host send list** (the list and its commit/abort bookkeeping come with step 3, so `applyTerrainEdit` can reserve its
  entry; encoding and sending with step 5): an ordered list of host state messages in `Replication` (not in the world
  outbox), holding the parts unencoded: `appendTerrainModification(space, parts)` from `applyTerrainEdit`; S23b appends
  its tombstone messages to the same list, so terrain parts and tombstones that bump one chunk's sequence reach the
  client in application order (the client's `+1` rule depends on it). The entries are encoded and sent by
  `repl->extract(networkOnly)` after the component changes and before the snapshot, to every regular peer, both in a
  real step's phase 8 (`extract(false)`) and at the `stepNetworkOnly` commit point (`extract(true)`); the list is
  cleared when the step or `stepNetworkOnly` commits (`Replication::commit`, S16's staged bookkeeping). An aborted step
  keeps the list (the applied edits of finished phases remain, §7.6), so the next completed step sends them; an aborted
  phase 8 drops its outbox entries and the next phase 8 resends. Without peers the list is cleared at the commit (single
  player keeps nothing). Edits between steps (world API, tools) are sent by the next step's phase 8.
  `Replication::hostStateList()` exposes the list read-only (the benchmark's encoder, section 9).
- **Chunk snapshot (E5)**: `void Replication::sendChunkSnapshot(uint8_t peer)` (S22 defines it with the terrain part;
  S23b inserts its `ChunkRecordState` messages before the terrain states, S23a sends `GenerationState` before it): one
  `TerrainChunkState` per chunk with `sequence > 0` of every space, in `SpaceId`, `ChunkId` order (with the dense delta
  only for terrain records). Order on a **join** (phase 8, `peersJoined()`): `Resync` (only if the epoch is not 0),
  `GenerationState` (S23a), S16's spawn set, then `sendChunkSnapshot(peer)` (the current state, which includes the
  host list's entries; those entries are not sent to that peer in the same pass). Order on a **host load**: section 4.
  This is the minimal pre-S26 modification snapshot: S22 needs it because S16 supports late joiners; S26 replaces it
  with per-loaded-chunk snapshots.
- **Client `Resync`**: the client clears its whole edit store (entries and blocks); every cleared chunk with deltas goes
  through `onTerrainChanged` with its whole-chunk box (visuals return to the base; one `TerrainEditedEvent` per such
  chunk), then the following `TerrainChunkState` messages apply.
- **Client receive** (phase 1, the handler of the raw message): untrusted input, validate-then-apply. Validation decodes
  into scratch and touches no store and allocates no block: the flags, epoch, space (exists, `OpenWorld`), chunk inside
  `[0, sizeChunks)^2`, box inside the tile, the codec's bounds (value count, decompressed size, `|delta| <= 2^24`),
  `sequence > 0`; parts of one message that share samples (borders) carry equal values there. **No check against the
  client's present neighbours** (the host is authoritative): a join or host-load snapshot sends one `TerrainChunkState`
  per chunk in `ChunkId` order, so the first of two adjacent edited chunks arrives while its neighbour is still absent
  or stale, and a neighbour check would reject it. The full border rule is a `TERR` load check only (section 4). Shared
  border samples are one copy (section 2): a later message of the neighbour writes the same values, and until it arrives
  the neighbour's samples already read the new border (transient within one snapshot). No base is read (the sum is
  clamped by `terrainSample`, section 2), so no hook is called. Any failure counts in `NetStats::rejected`, logs, and
  applies nothing (no block, no entry). A message of another epoch is dropped. Per part, against `sequence(chunk)`:
  `sequence <= current` is dropped (duplicate); `sequence == current + 1` is applied; `sequence > current + 1` is a gap:
  before S26 it is a protocol error that is logged with `SAIGA_LOG_E`, counted in the new `NetStats::modificationGaps`
  (the name S23b reuses for tombstones), and the part is still applied (absolute values keep the box correct; the
  samples of the missing message may be stale). The host + client variant fails on any gap. Applying the accepted parts
  of a message: reserve (allocate only the blocks in which a sample gets a non-zero delta that differs from the current
  one; the box is a superset of the changed samples), then write the deltas, `setSequence`, and one `onTerrainChanged`
  with the bounding box of the applied parts (equal to the host's changed box, section 3). `TerrainChunkState` replaces
  the chunk's deltas and sequence if its sequence is >= the current one (same reserve rule; `nonZeroSamples` of the
  chunk recomputed from its dense data after the write, those of neighbours with entries updated per written border
  sample, section 2; `onTerrainChanged` with the whole-chunk box when the chunk had or gets deltas).
- Clients never call the kernel. Equal heights on host and client follow from equal message streams plus equal bases
  (same hook and binary in S22; across machines from S23, R-CONTENT-1).

### 6. Visual path (D3, E2, E3; `engine/terrain/{TerrainVisualPatch,TerrainDirtyQueue}`, `TerrainSystem`)

Base data stays in the `TerrainStore`; edits reach the sink in two ways, both through S20's calls (S22 changes nothing
in `saiga_vulkan`, `VulkanRenderSink` or `RecordingRenderSink`; the patch calls, their upload ordering, the node bounds
and their tests are S20's, E2).

- **Merge at the send point** (`TerrainVisualPatch::mergeTile` / `mergePyramid`, called from S20's `sendTile` /
  `sendPyramid`): every tile and pyramid that reaches the sink (a streamed result in `updateView` or `completeView`, a
  new sink receiving everything, S21's re-send after a view-space switch) passes the merge. Without a terrain record
  among the covered chunks the base object is sent unchanged (the shared `shared_ptr`, no copy). Otherwise a merged copy
  is sent: the coarse samples of the tile that coincide with allocated blocks get `clampTerrainHeight(base + delta)`
  (level `L`: base samples with indices divisible by `2^L`; pyramid: by `32 / sp`; the base is the object being sent),
  and `minHeight` / `maxHeight` are recomputed exactly over the merged values. A tile result that was requested before
  an edit and arrives after it therefore never undoes the edit (§12.3; the merge runs on the main thread at send time,
  no sequence check). The pyramid of a non-view space (generated lazily and kept CPU-cached, D5) is merged when it is
  sent.
- **Grass masks (E3, F1)**: masks of chunks with deltas are computed with `prototypeGrassMask(level0, layout, seaLevel,
  grassMinNormalY, grassMinAboveSea)` from merged level-0 heights, on S20's streaming pool: a job captures a copy of the
  chunk's merged level-0 heights (the store's level-0 tile, which is resident whenever its mask is wanted, plus the
  deltas, merged on the main thread: one add per sample), the chunk's terrain `revision` (section 2; not the sequence,
  so S23b's tombstones never drop a mask) and the GPU set it was submitted for. Its completion (main thread, in `drain`)
  keeps the result only if the chunk's mask is still in S20's desired mask set of the current GPU set (same view space
  and layout, within `grassRadius` and the `grassMaskLayers` truncation) **and** the chunk's revision is unchanged;
  then it stores it in `TerrainVisualPatch`'s edited-mask cache `{ChunkId -> mask, revision}` and sends it through
  `sendGrassMask`. A result whose revision changed is dropped and a recompute re-queued (unless one is queued already;
  `maskResultsDropped`); a result for a chunk that left the desired set or for an older GPU set is discarded without a
  re-queue (the chunk gets a new job if it becomes desired again). The `TerrainStore` keeps base masks only (D3).
  S20's mask job for a chunk that has deltas at submission takes this path instead of the base tile. `sendGrassMask`
  of a chunk with deltas (new sink, rebuild) sends the cached mask when its revision is current; else it sends nothing
  yet and queues a recompute (the chunk shows no grass for the few frames until the job completes; never base grass on
  a crater wall). Submissions are capped at `TerrainViewSettings::editMasksPerFrame` (default 2) per `updateView`; the
  rest stay queued. A cache entry is dropped with S20's `removeGrassMask` of the chunk (grass radius), on a GPU-set
  rebuild of another space, and when the chunk's deltas return to zero (`nonZeroSamples` 0): a chunk with a sent mask
  then gets the store's base mask through `sendGrassMask`, queued like a recompute.
- **Dirty queue** (`TerrainDirtyQueue`): `onTerrainChanged` adds the box (global base samples) only while a render sink
  is attached (headless worlds keep nothing):
  - to the **tile boxes** only if the box's space is the current view space (the GPU set's space); tiles of other
    spaces are not on the GPU and arrive merged when that space is viewed. The tile boxes are dropped on a view-space
    switch (the new set is sent merged);
  - to the **pyramid boxes of its space**, kept per space regardless of the view space and never cleared on a switch
    (D3; patching a just re-sent merged pyramid writes equal values, which is harmless).
  Each list merges a new box into an overlapping or touching one.
- **Patch step** (`patchEditedTiles`, in `TerrainSystem::updateView` after its sends; in `completeView` without a
  budget: `completeView` drains the whole dirty queue and waits for the queued mask jobs, so a complete point shows
  every edit): drains the view space's lists in insertion order until `TerrainViewSettings::editSamplesPerFrame`
  patched samples (default 2^18, about 1 MB of uploads; summed over all levels) are used. For each box: every sent tile
  of every streamed level that intersects it gets the sub-rectangle of its coincident samples as
  `patchTerrainTile(region, x0, z0, w, h, heights)`; the pyramid likewise as `patchTerrainPyramid`. The heights are
  `clampTerrainHeight(base + delta)` with the base read from the store's own tile at the same `RegionId` (a sent tile is
  always resident in the store: S20 removes it from the sink before evicting it) or from the store's CPU pyramid, and
  the delta from the blocks (0 where unallocated): **no hook call on the main thread** (asserted in a test through
  `baseHookSamples`). A box whose patches exceed the remaining budget is split by rows: the first rows that fit are
  patched now, the remaining rows stay queued as a smaller box at the front of the list. Tiles that are not sent are
  skipped (they get the merge when sent). Every level-0 chunk with a sent mask inside a patched box gets a mask
  recompute queued (above).
- Ordering: a tile set and patched in the same frame (a merged send, then a patch of a queued box over it) is S20's
  case of a patch behind a still-queued full copy (E2); the recording sink's copy ends equal to the merged heights
  (test).
- The water plane is one snapped plane at sea level (S20): nothing to widen; a pit dug below the sea level shows the
  plane above it (§20: global sea plane until S39). The grass follows the patched heights (it samples the height tiles)
  and the recomputed masks. The prototype trees (`PrototypeForest`, float placement) are not re-snapped before S23b
  (risk below).
- Spaces without terrain (`!hasTerrain()`) draw nothing (S20, D1/E6); their edits are simulation, save and replication
  only.
- Timers: profile scope `terrain/visual` (frame profile) covers the merge at the send points, the patch step and the
  mask completions (not S20's drain of tile results or generation). In a normal frame it is part of the frame's CPU
  work; inside a `completeView` it is part of S20's terrain-stream wait part (E9). `TerrainSystem::Stats` gains
  `patchedSamples`, `patchCalls`, `queuedBoxes`, `mergedSends`, `maskRecomputes`, `maskResultsDropped`.

### 7. World API and MCP tools (`engine/api/WorldApi.{h,cpp}`, `engine/tools/WorldTools.cpp`)

| Operation / tool | Contract |
|---|---|
| `edit_terrain(space?, op, center, radius, strength)` | changes: the terrain of the space (host only; op `"add"`, `"subtract"`, `"flatten"`, `"smooth"`; center `[x, y, z]`, y used only by flatten, 0 when omitted); applied at once between steps (refused inside a step, on clients, for an unknown or enclosed space, for an invalid brush with the field and bound in the error, `validateBrush`); returns `{changed_samples, chunks: [{chunk: [x, z], sequence, box: [x0, z0, w, h]}], box: [x0, z0, x1, z1]}` (chunk boxes in chunk-local samples, the global box in metres) (empty `chunks` when nothing changed); logged under `api` |
| `terrain_height(space?, x, z)` | reads: the exact surface height (§12.2 rule incl. edits) and the three integer samples; `edited: bool` (S20 adds no such tool) |
| `terrain_edits(space?, limit?, after?)` | reads: the chunks with `sequence > 0` in `ChunkId` order with sequence, `has_deltas`, the count of non-zero delta samples and the delta min/max; paged with `next` |

Registered through `registerWorldTools` (one tool per operation); `edit_terrain` is not registered with `allowEdits =
false`; `describe_world` lists the operations.

### 8. Scenario framework (`engine/tools/ScenarioWorld.{h,cpp}`, `ScenarioRunner.cpp`)

- Helpers: `editTerrain(const TerrainBrush&, SpaceId space = 0)` (on the host world; between steps; recorded in the
  failure report), `terrainHeight(double x, double z, std::optional<int> client = {}, SpaceId space = 0)` and
  `terrainSample(i, k, client, space)`.
- The host + client end condition also waits until every client's chunk sequences equal the host's (inside the
  existing bound of `2 * delaySteps + 3` steps without loss); `compareReplicas` compares, per space over the union of
  chunks with `sequence > 0` on either side, the sequence (through the accessor) and, for terrain records, the dense
  delta of every client with the host (exact; a one-sided chunk is a difference; a difference names the peer, chunk,
  sequence and the first differing sample) and fails on `NetStats::modificationGaps > 0`. This
  makes "host + client end with equal heights" a property of every scenario, not only the terraforming ones.
- `terraformTestTemplate(int spacing)` (test helper, added to S20's `tests/terrain_test_hooks.h` in its engine-only
  part): S20's `testSpaceTemplate(16, spacing, integerTestHook, std::nullopt)`: `OpenWorld`, `sizeChunks` 16, the
  given spacing, no sea level, S20's cheap integer hook `integerTestHook` (named by S20 for S22's render and edit
  tests; S22 adds no hook of its own, D1). S22's tests take expected heights from base reads (`terrainSample` before
  the edit), never from the hook's formula. The only other hook is local to `test_engine_terrain_edit.cpp`: one that
  returns `2^23` everywhere (the clamp test).

### 9. Benchmark (`samples/games/world/TerraformBenchmark.{h,cpp}`, `src/saiga/vulkan/world/WorldBenchmark.*`, `engine/app/EngineApp.*`, `tests/`)

- `game_world --benchmark terraform` (§17.7; window, 1920 x 1080 like `flyover`). `WorldBenchmark` (saiga_vulkan, no
  engine headers, D2) gets the name `terraform` in `names()` and its static pose; the edit schedule lives in the game
  (`TerraformBenchmark`, calls `World::editTerrain` between steps), which adds its keys to the result object the way
  `game_world` adds `step` and `recording` (#66's `result()`), and prints its lines after `summaryText`.
- Stepping (D9, E9): `game_world`'s benchmarks run inside `EngineApp::run()`, whose `frame()` advances the world by
  the measured wall-clock time, and a render hook cannot call `runFrames`. S22 adds
  `void EngineApp::setFixedFrameTime(double dt)` (0, the default: the measured time): when > 0, `frame()` advances the
  worlds (`advanceWorlds`) by `dt` instead of the measured frame time; the frame statistics, the fps counter, the input
  and `game_world`'s own interval measurement keep the wall clock, so `cpu_ms`, `frame_interval_ms` and
  `frame_work_ms` keep their meaning. `game_world --benchmark terraform` calls `app.setFixedFrameTime(1.0 / 60)` in its
  init hook (the world's dt: exactly one step per frame for the 60 warm-up and 600 measured frames); `runFrames` is
  unchanged. Test in `tests/test_engine_app.cpp`: headless, with a fixed frame time of the world's dt every `frame()`
  runs exactly one step, also when the frames take longer or shorter than dt.
- Residency (E9): the benchmark calls `EngineApp::setCompleteViewEachFrame(false)` (it opts out of S20's per-frame
  `completeView`: the camera is fixed, so the measured frames run `updateView` and the patch step under its budget,
  which is what a game frame does). `game_world` sets `residencyRange0 = max(400, 1.3 * terrainLodRange)` whenever the
  setting changes (S20); in the terraform benchmark the floor of that rule is `TerraformBenchmark::kResidencyRange0 =
  600` instead of 400 (game policy, within S20's 48 tiles per level), so the initial setting and any later setting
  change keep at least 600 m (ImGui is off in benchmarks, so no change happens during the run). The first warm-up frame
  is a complete point (`completeView`, its time in S20's terrain-stream wait part); asserted at the first measured
  frame: the level-0 tiles of the chunks `(14..16)^2` (every brush area) are resident; asserted over the measured
  frames: `Stats::pending == 0`, no `setTerrainTile`, and no hook read (the difference of
  `world.terrain().baseHookSamples()` is 0: every edit gathers its base from resident tiles; patches and merges only).
  The time of `updateView` incl. the patch step is part of each frame's CPU work (`frame_work_ms`);
  `terrain_visual_ms` is its `terrain/visual` part.
- Layout: 8 brushes, two on each of the 4 chunk corners `C_c = (3840 + 256 * (c % 2), 3840 + 256 * (c / 2))` m
  (c = 0..3; the content center's chunks 14..16). Brush `b = 2c + j`:

  | c | corner (m) | j = 0 (roughen) | j = 1 (level) |
  |---|---|---|---|
  | 0 | (3840, 3840) | Add, r 4 m | Flatten, r 12 m |
  | 1 | (4096, 3840) | Subtract, r 8 m | Smooth, r 16 m |
  | 2 | (3840, 4096) | Add, r 16 m | Flatten, r 8 m |
  | 3 | (4096, 4096) | Subtract, r 12 m | Smooth, r 4 m |

  (each op with two radii). Strength 0.25 m (Add/Subtract) or 0.5 (Flatten/Smooth); the Flatten target is
  `terrainHeight(C_c)` at the start. The brush center orbits `C_c` on a circle of radius `0.5 * r_b`, starting at angle
  `j * 180` degrees (the pair on opposite sides), advancing 1 m of arc per edit. The corner sample (shared by all 4
  chunks of the corner) is always inside the disc at `d = 0.5 r`: `t2 = 16384`, `w = 36864`, so every Add/Subtract edit
  changes it by `(256 * 36864) >> 16 = 144` units and changes **exactly 4 chunks** (asserted per edit from
  `TerrainEditResult::chunks`). The levelling brush of a corner works against its roughening brush (Flatten pulls the
  rising mound back to the fixed target, Smooth evens the deepening pit), so neither converges to no-ops; their no-op
  edits and chunks per edit are reported, not assumed.
- Schedule (by step): the edits run between steps, in `game_world`'s `beforeRender` hook after the frame's one step
  (before the frame's `updateView`, so the frame patches its own edit): brush `b` edits after steps with `step % 8 ==
  b`, so exactly one edit per measured frame: 600 edits (60 edits/s at 60 steps/s in total, 7.5 per brush and second),
  each brush 75 edits. `TerraformBenchmark` times each `World::editTerrain` call itself (`steady_clock` around the call:
  quantize, gather, kernel, commit, the send-list entry); this is `terrain_edit_ms`. The edits are not part of any step,
  so `step {median, p95}` excludes them. Expected: `chunks_edited >= 1200` (300 Add/Subtract edits x 4 chunks;
  asserted), plus the reported Flatten/Smooth chunks.
- Camera: eye `(3968, g + 300, 3940)`, target `(3968, g, 3968)` with `g = terrainHeight(3968, 3968)` at the start
  (absolute y, about 85 degrees down). At game_world's 60 degree vertical field of view and 16:9 every brush area
  (x, z in `[3814, 4122]`) is in view (the tightest point at `tan = 0.53` of `0.577` vertically) and within 400 m of
  the camera at equal ground height (at most 383 m; residency 600 m covers ground differences, and the assertion above
  checks it).
- Encoding (D9): `game_world` has no peers, so the phase-8 send never runs. Right after each `editTerrain` call
  (between steps) the schedule encodes the host send list's new entry (`Replication::hostStateList()`) with the
  phase-8 encoder (`encodeTerrainModification`) into a scratch buffer, timed separately; nothing is sent.
- `benchmark_terraform.json`: the common keys of §17.7 as written since #66 (`cpu_ms`, `frame_interval_ms`,
  `frame_work_ms`, `wait_ms`, `gpu_ms`, `stalled_frames`) and S20's terrain keys (`terrain_stream_ms` with max 0 over
  the measured frames: no `completeView` after the first warm-up frame; `wait_ms.terrain_stream`; the block `terrain:
  {uploads, upload_mb, resident_tiles, cpu_mb, gpu_mb}`) plus `edits`, `noop_edits {add, subtract, flatten, smooth}`,
  `chunks_edited`, `chunks_per_edit {flatten, smooth}`, `samples_changed`, `terrain_edit_ms {median, p95}` (per edit,
  timed by `TerraformBenchmark`), `base_hook_samples` (0), `terrain_encode_ms {median, p95}` (per edit),
  `edit_bytes_per_second` (the encoded `TerrainModification` bytes, as sent to one client), `terrain_visual_ms {median,
  p95}` (per frame: the `terrain/visual` scope, excl. GPU time), `patched_samples {median, p95}` and `upload_bytes
  {median, p95}` per frame, `mask_recomputes` (0 here: the camera is above the grass radius; masks are measured
  headless), `edit_store_bytes` at the end, `step {median, p95}`.
- Budget (amendment of §17.10, release build): `terrain_edit_ms` p95 <= 1 ms per edit; `terrain_visual_ms` p95 <= 1 ms
  per frame; the step p95 within the host-step budget (<= 4 ms; edits excluded, they are budgeted per edit);
  `edit_bytes_per_second` reported against the 256 kbit/s per-client upload budget (report only, §17.10). The baseline
  for the development GPU goes into `data/tests/benchmarks.json`. S28 re-runs it with collider patches.
- `tests/test_engine_terrain_benchmark.cpp` (non-gating, prints ms like the ECS benchmark; runs in CI without a GPU;
  headless, so no resident tiles: every gather and read of an edited sample uses the hook): the same schedule for 600
  edits with `terraformTestTemplate(1)` and with the prototype hook (`prototypeHeight`, the worst case): kernel only,
  `applyTerrainEdit` (with `baseHookSamples`), `terrainSample` of edited samples (hook path), message encode + client
  decode/apply, `hash()` with 50 edited chunks, save/load of 50 full-chunk edited chunks at 1 m (asserted: the load calls
  no hook), a join of a client to a host with 50 full-chunk edited chunks (`sendChunkSnapshot` encode + client apply),
  one grass mask from merged heights (`prototypeGrassMask`, one chunk at 1 m and at 2 m; the pool job's cost),
  and a recording-sink world with resident tiles for the patch step (`terrain_visual_ms` without a GPU).

### 10. `game_world` terraform mode (game policy, samples; D14)

The §22 row does not require a demo change. Minimal, because the 32 km world is where terraforming is visible and the
benchmark lives there:

- Key `T` toggles the terraform mode; while it is on, the left / right mouse button raises / lowers (`1`..`4` select
  add, subtract, flatten, smooth) at the terrain point under the mouse cursor (found by marching `terrainHeight` of
  the rendered world in game code). The game calls `app.startWorld().editTerrain(space, brush)` between steps (in
  `beforeRender`) at 10 edits/s while a button is held: `startWorld()` is the host world, also with `--coop` (S16b),
  where the rendered world is a client and an edit on it would stop the program (R-NET-1); the edit reaches the
  rendered client world through replication. Radius and strength are game constants shown and editable in ImGui
  (policy, R-MECH-1). No `PlayerAction`, no gameplay system, no data file.
- The `edit_terrain` MCP tool makes the mode scriptable (screenshots of a crater, local review). With `--coop` the
  MCP tools act on the client world, so `edit_terrain` answers with its client error there (the `T` mode still works).
- The flight, platformer and racing demos are unchanged (they keep working on S20's tiles; no edit path is active).

## Out of scope

- Collider patches (S28), nav patches (S45, consumer of the event), region modification summaries and far content
  (S27), re-snapping or tombstoning tier-1 instances (S23b: tombstones in `applyTerrainEdit`'s reserve and commit
  steps, re-snap from the `onTerrainChanged` queue, F2), placed pieces (S30), authored
  grass masks and their edits (S23 / S31; S22 only re-runs S20's prototype mask rule on merged heights; generator
  worlds combine the generator mask with the same rule in S23b, E3), water fill on dig (§20: global sea plane).
- The patch calls themselves, their upload ordering, node-bound widening and their Vulkan tests (S20, E2).
- Edits of non-resident or pending chunks with simulation radii (S24; in S22 every sample is evaluable on demand), the
  chunk store (S25), per-loaded-chunk snapshots, gap re-requests and `ChunkData` (S26).
- Brushes evaluated on clients, client-requested edits through game actions, undo, brush textures or falloff shapes
  other than the quartic, terraforming limits and protected areas (S44 plan).
- Drawing spaces without terrain, parallel brush evaluation, a base cache (R-PAR-1 / lazy base; added later if the
  benchmark exceeds the budget).

## Design notes

- **Integer brush, host only**: clients receive results, so host/client equality does not need a portable brush; the
  brush is integer anyway because (a) the committed result hash then holds on GCC, Clang and MSVC (CI), (b) the
  repeat variant and the MCP tools are reproducible without caveats about FMA contraction or `-ffast-math`.
- **Base data never mutated (D3)**: the `TerrainStore` shares its tile vectors with the sink and feeds `terrainHeight`
  and `TerrainSystem::sample`; patching those vectors would leak edits into the base (a host load or `Resync` rebuilding
  deltas over patched tiles would add old edits twice, and only with a sink, which CI does not exercise). Deltas are a
  separate layer: merged when sent, patched when changed.
- **Lazy base (deltas only)**: storing the base next to the deltas (round 1) made every apply path materialize: a
  full-chunk `TerrainChunkState` at 1 m is 17 x 17 blocks from the hook (about 50 ms), a late join or `TERR` load of 50
  edited chunks several seconds, and validation needed the base before anything could be allocated. With the base read
  from `TerrainSystem::sample` on demand, receive and load decode only, validation needs no base, and a block holds
  1 KB. The price is a hook call per read of an edited sample without a resident level-0 tile (as S20's miss path) and
  a hook gather per edit on headless hosts (risk below).
- **One copy per sample**: blocks keyed by global sample coordinates make shared borders consistent by construction;
  the per-chunk dense form exists only in saves, messages, hashes and comparisons, where the border rule is checked.
- **No state from reads**: the gather never allocates, so a Smooth ring or a no-op Flatten leaves no entry on the host
  that a client could not reproduce; every path allocates only blocks with a non-zero changed delta.
- **Absolute deltas in messages**: idempotent per box; a duplicate is harmless and S26's snapshot-then-live rule
  (`sequence <=` snapshot dropped) works unchanged.
- **Send list and change list instead of the world outbox and in-step events**: the world outbox and the event
  queues drop an aborted phase's entries while the applied edits of finished phases remain (§7.6); lists committed with
  the step cannot lose an edit or its event. Between-steps changes are emitted at once and skip the list, so no change
  is emitted twice.
- **No synchronous CPU pyramid write**: nothing in the simulation reads the pyramid before S27 (§12.2 says the pyramid
  serves rendering and far content), so the pyramid is merged and patched only on the visual path (amendment of §12.2);
  S27 decides how far content reads edits.
- **Caller only on the GPU side (E2)**: S20's int32 atlas has no per-tile reference, so a patch is a plain copy of
  integers and an edit below a tile's minimum needs nothing special beyond S20's bound widening.

## Implementation steps (in order; each step builds with the CI recipe and passes `ctest`)

Each step adds only tests that use code of that step or earlier ones; the test list below names the step of every
group that would otherwise be ambiguous.

1. `TerrainBrush`, `validateBrush`, `BrushKernel` with `floorDiv` / `roundShift16` and the clamp-then-compare rule
   (S20's `kTerrainHeightLimit` / `clampTerrainHeight`); `tests/test_engine_terrain_edit.cpp` with the kernel, clamp
   and `validateBrush` tests incl. the rounding reference and the committed hash (pure functions, no world).
2. `TerrainEditBlock` / `TerrainChunkEdits` / `TerrainEditStore` (delta blocks, per-chunk entries with
   `nonZeroSamples` and `revision`, the sequence accessor, dense form, content hash, `Stats`, `memoryBytes`);
   `World::terrainSample`, `terrainEdits`; `TerrainSystem::height` through `terrainSample` (the D1 early return only
   without deltas); `TerrainSystem::baseHookSamples()`; `terraformTestTemplate` in `tests/terrain_test_hooks.h`; store
   tests on the store API directly (writes, `bumpSequence`, `denseDelta`, counts, revision, `terrainSample` /
   `terrainHeight` over written deltas).
3. `gatherSamples`, `applyTerrainEdit` (steps 1-7 incl. the named S23b extension point, which stays empty),
   `editTerrain` (checks at the call; command / immediate), `onTerrainChanged` (change list only; the dirty-queue push
   comes with step 8), the change list and `pendingTerrainChanges`, `TerrainEditedEvent` (`EngineEvent::TerrainEdited`,
   emission between steps / at the commit / at the next `runStep` after an abort), the host send list in `Replication`
   (append of unencoded parts, `hostStateList()`, cleared by `Replication::commit`, kept on abort; nothing encoded or
   sent yet); statistics and the `terrain/edit` scope; `data/tests/engine/schema.json` regenerated with
   `SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_save --gtest_filter='SaveFormat.EngineSchemaIsCommitted'` (only
   that filter, so `save_v1.bin` is not rewritten); the edit-based store tests, world and event tests (except the
   host-load ones).
4. Delta codec (`TerrainDeltaCodec`); `TERR` section by S21's recipe (or the D7 mechanism itself if S22 merges first);
   load validation and commit (incl. the send-list and change-list clear), hash, `diffSaves` kind; save/load tests, the
   D1 save round trip, the host-load event test.
5. `TerrainModification` / `TerrainChunkState` encode/decode over the codec; encoding and sending of the send list in
   the replication pass (both `extract` paths, abort of phase 8); `sendChunkSnapshot` for joins and `onHostLoad`,
   client `Resync` clear; client handler (validate into scratch without a neighbour check, then reserve and apply,
   count recompute), `NetStats::modificationGaps`; `tests/test_engine_terrain_net.cpp`.
6. `ScenarioWorld` helpers, end condition and comparison; the engine scenarios.
7. World API operations and MCP tools with `handleMessage` tests.
8. Visual path (caller side only): `onTerrainChanged`'s dirty-queue push; merge in S20's `sendTile` / `sendPyramid`;
   dirty queue (view-space tile boxes, per-space pyramid boxes); patch step with row splitting and the base from the
   store's tiles; the recording-sink visual tests without grass; `TerrainEditRender.PatchBelowNodeMinimum`.
9. Edited grass masks (pool jobs, revision and desired-set check, cache, `sendGrassMask`, cap, base mask after a
   return to zero, `completeView` waiting for the mask jobs); the grass and new-sink mask tests; the golden
   `terrain_edit_crater` (image in a dedicated commit, lavapipe).
10. Benchmarks: `EngineApp::setFixedFrameTime` with its `test_engine_app` case, `tests/test_engine_terrain_benchmark.cpp`,
    the `terraform` pose in `WorldBenchmark`, `TerraformBenchmark` (per-edit timing, residency floor),
    `game_world --benchmark terraform`; measure, commit the baseline; `flyover` and `forest` before/after (R-TEST-2).
11. `game_world` terraform mode (key on `startWorld()`, MCP); screenshots (local), also with `--coop`.
12. `docs/engine_design.md` (amendments below), `AGENTS.md` (tests, test count), this file's status sections.

## Tests

`tests/test_engine_terrain_edit.cpp` (new, CI, no device; the step that adds each group in brackets):

- [1] Kernel: quantization (`floor` for negative and fractional centers, `x = -0.001` -> `-1`), weight values at the
  center (65536), at half radius (`t2 = 16384` -> `w = 36864`), near the rim (`u < 256` -> `w = 0`: affected but
  unchanged), at and beyond the radius (sample excluded); Add and Subtract change the center sample by exactly `s`;
  mirrored samples around a center on a sample get equal results; Flatten at strength 1 sets the center to `cy` exactly
  and never crosses the target anywhere; Smooth leaves a constant field and a linear ramp's interior unchanged; the
  result equals a reversed-iteration evaluation (pre-edit reads only); the rectangle is clipped at the space border
  (edit at `(0, 0)` and at `sizeChunks * 256`); tests count changed samples, not affected ones.
- [1] Smooth rounding against an exact integer reference: over a seeded noisy field (10^4 affected samples, strengths
  0.1, 0.5, 1, radii 4 and 16 m), every sample satisfies `|2 * 9 * 65536 * (h' - h) - 2 * (sum - 9 * h) * g| <=
  9 * 65536` (error at most 0.5 unit against the exact rational step), and the mean signed error
  `sum((9 * 65536 * (h' - h) - (sum - 9 * h) * g)) / (9 * 65536 * count)` lies within +-0.02 units (no sign bias).
- [1] Clamp (decided after the clamp): an Add at a sample at `+2^23` and a Subtract at `-2^23` change nothing (not counted
  as changed); an Add at `2^23 - 100` with `s = 1024` gives exactly `2^23` and counts as changed; a sample outside the
  limit at zero weight stays unchanged (pure kernel: such inputs do not occur in the engine).
- [1] `validateBrush`: rejects NaN and infinity in each field (also y for Add), radius <= 0 or > 128, strength out of
  range per op, op >= Count, x or z beyond the space by more than 128 m (`1e300` and `-1e300` included, checked before
  any quantization: UBSan clean on the Clang job), a Flatten target beyond 8192 m; accepts the bounds themselves.
- [2] Store API: delta writes allocate only blocks with a non-zero changed delta; `nonZeroSamples` follows writes in
  every chunk with an entry that contains the sample and is recomputed when an entry is created over border samples a
  neighbour's write already set; `revision` changes with every delta change of the closed range (also through a
  neighbour's shared border) and not with `bumpSequence`; `denseDelta`, the content hash, `memoryBytes`;
  `terrainSample` and `terrainHeight` over written deltas.
- [3] Store through edits: a brush on a chunk corner changes the shared samples once and increments all four sequences;
  a brush inside one chunk increments only that chunk; a Flatten on a field already at the target, a Smooth on a
  constant field and an Add on a space whose test hook returns `2^23` everywhere change nothing (no entry, no allocated
  block, no event, no send-list entry); blocks are allocated only for changed samples (a Smooth's read ring allocates
  nothing); an edit that returns a chunk's deltas to zero keeps its sequence with `has deltas` false; `bumpSequence`
  alone creates an entry without deltas; `denseDelta` of two neighbours agrees on the shared border; `terrainSample` and
  `terrainHeight` after an edit equal the §12.2 triangle rule over `clampTerrainHeight(base + delta)` at samples and at
  seeded random points incl. cell diagonals; the gather from a resident level-0 tile equals the gather from the hook
  (`baseHookSamples` 0 vs the read rectangle's size).
- Space without terrain (D1): [3] an edit on the default template (base 0) changes `terrainHeight` (the early return is
  gone once a chunk has deltas); [4] it round-trips through save/load.
- [3] World: `editTerrain` inside a phase-4 system applies at the phase-4 flush (a later system of the same phase sees the
  old height, phase 5 the new one); two edits in one phase apply in recording order; between steps at once; death tests
  (every build, `SAIGA_EXIT_ERROR`): `editTerrain` on a client, an invalid brush, an unknown space, an enclosed space
  (S21 if present); each also from inside a phase-4 system, where the program stops at the call (the message names
  the field or role), before anything is recorded.
- [3] Events (one path per change): a between-steps edit emits exactly one `TerrainEditedEvent` (readable in the next
  step, never in `pendingTerrainChanges`); an in-step edit appears in `pendingTerrainChanges` (read by a later phase-5
  system of the same step) and emits exactly one event at the commit (readable in the next step) with the changed box;
  regression for the aborted step: an edit applied at the phase-3 flush, then phase 4 throws; between steps
  `pendingTerrainChanges` still lists it; the next `runStep` emits it before `events.beginStep()`, so it is readable in
  that retried step; with a second abort of the retried step it is readable in the step after; over the whole run
  exactly one event per edit; [4] a host load clears the list and emits nothing.
- [4] Codec: round trips of rectangles (1 x 1, a part box, `N x N`) raw and zstd; every rectangle decodes on its own
  (the predictor starts at 0); the bounds (size, count, `|delta| <= 2^24`, bytes left over) are refused.
- [4] Save: round trip with edits in two chunks (equal `save()` bytes after the load, equal `hash()`); a chunk with a
  sequence and no deltas (`bumpSequence`, and deltas returned to zero) round-trips as a `TERR` entry without deltas;
  no `TERR` without chunks with `sequence > 0` (save bytes and `hash()` equal to those of a world of the same build
  without terrain edits; not to a pre-S22 world, whose event queues and schema lack `TerrainEdited`); `save_v1.bin`
  still loads; the section sits at its D7 position; the load calls no hook (`baseHookSamples`
  unchanged) and allocates only blocks with non-zero deltas; tampered sections (wrong `N`, wrong spacing, chunk out of
  range, unsorted chunks, sequence 0, unknown flag bits, deltas present but all zero, bad zstd, oversized values,
  `|delta| > 2^24`, unequal shared borders, a non-zero border sample without the neighbour's deltas) are refused with
  the world unchanged; `hash()` changes with an edit; `diffSaves` reports a differing chunk with its first sample.
- Visual (recording sink, `TerrainSystem` with a test camera, `terraformTestTemplate(2)`; [8] unless marked [9]):
  - after an edit the next `updateView` records a level-0 `patchTerrainTile` with `base + delta` for the changed box and
    coarse patches with the coincident samples, and a `patchTerrainPyramid`; the recording sink's copies then equal the
    merged heights; the `TerrainStore`'s tiles are unchanged (base, bitwise); the patch step calls no hook
    (`baseHookSamples` unchanged during `updateView`);
  - set then patch in one frame: an edit while a tile's result is pending; the frame that sends the (merged) tile also
    patches the queued box over it; the recording holds `setTerrainTile` before `patchTerrainTile` and the copy equals
    the merged heights;
  - a tile result requested before an edit and sent after it is a merged copy with exact `minHeight`/`maxHeight`; a new
    sink set after edits receives merged tiles and pyramid; [9] and the cached edited masks (never a base mask of an
    edited chunk);
  - a budget of 1000 samples spreads a large edit over several frames by rows and every sample arrives exactly once;
    `completeView` drains the whole queue in one call (no budget); [9] and waits for the mask jobs;
  - an edit in a space that is not the view space queues no tile boxes, only pyramid boxes, and the pyramid sent at the
    switch is merged (with S21 present; else a test of the per-space queue with a second template);
  - a host load re-patches the tiles of the chunks edited before and after (no eviction; the sink's copies equal the
    loaded state);
  - [9] grass: a mask of an edited chunk within the grass radius is recomputed on the pool from merged heights (a
    crater wall steeper than the slope rule loses grass); at most `editMasksPerFrame` submissions per frame; a second
    edit between submission and completion drops the first result (`maskResultsDropped`), re-queues, and the final
    mask matches the final heights; a `bumpSequence` alone between submission and completion keeps the result (F1); a
    result for a chunk that left the desired mask set (camera moved away) is discarded without a re-queue; an edit
    that returns a chunk's deltas to zero drops its cache entry and sends the store's base mask;
  - nothing is queued without a sink.
- [7] World API / MCP (`Poller`): `edit_terrain` (result incl. chunk-local boxes, `api` log line, errors: client, inside a
  step, invalid brush, center out of the space, enclosed space), `terrain_height` (`edited` flag, exact value),
  `terrain_edits` (paging, `has_deltas`); read-only registration has no `edit_terrain`.

`tests/test_engine_terrain_net.cpp` (new in step 5, CI, loopback pairs and networks):

- Encode/decode round trips (raw and zstd, `TerrainChunkState` with and without deltas), every validation failure (incl.
  unequal shared samples between parts of one message, sequence 0, `|delta| > 2^24`, a truncated zstd stream after
  valid parts) counted in `rejected` with nothing applied: no block allocated, no entry created, no hook called
  (`baseHookSamples` unchanged).
- Host authority over neighbours (regression: a neighbour check rejected the first of two adjacent edited chunks of a
  snapshot): a `TerrainChunkState` whose border differs from a stale present neighbour, and one whose neighbour is
  still absent, are applied; after the neighbour's own state both chunks' dense deltas, `nonZeroSamples` and
  sequences equal the host's.
- A host edit reaches the client in the next step at delay 0 and after 2 more steps at delay 2; the client's samples and
  sequences match; a 4-chunk edit arrives as one message, applies atomically, and is **one** `TerrainEditedEvent` on the
  host and one on the client with the same box; the client's `pendingTerrainChanges`, read by an in-step phase-4
  system in the step the message applies, holds that one box.
- Applying a part whose new deltas in a block are all zero (an edit back to the base in a not yet allocated block on the
  client after a `Resync`) allocates no block.
- Duplicates (`sequence <= current`) dropped; an injected gap is applied, logged and counted in `modificationGaps`; a
  message of an old epoch is dropped.
- Edits between steps are sent by the next step's phase 8, and by a host `stepNetworkOnly` (`extract(true)`) when that
  runs first; an aborted phase 4 after an applied edit in phase 3 keeps the entry and the next step sends it; an aborted
  phase 8 resends; a host without peers keeps no entries.
- A late joiner receives, after the spawn set, `TerrainChunkState` for every chunk with `sequence > 0` (incl. one
  without deltas made by `bumpSequence`, and the four chunks of a corner edit, whose shared samples arrive in four
  messages) and ends with the host's state (dense deltas, sequences, `nonZeroSamples`), also with an edit in the
  joining step; a host load with a client sends `Resync` then the states (`sendChunkSnapshot`) from `onHostLoad` before
  the next step, and the client ends with the loaded state (incl. a 4-chunk corner edit in the save, lower sequences,
  and a save without edits: the client's entries are cleared and it emits an event per cleared chunk with deltas).

`tests/test_engine_scenarios.cpp` (step 6; engine scenarios, all variants incl. save/load, repeat and host + client
delay 0, 2 and async):

1. `terraform_dig_and_fill` (`terraformTestTemplate(1)`): a test-only game object `Terraformer` (`TerraformerState
   {TerrainBrush brush; uint32_t remaining}`, its `update` calls `world().editTerrain` until `remaining` is 0) raises a
   mound by 5 x 1 m (Add, radius 6 m) at the sample (1000, 1000) in steps 10-14 (command path, phase 3); in step 20
   the scenario digs a 2 m pit with radius 12 m on the chunk corner (256, 256) through `call("edit_terrain", ...)`
   (immediate path); in step 30 it flattens the spot (2000, 600) (radius 8 m) to its base height + 3 m (strength 1)
   with `editTerrain` and checks in the same callback, right after the immediate edit, that the host's sample at the
   center equals the target exactly (a check of the state at that point, so it holds wherever the save/load swap
   falls); in step 31 it smooths the same spot (strength 1, radius 8 m), which changes samples of the bump the flatten
   left. All three spots lie inside single chunks away from the corner. Checks at the end (they read only the final
   state, so they hold wherever the runner places the save/load swap): the mound's center is exactly base + 5 m and its
   chunk's sequence 5, the pit's center exactly base - 2 m and the samples at 12 m unchanged, the four corner chunks'
   sequences are 1, the flatten spot's chunk sequence is 2 (flatten and smooth both changed it). Networked variants:
   the end comparison proves equal heights and sequences on every client.
2. `terraform_player_walks_into_crater` (`terraformTestTemplate(2)`): a player (client-owned in the networked variants)
   stands 20 m from a spot the host digs (Subtract 3 m, r 10 m) and walks into it (`move`); `stepUntil` (bounded) the
   player's y equals the host's `terrainHeight` at its position within 1e-4 m and lies at least 2 m below the base
   ground: the mover used the edit (on the client, the replicated one).

`tests/test_engine_terrain_render.cpp` (engine-linked render tests, D2: inside `MODULE_ENGINE` and the Vulkan guard;
S20 creates it, E12), CI on lavapipe. S22 adds no `test_vulkan_terrain` cases (the patch calls are S20's, E2):

- [8] `TerrainEditRender.PatchBelowNodeMinimum` (the one engine-path check of what S20's tests do not cover): a `World`
  with `terraformTestTemplate(2)` (S20's `integerTestHook`) and a `VulkanRenderSink`; after `completeView`, a pit 30 m
  below the tile's minimum is dug and `updateView` frames run until the dirty queue is empty and
  `terrainUploadsPending() == 0`; `TerrainAtlas::readLayerForTests` of the level-0 tile equals the merged heights; S20's
  probe (`TerrainPass::probeVertices`) at the pit's center sample returns the patched height (S20's tolerance) and its
  level-0 node is selected in that frame (not culled by the old bounds).
- [9] Golden `terrain_edit_crater` (the crater's grass depends on the edited masks): the same world; a crater (Subtract
  6 m, r 20 m) applied before the first `completeView` (merge path) and a raised ring (8 Add edits of 2 m, r 6 m, on a
  24 m circle around the crater) applied after it, followed by `updateView` frames until the queue is empty (patch
  path); compared with `data/tests/golden/terrain_edit_crater.png` (created on lavapipe in a dedicated commit, AGENTS.md
  recipe).

Benchmarks (R-TEST-2): `tests/test_engine_terrain_benchmark.cpp` (non-gating), `game_world --benchmark terraform`
(local, twice), `flyover` and `forest` before and after (local).

## Done when

- The S22 row of §22: terraform scenarios pass incl. the save/load variant; host + client end with equal heights at
  delay 0 and 2 (and async) (CI); the terraforming benchmark (integer result, encoding, visual patching) within the
  budget of §17.10 as amended, with its assertions (4 chunks per Add/Subtract edit, `chunks_edited >= 1200`, no hook
  reads, no tile sends in the measured frames) holding (local, reported in the PR with the baseline).
  **Done** locally: all six variants of both scenarios pass; the benchmark's p95 values are 0.039 ms per edit and
  0.015 ms per frame, every assertion held, the baseline is committed (Results). CI to confirm.
- The committed kernel hash passes on all CI compilers; `TerrainEditRender.PatchBelowNodeMinimum` and the golden
  `terrain_edit_crater` pass on lavapipe (CI). **Done** locally with GCC 15 and Clang 19 and on the local lavapipe
  (MSVC and CI's lavapipe: CI).
- `flyover`/`forest` within 5 % of the baseline or explained. **Done**: within 1 % of a reference build of the plan
  commit (Results).
- `game_world` terraform mode screenshots reviewed (local); the tech demos' `--scenarios` still pass (local). **Done**
  (the crater through `edit_terrain`; the T mode by hand not exercised without an interactive session); flight,
  platformer and racing scenarios pass.
- `docs/engine_design.md` updated (amendments below), `AGENTS.md` (new tests, test count), this file's Status, Design
  changes and Results. **Done.**

## Verification

- CI recipe of AGENTS.md, then `cd build && ctest --output-on-failure --parallel 4`;
  `ctest -R "test_engine_(terrain_edit|terrain_net|scenarios|save|replication|app)"`; the render tests on lavapipe:
  `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R test_engine_terrain_render`.
- Samples (Clang configuration with samples ON): the demos' `--scenarios`.
- Benchmarks: the bench recipe of AGENTS.md, `cmake --build build-bench --target game_world`, then in
  `build-bench/samples/games` (where `game_world` is built): `./game_world --benchmark terraform` (twice),
  `--benchmark flyover`, `--benchmark forest`; compare with `data/tests/benchmarks.json` (the #66 summary for the
  common keys, the terraform lines for the rest).
- Local look: `SAIGA_SCREENSHOT=/tmp/terraform.png ./game_world` after `edit_terrain` calls through MCP (the
  `saiga-world` server of `game_world`), and the `T` mode by hand; reviewed.

## Files

`src/saiga/engine/terrain/{TerrainBrush.h,BrushKernel.{h,cpp},TerrainEditBlock.h,TerrainChunkEdits.h,
TerrainEditStore.{h,cpp},TerrainEdit.{h,cpp},TerrainDeltaCodec.{h,cpp},TerrainMessages.{h,cpp},
TerrainVisualPatch.{h,cpp},TerrainDirtyQueue.{h,cpp}}`, `src/saiga/engine/terrain/TerrainSystem.{h,cpp}` (S20's:
`height` through `terrainSample`, `baseHookSamples()`, the merge in the send point, the patch step, edited masks,
settings and stats), `src/saiga/engine/app/EngineApp.{h,cpp}` (`setFixedFrameTime`),
`src/saiga/engine/world/{World.h,World.cpp,Events.h}` (event, change list, commit/abort, `runStep`'s emission before
`beginStep`), `src/saiga/engine/net/{NetIds.h,EngineMessages.cpp,Replication.{h,cpp}}` (send list,
`sendChunkSnapshot`), `src/saiga/engine/save/{SaveFormat.h,SaveReader.{h,cpp},WorldSave.cpp,WorldLoad.cpp,
SaveDiff.cpp}`, `src/saiga/engine/api/WorldApi.{h,cpp}`, `src/saiga/engine/tools/{WorldTools.cpp,
ScenarioWorld.{h,cpp},ScenarioRunner.cpp}`, `src/saiga/vulkan/world/WorldBenchmark.{h,cpp}` (the `terraform` name and
pose), `samples/games/world/{main.cpp,TerraformBenchmark.{h,cpp}}`, `tests/test_engine_terrain_edit.cpp`,
`tests/test_engine_terrain_net.cpp`, `tests/test_engine_terrain_benchmark.cpp`, `tests/test_engine_scenarios.cpp`,
`tests/test_engine_terrain_render.cpp` (S20's file, S22's cases), `tests/terrain_test_hooks.h` (S20's file:
`terraformTestTemplate`), `tests/test_engine_app.cpp` (the fixed frame time), `tests/CMakeLists.txt`,
`data/tests/engine/schema.json`, `data/tests/golden/terrain_edit_crater.png`, `data/tests/benchmarks.json`,
`docs/engine_design.md`, `AGENTS.md`, this file. Not changed: `ChunkId.h` (`kTerrainHeightLimit` is S20's), the
sinks, `RenderScene`, `TerrainAtlas`, `test_vulkan_terrain.cpp` (S20, E2).

## Risks

- **S20's / S21's merged interfaces** (the send point incl. `sendGrassMask`, re-send on a space switch, the patch
  calls, `setCompleteViewEachFrame`, `integerTestHook` / `testSpaceTemplate`, the desired mask set, the
  optional-section recipe) may differ in detail from "S20 interfaces used": the plan is updated against the merged
  code before step 1.
- **Client transient during a snapshot**: without a neighbour check (host authority), a chunk's border samples read
  the new values as soon as a neighbour's `TerrainChunkState` applies, before the chunk's own state arrives (normally
  in the same receive; at most the snapshot's own delivery time). The end state is the host's; the comparison of
  every host + client scenario checks it.
- **Hook reads without resident tiles (lazy base)**: on headless hosts every edit gathers its read rectangle from the
  hook (prototype hook, 0.7 us per sample: r 16 m at 1 m is 33^2 samples, about 0.8 ms; r 128 m is 257^2 samples,
  about 46 ms), and every read of an edited sample costs a hook call (as S20's unedited miss path). In rendering apps
  the brush area is in view and the gather copies from the resident tiles (asserted in the benchmark). Reported by the
  headless benchmark test; the fallback is a per-block base cache filled on the pool, or `parallelFor` with an R-PAR-1
  amendment. Receive, join and load never call the hook (the round-1 materialization stall of seconds for a late join
  with 50 edited chunks is gone; measured in the headless test).
- **Bandwidth**: the benchmark's 60 edits/s may exceed the 256 kbit/s per-client report value; it is reported, not
  gated (§17.10); delta encoding plus zstd keeps a typical edit at a few hundred bytes.
- **Grass transients**: a chunk with deltas shows no grass between a new sink or a re-send and the completion of its
  mask job (a few frames); a dropped result (revision changed) delays the mask by another job.
- **Prototype trees** in `game_world` keep their heights over craters and mounds until S23b's re-snapping (visible in
  the terraform mode; accepted, noted in the PR).
- **Conservative bounds** only widen until a tile is set again (S20); many edits in one tile can make its node bounds
  loose (slightly more nodes drawn, deeper skirts). Acceptable; exact recompute on the next send.
- **Pits below the sea level** show the global water plane above them (§20 until S39).
- **Hash cost** of edited chunks is paid only when `hash()` is called and only for dirty chunks; scenarios that edit
  every step re-hash one to four dense tiles per step (about 0.3 ms each at 1 m spacing), acceptable for tests.
- **zstd determinism**: `save()` bytes are equal within one build (fixed level, same library), which is all §15.2
  requires.

## Open questions and proposed design-doc amendments

Open questions (decided in the plan review):

1. Channel: `State` (chosen, ordered with `Resync`) or `ChunkData` with its own epoch handling now?
2. Engine safety bounds `kMaxBrushRadius = 128 m`, `kMaxBrushStep = 64 m`: constants, or `SpaceTemplate` parameters?
3. One fixed falloff (quartic) and a 3 x 3 Smooth kernel, or a falloff field in `TerrainBrush` now (changes §12.2's
   struct)?
4. `TerrainEditedEvent` as the broadcast for game reactions, plus `onTerrainChanged` / `pendingTerrainChanges` for
   engine consumers (chosen), or an engine listener API?
5. Gap handling before S26: logged + counted + applied (chosen), or a fatal error?
6. Lazy base (chosen: deltas only, hook reads without resident tiles) or a per-block base cache filled on the pool?

Answered in review round 1: tech demo scope (D14: `game_world` terraform mode by key and MCP only, no flight craters);
coarse levels and the pyramid are point-sampled (S20). Answered in review round 2: the height limit `+-8192 m` is S20's
(E7, S20's A18); the GPU format and the patch implementation are S20's (E2).

Amendments of `docs/engine_design.md` (applied with the stage, R-DOC-2):

- §12.2: `TerrainBrush` with `Count`, the field semantics (y only for Flatten; strength units per op), the bounds checked
  before quantization (incl. the center within the space ± `kMaxBrushRadius`), the quantization, the quartic integer
  falloff, the op formulas and rounding (Smooth in one rounding to the nearest), the strict-inside rule, pre-edit reads,
  the clamp to S20's `kTerrainHeightLimit` with "changed" decided after the clamp; `kMaxBrushRadius`, `kMaxBrushStep`;
  edits never change base tile data: deltas are merged at the send point and patched afterwards through S20's
  `patchTerrainTile` / `patchTerrainPyramid`, grass masks of edited chunks recomputed from merged heights; "the
  pyramid's CPU copy is updated synchronously with the edit" is dropped (nothing in the simulation reads it before
  S27); `World::terrainSample` (base + delta, clamped), `World::terrainEdits`; edits are allowed on open-world spaces
  without terrain (base 0, not drawn); role, space and brush checked at the call; "the edit reports the affected
  tier-1 instances and pieces" -> S22 reports the changed box through `TerrainEditedEvent` and `onTerrainChanged`,
  S23b adds its tombstones (computed in the edit's reserve step, written in its commit) and re-snaps from
  `onTerrainChanged`, S30 adds pieces.
- §12.1: the in-memory modification state (16 x 16 delta blocks keyed by space sample, one copy per sample, no stored
  base; per-chunk entries with the sequence), dense per chunk in saves and messages; a sequence increments for every
  chunk whose closed sample range contains a changed sample (and for S23b's tombstones); a chunk has state iff its
  sequence is > 0; the sequence stays in `TerrainEditStore` for all spaces (E10); a local terrain revision per chunk
  (changed only with its deltas, not saved or sent) keys visual staleness such as the edited grass masks (F1).
- §7.7 / §18.2: `TerrainEditedEvent` (`EngineEvent::TerrainEdited`, `Replicate::Never`, host and clients; one event per
  applied change: at once between steps, at the commit for in-step changes, at the next step's start after an abort).
- §8.1: phase 8 (host) sends the ordered host state list (terrain modifications, S23b's tombstones); clients apply them
  in phase 1 (exists).
- §13.1 / §13.2: the S22 messages `TerrainModification` and `TerrainChunkState` (sequence, optional deltas) on
  `Channel::State` with the epoch; S22's minimal snapshot `sendChunkSnapshot` for late joiners (after the spawn set) and
  host loads (from `onHostLoad` after `Resync` and S23a's `GenerationState`); the delta codec (row-major zigzag varint
  differences, predictor reset per part and chunk); clients validate each message on its own (no check against
  present neighbours: the host is authoritative); the client's gap rule before S26 (logged, counted, applied); S26
  moves snapshots to `ChunkData` with re-requests.
- §13.6/§17.3: `NetStats::modificationGaps`; the host + client end condition and comparison include the terrain
  modification state of every scenario.
- §15.1: the optional `TERR` section (layout above, section version 1, position of D7; per chunk the sequence and
  optional deltas); its chunk payloads are zstd compressed (an exception to "blobs are uncompressed"); §17.4:
  `world.hash()` covers the terrain modification state from S22 (not S25), nothing is appended without chunks with a
  sequence (the new event queue still changes every hash value against pre-S22 builds); `diffSaves` terrain kind.
- §16.2: `edit_terrain`, `terrain_height`, `terrain_edits`.
- §17.2: `ScenarioWorld::editTerrain`, `terrainHeight`, `terrainSample`.
- §8.3: `EngineApp::setFixedFrameTime(double dt)` (the worlds advance by a fixed time per frame inside `run()`; the
  frame statistics keep the wall clock); item 3 of "One frame" names it next to `runFrames`.
- §17.7: the `terraform` benchmark definition (opt-out of the per-frame `completeView`, one step per frame through
  `setFixedFrameTime`, the residency floor of 600 m, edits between steps timed per edit, the corner layout with its
  asserted chunk counts, the camera, separate encoding) and JSON keys on top of #66's and S20's terrain keys,
  `tests/test_engine_terrain_benchmark.cpp`.
- §17.10: `terrain_edit_ms` p95 <= 1 ms per edit and `terrain_visual_ms` p95 <= 1 ms per frame (desktop reference,
  release; the step budget excludes edits between steps); edit bandwidth reported against the upload budget.
- §18.4: `game_world` terraform mode (key and MCP; edits go to `startWorld()`, so `--coop` works); §3, §22 (S22
  status).

Contradictions found while planning (resolved by the amendments above):

- §13.2 brings modification snapshots with S26, but S16 already supports late joiners and host loads with clients;
  without a minimal snapshot in S22 a late joiner would miss every earlier edit.
- §17.4 adds chunk states to `hash()` only from S25; the repeat variant would not check terrain determinism in S22-S24.
- §13.1 assigns large reliable messages to `ChunkData`, which is not ordered with `Resync` (State); S22 keeps terrain
  on State.
- §12.2 "its CPU copy is updated synchronously with the edit": the pyramid is base data shared with the sink (D3); no
  simulation consumer before S27.
- §12.2 "the edit reports the affected tier-1 instances and pieces": nothing to report before S23b/S30.
- §15.1 "blobs are uncompressed": `TERR` payloads are zstd.

## Design changes

The design-doc amendments listed above are made in `docs/engine_design.md` (§3 rows, §7.7, §8.1, §8.3, §12.1, §12.2,
§13.2, §13.6, §15.1, §16.2, §17.2, §17.3, §17.4, §17.7, §17.10, §18.4, §22). Changes against the plan made during the
implementation (each the smallest sound fix; details in the review log):

- Names of the merged S20/S21 code: the grass rule is `generateGrassMask` (the plan's `prototypeGrassMask`, which is
  `saiga_vulkan`'s; the engine wraps it), `SpaceTemplate::sizeMeters()`, `TerrainSystem::store(space)`.
- `SampleRect` (the inclusive sample rectangle of the kernel, the changed boxes and the dirty queue) is its own header
  `engine/terrain/SampleRect.h`; `TerrainChange {space, box}`, `TerrainPart` and `TerrainEditResult` are in
  `TerrainEdit.h` (small value types).
- `World::checkTerrainEdit(space, brush, error)` (the non-fatal form of `editTerrain`'s checks, used by the world API and
  the terraform mode), `World::terrainEditsForUpdate()` (engine-internal: the edit path, the handlers, loads, tests),
  `World::reserveTerrainChange(n)` (the change-list capacity of the reserve steps), `Replication::clearHostStates()`
  (a host load).
- `TerrainEditStore` keeps reserve and commit apart with `reserveFor` / `reserveBlock` (allocation) and `makeEntry` /
  `insertEntry` (a `std::map` node allocated in the reserve step and inserted without allocation); a load builds a
  staged store and swaps it in (`replaceState`, the counter stays monotone).
- `TerrainModification` parts carry a varint length before their encoded deltas (the plan's table had none; needed to
  delimit a zstd frame per part); zstd is chosen for messages with at least 256 values.
- The gap of a received part is logged with `SAIGA_LOG("net", ...)` (`SAIGA_LOG_E` needs an entity).
- The flush scope is `step/phase:<p>/commands/terrain_edit` (the plan's `terrain/edit` would need a parent scope).
- `RecordingRenderSink` logs `patchTerrainTile` / `patchTerrainPyramid` in its opt-in call log (the "set then patch in
  one frame" test reads the order); nothing else in the sinks changed.
- The visual time is `TerrainSystem::Stats::lastVisualMs`, added by `EngineApp` to the frame profile as
  `frame/render/terrain_visual`; further stats `patchedSamplesLastUpdate`, `maskJobsLastUpdate`, `maskQueue`,
  `maskJobsInFlight`, and `TerrainSystem::dirtyQueue()` / `visualPatch()` for tests.
- The headless benchmark test runs the schedule on a 32-chunk template (the corners at 3840..4096 m lie inside;
  `terraformTestTemplate` has 16 chunks, whose last corner is the space corner).
- The scenario `terraform_dig_and_fill` used the `editTerrain` helper for the pit in step 6 and switched to
  `call("edit_terrain", ...)` in step 7 (the tool comes with step 7).

## Results

Measured on the RTX 5080 Laptop GPU (driver with validation for the tests) and on lavapipe (Mesa with LLVM 21.1.8).

- **Tests.** ctest 77/77 on the GPU (local build with samples; with the CI options 75 tests: 72 + `test_engine_terrain_edit`
  (45 cases), `test_engine_terrain_net` (9), `test_engine_terrain_benchmark` (3)); every `test_engine_*` /
  `test_vulkan_*` on lavapipe passes (37/37). The committed kernel hash passes with GCC 15 and Clang 19 (a Clang
  `-Werror` build of the engine, the touched tests and `game_world`; MSVC is left to CI). The terraform scenarios
  pass in all six variants (Plain, Repeat with `varyThreads`, SaveLoad with `strictContinuation`, HostClient,
  HostClientDelay, HostClientAsync); the end comparison now covers the terrain state of every scenario.
- **Golden** `terrain_edit_crater` (lavapipe; a crater merged at the first complete point, a ring of 8 mounds patched
  afterwards, the edited grass masks): created in its own commit, identical over repeated lavapipe runs; the RTX differs
  by mean 1.35/255 (not gating). `TerrainEditRender.PatchBelowNodeMinimum` passes on both devices.
- **Scenarios of the tech demos**: flight 18, platformer 36, racing 18 runs passed (`--scenarios`, incl. the host +
  client variants and their new terrain comparison); `game_world` has no scenarios.
- **Screenshots** (local, reviewed): `game_world` headless over MCP: a crater (subtract 18 m, r 45 m), a ring of ten
  mounds, a smooth and a flatten through `edit_terrain` (`s22_world_crater.png`; the prototype trees keep their heights
  over the crater: the S23b risk); `game_world` (the World window with the terraform toggle), flight, racing and
  platformer `--demo` as before; with `--coop` `edit_terrain` answers "editTerrain on a client" as planned (the T mode
  by hand was not exercised: no interactive session).
- **Terraform benchmark** (build-bench, Release without asserts and `-march=native`, two runs; baseline committed in
  `data/tests/benchmarks.json`): 600 edits, 2396 chunks (every Add / Subtract edit 4 chunks; Flatten 4.0, Smooth 3.97 per
  edit), 179 436 samples changed, no no-op edits; `terrain_edit_ms` median / p95 0.020 / 0.039 (budget p95 <= 1 ms);
  `terrain_encode_ms` 0.020 / 0.027; `terrain_visual_ms` 0.011 / 0.015 (budget p95 <= 1 ms); patched samples per frame
  511 / 1369, upload bytes 2504 / 6032; `base_hook_samples` 0, no tile sends or pending requests in the measured
  frames, `terrain_stream_ms` 0; step median / p95 0.0023 / 0.0028 ms; `edit_bytes_per_second` 48 481 (above the 32 000
  B/s = 256 kbit/s report value at 60 edits/s, reported, not gated); edit store 55 KB; frame interval 1.00 ms (983 fps),
  CPU busy 0.20 ms, GPU busy 0.28 ms.
- **Headless benchmark test** (`test_engine_terrain_benchmark`, RelWithDebInfo, no resident tiles): the schedule with the
  integer hook: `applyTerrainEdit` 0.024 / 0.049 ms, the kernel 0.004 / 0.013 ms, encode 0.016 ms, client decode +
  apply 0.016 ms; with the prototype hook (the worst case: 502 hook samples per edit) 0.21 / 0.43 ms per edit and
  0.35 us per edited `terrainSample`. 50 full-chunk edited chunks at 1 m: `hash()` 13.7 ms (cached afterwards 0.003
  ms), save 5.7 ms (40 KB), load 65 ms and a late join's client apply 65 ms, both without a hook call. A grass mask from
  merged heights: 0.84 ms (1 m), 0.21 ms (2 m). The patch step on a recording sink with resident tiles: terrain/visual
  0.080 / 0.158 ms per frame.
- **flyover / forest** (R-TEST-2; build-bench; before = a reference build of the S22 plan commit `115548d2`, run
  interleaved with S22, two runs each; medians of the runs):

  | | before | after |
  |---|---|---|
  | flyover `cpu_ms` (interval) | 3.88 / 3.87 | 3.89 / 3.92 |
  | flyover GPU busy / trees / terrain + water | 2.53 / 1.40 / 0.19 | 2.51 / 1.39 / 0.19 |
  | flyover `frame_work_ms` / `cpu_work_ms` | 0.220 / 0.184 | 0.221 / 0.184 |
  | forest `cpu_ms` | 2.180 / 2.181 | 2.187 / 2.188 |
  | forest GPU busy / trees / grass | 2.19 / 0.55 / 0.68 | 2.19 / 0.55 / 0.68 |
  | forest `frame_work_ms` / `cpu_work_ms` | 0.170 / 0.136 | 0.168 / 0.137 |

  All within 1 %. An earlier pair of S22 runs right after other benchmarks showed forest `frame_work_ms` 0.22 against
  0.17 of an older build; the interleaved runs of the same machine state show no difference (it was run-order noise).

## Review log

Plan review rounds and implementation reviews, each with the findings and how they were resolved.

- **Implementation** (2026-11-21, Claude Opus 5.5): the plan was checked against the merged S20 and S21 code before
  step 1. Deviations (see Design changes): the merged names (`generateGrassMask`, `sizeMeters`, `TerrainSystem::store`);
  S21's optional-section recipe existed and `TERR` was added by it (`kTagTerrain`, the table entry after `SPCE`, a raw
  `ParsedSave::Range`, the loader pass, the `diffSaves` case; S21's order test named `TERR` as an "unknown" tag and was
  changed to `ZZZZ`); the varint length per message part; `SAIGA_LOG` for gaps; the flush scope name; the recording
  sink's patch log lines; the step 6 / 7 order of the scenario's `call("edit_terrain")`; the 32-chunk template of the
  headless benchmark test; Clang fixes found by a local Clang build (an `optional::emplace` of the nested `Range`
  aggregate, a lambda capture). The "before" numbers of flyover / forest come from a separate release build of the plan
  commit (`git archive` of `115548d2` with the submodules linked), run interleaved with the S22 build.

- **Review round 1** (2026-11-21, local Opus + Fable, 20 merged findings; cross-plan decisions D1-D14):
  1. Edited data leaked into S20's shared base tiles/pyramid -> fixed (D3): `TerrainStore` never mutated; deltas merged
     at every send point and patched afterwards; no synchronous CPU pyramid write (§12.2 amended); a load re-queues
     dirty boxes instead of evicting.
  2. Sink call did not fit S20's atlas -> fixed (D4): `patchTerrainTile` / `patchTerrainPyramid` with int32 heights;
     the atlas converts with its fixed page-table reference and widens node bounds at every affected level;
     `RecordingRenderSink` applies patches; water widening removed (one plane).
  3. Read allocation created host-only state -> fixed: the gather never allocates; blocks only for changed samples;
     a chunk has state iff `sequence > 0`; hash, `TERR`, comparison and API iterate records only; tests for no-op
     Flatten and Smooth.
  4. Unbounded center before quantization -> fixed (D12): `validateBrush` takes the space and bounds x/z, Flatten y,
     all before quantizing; tests incl. `1e300`.
  5. Hook-less spaces -> fixed (D1): edits allowed with base 0, early return only without records; scenarios use
     `terraformTestTemplate` with a cheap integer hook and stated spacing.
  6. Module boundaries -> fixed (D2): golden in the engine-linked `test_engine_terrain_render.cpp`; the schedule in
     `samples/games/world/TerraformBenchmark`; only patch tests in `test_vulkan_terrain`.
  7. Benchmark did not measure its claims -> fixed (D9): brushes orbit chunk corners (every edit crosses a border,
     ~18 % cover 4 chunks), one step per frame via `runFrames`, 60 edits/s stated, encoding reported separately,
     absolute camera pose, no streaming in measured frames.
  8. Materialization cost -> fixed: materialize from the resident level-0 tile when present, reported separately
     (`terrain_materialize_ms`, headless test with the prototype hook).
  9. Duplicated borders -> fixed: blocks keyed by global sample, one copy per sample; dense per chunk only for save,
     messages, hash; border rule checked in loads and messages.
  10. Kernel text -> fixed: `w` in `[0, 65536]`; clamp only changed samples; base checked against the limit at the
      gather; tests count changed samples.
  11. Smooth bias -> fixed: `floorDiv(2 * sum + 9, 18)`; 100-pass mean test. (`roundShift16` ties documented as toward
      +infinity, which is what `floorDiv(v + 32768, 65536)` does.)
  12. Client violation and event location -> fixed: `SAIGA_EXIT_ERROR` for role/space/brush; `EngineEvent::TerrainEdited`
      in `world/Events.h`, IDs appended in merge order (D10).
  13. Aborted step lost the event -> fixed: in-step changes go to a change list emitted at the step's commit, kept on
      abort; regression test.
  14. Placeholder S20 interfaces -> fixed: "S20 interfaces used" with the real names; `editSamplesPerFrame` in
      `TerrainViewSettings`; `TerrainSystem::height` through `terrainSample`; lock-free read rule; Files list fixed.
  15. S21 interplay -> fixed (D3/D5): pyramid boxes per space, kept across switches; merged pyramid at send.
  16. Host load/join details -> fixed: states sent from `onHostLoad` right after `Resync`; client `Resync` clear and
      state application go through `onTerrainChanged` (events, dirty queue); TERR position per D7; empty hash
      contribution without records; `SaveReader` in Files; §15.1 zstd amendment; sequence behind an accessor.
  17. Grass after edits -> fixed: masks of edited chunks computed from merged heights at send and recomputed for
      patched chunks (`editMasksPerFrame`).
  18. Scenario wording and budget splitting -> fixed: checks read only the final state; boxes split by rows.
  19. S23 consistency -> fixed: design note (c) dropped; `onTerrainChanged` and `terrainChangesThisStep` expose
      applied changed boxes (D13); gap counter named `modificationGaps`.
  20. Demo scope -> fixed (D14): key + MCP only; `PlayerAction`, `terraformSystem`, `data/world/terraform.json`, the
      game scenario and flight craters dropped; `game_world --scenarios` is no longer a Done-when item.
  Rejected: none.
- **Review round 2** (2026-11-21, local Opus + Fable, 13 merged findings; cross-plan decisions E1-E14):
  1. Old S20 renderer (R32F, `{layer, minHeight}`, float conversion, fixed per-tile reference) -> fixed (E2): interface
     item 7 and section 6 describe S20's int32 atlas; S22 is only the caller (merge, dirty queue, patch step, masks);
     the fixed-reference design note, its §12.2 amendment and contradiction, "float only in GPU tile data" and the
     `test_vulkan_terrain` patch tests dropped; one engine-path check `TerrainEditRender.PatchBelowNodeMinimum` (node
     stays selected, probe returns the patched height); upload ordering is S20's; a recording-sink "set then patch in one
     frame" test added.
  2. Benchmark geometry -> fixed (E9): 4 corners with an Add/Subtract and a Flatten/Smooth brush each, orbit radius
     `0.5 r` (the corner sample is always changed: exactly 4 chunks per Add/Subtract edit, asserted; `chunks_edited >=
     1200` asserted); levelling brushes work against the roughening ones (no convergence), no-ops reported; corners
     pulled in to x, z in 3840..4096 with a camera that sees every brush within 400 m, `residencyRange0 = 600`, resident
     level-0 tiles and `baseHookSamples == 0` asserted; the benchmark opts out of the per-frame `completeView`;
     `completeView` drains the whole queue; `terrain/visual` owns merge, patch and mask completions; JSON on #66's keys.
  3. Double emission and host/client event count -> fixed: one path per change (between steps at once, never in the
     list; in-step through the list, emitted at the commit); one `onTerrainChanged` per edit (host) and per message
     (client, bounding box of the parts = the host's box); renamed to `pendingTerrainChanges`; kept entries emitted at
     the next `runStep` before `events.beginStep()`; the net test reads the list from an in-step system.
  4. Snapshot order -> fixed (E5): `Replication::sendChunkSnapshot(peer)` defined here; join and host-load orders
     stated.
  5. Scenario flatten check broken by the step-31 Smooth -> fixed: the flatten result is checked in the step-30
     callback right after the immediate edit; the end checks the flatten chunk's sequence 2.
  6. Materialization before validation and its cost -> fixed by a lazy base: blocks hold deltas only, the base is
     read from `TerrainSystem::sample`; receive and load validate in scratch without a base, allocate only blocks with a
     non-zero changed delta, and never call the hook; `terrainSample` clamps the sum; the hook cost of headless gathers
     is a stated risk and measured (incl. join and load of 50 full chunks).
  7. `kTerrainHeightLimit` / `clampTerrainHeight` are S20's -> fixed (E7): dropped from step 1 and Files; the gather's
     unreachable `SAIGA_EXIT_ERROR` became a `SAIGA_ASSERT`; out-of-limit inputs remain a pure kernel test.
  8. "Changed" before the clamp -> fixed: decided after the clamp (`h' = v == h ? h : clamp(v)`); Add at the limit is a
     no-op; tests added.
  9. Smooth mean test not a kernel property -> fixed: Smooth now rounds once
     (`floorDiv(2 * (sum - 9h) * g + 9 * 65536, 18 * 65536)`), tested per sample against an exact integer reference
     (error <= 0.5 unit) and for sign bias (mean within +-0.02 units).
  10. Host send list -> fixed: an ordered list of host state messages (S23b appends tombstones), sent by
      `repl->extract(networkOnly)` in phase 8 and at the `stepNetworkOnly` commit point.
  11. Grass masks -> fixed (E3): recomputed on the pool from a copy of the merged heights, dropped if the sequence
      changed, capped per frame, cached in `TerrainVisualPatch`, sent through `sendGrassMask`; S20's real
      `prototypeGrassMask` signature; mask cost measured headless (the benchmark camera is above the grass radius).
  12. Records vs S23b -> fixed (E10): the sequence stays in `TerrainEditStore` (`sequence`, `bumpSequence`,
      `setSequence`); `TERR` and `TerrainChunkState` carry the sequence plus optional deltas; "terrain record" = a chunk
      with deltas; iteration sets stated; `TerrainEditResult` carries the chunk-local box per chunk; the per-chunk
      struct is `TerrainChunkEdits` in `engine/terrain/TerrainChunkEdits.h` (the name S23b lists).
  13. Dirty queue -> fixed: tile boxes only for the view space (dropped on a switch), pyramid boxes per space; patch
      heights from the store's own tile or CPU pyramid plus deltas, never a hook call on the main thread (tested).
  Rejected: none.
- **Review round 3** (2026-11-21, local Opus + Fable, 13 findings; cross-plan decisions F1-F4, F1 and F2 bind S22):
  1. Neighbour check rejected adjacent snapshot chunks -> fixed: messages are validated on their own (within-message
     border equality only, host authoritative); the full border rule stays a `TERR` load check; `nonZeroSamples`
     recomputed by a dense scan when an entry is created or replaced; regression test, and a 4-chunk corner edit in the
     late-joiner and host-load tests; client transient listed under Risks.
  2. Benchmark stepping inside `run()` -> fixed: `EngineApp::setFixedFrameTime(double)` (worlds advance by the fixed
     time, statistics keep the wall clock), `test_engine_app` case; EngineApp in Files, step 10 and a §8.3 amendment
     next to §17.7.
  3. Step order -> fixed: the send list (append, commit, abort) moves to step 3, encoding and sending stay in step 5;
     the delta codec moves to step 4 (`TERR` uses it); `terraformTestTemplate` to step 2; the dirty-queue push of
     `onTerrainChanged` comes with step 8; every test group carries the step that adds it ([n]), grass parts in step 9.
  4. Benchmark edits between steps -> fixed: `TerraformBenchmark` times each `editTerrain` call (`terrain_edit_ms` per
     edit); the step and its budget exclude edits.
  5. Residency override and JSON keys -> fixed: the terraform benchmark raises the floor of `game_world`'s
     `residencyRange0` rule to 600 m (`TerraformBenchmark::kResidencyRange0`), so no setting change undercuts it; S20's
     keys `terrain_stream_ms`, `wait_ms.terrain_stream` and `terrain: {uploads, upload_mb, resident_tiles, cpu_mb,
     gpu_mb}`.
  6. Owner of `baseHookSamples` -> fixed: `TerrainSystem::baseHookSamples()`, a relaxed atomic counted in
     `TerrainSystem::sample`'s height-source path for every caller; edit statistics in `TerrainEditStore::Stats`.
  7. Hash claim -> fixed: registering `TerrainEdited` changes every hash and the schema; equality is stated against a
     world of the same build without edits (§4, save tests, §17.4 amendment).
  8. Mask completion -> fixed: kept only if the chunk's mask is still in S20's desired set of the current GPU set and
     its revision is unchanged (a revision change re-queues, leaving the set does not); keyed on F1's revision; the
     cache entry is dropped and the base mask sent when deltas return to zero; tests.
  9. Codec predictor -> fixed: one delta codec, predictor 0 at the start of every part and chunk; codec tests.
  10. Validation point -> fixed: role, space and brush checked at the `editTerrain` call, also inside a step; death
      tests from a phase-4 system.
  11. `--coop` -> fixed: the `T` mode edits `app.startWorld()` (the host world with `--coop`); the MCP tool on the
      client world answers with its client error.
  12. F2 (stale S23b text) -> fixed: steps 4 and 5 of `applyTerrainEdit` name S23b's extension point (tombstones
      computed and reserved in the reserve step, written in the commit); `onTerrainChanged` feeds only S23b's re-snap
      queue; Scope 3, Out of scope and the §12.2 amendment updated. F1 -> fixed (`TerrainChunkEdits::revision`).
  13. S23a wording ("materialized edit blocks", base re-read) is S23a's to fix; S22 states in interface item 2 that
      `TerrainSystem::sample` stays a cheap point query safe in `parallelEach` (S23a's source keeps it).
  Also: S20's round-3 item 14 -> S22 uses S20's `integerTestHook` through `testSpaceTemplate` (no own `testSlope`
  hook); the schema regeneration uses only `--gtest_filter='SaveFormat.EngineSchemaIsCommitted'`; `TERR` cites S21's
  recipe names (`kTagTerrain`, `{tag, optional, maxVersion}`, `ParsedSave::Range`, `SectionWriter::beginOptional`).
  Rejected: none.
- **Implementation review** (2026-11-22, rounds 1 and 2, one reviewer over the S20-S23c stack): round 1 (8, minor):
  without a sink `completeView` could never complete an edited chunk (its mask queue drops entries without a sink while
  `missing()` wanted a ready mask) -> fixed (`S22 review`): without a sink the store's base masks of edited chunks are
  accepted (edited masks are made for a sink only). Round 1 (9): the replication's terrain handlers after a refused
  `GenerationState` -> fixed in S23a, where `generationMismatch` exists. Round 2 (11, minor test): the
  `terrain_edit_crater` golden waited at most 200 frames for the background mask jobs -> a 30 s deadline loop (and its
  `M_PI` -> `kPi`, as S20's tests). Round 2 (12, nit): the `T` mode in an enclosed space reported the edit error ten
  times per second -> spaces that are not open-world are skipped before the ray march. The S20 fixes below this branch
  were rebased in (one conflict: `TerrainSystem::height` keeps D1's edit rule before the NaN check).
