# S23a: Generation pipeline

Status: planned.

Sections follow the stage plan template of S05; "Implementation steps", "Risks and open questions", "Design-doc
amendments" and "Review log" are added for this stage. S23 of §22 is split into three stages with their own plans
(decision D13 of review round 1): **S23a** (this plan: the pipeline, hashes, saves, stamping), **S23b**
(`S23b_tombstones_content.plan.md`: tombstones, content render sets, re-snap and tombstone-on-edit, MCP tools, the
`generated_site` golden) and **S23c** (`S23c_overworld_generator.plan.md`: the flight demo's integer generator and
the game hashes). The §22 row is amended here (amendment 1).

## Goal

Every space's content comes from a deterministic, integer-only generation pipeline (`docs/engine_design.md` §12.1,
§15.3, R-CONTENT-1):

- a fixed-point toolkit and an integer lint that every CI job runs;
- the `Generator` interface, which sees only an integer view of its space template;
- a world plan per space instance (sites of authored places, space instances, integer plan grids), all computed at
  world construction;
- chunk generation (base heights with the site patches composed by the engine, tier-1 candidates, grass masks);
- prefabs (versioned JSON files) and their stamping from the plan alone (entities with components, `SpaceLink`
  entrances), host only, in phase 7;
- a shared content cache, pinned generator versions, the `GENR` and `CREC` save sections and the pre-S26 generation
  state message with which a client checks that it generates what the host generates (and refuses otherwise);
- content hashes committed for an engine test generator and reproduced by GCC, Clang and MSVC in CI, with the content
  code compiled with UBSan (trapping) on the Clang job; the hash-file check is an engine function games reuse.

S20's terrain tiles, `terrainHeight` and S22's edit base take their heights from the pipeline from this stage on.

## Depends on

S20, S21 and S22 as revised after review round 2 (decisions D1-D14 of round 1 and E1-E14 of round 2; names below
follow them and are checked against the final code in step 1):

- S20 (`S20_height_tiles.plan.md`): `SpaceTemplate` (`engine/world/SpaceTemplate.h`; D1: a template without terrain
  source is flat at 0 and draws no terrain) and `SpaceTemplate::hasTerrain()` (E6: every "has terrain" rule keys on
  it), `kHeightUnit`, `kTerrainHeightLimit` (2^23 units) and `clampTerrainHeight` (E7), `terrainSurfaceUnits(h00, h10,
  h01, h11, fx, fz, fracBits)` (`core/math/TerrainSurface.h`), the callable height source `TerrainHeightSource` that
  S20's generation functions and `TerrainSystem::sample` take (E7), `TerrainSystem` (`World::terrain()`: `sample`, which
  returns 0 for a space without terrain,
  `height`, `updateView`, `completeView`, `readiness`, D6/E8; `clearTerrain` on the sink, E1), `TerrainStore` (base
  data only, D3), `TerrainTileData` / `GrassMaskData`, the pure tile / pyramid / grass-mask functions below the engine
  (D2) and their engine wrapper `TerrainGeneration`, the send point with `sendGrassMask` (E3), the `StreamingPool` of
  D8 (`submit(job, priority, completion)`, `drain()`, jobs capture copies, never the `World`), the float
  `prototypeHeight` hook (`TerrainHeightHook`) in `vulkan/world/PrototypeTerrain` and the prototype `SpaceTemplate`
  built in `samples/games/common/` (D2).
- S21 (`S21_spaces_transitions.plan.md`): `World::spaceCount()`, `spaceTemplate(SpaceId)`, `spaceSeed(SpaceId)`;
  space-state entities per space; `SpaceKind::Enclosed` and the enclosed spaces' chunk range `[0, sizeChunks)^2` with
  the site origin at the centre of the middle chunk, `(sizeChunks / 2) * 256 + 128` m on both axes (D12, E4);
  `SpaceLink`; `moveToSpace`; `requestTransition`; the `SPCE` section (with its prefix rule: a save listing fewer
  spaces loads) and the optional-section mechanism of D7; template validation with `SAIGA_EXIT_ERROR` (D1).
- S22 (`S22_terraforming.plan.md`): `TerrainEditStore` with the per-chunk modification sequence for all spaces
  (`sequence(ChunkId)`, `bumpSequence`, E10) and the lazy base (the store holds deltas only; the base of every edited
  sample is read through `TerrainSystem::sample` when needed, never stored), `World::terrainSample`, `applyTerrainEdit`
  and its per-chunk changed boxes, `onTerrainChanged`, `TERR`, `TerrainModification` / `TerrainChunkState` on
  `Channel::State` with the S16 epoch, `Replication::sendChunkSnapshot(peer)` (E5: the one function that sends a joining
  peer's or a host load's chunk snapshot; S23a and S23b extend it).
- Existing: S06 `Hash` / `Hasher` / `hashCombine` / `fnv1a64`, `JobPool`; S10 `ClassList`, `World::spawnByClass`, the
  world API's `spawn` (`WorldApi.cpp`, the post-`onSpawn` `mergeJson` path) and `checkComponentValues`; S11 saves,
  schema and data-file rules; S12 scenarios and their variants; S15/S16 messages, `Resync`, the spawn set; S19b
  `RenderSink` / `RecordingRenderSink`; `core/util/json11` (in this repository) and `Saiga::ParseNumber`
  (`core/util/tostring.cpp`, locale independent).

## Rules that apply

R-CONTENT-1 (integer and fixed-point generation, exact conversions, no tunable data files in generators), R-DET-1,
R-MECH-1 (the engine provides the pipeline, stamping, hashing and the toolkit; games own generators, prefabs and every
policy), R-REG-1/2 (`WorldSetup::generators`; names only at the data/save/network boundaries), R-FILE-1, R-STATE-1/2
(stamped entities are ordinary saved entities; chunk records are state), R-SYS-1/2 (stamping is a phase-7 system that
spawns through commands), R-NET-1/2 (clients never stamp and never choose versions), R-PAR-1 (generation on the
streaming pool; the OpenMP deviation stays narrowed to the prototype fixture), R-DATA-1 (prefab files), R-SAVE-1
(sections with their own format versions, D7), R-TEST-1/2, R-MATH-1/2, R-DOC-1/2, and the portability rules of
AGENTS.md (GCC, Clang, MSVC).

## Scope

### 1. Fixed-point toolkit (`engine/content/gen/FixedMath.h`, `gen/TrigTable.{h,cpp}`, `gen/IntNoise.{h,cpp}`)

Engine mechanisms every generator uses; integer only, header-light, compiled and tested on all three CI compilers.

- `FixedMath.h` (header only, `constexpr` where possible):
  - `floorDiv(int64 a, int64 b)`, `floorMod(a, b)` for `b > 0` (rounding toward negative infinity; `/` and `%`
    truncate and would mirror content at the axes).
  - `floorShift(int64 a, int s)`, `0 <= s < 63`: an arithmetic right shift; the header has
    `static_assert((int64_t(-3) >> 1) == -2)` (implementation defined in C++17; GCC, Clang and MSVC shift
    arithmetically).
  - `mulShift(int64 a, int64 b, int s)` = `floorShift(a * b, s)`; with `SAIGA_ASSERTS` the **product** is asserted to
    fit: `a != INT64_MIN`, `b != INT64_MIN` and `b == 0 || |a| <= INT64_MAX / |b|` (not a bound on each factor: the
    trig series multiplies `x^2` of about `2.65 * 10^9` by Q30 coefficients). No `__int128` (MSVC has none).
  - `lerpQ16(a, b, t)` = `a + floorShift((b - a) * t, 16)`, `t` in `[0, 65536]`.
  - `smoothQ16(t)`, `t` in `[0, 65536]`: the quintic fade `6t^5 - 15t^4 + 10t^3` as a stepwise Horner scheme in Q16,
    `r = 6t - 15 * 65536; r = floorShift(r * t, 16) + 10 * 65536; r = floorShift(r * t, 16)` three times. Every
    product is below `2^17 * 2^20` (no `t^5` intermediate, which would overflow int64 in Q16). Exact endpoints:
    `smoothQ16(0) == 0`, `smoothQ16(65536) == 65536`. The floor shifts make it **not** monotone: 1601 inputs step down
    by 1 LSB (the first at `t = 48775`); every value lies in `[0, 65536]`, `s(t + 1) >= s(t) - 1`, and it stays within
    2 LSB of the exact polynomial (all three tested). The formula is fixed (it enters the committed hashes); callers
    need continuity, not strict monotonicity.
  - `isqrt64(uint64)`: the exact floor of the square root, bitwise (digit by digit), no float.
  - `clampI(v, lo, hi)`; `ceilDiv`.
  - Units: `kPosUnitsPerMetre = 256` (candidate and prefab x/z, 1/256 m) and `kHeightUnitsPerMetre = 1024` (S20's
    `kHeightUnit`).
  - The rules (comment block, enforced by review, the lint and UBSan): multiply narrow unsigned types only after a
    cast to `uint32_t`/`int64_t` (`uint16_t * uint16_t` promotes to `int`); shift left only unsigned values or
    non-negative values known to fit; no `std::abs(INT64_MIN)`; floor helpers for possibly negative values; sort
    only on unique keys (`std::sort` is not stable); no `long` (32 bits on Windows) and no `size_t` in content code
    (both refused by the lint; container sizes are converted with an explicit `uint32_t(...)` cast at the call);
    no floating literals and no libm, also not through a header that includes `<cmath>` (the lint refuses the
    literals and the function names, Scope 10); every narrowing conversion (yaw and slot arithmetic in `uint16_t` /
    `uint8_t`, `int64_t` to `int32_t`) is an explicit cast (`uint16_t(...)`): CI builds with
    `SAIGA_WARNINGS_AS_ERRORS=ON`, where MSVC's C4244 / C4267 are errors.
- `TrigTable`: a quarter-wave sine table for a circle of 2^17 steps (32769 int32 entries in Q30, 128 KB):
  - `sinQ30(uint32 a17)`, `cosQ30(uint32 a17)`;
  - filled once in a function-local static (thread-safe initialization) by an integer-only Taylor series of degree 15
    in Q30 with floor shifts, so it is identical on every compiler without a committed 32769-entry source file;
  - error bound (derived in a comment next to the series and tested, Tests): the truncation after degree 15 is below
    `(pi/2)^17 / 17! < 2^-37`; the Horner scheme `x * (1 - x^2/6 * (1 - x^2/20 * (...)))` makes at most one floor
    error of 1 LSB per step (8 steps plus `x^2`), and every later step's error is scaled by `x^2 / (2k (2k+1)) <
    0.42`, so the total stays below 4 LSB; the test allows 8 LSB and prints the observed maximum;
  - a `uint16_t yaw` (65536 = one turn) gives `sin/cos(yaw)` at index `2 * yaw` and the half angle of a yaw quaternion
    at index `yaw`.
- `IntNoise` (unsigned or int64 arithmetic, no float):
  - `uint64_t contentHash64(uint64 seed, uint64 a, uint64 b, uint64 c, uint64 d)`: splitmix64 finalizers chained over
    the inputs; the only randomness in content code (counter based: a candidate is a pure function of its inputs,
    §12.1 (b)). `Rng` (PCG32, sequential) is not used in generators.
  - `int32_t gradientNoise(uint64 seed, int64 x, int64 z, int cellShift)`, `1 <= cellShift <= 40`:
    - lattice cells of `2^cellShift` input units (cell index `floorShift(x, cellShift)`, fraction
      `x - cell * 2^cellShift` converted to Q16 by a left shift for `cellShift <= 16`, else a right shift);
    - 16 gradient directions at multiples of 22.5 degrees, Q16 components from the trig table (unit length up to
      rounding), chosen by `contentHash64(seed, cellX, cellZ, 0, 0) & 15`;
    - the four corner dot products blended with `smoothQ16` and `lerpQ16`.
    - Scale: the maximum magnitude of 2D gradient noise with unit gradients and this fade is `sqrt(2) / 2` of the cell
      fraction unit (attained at the cell centre with all four gradients pointing at it), so the raw value is
      multiplied by `92682 / 65536` (`sqrt(2)` in Q16, `floorDiv`) and clamped to `[-65536, 65536]` (the clamp only
      absorbs rounding; the test samples the centres of 2^16 cells, where all four gradients lie within 22.5 degrees of
      the diagonal towards the centre in about one cell of 850 (`(3/16)^4`), giving at least `cos(22.5 deg) * 65536 >
      60000`; it checks that the observed extremes of both signs lie in `[60000, 65536]` in magnitude).
  - `fbm(seed, x, z, cellShift, octaves, gainShift = 1)`: octave `i` uses `cellShift - i`, amplitude shifted right by
    `i * gainShift`, seed `contentHash64(seed, i, 0, 0, 0)`. Asserted in every build: `1 <= octaves <= cellShift`
    (the smallest cell is at least 2 units, so `cellShift - i >= 1`), `octaves <= 16`, `i * gainShift <= 16`. The sum
    is normalized by the geometric amplitude sum (integer) back to `[-65536, 65536]`.
  - `ridged(...)`: `65536 - |n|` per octave with fbm's octave rules.
  - `valueNoise(seed, x, z, cellShift)`.
- No lint here: the integer lint is a test (Scope 10).

### 2. Generator interface and plan types (`engine/content/Generator.h`, `WorldPlan.h`, `GeneratorList.{h,cpp}`)

As §12.1, with the additions marked (A) (amendments 2-4):

`SpaceKind` moves from S20's `SpaceTemplate.h` (`engine/world/SpaceTemplate.h`) into its own header
`engine/world/SpaceKind.h` (included by `SpaceTemplate.h`), so `Generator.h` and generator code can use it without
including `SpaceTemplate.h` (Scope 10).

```cpp
using GeneratorId = uint16_t;                     // engine/ecs/Ids.h; 0 = the engine's FlatGenerator (A)
enum SlotFlags : uint32_t { ReSnapOnEdit = 1, TombstoneOnEdit = 2, Cosmetic = 4 };   // (A) per slot, §12.2, §17.9
struct GeneratorSlot { std::string name; uint32_t flags = ReSnapOnEdit;              // (A) name and flags: content,
                       std::vector<std::string> meshes; };                           // saved in GENR; meshes:
                                                  // presentation only (not hashed, not saved), see Design notes
// (A) The only part of a SpaceTemplate a generator sees: integers and the name. Generators never include
// SpaceTemplate.h (the lint refuses it in content code), so a double field cannot leak into content.
struct GenerationTemplate { std::string name; SpaceKind kind = SpaceKind::OpenWorld; uint16_t templateIndex = 0;
                            int32_t baseSpacing = 2, sizeChunks = 128; std::optional<int32_t> seaLevel; };
class Generator
{
   public:
    virtual ~Generator() = default;
    virtual const char* name() const = 0;          // (A) saves, messages and the hash file name generators by name
    virtual uint16_t version() const = 0;          // the newest version; every older one stays callable (§15.3)
    virtual std::vector<GeneratorSlot> slots() const = 0;   // (A) append-only across versions
    virtual bool integerOnly() const { return true; }       // (A) false only for the float prototype fixture
    virtual bool legacyTerrain() const { return false; }    // (A) heights equal the pre-S23 hook of its template
    virtual bool hasTerrain() const { return true; }        // (A, E6) false: flat at 0, no terrain drawn or streamed
    // (A) In/out: the pipeline passes a plan with seed and space set (everything else empty); plan() fills sites,
    // spaces and grids and must not change seed or space (asserted). It may call height(version, plan, x, z) for
    // natural heights (site slopes, water tests, baseHeight) once the grids height() reads are filled (its own order).
    virtual void plan(uint16_t version, WorldPlan& plan) const = 0;
    virtual int32_t height(uint16_t version, const WorldPlan& plan, int64_t x, int64_t z) const = 0;
                                                  // natural terrain without site patches (A: the pipeline composes
                                                  // them); reads plan.seed, plan.space and the plan's grids
    virtual bool candidate(uint16_t version, const WorldPlan& plan, ChunkId chunk, uint32_t slot, uint32_t counter,
                           Candidate& out) const = 0;
    virtual uint32_t candidatesPerChunk(uint32_t slot) const = 0;
    virtual uint32_t keptCount(uint32_t slot, int level) const = 0;   // declared now, used by S27
    virtual void grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const = 0;
};
struct Candidate { uint16_t meshSlot = 0; int32_t x = 0, z = 0;  // 1/256 m, chunk relative, [0, 65536)
                   uint16_t yaw = 0; uint16_t scale = 1024;     // yaw: 65536 = one turn; scale: 1/1024
                   uint32_t flags = 0; };                       // (A) game bits, hashed, not interpreted
struct GrassMask { int32_t size = 0; std::vector<uint8_t> density; };   // S20's GrassMaskData layout: one texel
                                                  // per base cell (256 / baseSpacing per side), row-major z then x
struct PlanGrid { int32_t cellSize = 0;                         // metres, a power of two
                  int32_t originX = 0, originZ = 0, width = 0, height = 0;   // in cells
                  std::vector<int32_t> values;                  // row-major; at(cx, cz) clamps to the border
                  int32_t at(int64_t cellX, int64_t cellZ) const; };
struct Site { uint32_t key = 0;                                 // unique in the plan
              std::string prefab; uint16_t prefabVersion = 1;
              int64_t x = 0, z = 0;                             // the prefab origin, integer metres in the space
              uint8_t quarterTurns = 0;                         // (A) rotation in 90 degree steps, 0..3
              int32_t baseHeight = 0;                           // height units; prefab y = 0 lies here (the
                                                                // generator's choice; convention: natural height
                                                                // at the site origin)
              int32_t footprint[4] = {};                        // [x0, z0, x1, z1) in local metres, checked
            };
struct SpaceInstance { uint16_t templateIndex = 0; uint64_t seed = 0;
                       SpaceId parent = 0; uint32_t siteKey = 0; uint8_t link = 0;   // (A) parent/link
                       uint8_t origin = 0; };                   // (A) 0 overworld, 1 listed, 2 site, 3 unreferenced
struct WorldPlan { // Set by the pipeline before plan() (unchanged by it, asserted):
                   uint64_t seed = 0; GenerationTemplate space;      // (A) read by plan/height/candidate/grassMask
                   // Filled by plan():
                   std::vector<Site> sites; std::vector<SpaceInstance> spaces;
                   PlanGrid biomes, rivers; PlanGrid cloudCoverage;   // [S42] not hashed
                   // Filled by the pipeline after plan() returns (empty before, asserted):
                   std::map<ChunkId, std::vector<uint32_t>> sitesByChunk; };   // site indices per chunk, sorted
class GeneratorList { public: void add(GeneratorId id, std::shared_ptr<const Generator> generator); };
```

- `WorldSetup::generators` (`void (*)(GeneratorList&)`) and `SpaceTemplate::generator` (`GeneratorId`, default 0) as
  §7.9. The transitional S20 hook field of `SpaceTemplate` is removed (amendment 5) in step 7, together with the move of
  its users (until then the pipeline's height source is used only for templates with a non-zero `generator`, step 6).
  Generators are held by `shared_ptr<const Generator>` so streaming-pool jobs can keep them alive (D8: jobs never
  capture the `World`).
- `SpaceTemplate::fullRadius` (default 1000 m, §7.9, §12.3) is added now (stamping uses it; S24 adds the other radii).
  `ScenarioWorld` lowers it to `min(value, 128 m)` on its copy of the setup (§17.2); a scenario that needs content
  farther away moves its `Player` there (S23c's ruin scenario holds the plane above the site).
- `ScenarioWorld` gets `const WorldSetup& setup() const` (F3 of review round 3): the setup its worlds are built from
  (incl. the 128 m override and the per-setup `sharedContent`), so a scenario can pass it to `checkContentHashes`
  (S23c). The private member of that name today (`ScenarioWorld.h`, `WorldSetup setup;`) is renamed `worldSetup`.
- Engine generators (`engine/content/gen/`, linted):
  - `FlatGenerator`, registered as ID 0 by the engine: height 0, no slots, an all-zero grass mask, an empty plan,
    `legacyTerrain() == true`, `hasTerrain() == false`. Every setup without generators keeps S20's flat default (D1:
    no terrain drawn).
  - `AuthoredSpaceGenerator(name, prefab, prefabVersion)`: for a space that is prefabs only (a dungeon). Its plan holds
    one site at the centre of the enclosed space's middle chunk (S21, D12, E4: chunk range `[0, sizeChunks)^2`, site
    origin `(sizeChunks / 2) * 256 + 128` metres in x and z, `baseHeight` 0); no slots; height 0; `hasTerrain() ==
    false` (enclosed spaces have no height map, §12.4); `legacyTerrain() == false` (it has sites, so its worlds write
    generation state, below).
- **"Has terrain"** (E6): `SpaceTemplate::hasTerrain()` (S20: hook != null) cannot stay a template member once the hook
  field is gone; it becomes `World::spaceHasTerrain(SpaceId)` = `generator(template).hasTerrain()`. Step 7 switches
  every S20/S21 call site (in step 6, before the switch, `SpaceTemplate::hasTerrain()` is `hook != null ||` the
  template's non-zero generator has terrain): `TerrainSystem` (terrain and sea drawn, nothing requested and
  `clearTerrain` sent for a space without terrain, E1; `sample` returns 0; `readiness` 1), `describe_world.has_terrain`,
  S21's enclosed-template validation (an enclosed template whose generator has terrain is a setup error,
  `SAIGA_EXIT_ERROR`) and its loading rule, and S22's edit refusal for enclosed spaces (keyed on `SpaceKind`,
  unchanged). S20's tests `.FlatWithoutHook` and S21's enclosed tests keep their expectations (a template without a
  generator is `FlatGenerator`).
- **"Writes generation state"** (one predicate for `GENR`, `GenerationState` and the generation part of `world.hash()`,
  round 3): `GenerationPipeline::writesGenerationState()` = some template's generator has `legacyTerrain() == false`, or
  `ChunkRecords` is non-empty. Records need sites or slots, which only non-legacy generators have (registration check
  below), so the predicate is fixed per setup: `FlatGenerator` worlds and `HookGenerator` worlds (the prototype fixture,
  S20's and S22's hook tests) write no `GENR`, send no `GenerationState` and hash as before; every world with an integer
  generator or an `AuthoredSpaceGenerator` does all three. Scopes 7 and 8 refer to this predicate only.
- Registration checks (`SAIGA_EXIT_ERROR`, every build, death tests without `SAIGA_ASSERTS`, as D1's template
  validation):
  - slot count <= 200 (slot values 200-255 are reserved for engine keys: S23b's static parts use 255, S30's pieces;
    `StaticInstance` packs the slot into 8 bits, §10.1); `candidatesPerChunk(slot) <= 2^24`;
  - unique slot names; exactly one of `ReSnapOnEdit` / `TombstoneOnEdit` per slot;
  - `keptCount(slot, 0) == candidatesPerChunk(slot)`, `keptCount` non-increasing in the level;
  - unique generator names and IDs; every template's generator registered;
  - a generator with `legacyTerrain() == true` has no slots, and its plans have no sites (a plan error).
- Thread safety: all `Generator` functions are `const` and called concurrently (streaming pool, `JobPool`); a
  generator keeps no mutable state. The hash test runs them from three workers.
- `HookGenerator` (`engine/terrain/HookGenerator.{h,cpp}`: beside S20's terrain code, **not** in `engine/content/`,
  not linted): `HookGenerator(std::string name, TerrainHeightHook hook, std::optional<HookGrassRule> grass,
  bool integerOnly)` adapts an S20 height hook (`TerrainHeightHook`, `int32_t (uint64_t seed, int64_t x, int64_t z)`,
  called with `plan.seed`) to the interface: `height` returns `clampTerrainHeight(hook(...))` (E7: the clamp S20
  applies to every hook result stays in the hook path, so S20's `.HookResultsClamped` stays valid; the pipeline's
  range assert of Scope 4 is for integer generators' unclamped results); no slots, no sites, `legacyTerrain() ==
  true`, `hasTerrain() == true`; `grassMask` evaluates the hook at the
  chunk's samples and applies S20's grass rule (`prototypeGrassMask` of `vulkan/scene/TerrainTiles`, D2) with the given
  thresholds (`grassMinNormalY`, `grassMinAboveSea`), or gives an all-zero mask without a rule. It replaces the
  template hook field for every engine-side user (step 7): the prototype fixture of the games (name `prototype`,
  `integerOnly == false`) and S20's and S22's engine tests (the prototype hook inline, or their integer test hooks
  with `integerOnly == true`); their heights, and so their tests and goldens, are unchanged.

### 3. Prefabs (`engine/content/Prefab.h`, `PrefabStore.{h,cpp}`, `engine/api/ComponentJson.{h,cpp}`)

- Files: `<WorldSetup::dataDirectory>/prefabs/<name>.json` is version 1 and `<name>.v<N>.json` is version N
  (§15.3), read with `readJsonFile` under R-DATA-1: missing optional keys keep defaults; unknown keys, malformed or
  out-of-range values and a missing file are errors naming file and key (`SAIGA_EXIT_ERROR`, every build). Not data
  files of §14: never reloaded, never read by `world.data`.
- Format (`"format": 1`); all coordinates prefab-local, rotated with the site's quarter turns:

```json
{
  "format": 1,
  "footprint": [-16, -16, 16, 16],
  "terrain_patch": { "origin": [-12, -12], "size": [25, 25],
                     "heights": [/* 625 ints: height units relative to the site's baseHeight */],
                     "weights": [/* 625 ints in 0..256: 256 = patch, 0 = natural terrain */] },
  "exclusion": [[-14, -14, 14, 14]],
  "entities": [
    { "class": "TestChest", "position": [2.5, 0.0, -1.25], "yaw_deg": 90,
      "components": { "Health": { "value": 50 } } },
    { "class": "TestEntrance", "position": [0, 0, 8],
      "space_link": { "to": "link", "link": 0, "arrival": [0, 1, 0], "arrival_yaw_deg": 180 } }
  ],
  "static_geometry": [ { "mesh": "test/pillar", "position": [4, 0, 4], "yaw_deg": 0 } ],
  "spaces": [ { "template": "test_dungeon" } ]
}
```

- Units and exactness:
  - `footprint`, `terrain_patch.origin/size` and `exclusion` are integer metres; rectangles are half open
    `[x0, z0, x1, z1)`; patch samples are 1 m apart. Everything (patch, exclusion, entities, static parts) lies inside
    the footprint (checked at load).
  - Positions are metres; x and z must be exact multiples of 1/256 m, y of 1/1024 m: the parser checks `v * 256` (or
    `* 1024`) for integrality and range; an inexact value (`0.1`) is an error.
  - JSON numbers are parsed without the C locale: json11's `parse_number` calls `Saiga::ParseNumber` (`from_chars`,
    or a classic-locale stream) on the validated number text instead of `strtod`, and `dump(double)` formats with a
    `std::ostringstream` imbued with `std::locale::classic()` and `std::setprecision(17)` instead of
    `snprintf("%.17g")` (decimal comma under `de_DE`). The stream's default float format is `%g`, so the output is
    byte-identical to today's in the C locale (`3000000` stays `3000000`, not the shortest form `3e+06` of
    `NumberToStringRoundtrip`): no JSON writer, MCP reply or test expectation changes. This fixes every JSON reader and
    writer of the repository (MCP, data files, prefabs); short decimals exact in binary parse exactly.
  - 64-bit integers in strings (the seed `"12345"` and the `"0x..."` hashes of the hash file) are parsed with
    `std::from_chars` into `uint64_t` (base 10, or base 16 after the `0x` prefix; the whole string must be consumed):
    `ParseNumber` has no unsigned 64-bit overload and seeds above 2^63 must parse.
  - `yaw_deg` is an integer number of degrees, converted to the 65536-step yaw by `floorDiv(deg * 65536 + 180, 360)`.
  - **Quarter turns** (fixed here because they enter the committed hashes): a site's rotation is `quarterTurns`
    times 90 degrees about +y in the yaw direction (yaw `θ` maps local `(x, z)` to `(x cos θ + z sin θ, -x sin θ +
    z cos θ)`, the rotation of the quaternion `(cos θ/2, 0, sin θ/2, 0)`). A local point `(x, z)` (any unit) maps to
    `q = 1: (z, -x)`, `q = 2: (-x, -z)`, `q = 3: (-z, x)`, then the site origin is added; the inverse of `q` is `4 - q`
    (mod 4). Point data (patch samples, entity and static-part positions) is rotated point by point, exactly. Every
    **point-in-area test** inverse-rotates the point into the site's local frame and tests it against the local
    half-open rectangle (candidates against exclusions, grass cell centres, the patch lookup of Scope 4), so no rotated
    half-open rectangle is ever formed (a rotation turns `[x0, x1)` into `(-x1, -x0]`). Where a rotated **area** is
    needed (`sitesByChunk`, the inside-space test, the overlap test), the pipeline rotates the closed integer bounds
    `[x0, x1] x [z0, z1]` of the rectangle's corners (`q = 1: [z0, z1] x [-x1, -x0]`, ...): conservative by at most
    the boundary line (a footprint ending exactly on a chunk border is also listed in the next chunk, which has nothing
    of it to stamp; footprints that only touch count as overlapping; a footprint ending exactly on the space edge is
    inside). Yaw composition: a stamped entity's or static part's yaw is `uint16_t(prefab yaw +
    quarterTurns * 16384)` (wrapping).
- `components` are JSON in the `visit` form (as the world API's `spawn`), kept as JSON (`PrefabStore` stores them
  unmodified). They are **validated at plan time** with `checkComponentValues` (moved from `WorldApi.cpp` into
  `engine/api/ComponentJson.{h,cpp}` together with `addComponentValues` and the merge loop of `WorldApi::spawn`, so
  stamping and the API share one path): an unknown class, component or field, a special component (`Transform`,
  `Behaviour`, ...), a wrong type is fatal, naming prefab, entity and field. Prefabs additionally refuse `Parent`
  (`checkComponentValues` lets it through with an entity reference, which a prefab cannot hold), `NetOwner` and
  `SpaceLink` (set only through `space_link`). These checks are **phase B** of the pipeline (Scope 4): they run in the
  `World` constructor after class registration and after `WorldSetup::renderAssets` (today the constructor's last
  statement, `World.cpp:107`), so a `RenderInstance` value is checked against the registered meshes; a load runs them
  on the loaded pins' plans against the same tables. Everything about a prefab that needs no `World` (format, units,
  rectangles, patch, `spaces`, `pieces`) is checked in phase A when the file is read. They are **applied after
  `configure()` / `onSpawn()`** like the API's `spawn`: a component the class added itself is merged with `mergeJson`
  (so a class that adds `Health` in `onSpawn` gets the prefab's value instead of the "already has Health" assert), any
  other is added with its default and merged. No binary pre-decoding (it would fill unspecified fields with defaults).
- **Flush-safe add**: the shared path's add today calls `World::addDefaultNow`, which asserts `!inStep()`
  (`World.cpp:171`); stamping applies it inside the phase-7 flush. A new engine-internal `World::addDefaultById(Entity,
  ComponentId)` follows the typed `addNow`'s rules: allowed between steps and while the world applies commands
  (`flushing`), refused while deferring (`SAIGA_ASSERT`), a no-op on an entity destroyed earlier in the same flush,
  the special components refused as in `addDefaultNow`. `addComponentValues` uses it; `addDefaultNow` stays the API's
  between-steps entry. Test: a stamp during a step with a prefab component the class does not add itself.
- `space_link`: `to` is `"link"` (the site's n-th entry of `spaces`) or `"parent"` (back to the space that created
  this instance). `"link"`: `arrival` is in the target space's coordinates; `"parent"`: relative to the creating
  site's origin, rotation and base height, converted exactly. The entity gets S21's `SpaceLink {target, arrival,
  arrivalRotation}` at stamping.
- `spaces`: the space instances a site creates (template by `SpaceTemplate::name`), allowed only in prefabs of space-0
  sites; nested instances are a plan error.
- `terrain_patch` in the prefab of a site of an enclosed space (or of any space whose generator has no terrain) is a
  plan error (no height map to patch, §12.4).
- `pieces` is an error until S30 ("placed pieces in prefabs come with S30").
- `PrefabStore` is shared through `ContentCache` (Scope 5), keyed by (directory, name, version). Each `Prefab` gets a
  content hash over its integer content (footprint, patch, exclusion, static geometry names and poses, entity class
  names, integer poses, link data, component names; not component values, Design notes); it enters the plan hash.

### 4. Generation pipeline (`engine/content/GenerationPipeline.{h,cpp}`, `ChunkContent.h`, `ContentConvert.{h,cpp}`, `gen/SitePatch.{h,cpp}`, `gen/ChunkGeneration.{h,cpp}`)

One pipeline per `World`; it owns the generator list, the pinned versions and every plan.

- **Seed** (D11): `World::generationSeed()` is `WorldSetup::seed`, always. The seed is part of the setup: a load
  whose saved seed differs is a load error, and a client whose setup seed differs from the host's is refused
  (Scope 7). There is no run-time rebuild of the space table.
- **Pinned versions**: `std::map<std::string, uint16_t> pinned` by generator name (the plan/terrain version) and
  `current` (the placement version new chunk records get: the generator's `version()`, on the host and on clients;
  a client whose versions differ from the host's is refused, Scope 7, E11). A new world pins `version()` of every
  registered generator; a load sets the pins from `GENR` (Scope 8) and keeps `current = version()`.
- **Two construction phases** (the `World` constructor today, `World.cpp` ~60-107: event registration,
  `registerEngineSystems`, the game's systems, classes, messages, data files, `setupProfiler` (one profiler scope per
  registered system), `Replication`, `createStateEntities`, `renderAssets`):
  - **Phase A**, pure (`static GenerationPlans GenerationPipeline::build(const WorldSetup&, const GenerationPins&,
    ContentCache&)`, no `World`): the generator list and its registration checks, the template views, the prefab
    files of every plan (read and checked as far as they need no `World`, Scope 3), the space list, every plan with
    its validation (below), `sitesByChunk`, the stamp and static lists' inputs and the plan hashes. It runs **first in
    the constructor**, before `registerEngineSystems`, so the space list exists for S21's space-state entities
    (`createStateEntities`) and `registerEngineSystems` knows whether some plan has sites (it appends
    `chunkStampSystem` only then, Scope 9; being registered before `setupProfiler`, it gets its profiler scope).
    Standalone users: `checkContentHashes` (the file's pins and placement versions, Scope 11) and a load's validation
    (the saved pins, Scope 8) call it without a `World`.
  - **Phase B** (`GenerationPipeline::validateEntities(const World&, const GenerationPlans&)`), last in the
    constructor, after `renderAssets`: the prefab entities' classes (`ClassList`), component values and refusals and
    `RenderInstance` meshes (Scope 3). A load runs phase A with the saved pins and phase B on its result, both before
    the commit (validate-then-commit, §15.1).
- **Plans, all computed eagerly** in phase A. Nothing computes a plan lazily: `terrainHeight` runs inside
  `parallelEach` and the pool, and plans are immutable `shared_ptr<const WorldPlan>` after construction.
  - Space 0's plan: the pipeline creates a `WorldPlan` with `seed` and `space = view(template 0)` set, then calls
    `generator(template 0).plan(pinned, plan)` (in/out, Scope 2); it asserts that `seed`, `space` and `sitesByChunk`
    are unchanged afterwards. A generator that needs natural heights inside `plan()` (slope or water tests, a site's
    `baseHeight`) calls its own `height(version, plan, x, z)` after filling the grids that `height()` reads.
  - The space list, built by the pipeline (amendment 3):
    1. `spaces[0]` = `{template 0, seed, origin overworld}`, pushed by the pipeline; a generator lists no space 0.
    2. The instances the generator listed in `plan.spaces`, in its order (seeds chosen by the generator; convention:
       `hashCombine(hashCombine(seed, fnv1a64("listed")), i)`).
    3. Per site in site order, per entry of its prefab's `spaces`: `{template, seed = siteSeed, parent = 0, siteKey,
       link}` with `siteSeed = hashCombine(hashCombine(seed, fnv1a64("site")), key)` for link 0 and
       `hashCombine(siteSeed(link 0), link)` for link > 0 (D11 domain separation: a site key can never collide with a
       listed index or a template index).
    4. Every template not referenced by 1-3, in template order: `{template, hashCombine(seed, templateIndex), origin
       unreferenced}`. This keeps S21's "one instance per template" for every generator (e.g. `game_world`'s
       `islands` on the prototype generator), not only for `FlatGenerator`; with `FlatGenerator` on template 0 and no
       sites the list is exactly S21's (`SpaceId` = template index, S21's seeds).
  - Prefix rule (with S21's `SPCE`): a template appended to the setup adds an unreferenced instance at the end of the
    list, so a save with fewer templates and spaces loads (its spaces are a prefix of the current list; the new ones
    start without state). A change that inserts an instance in the middle (a generator listing a new template) is a
    load error (Scope 8).
  - Every other instance's plan: the same in/out call with that instance's seed and `view(its template)`; its
    `spaces` must be empty and its sites' prefabs must have no `spaces` (plan errors).
  - Validation per plan (fatal in the constructor, a load error in a load): unique site keys; `quarterTurns <= 3`;
    rotated footprints inside the space (open world: `[0, sizeChunks * 256)^2`; enclosed: S21's chunk range, D12) and
    not overlapping each other (integer rectangle test after rotation, Scope 3 conventions); prefabs that exist, fit
    their footprint and have valid components (Scope 3); no `terrain_patch` in a space without terrain; site
    `baseHeight` and the patched heights within `±kTerrainHeightLimit`.
  - The pipeline fills `sitesByChunk`: a site is listed in every chunk that the closed integer bounds of its rotated
    footprint intersect (Scope 3 convention; a site may be listed in a chunk it only touches).
  - **Stamp and static-part lists** (`stampList(ChunkId)`, `staticList(ChunkId)`, Scope 9) are part of the pipeline
    (step 6), because they enter the committed hashes (Scope 6).
  - Plan cost is reported (Tests: benchmark), because every instance plan is computed at construction.
- **S21's accessors become plan-backed**: `spaceCount()` = the list size; `spaceTemplate(i)` = the instance's template;
  `spaceSeed(i)` = the instance's seed (space 0 = the world seed, S20 A3); S21's space-state entities are created for
  every instance; `SPCE` lists the instances (template name per `SpaceId`) and the load checks it against the plan.
- **Height** (`int32_t height(SpaceId, int64 x, int64 z) const`): the natural `generator.height(pinned, plan, x, z)`
  with the site patches composed by `SitePatch` for every site of `sitesByChunk[chunk of (x, z)]` whose patch contains
  (x, z):
  - the point is inverse-rotated into the site's frame, `(x - site.x, z - site.z)` by `4 - quarterTurns` (Scope 3
    convention; an index permutation, exact), and tested against the local patch rectangle `[originX, originX +
    sizeX) x [originZ, originZ + sizeZ)`;
  - inside: `h = natural + floorShift((site.baseHeight + patch - natural) * w, 8)` in int64; outside: `h = natural`;
  - an integer generator's result (natural and patched) is asserted (every build) to lie in `±kTerrainHeightLimit`
    (a generator bug, not clamped); `HookGenerator` clamps its hook results itself (E7, Scope 2).
  - `SitePatch` (in `gen/`, linted) works on a plain integer struct `PatchData {originX, originZ, sizeX, sizeZ,
    heights, weights}` (`int32_t` vectors); the pipeline (outside `gen/`) converts each `Prefab`'s patch once, so
    `gen/` never includes `Prefab.h` (json11).
  - This is the point-evaluable height of §12.1 (a). S20's terrain source calls (`TerrainGeneration` of every level,
    the pyramid, `TerrainSystem::sample` on a miss, so `terrainHeight` and S22's lazy base: the base of an edited
    sample read through `TerrainSystem::sample` when no level-0 tile is resident) take S20's `TerrainHeightSource`
    (E7); the pipeline passes one that calls `pipeline.height(space, x, z)`. S20's signatures are not reopened, and
    `vulkan/scene/TerrainTiles` is not touched.
  - **A cheap point query, safe in `parallelEach`** (S22 calls `TerrainSystem::sample` per edited sample on a miss,
    `terrainHeight` runs inside system bodies): `height` is `const`, takes no lock, touches no cache (the content
    cache holds chunks, not points) and allocates nothing; per call one lookup in the plan's immutable `sitesByChunk`,
    at most a few patch tests, and the generator's `height()` (a few noise octaves). Its cost against S20's hook is
    benchmarked (Tests).
  - Pool jobs capture a `HeightSource` value (wrapped as the `TerrainHeightSource`): `shared_ptr<const Generator>`,
    `shared_ptr<const WorldPlan>`, the pinned version and the converted patches (`shared_ptr<const PatchData>`), never
    the `World` (D8).
- **Chunk generation** (pure, thread safe), split by the version that keys it:

```cpp
// Plan-version content (keyed by generator, pinned version, plan hash, chunk); heights shared with S20's level-0 tile
struct ChunkBase { ChunkId id; std::shared_ptr<const std::vector<int32_t>> heights;   // S20's level-0 layout,
                                                                                      // open world only
                   std::vector<int32_t> exclusion; ContentHashes hashes; };          // exclusion rectangles in
                                                                                      // chunk-relative 1/256 m
// Placement-version content (adds the chunk's placement version)
struct TierOneInstance { InstanceKey key; uint16_t meshSlot; int32_t x, z;   // 1/256 m chunk relative
                         int32_t y;                                          // height units, absolute
                         uint16_t yaw, scale; uint32_t flags; };
struct ChunkPlacement { ChunkId id; uint16_t placementVersion;
                        std::vector<TierOneInstance> instances;   // sorted by (slot, counter)
                        GrassMask grass; ContentHashes hashes; };
// Plan only, no chunk generation (Scope 9): the stamp and static-part lists of a chunk
struct StampRecord { uint32_t siteKey; uint32_t entity; int64_t x, z; int32_t y; uint16_t yaw; };  // x, z 1/256 m,
                                                  // y height units = site.baseHeight + prefab y; entity: prefab index;
                                                  // yaw = prefab yaw + quarterTurns * 16384 (Scope 3)
struct StaticPart  { uint32_t siteKey; uint32_t part; std::string mesh; int64_t x, z; int32_t y; uint16_t yaw; };
```

- **Accessors** (named here because S23b uses them; `ChunkId` carries the space):
  - synchronous (main thread: tombstone validation, tombstone-on-edit, `Players` content passes, `completeView`,
    `World::contentHash`; a cache hit or inline generation, never a wait, Scope 5):
    `std::shared_ptr<const ChunkBase> GenerationPipeline::chunkBase(ChunkId) const` and
    `std::shared_ptr<const ChunkPlacement> GenerationPipeline::chunkPlacement(ChunkId, uint16_t placementVersion)
    const` (the base of the same chunk is taken from the cache or generated first);
  - pool form (D8): `ContentSource GenerationPipeline::contentSource(SpaceId) const`, a copyable value (generator,
    plan, pinned version, converted patches, the cache's `shared_ptr`; never the `World`) with
    `void ContentSource::requestPlacement(StreamingPool&, ChunkId, uint16_t placementVersion, int priority,
    std::function<void(std::shared_ptr<const ChunkPlacement>)> onMainThread)`: it submits a pool job (in-flight
    registration when the job starts, attach to a running generation of the same key) whose completion runs at the
    pool's `drain()` on the main thread;
  - `stampList(ChunkId)`, `staticList(ChunkId)` (plan only, Scope 9) and `height(SpaceId, x, z)` (above).
- Generation steps of `chunkBase` (1) and `chunkPlacement` (2-4):

  1. Heights: the level-0 tile at the template's base spacing from `height()` at every sample incl. the shared border.
     Open-world requests are asserted to lie in `[0, sizeChunks)^2` (D12); enclosed chunks have no heights.
  2. Tier 1, per slot, per `counter < candidatesPerChunk(slot)` in order: `candidate()`; reject false; assert
     `0 <= x, z < 65536`; reject inside an exclusion rectangle (the candidate's position inverse-rotated into each
     site's local frame and tested against the local rectangle in 1/256 m, exact); y =
     `terrainSurfaceUnits` of the containing cell's four base samples with `fx`, `fz` in 1/256 m (`fracBits = 8 +
     log2(baseSpacing)`); key `{slot, counter}`.
  3. Grass: `generator.grassMask(placementVersion, ...)` (size asserted `256 / baseSpacing`); cells whose centre lies
     in an exclusion rectangle are 0. Rendering keeps S20's grass rule until S23b switches `TerrainSystem` to these
     masks.
  4. Hashes (Scope 6 table): `heights` in `ChunkBase`, `tier1` and `grass` in `ChunkPlacement`, `stamps` and
     `statics` over the plan lists; the total combines them.
- **Exact conversions** (`ContentConvert`, outside `gen/`, the only place generated integers become floating point):
  - position `chunkOrigin + (x / 256, y / 1024, z / 256)`: int to double, a power-of-two scale, an addition exact
    below `2^53 / 256`;
  - entity rotation (host only, simulation state, not hashed): `Quat(cosQ30(yaw), 0, sinQ30(yaw), 0)` scaled by
    `2^-30` and **normalized in double** (the table's norm error of up to `2^-28` would otherwise sit in saved and
    replicated state; the hashed data stays the integer yaw);
  - scale `scale / 1024`. (Render instances: S23b.)

### 5. Content cache (`engine/content/ContentCache.{h,cpp}`)

- Thread safe (one mutex; work runs outside the lock). Entries: plans and prefabs (never evicted), `ChunkBase` and
  `ChunkPlacement` (LRU by a byte budget, default 256 MB).
- **Keys**: a generator registration identity `GeneratorIdentity {name, version(), hash of the slot table (names and
  flags)}`, then:
  - plans: the pinned version, the instance seed and the hash of the template view (the fields the plan hash covers);
  - `ChunkBase`: the pinned version, the space's **plan hash** (Scope 6; it covers seed, template view, sites and
    prefab content hashes, so two setups sharing a cache can never confuse two plans with equal seed and template
    index) and the chunk;
  - `ChunkPlacement`: the placement version, the plan hash and the chunk.
  Heights are cached under the plan-version key only (one height generation per chunk, whatever the placement
  versions). The first world using a cache records each name's identity; a world registering another
  identity under the same name on the same cache stops with `SAIGA_EXIT_ERROR` (a shared cache would otherwise serve
  one generator's chunks to another, e.g. a test's v1 and v2 builds).
- **In-flight rule** (D8):
  - a streaming-pool job registers its key as in flight **only when it starts executing**, never when queued, so a
    cancelled queued job leaves nothing behind;
  - another pool job asking for an in-flight key attaches to it (its completion runs when the running generation
    finishes; running jobs always finish, cancellation is only checked before start);
  - a **synchronous requester** (the main thread: `completeView`, scenario worlds, `World::contentHash`, the hash
    test) never waits: on a miss it generates inline, even if a pool job is generating the same key;
  - **first insert wins**: a second result for a key is dropped (with `SAIGA_ASSERTS` its section hashes are compared
    with the inserted one's and must be equal).
- `WorldSetup::sharedContent` (`shared_ptr<ContentCache>`, like `sharedData`) shares one cache between the worlds of
  a setup. The scenario runner and `ScenarioWorld` create one per setup (amendment 12: §17.2's "per test process" is
  "per setup"; equal setups in one process may share it explicitly).

### 6. Content hashes (`engine/content/gen/SectionHash.{h,cpp}`, `engine/content/ContentHash.{h,cpp}`)

Placement (round 3, lint): `ContentHashes` (the five section hashes and the total) and the section and total hash
functions live in `gen/SectionHash.{h,cpp}` (linted, included by `ChunkContent.h` and `gen/ChunkGeneration`), because
chunk generation computes them; only the plan hash and the combined plan hash (they read `WorldPlan`, the prefab
content hashes and generator names) stay in `ContentHash.{h,cpp}` outside `gen/`.

FNV-1a 64 through `Hasher` (S06, little-endian). Every section is its own `Hasher` seeded with
`fnv1a64("saiga.content.<section>.v1")`. `u16` values are hashed as `addU32`, `u8` as `addU8`, strings with
`addString` (u64 length, bytes), signed values with `addI32` / `addI64`. Exact byte layouts (fixed before the first
committed file; a change is a new `.v2` seed name and a new file entry):

| Section | Input, in order |
|---|---|
| `heights` | `u32` samples per side, `u32` base spacing, every sample `i32`, row-major z then x; enclosed: only `u32 0`, `u32 0` |
| `tier1` | `u32` instance count; per instance in (slot, counter) order: `u32` slot, `u32` counter, `u32` meshSlot, `i32` x, `i32` z, `i32` y, `u32` yaw, `u32` scale, `u32` flags |
| `grass` | `u32` size, every density byte (`u8`), row-major z then x |
| `stamps` | `u32` record count; per record in (site index, entity index) order: `u32` site key, `u32` entity index, `string` prefab name, `u32` prefab version, `string` class name, `i64` x, `i64` z (1/256 m, space coordinates), `i32` y (height units), `u32` yaw |
| `statics` | `u32` part count; per part in (site index, part index) order: `u32` site key, `u32` part index, `string` mesh name, `i64` x, `i64` z, `i32` y, `u32` yaw |
| total | `u32` space, `i32` chunk x, `i32` chunk z, `u32` pinned version, `u32` placement version, then the five section hashes (`u64`) in the order above |

Plan hash (`saiga.content.plan.v1`): `u64` seed, `u32` pinned version, the template view (`string` name, `u8` kind,
`u32` template index, `i32` base spacing, `i32` size in chunks, `u8` has sea level, `i32` sea level or 0); `u32` site
count, per site in plan order: `u32` key, `string` prefab, `u32` prefab version, `u64` prefab content hash, `i64` x,
`i64` z, `u8` quarter turns, `i32` base height, four `i32` footprint values; `u32` instance count, per instance in list
order: `u32` template index, `u64` seed, `u32` parent, `u32` site key, `u8` link, `u8` origin; then `biomes` and
`rivers`, each `i32` cell size, `i32` originX, originZ, width, height, `u32` value count, every value `i32`.
`cloudCoverage` is not hashed (§12.1). Slot mesh tables are not hashed (presentation, Design notes).

Combined plan hash (`saiga.content.plans.v1`, E11): `u32` space count N, then per `SpaceId` `0..N-1`: `u32` template
index, `string` generator name, `u64` that space's plan hash. `GENR` stores N and this hash; a load recomputes it over
the first N spaces of the current list (prefix rule, Scope 4).

- `World::contentHash(ChunkId)` generates the chunk synchronously (cache) and returns the total; the stamp and static
  lists come from the plan.

### 7. Generation state on clients (`engine/net/EngineMessages.*`, `engine/content/GenerationPipeline.*`)

A client computes terrain heights and content from its own pipeline, so it must generate exactly what the host
generates. Before S26's handshake, the host sends what the client checks as a raw engine message (after S22's messages
in `EngineMessage`), host -> client, `Channel::State`, reliable, with the S16 epoch. S23a only **validates and
refuses** (E11); adopting the host's pins is S26's (its handshake).

- `GenerationState`: epoch (u32), seed (u64), generator count (u16); per generator in name order: name (string),
  pinned version (u16), current placement version (u16), slot count (u16) and the slot names and flags; then the
  space count N (u32) and the combined plan hash (u64, Scope 6).
- **Only when needed**: sent only by a host whose world `writesGenerationState()` (Scope 2: a non-legacy generator, or
  records). Flat and hook worlds send nothing, so the message and byte counts of the existing replication and terrain
  tests are unchanged.
- **Order** (E5, as S22's section 5 states it): on a join (phase 8): `Resync` (only if the epoch is not 0),
  `GenerationState`, the spawn set, then `Replication::sendChunkSnapshot(peer)` (S22's function: S23b's
  `ChunkRecordState`s, then S22's `TerrainChunkState`s). After a host load (inside `onHostLoad`, right after
  `Resync`): `GenerationState`, then `sendChunkSnapshot(peer)`; the loaded entities follow as spawns in the next phase
  8. The client handles `GenerationState` first (the base heights of the following terrain states depend on it).
- Client, validate-and-refuse (untrusted input; a failure is counted in `NetStats::rejected`, logged with
  `SAIGA_LOG_E` category `content` naming the first difference, e.g. "host uses generator test_terrain pinned v1, this
  client pins v2", sets `World::generationMismatch()`, and applies nothing; the host + client variants fail on it):
  - the seed equals its setup seed (D11);
  - the generator names are exactly its registered ones;
  - per generator: the host's pinned version equals the client's pin and the host's current version equals the
    client's `version()` (a client of another binary version, or a host that loaded a save of an older version, is
    refused until S26);
  - every slot table equals its own (names and flags);
  - N equals its space count and the combined plan hash equals its own (the same plans, so the same space table and
    base heights).
  - A valid message changes nothing on the client (it already generates the host's content); it records that the
    check passed (`generationChecked()`, read by S23b's record validation and the tests).
- Host: when the first peer joins and a used generator has `integerOnly() == false`, the host logs once
  (`SAIGA_LOG_E`, `content`): "generator 'prototype' is not integer-only: clients on other machines or compilers may
  see other terrain". (The prototype fixture's co-op runs in one process, S16b.)

### 8. Records, pinned versions and saves (`engine/content/ChunkRecords.{h,cpp}`, `engine/save/`)

- `ChunkRecords` (World-owned, ordered map `ChunkId -> ChunkRecord`, iterated in `ChunkId` order):
  `struct ChunkRecord { uint16_t placementVersion; };` plus the host-only set `stamped` (`std::set<ChunkId>`, the
  "chunks whose prefab entities are stamped" of §12.1). A record fixes the chunk's placement version. S23b adds the
  tombstones to the record (E10: the modification sequence stays in S22's `TerrainEditStore` for every space; no
  sequence in records or `CREC`).
- **Which chunks have a record** (E10: records only where state needs them):
  - a chunk that the host stamps (Scope 9) gets a record with the current version when it is stamped (so a later
    generator upgrade does not move the tier-1 content around a stamped site). This is a **deliberate extension of
    E10** (whose `CREC` holds only chunks with tombstones or a non-current version): the host must save which chunks
    are stamped, so `CREC` also holds stamped records at the current version with the host-only `stamped` byte; they
    are never sent to clients (S23b's `ChunkRecordState` skips records that equal the effective record);
  - S23b adds the chunks with tombstones;
  - a load keeps the saved records (and so a placement version below the current one).
  - No other chunk gets a record: not a chunk a player only visited, not a terrain edit (its sequence lives in S22's
    store), not a chunk of a space whose generator has neither slots nor sites. A chunk without a record is generated
    with the current version by every peer. Flat, hook and prototype worlds therefore have no records and no `CREC`
    (existing saves, hashes and tests unchanged). Placement drift after an upgrade in visited but unmodified chunks is
    accepted until S25 (Risks 6).
- Save blob (`SaveFormat.h`, D7 order `HEAD, SCHM, SPCE?, GENR?, TERR?, CREC?, ENTS, ...`; each with its own format
  version; the blob version stays 1):
  - `GENR` (format 1), written when `writesGenerationState()` (Scope 2): section
    format (u32 = 1); seed (u64); template count (u32), per template in template order the generator name (string);
    generator count (u32), per generator in name order: name (string), pinned version (u16), current version (u16),
    slot count (u16), per slot name (string) and flags (u32); space count N (u32) and the combined plan hash (u64,
    Scope 6, E11).
  - `CREC` (format 1), only when records exist: section format (u32 = 1); space count (u32); per space with records in
    `SpaceId` order: space (u16), record count (u32); per record in `ChunkId` order: x, z (i32), placement version
    (u16), stamped (u8). (S23b's format 2 appends the tombstones per record.)
- Load checks (before the commit; each a load error naming the item, the world unchanged):
  1. the saved seed equals the setup seed ("save seed 42, setup seed 7: the seed is part of the setup");
  2. per saved template, the saved generator name equals the registered one ("space template 'overworld' used
     generator 'prototype', now 'overworld'"); templates beyond the saved count are new (prefix rule);
  3. a pinned or current version greater than `version()` ("save from a newer generator <name> v<N>");
  4. a slot table that is not a prefix-equal match (slots are append-only; the error names the first differing
     slot);
  5. the plans recomputed with the saved pins give the saved combined plan hash over the first N spaces ("generator
     <name> v<N> produces another plan: a released version changed");
  6. **the saved pins give the same space table** (template per `SpaceId`) as the world's current one, over the first
     N spaces, and that table equals `SPCE` (E11: "the save's generator versions produce another space list"); there
     is no run-time rebuild of the space table, so a save whose pins would list other spaces cannot load;
  7. record versions between the pinned and the current version, chunks inside their space's range (D12), records only
     in spaces whose generator has slots or sites;
  8. **a save without `GENR`** (written before S23): accepted only if every template's generator has
     `legacyTerrain() == true` (`FlatGenerator` and every `HookGenerator`, incl. the prototype fixture: heights equal
     the hook the template had); otherwise "save from before S23: template '<name>' now uses generator '<g>'" (S22's
     deltas would otherwise be re-based onto new terrain). It pins the current versions.
  - Then the pipeline state (pins, plans, records, `stamped`) is committed with the world.
- **A host load with other pins** (same space table, other heights or sites, e.g. a v1 save loaded by a binary whose
  newest version is 2 with a plan change): the commit replaces the plans, then drops everything derived from the old
  pins: `TerrainSystem`'s base store and pyramid caches (D3) and the sink's terrain (`clearTerrain`, E1; the camera's
  tiles and pyramid are requested again from the new heights). S22's store holds deltas only (lazy base), so nothing
  of it is derived from the old pins: once the base tiles are dropped, every edited sample reads the loaded pins' base
  through `TerrainSystem::sample`, which is the base the loaded `TERR` deltas were saved against.
  Connected clients receive the new `GenerationState` after `Resync` and are refused (their pins differ, Scope 7).
- `world.hash()` (one byte table; S23b appends its part, S22's terrain block comes first): only when
  `writesGenerationState()` (flat and hook worlds' hashes, incl. S22's committed result hash, are unchanged), after
  S22's block: `u64` seed;
  `u32` generator count, per generator in name order: `string` name, `u32` pinned, `u32` current; `u32` record count,
  per record in (`SpaceId`, `ChunkId`) order: `u32` space, `i32` x, `i32` z, `u32` placement version, `u8` stamped
  (host and replay comparisons; S23b appends per record `u32` tombstone count and per key `u32` slot, `u32` counter).
  The host + client comparison of S16/S22 does not compare records in S23a (clients have none; S23b compares effective
  records).
- `diffSaves` reports `GENR` and record differences by name.
- `data/tests/engine/save_v1.bin` is unchanged and still loads (flat setup: no `GENR`, legacy rule). A new golden save
  `data/tests/engine/save_s23a.bin` (test generator, stamped chunks, an edited chunk) guards the `GENR` / `CREC`
  formats.

### 9. Stamping (`engine/content/ChunkStampSystem.cpp`, `content/Systems.h`, `engine/world/World.*`)

- `chunkStampSystem`: phase 7, `RunOn::Host`, registered first in the phase. It is registered only when some plan has
  sites (otherwise no system, no work, no profiler scope: scenarios and games without sites are unchanged):
  `registerEngineSystems` appends it after phase A of the pipeline has run (Scope 4), so it exists before
  `setupProfiler` creates the per-system scopes.
- Per step: per space, the chunks whose square intersects the disc of `SpaceTemplate::fullRadius` around any `Player`
  in that space (double distances; simulation state on the host, not content), restricted to the space's chunk range,
  visited in `ChunkId` order (only chunks listed in `sitesByChunk` can have a stamp list). For a chunk not in
  `stamped` with a non-empty stamp list, it is stamped, added to `stamped` and gets a record with the current
  version if it has none (Scope 8).
- **The stamp list comes from the plan only**: `stampList(ChunkId)` = for every site of `sitesByChunk[chunk]` and every
  prefab entity whose integer space position (rotated by the quarter turns, offset by the site origin) lies in this
  chunk, sorted by (site index, entity index), with `y = site.baseHeight + prefab y` and the composed yaw (Scope 3). No
  chunk generation; the list is cached per chunk under the plan key. `staticList(ChunkId)` is built the same way from
  the prefabs' `static_geometry`. Both are implemented with the pipeline in step 6 (they are hashed). (Before this
  split, stamping generated every chunk within 1000 m synchronously: about 50 chunks at up to 50 ms each.)
- Spawning (type-erased, through commands, R-SYS-2): a new engine-internal `World::spawnByClassDeferred(info, space,
  position, rotation, std::function<void(World&, Entity)> afterStart)` reserves the entity now (`beginSpawn`) and
  records one command (`makeCommand`) that runs `finishSpawn`, `startObject` (`configure`, `onSpawn`) and then
  `afterStart`, as `World::spawnIn<T>` does for typed spawns. Between steps it applies at once (as `spawnIn<T>`), so
  the existing `spawnByClass` becomes a wrapper of it. Per stamp record:
  - the class by name through `ClassList` (R-REG-2: the data boundary);
  - position and rotation from the exact conversions (rotation normalized in double), in the chunk's space;
  - `afterStart`: the prefab's component JSON through the shared merge path of Scope 3 with the flush-safe add
    `addDefaultById` (it runs inside the phase-7 flush; a component that cannot be applied any more, e.g. `onSpawn`
    destroyed the entity, is logged with `SAIGA_LOG_E` and counted, not fatal);
    then `SpaceLink.target` from the plan: `link` -> the instance with that (site key, link); `parent` -> the
    instance's parent space, arrival converted from the parent site.
  - The record and the `stamped` entry are written by a command recorded before the chunk's spawn commands, so all
    apply at the same phase-7 flush (a phase that does not reach its flush applies none of them).
- Clients never stamp: stamped entities are replicated (§13.2), and the system is `RunOn::Host`.
- Space transitions: `moveToSpace` / `requestTransition` to an instance space need nothing new (all plans exist); the
  arrival chunks are stamped in phase 7 of the arrival step.
- Static geometry of enclosed spaces is drawn from S23b on (render sets); colliders come with S28.

### 10. Integer lint and CI (`tests/test_engine_content.cpp`, `.github/workflows/tests.yml`, `src/saiga/engine/CMakeLists.txt`)

- **Lint as a gtest scanner** (`ContentLint.*` in `test_engine_content`, so it runs on all three CI jobs; it reads the
  source tree through `SAIGA_PROJECT_SOURCE_DIR`, written into the generated header by
  `SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON`, which all three CI jobs set). With an empty project directory the test
  **fails** with the message "ContentLint needs SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON" (not a skip: the Ubuntu jobs fail
  on skips anyway, and a silent pass would hide a misconfigured job).
  - Scanned set (the generator files): every file under `src/saiga/engine/content/gen/`, the interface headers
    `src/saiga/engine/content/{Generator.h,WorldPlan.h,ChunkContent.h}`, `src/saiga/engine/world/SpaceKind.h`,
    `tests/content_test_generator.h`, `samples/games/*/content/` and `samples/games/common/content/` (the sources exist
    on every job even where samples are not built). `tests/content_test_setup.h` (World, classes, meshes) is not a
    generator file and is not scanned. A listed file or directory that does not exist is skipped (the scanner is
    added in step 3, the interface headers in steps 4 and 6, the test generator in step 8, game `content/`
    directories in S23c); each is scanned from the step that adds it. The test prints the scanned file count.
  - Preprocessing: a small tokenizer first normalizes CRLF to LF and joins backslash-newline continuations (keeping
    the original line numbers), then strips `//` and `/* */` comments, string literals (incl. escapes), character
    literals and raw strings `R"d(...)d"`, keeping line numbers. It also classifies **pp-numbers** (a digit, or `.`
    followed by a digit, then `[0-9A-Za-z_.']` and `e+ e- E+ E- p+ p- P+ P-`): a pp-number not starting with `0x` /
    `0X` that contains `.` or an exponent `e` / `E`, or a hexadecimal one with a `p` / `P` exponent, is a **floating
    literal** and refused (`0.7`, `1e3`, `.5f`, `2.`, `0x1p3`; not `0x1e3`, `10'000`, `65536u`). (ECMAScript
    `std::regex` has no look-behind, so this check is in the tokenizer, not a regex.)
  - Token rules (`std::regex`, ECMAScript, word boundaries, over the stripped text), each finding printing file, line
    and token:
    - types and headers: `float`, `double`, `long`, `size_t`, `__int128`, `<cmath>`, `<math.h>`, `<random>`,
      `<chrono>`, `unordered_\w+`, `\w+_distribution`, `\brand\s*\(`, `\bRng\b`;
    - libm and floating helpers, **qualified or not** (`ChunkId.h` includes `math.h` and `<cmath>`, so the names are
      reachable without an include): `\b(std::)?(sin|cos|tan|asin|acos|atan2?|sinh|cosh|tanh|exp2?|expm1|log(2|10|1p)?|
      pow|sqrt|cbrt|hypot|fma|fmod|remainder|round|lround|llround|floor|ceil|trunc|rint|nearbyint|ldexp|frexp|modf|
      fabs)\s*\(`, `std::hash`;
    - double-valued engine names from allowed headers: `kHeightUnit`, `terrainSurfaceDouble`, `chunkSize`,
      `chunkOrigin`, and the double math types of `math.h` (`\b(Vec[234]|vec[234]|Quat|quat|Mat[34]|mat[34])\b`);
    - `SpaceTemplate` (generators see only `GenerationTemplate`).
  - Include allow-list (an `#include` line naming anything else is a finding): `saiga/engine/content/gen/*`,
    `saiga/engine/content/{Generator.h,WorldPlan.h,ChunkContent.h}`, `saiga/engine/world/SpaceKind.h`,
    `saiga/engine/ecs/Ids.h`, `saiga/engine/core/Hash.h`, `saiga/core/math/ChunkId.h`,
    `saiga/core/math/TerrainSurface.h`, `saiga/core/util/assert.h`, `saiga/export.h`, `<cstdint>`, `<cstddef>`,
    `<vector>`, `<array>`, `<string>`, `<string_view>`, `<map>`, `<optional>`, `<algorithm>`, `<memory>`, `<utility>`,
    `<limits>`, a game's own `content/` headers, and for the test generator `tests/content_test_*.h`. Headers reached
    transitively are not scanned; what they make reachable is covered by the token rules above. `Prefab.h` (json11) is
    not allowed: `SitePatch` takes `PatchData` (Scope 4).
  - Self-test `ContentLint.Fixture`: `tests/content_lint_fixture/bad_generator.h` contains each refused token, each
    floating-literal form, an unqualified and a qualified libm call, `size_t`, `kHeightUnit`, `terrainSurfaceDouble`
    and a forbidden include once in code, plus the same tokens in a comment, a string literal and a raw string, plus
    the allowed look-alikes (`0x1e3`, `floorDiv(`, `10'000`); the scanner must report exactly the code lines (a
    committed expected list in the test).
- **UBSan on the Clang job**: CMake option `SAIGA_CONTENT_UBSAN` (default OFF; Clang and GCC only, an error with
  MSVC) adds `-fsanitize=signed-integer-overflow,shift,integer-divide-by-zero,bounds,null,alignment,bool,enum,return,
  unreachable -fsanitize-trap=all` as `COMPILE_OPTIONS` of the source files under `src/saiga/engine/content/` (in
  `src/saiga/engine/CMakeLists.txt`) and to `test_engine_content` (which compiles `content_test_generator.h`). The
  flags are checked with `check_cxx_compiler_flag` at configure time (a compiler without them is a configure error
  naming the flag, not a silent no-op).
  - Trapping needs no sanitizer runtime, so the shared library and every other target link as before; undefined
    behaviour aborts the test (`SIGILL`), which ctest reports as a failure. The Clang CI job configures with
    `-DSAIGA_CONTENT_UBSAN=ON`. Locally the same option, or `-fsanitize=undefined` with the runtime, gives messages.
  - What it covers: signed overflow, shifts, division and the other listed checks in those translation units;
    `bounds` covers only arrays of known size (C arrays, `std::array` element access through them), **not**
    `std::vector` indexing (no claim is made for it; vector indices are covered by the tests' expectations and by the
    hashes). The inline functions of `gen/FixedMath.h` are instrumented where they are instantiated in an instrumented
    unit; a copy instantiated in a non-instrumented unit of the library could be the one the linker keeps, so the
    guarantee for the toolkit rests on `test_engine_content`, whose own (instrumented) unit instantiates and runs every
    toolkit function and the test generator.
- `content_hashes.json` stores the seed as a string (`"12345"`): json11 numbers are doubles and a 64-bit seed would
  lose bits.

### 11. Committed hashes, the hash check and the test setup (`tests/content_test_generator.h`, `tests/content_test_setup.h`, `engine/content/ContentHashFile.{h,cpp}`, `data/tests/engine/`)

- `tests/content_test_generator.h` (header only, linted; included by `content_test_setup.h`):
  - `TestTerrainGenerator(maxVersion, variant = Committed)`, name `test_terrain`, versions 1 and 2, used by **both**
    open-world templates; `plan()` branches on `plan.space.name`:
    - height: fbm plus ridged noise, river channels carved below `seaLevel` from the `rivers` plan grid (filled first
      in `plan()`, so `height()` can be called while placing sites);
    - slots `test_tree` (TombstoneOnEdit, two meshes `test/tree_a`, `test/tree_b`, a hierarchical jittered grid of
      2 m cells: `candidatesPerChunk == 16384` (128 x 128 per chunk, pinned: S23b's message bound test tombstones
      10000 keys of one chunk), about one in 16 accepted, `keptCount` halving per level) and `test_rock` (ReSnapOnEdit,
      mesh `test/rock`, an 8 m jittered grid: `candidatesPerChunk == 1024`);
    - grass from slope and height;
    - plan of `overworld`: site 0 = prefab `test_ruin` at the interior point (1100, 1300) with one quarter turn (chunk
      (4, 5); it creates one `test_dungeon` instance); site 1 = `test_shrine` at (1536, 1536) with two quarter turns,
      its footprint across the corner of chunks (5..6, 5..6); further `test_shrine` sites from `contentHash64` on a
      **4 km** grid (at most 64 sites in the 32 km space) with random quarter turns, skipped where the rotated
      footprint would leave the space or overlap another site; every `baseHeight` is the natural height at the site
      origin (`height()` inside `plan()`);
    - it lists one instance of the open-world template `test_valley` (base spacing 1, 16 chunks), so listed
      instances, base spacing 1 (`fracBits` 8) and the unreferenced-template rule are covered;
    - plan of `test_valley`: no listed spaces, two `test_shrine` sites (prefabs without `spaces`), one with three
      quarter turns;
    - version 2 changes only placement (another jitter seed) and the grass threshold; plan and height equal v1.
    - `variant` (test-only, never in the committed file): `PlanChangeV2` moves site 1 by 256 m in v2 (same space
      table, other heights), `TableChangeV2` lists a second `test_valley` instance in v2 (another space table) and
      `ReleasedChangeV1` moves site 1 in v1 (a "released version changed" bug); used by the load tests of Scope 8.
    - It exercises every toolkit function, all four quarter turns and partial patch weights. Negative coordinates are
      covered by the toolkit tests only (D12: content requests lie inside the space).
  - `AuthoredSpaceGenerator("test_dungeon", "test_dungeon_rooms", 1)` for the enclosed `test_dungeon` template.
- `tests/content_test_setup.h` (E12; not linted; shared by `test_engine_content`, `test_engine_scenarios`, S23b's
  tests and the `generated_site` golden): `contentTestSetup(maxVersion, std::shared_ptr<ContentCache> cache = new)`
  returns the `WorldSetup` with the templates, the generators, `dataDirectory = SAIGA_PROJECT_SOURCE_DIR
  "/data/tests/engine"`, the test classes `TestChest` (adds `Health` in `onSpawn`, `RenderInstance` mesh
  `test/chest`), `TestMarker` (adds nothing; its prefab entity carries a `Health` value) and `TestEntrance`, and
  `renderAssets` registering the test meshes by name (`test/tree_a`, `test/tree_b`, `test/rock`, `test/pillar`,
  `test/chest`: small inline `MeshLods` built in the header, a box, a cone, a flattened box, each with LOD end
  distances of 4 km so a test camera never drops them).
- Prefabs `data/tests/engine/prefabs/test_ruin.json`, `test_ruin.v2.json` (version selection; plan v1 pins prefab v1),
  `test_shrine.json`, `test_dungeon_rooms.json`.
- The test template set: `overworld` (`test_terrain`, 128 chunks, base spacing 2), `test_valley` (`test_terrain`),
  `test_dungeon` (enclosed, 16 chunks, `test_dungeon`). Space IDs: 0 overworld, 1 `test_valley` (listed), 2
  `test_dungeon` (site 0).
- Committed file `data/tests/engine/content_hashes.json`, one entry per setup and version combination; plans are keyed
  by `SpaceId` and name their generator:

```json
{ "format": 1,
  "toolkit": { "trig_table": "0x…", "noise_grid": "0x…", "smooth_q16": "0x…" },
  "setups": [
    { "name": "test", "seed": "12345",
      "generators": { "test_terrain": { "plan_version": 1, "placement_version": 1,
                                        "slots": ["test_tree", "test_rock"] },
                      "test_dungeon": { "plan_version": 1, "placement_version": 1, "slots": [] } },
      "plans": { "0": { "generator": "test_terrain", "hash": "0x…" },
                 "1": { "generator": "test_terrain", "hash": "0x…" },
                 "2": { "generator": "test_dungeon", "hash": "0x…" } },
      "combined_plan_hash": "0x…",
      "digest_block": [0, 0, 8, 8],
      "chunks": { "0:0:0": {"total": "0x…", "heights": "0x…", "tier1": "0x…", "grass": "0x…",
                            "stamps": "0x…", "statics": "0x…"},
                  "0:4:5": {…}, "0:5:5": {…}, "0:6:6": {…}, "0:127:127": {…}, "0:117:3": {…},
                  "1:0:0": {…}, "1:15:15": {…}, "2:8:8": {…} },
      "digest": "0x…" },
    { "name": "test", "seed": "12345", "generators": { "test_terrain": { "plan_version": 1,
                                                                         "placement_version": 2, … }, … }, … } ] }
```

  - Chunks: the origin chunk, the two site chunks incl. the corner site, the last chunk of the space, a far chunk
    (about 30 km from the origin), two chunks of the base-spacing-1 valley and the dungeon's site chunk `2:8:8` (E4:
    origin 2176 m, the centre of chunk 8; D12: all inside their spaces).
  - `digest`: one `Hasher` over the totals of the chunks `[0, 8)^2` of space 0 in `ChunkId` order, followed by every
    site chunk of space 0 **not** in that block, each once, in `ChunkId` order (a site spanning four chunks adds four).
    With at most 66 sites that is 64 + at most about 120 chunks: a broad cross-compiler net next to the per-section
    entries, which localize a difference (heights equal and tier1 different means candidate or surface code). The test
    prints its run time; target below 10 s per CI job (generated on a `JobPool(3)`).
  - `toolkit`: the trig table, a 256 x 256 grid of `fbm` / `ridged` / `gradientNoise` / `valueNoise` samples at
    negative and positive coordinates, all 65537 `smoothQ16` values.
- **`checkContentHashes`** (`engine/content/ContentHashFile.{h,cpp}`, engine function, E12): `ContentHashReport
  checkContentHashes(const WorldSetup& setup, const std::string& file, bool update)`. Per setup entry of the file it
  runs **phase A only** (Scope 4: no `World`, no class, component or mesh checks) for the setup with the entry's seed
  and versions (pins and placement versions forced to the entry's values; a version above the registered `version()`
  is an error), compares every plan hash, the combined plan hash,
  every listed chunk's sections and total, and the digest (the digest's block is stored in the file:
  `"digest_block": [x0, z0, x1, z1]`), and `toolkit` when present. The report lists every mismatch (entry, chunk,
  section, expected, actual). With `update` it rewrites the file in place (the given path; keeps the entry list and
  the chunk IDs, recomputes the values) and reports failure, so an update run never passes by accident. The entry
  list, chunk IDs and digest blocks are authored, so a **missing file is an error** naming the path, also with
  `update`; a new file is committed first as a skeleton with every value `"0x0"` (S23c does so for flight), which the
  update run fills. The function writes to the path it is given (its own tests update a temporary copy); the callers
  that update committed files (`ContentHash.CommittedHashesReproduce`, S23c's `flight_content_hashes`) build the path
  from `SAIGA_PROJECT_SOURCE_DIR` and fail with a message instead of updating when it is empty, so an update never
  writes a copy outside the source tree. Hex strings and the seed are parsed with `std::from_chars` (Scope 3).
- `ContentHash.CommittedHashesReproduce` calls `checkContentHashes(contentTestSetup(2), <source>/data/tests/engine/
  content_hashes.json, SAIGA_UPDATE_CONTENT_HASHES set)`; a mismatch prints the report. Chunks of one entry are
  generated on a `JobPool(3)` (thread independence on every job). Games call the same function from a scenario (S23c).
- Update procedure (AGENTS.md): `SAIGA_UPDATE_CONTENT_HASHES=1 ./build/tests/test_engine_content
  --gtest_filter='ContentHash.*'` rewrites the file in the source tree and fails the run (it cannot pass by accident);
  only in a dedicated commit whose message gives the reason; an existing (setup, versions) entry is never regenerated
  for a released version (§12.1: a change is a bug; the one planned regeneration is S30's); a new version adds an
  entry. Games keep their own file (S23c).

## Out of scope

- Tombstones, their messages, `ChunkRecordState`, content render sets (tier 1, static parts), grass masks from the
  generator for rendering, re-snap and tombstone-on-edit, `World::tierOnePosition`, the MCP tools `chunk_info` /
  `list_sites` / `tombstone` and the `describe_world` / `world_status` additions, the `generated_site` golden: S23b.
- The flight `OverworldGenerator`, `data/flight/content_hashes.json` and its scenario: S23c.
- Adopting the host's pins on a client (S26's handshake; S23a refuses, E11).
- Streaming radii, prefetch, results applied when ready, arrivals (S24). Records of visited but unmodified chunks and
  the "every generated chunk" set of §15.3: S25.
- The chunk store, stored bases, **retiring a version** (needs stored bases): S25.
- The handshake (S26 replaces `GenerationState`), per-chunk snapshots, re-requests (S26).
- Far content and `keptCount` use (S27). Colliders (S28). Pieces (S30). Promotion (S31).
- A runtime content-hash check between peers (§20). Prefab editor tooling (§14).

## Design notes

- **Integer only, enforced three ways**: (1) the lint test refuses float, libm, platform-width types, unordered
  containers and sequential RNGs in content code, and refuses `SpaceTemplate` there; (2) the committed hashes catch
  everything else on GCC, Clang and MSVC; (3) UBSan (trapping) on the Clang job catches signed overflow, invalid
  shifts, division by zero and out-of-bounds indexing of arrays of known size (not `std::vector`, Scope 10), which are
  undefined behaviour and could differ between optimizers. Floating-point
  flags (`/fp:fast`, FMA contraction) cannot change integer content, so the content compiles with each job's defaults.
- **A lint in C++ instead of a CMake script**: CMake's regex has no word boundaries or non-greedy matches and
  `file(STRINGS)` mangles `;` and `[`; `WILL_FAIL` plus a pass regex inverts poorly. A gtest scanner runs on every job
  with the other content tests, strips literals properly and has a self-test.
- **The integer template view** makes the plan hash complete: `plan()` receives exactly the fields that are hashed. A
  generator cannot read `fullRadius` or another double of `SpaceTemplate`, so the lint does not need to understand
  member access.
- **Seed in the plan**: `height`, `candidate` and `grassMask` receive only the plan; `plan.seed` (set by the pipeline,
  hashed) is how they reach the instance seed. Without it they could only use constants. The plan is passed to
  `plan()` in/out with seed and space already set, so `plan()` can call `height()` for natural heights while placing
  sites; a separate `naturalHeight(seed, template, grids, x, z)` helper would duplicate the signature for no gain.
- **Clients check, they do not adopt** (E11): adopting the host's pins would rebuild plans, the terrain store and the
  edit base on a client mid-session; refusing keeps S23a simple and safe until S26's handshake negotiates versions
  before anything is loaded. The combined plan hash catches every plan difference with one u64.
- **The seed is setup, not state** (D11): changing it would change the space table, the space-state entities, the
  plans and every tile after construction, on a host and on a client receiving it after the spawn set. Saves and the
  client check it instead.
- **Stamping from the plan**: a stamp needs the site's pose, the prefab's entity poses and the site's base height,
  all in the plan. Prefab y is relative to `baseHeight`, which the terrain patch also uses, so entities stand on the
  patch without sampling generated terrain. Chunk generation stays on the visual path (S20 tiles, S23b sets).
- **Component values as JSON, applied after `onSpawn`**: the world API's `spawn` already defines the semantics (merge
  into a component the class added, add otherwise). Pre-decoding to binary visit bytes would fill unspecified fields
  with defaults and collide with classes that add the component themselves.
- **Component values are not in the content hash**: stamped entities exist only on the host and are replicated
  (§13.2); their placement and identity are hashed. A changed value in a released prefab is a review rule (§15.3:
  released prefab files are never edited).
- **Slot mesh tables are presentation**: they map `meshSlot` to mesh names for rendering only and may change (a new
  mesh); slot names and flags are content (saved in `GENR`, validated on clients).
- **Site patches and exclusions are composed by the engine** (amendment 4): one function gives `height()`, the tiles,
  `terrainHeight`, S27's far sets and S30's far meshes the same result; a generator cannot forget a site or blend
  differently in its far path. Quarter turns keep patches exact index permutations.
- **Counter-based randomness** (`contentHash64(seed, slot, chunk, counter)`) makes each candidate a pure function of
  its key (§12.1 (b), S27's far sets).
- **Eager plans**: plans are small (sites and grids), and computing them at construction keeps every reader const and
  thread safe; lazy creation from `terrainHeight` inside `parallelEach` or from pool jobs would race.
- **One version counter per generator**: the plan and terrain at the pinned version, candidates and grass at the
  chunk's placement version (>= pinned). `candidate(p, plan)` must accept a plan of any version <= p (§15.3).
- **The hash test uses an engine test generator**: the games are samples, not built on the GCC and MSVC jobs and not
  run by ctest (R-TEST-1). A game generator's cross-compiler guarantee rests on the toolkit (all compilers), the lint
  (which scans games' `content/`) and the game's own committed hash scenario (local, S23c).
- **The prototype fixture stays float** (a `HookGenerator` named `prototype`, step 7): moving `game_world` and racing
  to an integer generator would regenerate goldens and benchmark baselines and move racing's track without showing
  anything new; they move with S27 (amendment 13). Its non-integer status is explicit (`integerOnly() == false`, the
  host warning).

## Implementation steps

Each step builds (warnings as errors, as CI) and passes ctest on its own; the hook field of `SpaceTemplate` stays until
step 7, and records and saves follow stamping (round 3).

1. **Reconcile with S20-S22** (their final code): `SpaceTemplate` (incl. `SpaceKind`), its `prototypeHeight` field
   (`TerrainHeightHook`) and `hasTerrain()`, `TerrainHeightSource` and `clampTerrainHeight`, `TerrainGeneration`,
   `TerrainSystem::sample` / `readiness` / `clearTerrain`, `StreamingPool`, S21's accessors, `SPCE` and its prefix rule,
   the D7 section mechanism and the enclosed chunk range, S22's edit store (`sequence(ChunkId)`, `bumpSequence`, the
   lazy base), `terrainSample`, `onTerrainChanged`, `TERR`, messages and `Replication::sendChunkSnapshot`. Fix the
   names in this plan's Files list.
2. **json11 locale independence**: `parse_number` through `Saiga::ParseNumber`, `dump(double)` through a
   classic-locale stream with precision 17; `Json.NumbersIndependentOfCLocale` in `tests/test_core_json.cpp` (new) or
   the existing json test file.
3. **Toolkit**: `gen/FixedMath.h`, `gen/TrigTable.*`, `gen/IntNoise.*`; `test_engine_content.cpp` with the toolkit
   tests, the lint scanner (missing scan entries skipped) and its fixture; `SAIGA_CONTENT_UBSAN` (with
   `check_cxx_compiler_flag`) and the Clang CI job flag.
4. **Interface and engine generators** (nothing uses them for heights yet): `engine/world/SpaceKind.h` (moved out of
   S20's `SpaceTemplate.h`), `Generator.h` (with `GenerationTemplate`, `hasTerrain`, the in/out `plan`), `WorldPlan.h`,
   `GeneratorList.*` (registration checks, run by the `World` constructor), `gen/FlatGenerator.*`,
   `gen/AuthoredSpaceGenerator.*`, `engine/terrain/HookGenerator.*` (the class and its unit tests, not registered by
   anyone yet), `WorldSetup::generators` / `sharedContent`, `SpaceTemplate::generator` / `fullRadius`,
   `ScenarioWorld::setup()` (F3) and the 128 m override. The hook field stays.
5. **Prefabs**: `Prefab.h`, `PrefabStore.*`, `engine/api/ComponentJson.*` (moved from `WorldApi.cpp`, `WorldApi::spawn`
   uses it; `World::addDefaultById`, the prefab refusals of `Parent` / `NetOwner` / `SpaceLink`), the test prefabs.
6. **Pipeline, cache and hashes**: `GenerationPipeline.*` (phase A first in the constructor, phase B after
   `renderAssets`; the in/out call, the quarter-turn conventions with closed rotated bounds, `stampList` /
   `staticList`, the accessors), `gen/SitePatch.*` (`PatchData`), `gen/ChunkGeneration.*`, `gen/SectionHash.*`,
   `ChunkContent.h`, `ContentCache.*` (plan-hash keys), `ContentHash.*` (plan and combined plan hash),
   `ContentConvert.*`; plan-backed S21 accessors and space-state entities (with `FlatGenerator` the list equals S21's,
   so S21's tests are unchanged). **Interim height source**: `TerrainSystem` takes the pipeline's
   `TerrainHeightSource` for a template whose `generator` is not 0, the template's hook otherwise; `SpaceTemplate::
   hasTerrain()` is `hook != null` or the generator's `hasTerrain()`. Every existing world (generator 0) is unchanged.
7. **The switch**: the hook users move to `HookGenerator` (the common template builder in `samples/games/common/`
   (D2) registers `HookGenerator("prototype", prototypeHeight, {S20's grass thresholds}, false)` and sets the
   template's `generator`, used by `game_world`, racing and flight (flight until S23c) in their own setups; S20's and
   S22's engine tests that set a template hook register a `HookGenerator` with the same hook; S20's prototype template
   built inline from the `PrototypeTerrain` constants keeps its hook and thresholds); `World::spaceHasTerrain` replaces
   `SpaceTemplate::hasTerrain()` at every S20/S21 call site (Scope 2); the template hook field and the interim branch
   are removed. `test_vulkan_world` and `samples/vulkan/` stay on the float hook through D2's pure functions (they
   cannot see engine types). Heights, tests and goldens are unchanged (checked in this step).
8. **Hash test**: `tests/content_test_generator.h`, `tests/content_test_setup.h`, `ContentHashFile.*`
   (`checkContentHashes`, phase A only), the committed `content_hashes.json` (skeleton, then generated once on any
   compiler; the CI run on the other two proves it), the AGENTS.md procedure.
9. **Stamping**: `World::spawnByClassDeferred`, `chunkStampSystem` (appended by `registerEngineSystems` when some plan
   has sites), `ChunkRecords` in memory (the stamped records and the `stamped` set), `SpaceLink` resolution; the
   `Stamp.*` unit tests (between steps and during a step; no save involved).
10. **Records and saves**: `GENR`, `CREC`, the load checks (phase A with the saved pins and phase B before the commit,
    the space-table check, the host re-pin drop), `world.hash` (the generation part under `writesGenerationState()`),
    `diffSaves`, `save_s23a.bin` (from step 9's stamping); the content scenarios `prefab_site_is_stamped` and
    `site_entrance_links_instance_space` (their save/load variant needs `CREC`).
11. **Generation state**: the `GenerationState` message (only when `writesGenerationState()`) at S22's two places
    (before the spawn set on a join, after `Resync` on a host load), the client's validate-and-refuse, the host
    warning, the `GenerationState.*` tests.
12. **Docs**: `docs/engine_design.md` (amendments below) and `AGENTS.md` (`engine/content/`, the hash procedure,
    `SAIGA_CONTENT_UBSAN`, the new tests and the test count).

## Tests

`tests/test_engine_content.cpp` (new, `saiga_test(test_engine_content.cpp "saiga_engine")` inside `MODULE_ENGINE`, no
device):

- `FixedMath.*`: `floorDiv` / `floorMod` / `floorShift` against a reference over `[-1000, 1000]^2` and at the int64
  limits; `mulShift` rounding toward negative infinity; `isqrt64` exact (`r^2 <= v < (r+1)^2`) for all `v < 2^20`, for
  the squares `k^2` and `k^2 ± 1` of every `k < 2^16`, of 10^5 `contentHash64`-sampled `k < 2^32`, and for `2^64 - 1`
  (about 4 * 10^5 checks, not 4 * 10^9); `smoothQ16` over all 65537 inputs: exact endpoints, every value in
  `[0, 65536]`, `s(t + 1) >= s(t) - 1`, within 2 LSB of the polynomial evaluated in `double` (the test may use
  floating point), and the number of 1-LSB decreases printed (1601; not a failure); `mulShift` of the trig series'
  largest `x^2` (about `2.65 * 10^9`) by a Q30 coefficient passes, an overflowing product stops (death test with
  `SAIGA_ASSERTS`); `lerpQ16` endpoints.
- `TrigTable.*`: `sinQ30` within 8 LSB of `std::sin` (tests may use libm; the derived bound is 4 LSB, Scope 1, and the
  observed maximum is printed) over all 2^17 angles; exact symmetries
  (`sin(a + quarter) == cos(a)`, `sin(-a) == -sin(a)`); a yaw quaternion converted by `ContentConvert` has norm 1 within
  `1e-15`.
- `IntNoise.*`: ranges (`[-65536, 65536]`; observed extremes of both signs of `gradientNoise` at the centres of 2^16
  cells at least 60000 in magnitude, Scope 1); continuity per cell size; continuity across 0 for negative coordinates
  (floor, not truncation);
  independent octave seeds; `fbm` with `octaves > cellShift` stops (death test).
- `ContentLint.Sources` and `ContentLint.Fixture` (Scope 10); `Sources` fails with its message when
  `SAIGA_PROJECT_SOURCE_DIR` is empty and skips scan entries that do not exist yet; the fixture has CRLF line ends
  in part and a refused token split by a backslash-newline (reported at its first line).
- `ContentHashFile.*`: seeds and hashes parse with `from_chars` (a seed above 2^63, `"0x"` prefix, trailing garbage
  refused); an update run rewrites a temporary copy and reports failure; a skeleton copy (values `"0x0"`) is filled by
  an update run and then passes; a missing file is an error naming the path (also with `update`); a changed value is
  reported with entry, chunk and section; the check runs without a `World` (phase A only).
- `Prefab.*`: v1 and `.v2` selected by version; a missing version names the file; unknown key, inexact coordinate
  (`0.1`), weight outside 0..256, patch size not matching the arrays, content outside the footprint, `pieces`, nested
  `spaces` in an instance-space prefab: fatal errors naming file and key (death tests, every build); an unknown
  component field, a special component (`Transform`) and an unknown class are fatal at plan time naming prefab, entity
  and field; `Parent`, `NetOwner` and `SpaceLink` in `components` are fatal; a `RenderInstance` value naming an
  unregistered mesh is fatal, a registered one (from `renderAssets`) passes (plan checks run after `renderAssets`); a
  `terrain_patch` in a prefab of an enclosed-space site is a plan error; the prefab content hash changes with a patch
  value and not with whitespace or a component value; a prefab read under `de_DE.UTF-8` (`setlocale(LC_NUMERIC)`, as
  `Ini.IndependentOfCLocale`) gives the same values and hash.
- `Json.NumbersIndependentOfCLocale` (`tests/test_core_json.cpp`): under `de_DE.UTF-8` `1.5` parses as 1.5 and
  `dump()` writes `1.5`; `dump()` of 3000000, 0.1 and 1e300 is byte-equal to the `%.17g` output of the C locale.
- `Generator.*` registration (death tests): more than 200 slots, duplicate slot names, both edit flags, `keptCount(slot,
  0) != candidatesPerChunk`, an unregistered template generator, a duplicate name.
- `Pipeline.*`:
  - site composition: patch value at full weight, partial weight floor-rounded, natural outside, all four quarter
    turns (an asymmetric marker at local (3, 1) lands at (1, -3), (-3, -1), (-1, 3) for q = 1, 2, 3), entity yaw
    composition; the patch's first and last sample rows and columns are patched and the next sample outside is natural
    for every q (the point-in-local-rectangle rule); a footprint rotated outside the space and overlapping footprints
    are plan errors, touching footprints too (closed bounds); a site whose rotated footprint ends exactly on a chunk
    border is listed in both chunks and the second chunk's stamp list is empty;
  - construction order: phase A runs before `registerEngineSystems` (a setup with sites has `chunkStampSystem` and its
    profiler scope; a flat setup has neither, so its system and scope lists are unchanged) and before
    `createStateEntities` (a space-state entity per instance); phase B after `renderAssets` (the `RenderInstance`
    check of the Prefab tests); `GenerationPipeline::build` runs without a `World`;
  - `writesGenerationState()`: false for `FlatGenerator` and `HookGenerator` setups, true for the test setup and for a
    flat overworld with an `AuthoredSpaceGenerator` space; a `legacyTerrain()` generator with a slot is a
    registration error (death test);
  - `plan()` is called with seed and space set and may call `height()` (the test generator's `baseHeight` equals the
    natural height at the site origin); a generator that changes `plan.seed` stops (death test);
  - `spaceHasTerrain`: false for `FlatGenerator` and `AuthoredSpaceGenerator` spaces, true for `test_terrain` and a
    `HookGenerator`; a `HookGenerator` whose hook returns `INT32_MAX` gives `kTerrainHeightLimit` (no assert);
  - candidates inside an exclusion rectangle are rejected and grass cells inside are 0;
  - every candidate's y equals the surface rule recomputed from `height()`; instances sorted, x and z in range;
  - the space list: `spaces[0]` is the overworld; listed instance, site instance (seed with the "site" domain), link >
    0 seed, unreferenced templates in template order; `FlatGenerator` gives S21's list and seeds exactly; a prototype
    setup with a second template keeps it (unreferenced rule); an instance plan with `spaces` is an error;
  - all plans exist after construction; `terrainHeight` from 4 `parallelEach` workers equals the serial result;
  - a chunk request outside `[0, sizeChunks)^2` stops (death test with `SAIGA_ASSERTS`);
  - thread independence: 16 chunks serially and from a `JobPool(3)` in reverse order give equal bytes and hashes;
  - stamp lists come from the plan without generating a chunk (the cache's generation counter stays 0);
  - exact conversion: a position 30 km from the origin round-trips to the integer units.
- `ContentCache.*`: the same `shared_ptr` for equal keys; LRU eviction by budget; heights generated once for two
  placement versions; a held pool job (a test job blocks a worker on a latch) generating key K while the main thread
  requests K synchronously: no wait, inline generation, the first insert wins, both results hash equal; a queued job
  cancelled before start leaves no in-flight entry (a later request generates); a second identity under the same name
  on one cache stops (death test).
- `ContentHash.CommittedHashesReproduce` (the S23a acceptance test) and `ContentHash.PlacementVersionsCoexist` (plan
  v1 with placement v1 and v2: heights, stamps and statics equal; tier1 and grass differ; both committed).
- `ContentSave.*`: `GENR` / `CREC` round trip; records exist only for stamped chunks (a visited chunk without sites
  and an edited chunk get none; a flat world writes no `GENR` / `CREC`); a save loaded into a setup with another seed
  is a load error and the world is unchanged; a world created with `TestTerrainGenerator(1)` (setup A, own cache) with
  site 0 stamped and loaded into `TestTerrainGenerator(2)` (setup B, own cache) keeps plan v1 and the stamped chunk
  (4, 5) at placement v1 (its `tier1` hash equals the committed v1 entry), while the unrecorded chunk (0, 0) gets v2
  (equal to the committed v2 entry), so the test cannot pass on a shared cache; pinned 3 > max 2, a changed slot table
  (names the slot), a changed plan with the same pins (a test variant whose v1 differs), another generator name for a
  template: load errors with the world unchanged; `TableChangeV2`: a v1 save loaded into the v2 binary (whose own
  list has another space) is a load error naming the space list; `PlanChangeV2`: a v1 save loads into the v2 binary,
  `terrainHeight` at the moved site equals v1's, the recording sink received `clearTerrain` and a new pyramid, and
  `terrainSample` of an edited sample there equals the v1 base plus the loaded delta (S22's lazy base); a save with
  fewer templates (a setup with an appended template) loads (prefix rule); a pre-S23 save (`save_v1.bin`, flat) loads; a
  pre-S23 save with `TERR` into a setup whose template moved to `TestTerrainGenerator` is a load error; `world.hash`
  changes with a record, is unchanged for a flat world, and is equal after save/load; `save_s23a.bin` loads.
- `GenerationState.*` (loopback host and client, each peer with its own `ContentCache` where the identities differ):
  equal setups pass (`generationChecked()`, nothing changes on the client); another seed, a host that loaded a v1 save
  (pins v1) with a v2 client, a client binary with an older `version()` than the host's current, a slot-table
  mismatch, a different combined plan hash: rejected, counted, nothing applied, `generationMismatch()` set and the
  log names the difference; a flat or hook world sends no `GenerationState` (the message counts of S16's and S22's
  tests are unchanged); on a join the message precedes the spawn set and after a host load it follows `Resync` (E5
  order: an entity spawned in an instance space arrives in an existing space); the non-integer warning is logged once.
- `Stamp.*`: a class that adds `Health` in `onSpawn` stamped with a prefab `Health` value has that value (no assert);
  a stamp during a step with a prefab component the class does **not** add itself (`TestMarker`) applies at the
  phase-7 flush through `addDefaultById` (no `inStep` assert; the entity is reserved before); between steps at once;
  a stamp whose `onSpawn` destroys the entity is logged and counted.

`tests/test_engine_scenarios.cpp`: `registerContentScenarios` on `contentTestSetup(2)` (`tests/content_test_setup.h`:
the test generator, prefabs, meshes and the test-only classes `TestChest` (adds `Health` in `onSpawn`), `TestMarker`
and `TestEntrance`), as a parameterized gtest over all variants (plain, repeat with `varyThreads`, save/load with
`strictContinuation`, `HostClient`, `HostClientDelay`, `HostClientAsync`):

- `prefab_site_is_stamped` (the S23a stamp acceptance):
  - a player spawns 2 km from site 0: nothing stamped (`count<TestChest>() == 0`);
  - the player is teleported next to the site (`w.edit(player, ...)` on the owner's world) and the scenario steps
    until the host sees it there; then the chest stands at the prefab position converted exactly (`==` on doubles),
    with the prefab's `Health` value; `terrainHeight` at the patch centre equals `baseHeight + patch` exactly; natural
    terrain outside the footprint equals the generator's height;
  - the player leaves and returns: still one chest; the corner site 1 is stamped once although it spans four chunks;
  - on clients: one replicated chest, no stamping, equal `terrainHeight` at the patch centre; in save/load the
    stamped set survives (no second chest).
- `site_entrance_links_instance_space`: `TestEntrance.SpaceLink.target` is the plan's instance of site 0;
  `requestTransition` puts the player in that space, where `test_dungeon_rooms`' entities are stamped; its exit link
  returns the player to the arrival converted from the site's origin and quarter turn (exact).

Existing tests that must stay green unchanged: S20's and S22's terrain tests and S20's engine goldens (on
`HookGenerator`; `.HookResultsClamped` and `.FlatWithoutHook` through `HookGenerator` / `FlatGenerator`), S21's space
tests (plan-backed accessors, `FlatGenerator` list, enclosed spaces through `spaceHasTerrain`), `test_engine_save`
(`save_v1.bin`), S16's and S22's replication tests (flat and hook worlds send no `GenerationState`; their message and
byte counts and S22's committed result hash are unchanged), every scenario without sites (no stamp system
registered), and every JSON writer's output (json11's `dump` is byte-identical in the C locale).

Benchmarks (R-TEST-2; the release build of AGENTS.md; reported, not gating):

- plan time of the test setup and of `game_world`;
- chunk generation time per section for the test generator (report target <= 50 ms per 1 m chunk single-threaded on
  the desktop reference, amendment 11);
- the phase-7 cost of `chunkStampSystem` with nothing to stamp and with one site entering;
- `terrainHeight` through the pipeline against S20's hook (prototype setup);
- `game_world --benchmark flyover` and `forest` before and after (same prototype content through the prototype
  `HookGenerator`: within 5 %, R-TEST-2).

## Done when

The S23a row of §22 (amendment 1):

- **Committed content hashes reproduced on all CI compilers**: `ContentHash.CommittedHashesReproduce` and
  `PlacementVersionsCoexist` pass on the GCC, Clang (with `SAIGA_CONTENT_UBSAN=ON`) and MSVC jobs, and
  `ContentLint.*` passes there (CI).
- **Stamp scenario**: `prefab_site_is_stamped` passes in every variant (CI).

Also: the remaining tests above (CI); `game_world`, racing and flight unchanged (`--scenarios`, `--demo` screenshots;
local); `test_vulkan_world` and S20's engine goldens unchanged (CI); the benchmark numbers (local, in the PR);
`docs/engine_design.md` and `AGENTS.md` updated.

## Verification

- The CI recipe of AGENTS.md with `-DSAIGA_WARNINGS_AS_ERRORS=ON` (as all three CI jobs: the narrowing casts of the
  yaw and slot arithmetic, MSVC C4244 / C4267, are read from the MSVC job), plus a Clang build with
  `-DSAIGA_CONTENT_UBSAN=ON`; samples ON for the games. Each implementation step is checked this way before the next.
- `ctest -R "test_engine_(content|scenarios|save|spaces|terrain.*)|test_core_json" --output-on-failure`.
- The games as in Tests; MSVC is read from CI (no local MSVC).

## Files

New (`src/saiga/engine/content/`): `Generator.h`, `WorldPlan.h`, `GeneratorList.{h,cpp}`, `Prefab.h`,
`PrefabStore.{h,cpp}`, `GenerationPipeline.{h,cpp}`, `ChunkContent.h`, `ContentCache.{h,cpp}`,
`ContentHash.{h,cpp}`, `ContentHashFile.{h,cpp}`, `ContentConvert.{h,cpp}`, `ChunkRecords.{h,cpp}`,
`ChunkStampSystem.cpp`, `Systems.h`; `engine/world/SpaceKind.h`;
`gen/FixedMath.h`, `gen/TrigTable.{h,cpp}`, `gen/IntNoise.{h,cpp}`, `gen/SitePatch.{h,cpp}`,
`gen/ChunkGeneration.{h,cpp}`, `gen/SectionHash.{h,cpp}`, `gen/FlatGenerator.{h,cpp}`,
`gen/AuthoredSpaceGenerator.{h,cpp}`; `engine/api/ComponentJson.{h,cpp}`.

Changed:

- core: `core/util/json11.cpp` (numbers without the C locale).
- engine: `engine/ecs/Ids.h` (`GeneratorId`), `engine/world/{World,SpaceTemplate}.*` (setup fields, `generationSeed`,
  `contentHash`, `generationMismatch`, `generationChecked`, `spaceHasTerrain`, `spawnByClassDeferred`,
  `addDefaultById`, plan-backed spaces, pipeline phase A first in the constructor and phase B after `renderAssets`,
  `registerEngineSystems` appending `chunkStampSystem`; `SpaceKind` moved out),
  `engine/terrain/{TerrainSystem,TerrainGeneration,TerrainEditStore}.*` (the pipeline's height source, `hasTerrain`
  call sites, the re-pin drop on a host load), `engine/net/Replication.cpp` (`GenerationState` on joins and host
  loads), S21's enclosed
  validation and `describe_world.has_terrain` (`spaceHasTerrain`),
  `engine/terrain/HookGenerator.{h,cpp}` (new),
  `engine/save/{SaveFormat.h,WorldSave.cpp,WorldLoad.cpp,SaveDiff.cpp}`, `engine/net/EngineMessages.{h,cpp}`
  (`GenerationState`), `engine/api/WorldApi.cpp` (uses `ComponentJson`), `engine/tools/{ScenarioWorld,
  ScenarioRunner}.*` (`sharedContent` per setup, `fullRadius` override, `ScenarioWorld::setup()` with the member
  renamed `worldSetup`, F3), `src/saiga/engine/CMakeLists.txt`
  (`SAIGA_CONTENT_UBSAN`), root `CMakeLists.txt` (the option).
- tests: `tests/content_test_generator.h`, `tests/content_test_setup.h`, `tests/test_engine_content.cpp`,
  `tests/content_lint_fixture/bad_generator.h`, the json locale test, S20's / S22's engine terrain tests
  (`HookGenerator`), `tests/test_engine_spaces.cpp`, `tests/test_engine_scenarios.cpp`, `tests/CMakeLists.txt`.
- data: `data/tests/engine/{content_hashes.json,save_s23a.bin,prefabs/*.json}`.
- games: the common template builder in `samples/games/common/` (registers the prototype `HookGenerator`); the
  registration in `samples/games/{world,racing,flight}/`.
- CI: `.github/workflows/tests.yml` (Clang job: `-DSAIGA_CONTENT_UBSAN=ON`).
- docs: `docs/engine_design.md`, `AGENTS.md`, this file.

## Risks and open questions

1. **S20-S22 interfaces** are taken from their revised plans (D1-D14, E1-E14); step 1 reconciles them with the final
   code.
2. **The height source** (E7): S20's functions take `TerrainHeightSource`; S23a passes one and reopens no signature.
   The pipeline's source costs a virtual call per sample (measured against S20's hook, Benchmarks).
3. **Trapping UBSan on a shared library**: per-file `COMPILE_OPTIONS` with `-fsanitize-trap=all` need no runtime; if a
   Clang version still pulls a runtime symbol (e.g. through `-fsanitize=bounds` with a non-trap handler), the option
   drops that check and the PR says so.
4. **Eager plans with many instances**: a game listing thousands of instance spaces pays all plans at construction.
   Measured (benchmark); plans are small. If it ever matters, instance plans become a construction-time parallel loop.
5. **Clients of another version are refused** (E11) until S26's handshake, also after a host loads a save of an older
   generator version (its pins then differ from every fresh client's). Accepted: before S26 every peer runs the same
   binary; the refusal names the generator and versions.
6. **Records only for stamped (and, from S23b, tombstoned) chunks** before S25 (E10): placement drift after a
   generator upgrade is possible in chunks that were only seen or visited, or only terrain-edited (their re-snapped
   instances follow the new placement); S25's chunk store records every generated chunk.
7. **The prototype fixture stays float** for `game_world`, racing, the goldens and the baselines until S27
   (amendment 13); its co-op runs in one process. A cross-machine session on it gets the host warning.
8. **Game generators are not CI-tested on GCC/MSVC** (samples): accepted (Design notes, amendment 7).

## Design-doc amendments

To be applied in the implementation change (R-DOC-2):

1. **§22 split** (D13): the S23 row is replaced by three rows:
   - `S23a | Generation pipeline | fixed-point toolkit and integer lint, Generator interface (integer template view),
     world plans with sites and space instances (eager), chunk generation, prefabs, stamping from the plan (entities,
     SpaceLinks), content cache, pinned versions, GENR/CREC, GenerationState, content hashes on all CI compilers with
     UBSan on Clang (§12.1, §15.3) | S20, S21, S22 | committed content hashes reproduced on all CI compilers; stamp
     scenario (prefab entity and terrain patch present) in every variant (CI)`;
   - `S23b | Tombstones and content rendering | tombstones in chunk records with replication (TombstoneModification,
     ChunkRecordState; the sequence stays S22's), content render sets (tier 1, static parts), generated grass masks,
     re-snap and tombstone-on-edit from applied changed boxes, MCP tools, the generated_site golden (§12.1, §12.2,
     §12.5, §13.1, §16.2) | S23a | a tombstoned instance is hidden on the client in the host + client variants;
     generated_site golden with its count check (CI)`;
   - `S23c | Overworld generator (tech demo) | the flight demo's integer OverworldGenerator with ruin sites, the
     game's committed hashes and scenario (§12.1, §18.4) | S23b | flight_content_hashes and the flight demo
     screenshot over a ruin (local)`.
   - S24's dependency `S23` becomes `S23b`; **S27's dependency list gains `S23c`** (E13: S27 moves flight's far
     content onto the generator). References to "S23" elsewhere in the document mean the S23 stages; the §12.1 "S23
     scope" paragraph names S23a for stamping and S23b for tombstones.
   - §19 rows (by their first column; about lines 3192-3211 of today's file): "`FoliageGenerator`,
     `HeightField::generate`, `placeTrees`" -> stage `S23c` for the flight demo (the generator replaces them there)
     and `S27` for `game_world` and racing (amendment 13); "`GrassRenderer`, `grass.vert`": "generated masks (S23)"
     -> "(S23b)"; the §18 games row "level content from the world plan and tier-1 pieces (S23, S30)" -> "(S23a,
     S30)". The §22 S23 row (about line 3331) is the row replaced above. §18.4's flight sentence gains "[planned: S23c]
     on an integer generator with ruin sites" (amendment of S23c).
2. §12.1 `Generator`: `name()`, `slots()` (`GeneratorSlot {name, flags, meshes}`, append-only; meshes are
   presentation), `integerOnly()`, `legacyTerrain()`, `hasTerrain()` (E6: `World::spaceHasTerrain`); `plan(version,
   WorldPlan&)` in/out with `seed` and `space` (`GenerationTemplate`, the integer view) set by the pipeline before the
   call, so `plan()` may call `height()`; `Candidate::flags` are game bits; edit and cosmetic flags per slot;
   `GeneratorId` 0 is `FlatGenerator`; `AuthoredSpaceGenerator`; `SpaceKind` in its own header.
3. §12.1 space instances: the list = the overworld (pushed by the pipeline), generator-listed instances, per site per
   prefab `spaces` entry, then one instance per unreferenced template; site seeds are domain separated
   (`hashCombine(hashCombine(seed, fnv1a64("site")), key)`, link > 0 combined with the link; D11; replaces
   "hashCombine(world seed, site key)"); all plans are computed at world construction; no nested instances; `Site` and
   `SpaceInstance` fields (`quarterTurns`, `baseHeight`, `footprint`; `parent`, `siteKey`, `link`, `origin`); footprints
   inside the space (D12); enclosed sites at the centre of the middle chunk (E4); the quarter-turn and yaw-composition
   conventions (Scope 3); the prefix rule for appended templates.
4. §12.1 contract (a)/(c): `Generator::height` is the natural terrain; the pipeline composes the site patches
   (quarter turns, 0..256 weights) and applies the exclusions to candidates and grass.
5. §7.9: `SpaceTemplate`'s transitional hook field is removed; `generator` and `fullRadius` exist from S23a; the
   hook is adapted by `HookGenerator` (`engine/terrain/`, not content, not linted), which the games' common template
   builder registers for the prototype hook (D2) and engine tests use for their hooks.
6. §12.1 "S23 scope": stamping uses the stamp list from the plan (no chunk generation); the stamped set is a host-only
   set saved in `CREC`; chunk records (placement version, from S23b also tombstones; no sequence, which stays in S22's
   store, E10) exist only for stamped and tombstoned chunks and loaded older versions; `GENR` (with the combined plan
   hash, E11) and `CREC` (§15.1, D7 order).
7. §12.1 content hash: section byte layouts (Scope 6), the committed file, the digest, the update procedure, the engine
   test generator; it runs on the GCC, Clang (UBSan, trapping) and MSVC jobs; games keep their own file and scenario
   (S23c).
8. §12.1 / R-CONTENT-1: the lint test `ContentLint` (tokens, includes, `SpaceTemplate`), the `FixedMath.h` rules,
   `contentHash64` as the only randomness, `SAIGA_CONTENT_UBSAN`.
9. §14 / §15.3: the prefab format (units, exactness, locale-independent numbers, `yaw_deg`, `space_link`, `spaces`,
   `pieces` from S30); component values as JSON applied after `onSpawn` like the API's `spawn`; the prefab content hash
   excludes component values; prefabs live in `<dataDirectory>/prefabs/`.
10. §15.3 and §15.1: one version counter per generator; `candidate(p, plan)` accepts plans of versions <= p; `GENR`
    (seed, template -> generator names, pinned and current versions, slot tables, combined plan hash; written, like
    `GenerationState` and the generation part of `world.hash()`, only when `writesGenerationState()`: a non-legacy
    generator or records) and its load
    checks (seed is setup, D11; saved pins must give the same space list, E11; pre-S23 saves only with
    `legacyTerrain` generators); **retiring a version is [planned: S25]**.
11. §17.10: the report target for chunk generation time (<= 50 ms per 1 m chunk single-threaded, desktop reference).
12. §17.2: `WorldSetup::sharedContent` is the shared cache of generated content, one per setup (not per process);
    scenario terrain comes from the setup's generators (flat by default); scenario worlds lower `fullRadius` to at
    most 128 m; `ScenarioWorld::setup()` gives a scenario its setup (for `checkContentHashes`).
13. §3 and §19: the deviations "float noise" and "OpenMP" are narrowed to the prototype fixture (the `prototype`
    `HookGenerator` registered by `samples/games/common/`, `placeTrees`: `game_world`, racing, the goldens, the
    benchmarks) and their removal moves **from S23 to S27** (explicit deferral). The flight demo is the first game on an
    integer generator (S23c).
14. §13.1 / §13.2: `GenerationState` (pre-S26 subset of the handshake) on `Channel::State` with the epoch, only for
    worlds that write `GENR`, in E5's order (after `Resync`, before the spawn set on a join; right after `Resync`,
    before the chunk snapshot, on a host load); the client validates and refuses (`generationMismatch`), adopting is
    S26's (E11).
15. §12.1 (S20's limit, E7): integer generators' heights including site patches lie in `±kTerrainHeightLimit`,
    asserted; `HookGenerator` clamps hook results with `clampTerrainHeight`.
16. §12.1 / §18.1: generator code lives in `samples/games/<name>/content/` (or `common/content/`), prefabs in
    `data/<game>/prefabs/`.
17. §12.4 "Before S23" and S21's accessors: from S23a the instances come from the plan; `spaceTemplate(i)` and
    `spaceSeed(i)` are plan-backed; `SPCE` lists the plan's instances.

## Review log

Round 1 (Opus, Fable; merged findings in `review_r1_S23.md`; cross-plan decisions D1-D14). Findings 0-21 refer to
that file; those owned by S23b or S23c are logged there.

- 0 Split: done (D13); this plan holds the pipeline part; the §22 row amendment is amendment 1.
- 1 Seed and space list fixed at construction vs. a loaded or replicated seed: the seed is setup (D11); a different
  saved seed is a load error; the "regenerates the saved seed's heights" test is dropped; `GenerationState` (seed,
  pins, slot tables, plan hashes) precedes the spawn set and is validated; the client adopts only pins and only with
  an equal space table (Scope 7).
- 2 Generators never received the seed: `WorldPlan::seed` (set by the pipeline, hashed); the prototype generator calls
  the hook with `plan.seed` (amendment 2).
- 3 Hash chunks outside the space, site at the origin: D12 applied: requests asserted inside `[0, sizeChunks)^2`,
  in-range hash chunks, sites at interior points (one across a chunk corner), rotated footprints validated inside the
  space; negative inputs only in toolkit tests; enclosed chunk range from S21.
- 4 Synchronous generation of every chunk for stamping: stamp lists from the plan only (Scope 9); no chunk generation
  for stamping; measured in the benchmark. Tombstone validation and tombstone-on-edit (which need chunks) are S23b.
- 5 Pre-decoded components conflict with `onSpawn`: JSON kept, validated at plan time with `checkComponentValues`,
  applied after `onSpawn` with the shared merge path; deferred type-erased spawn `spawnByClassDeferred` specified; test
  with a class that adds `Health` itself.
- 6 Space list ambiguity and "one per template": `spaces[0]` pushed by the pipeline; D11 domain separation; the
  unreferenced-template rule for every generator.
- 7 Plan hash misses template inputs: `GenerationTemplate` view (name, kind, index, spacing, size, sea level), hashed;
  slot mesh tables declared presentation only; the lint refuses `SpaceTemplate` in content code.
- 8 Layering: D2 applied: the float adapter is `HookGenerator` in `engine/terrain/` (a non-content place, not linted,
  the hook as constructor argument), registered for the prototype hook by `samples/games/common/` and used by the engine
  tests (which cannot link games); `test_vulkan_world` and `samples/vulkan` stay on the hook; the §3 deferral to S27 is
  amendment 13; host warning for non-integer generators in networked sessions.
- 9 Lint as a CMake script: now a gtest scanner with literal stripping and a fixture self-test (Scope 10).
- 10 Thin cross-compiler guard: trapping UBSan on the Clang job for the content sources and the test; a digest over
  `[0, 16)^2` plus all site chunks; seed stored as a string.
- 11 Byte layouts: exact table for sections, total and plan hash (Scope 6).
- 12 Toolkit traps: `isqrt64` sampled explicitly; `fbm` asserts `octaves <= cellShift`; `smoothQ16` stepwise Horner;
  `gradientNoise` scale derived (`sqrt(2)` in Q16 plus a clamp) and tested.
- 13 Cache in-flight vs. synchronous stamping: D8 rule (Scope 5), held-job test; registration identity in the key and
  the one-identity-per-cache check; heights under the plan-version key; the v1/v2 load test uses separate caches and
  checks committed entries.
- 14 Lazy plans from const/multithreaded paths: all plans eager at construction; validate-then-commit on load.
- 15 Records and S22's sequence: S23a records hold only the placement version plus the host-only stamped set; the
  sequence move and the client record rules are S23b's (logged there).
- 16, 17, 19 (render sets, re-snap trigger, tombstone message validation and size): S23b. Finding 17's "skip content
  passes for generators with no slots and no sites": the stamp system is registered only when needed (Scope 9); the
  render part is S23b's.
- 18 Old saves re-based onto new terrain: `GENR` stores template -> generator names (mismatch = load error); a save
  without `GENR` loads only with `legacyTerrain` generators (Scope 8, check 7).
- 20 Quaternion norm error in state: the entity rotation is normalized in double (host only, not hashed); hashed data
  stays the integer yaw.
- 21 Locale: json11 number parsing and formatting made locale independent (`Saiga::ParseNumber`); a prefab test under
  `de_DE` (CI installs the locale).

Round 2 (Opus 6, Fable 7; merged findings in `review_r2_S23.md`; cross-plan decisions E1-E14 in `decisions_r2.md`,
which override D1-D14 where they differ). Findings 1-18 of the S23a part:

- 1 Merge asserts in a step: `World::addDefaultById`, a flush-safe type-erased add with the typed `addNow`'s rules,
  used by the shared merge path; test with `TestMarker`, a class that does not add the prefab's component (Scope 3,
  Tests).
- 2 Include allow-list vs. the engine's own headers: `SpaceKind` moved to `engine/world/SpaceKind.h`; the allow-list
  gains the content interface headers, `Ids.h`, `assert.h`, `export.h`, `<cstddef>`, `<utility>`, `<string_view>`,
  `<limits>` and the test headers; `SitePatch` takes a plain `PatchData`, so `gen/` never includes `Prefab.h`; the
  scanned set is the generator files plus the interface headers (Scope 10).
- 3 Lint holes: floating literals classified by the tokenizer's pp-number rule (hex excluded); libm names refused
  qualified or not; `size_t`, `kHeightUnit`, `terrainSurfaceDouble`, `chunkSize`, `chunkOrigin` and the double math
  types refused; each in the fixture.
- 4 Host load with other pins: E11 (another space table is load error 6); a host re-pin with the same table drops the
  base store, the sink's terrain (`clearTerrain`) and the materialized edit blocks; tests with the `PlanChangeV2` and
  `TableChangeV2` variants (Scope 8).
- 5 S20 conflicts: E7 (`HookGenerator` clamps, S20's `.HookResultsClamped` valid, the assert only for integer
  generators; S20's `TerrainHeightSource`, no signature reopened, `TerrainTiles` untouched) and E6
  (`Generator::hasTerrain`, `World::spaceHasTerrain`, the S20/S21 call sites listed in Scope 2).
- 6 S21: E4 origin (`2:8:8` in the hash file follows); the prefix rule for `SPCE` and `GENR` stated (Scope 4, Scope 8
  check 2, the combined hash over the first N spaces).
- 7 §22: S27 depends on S23c (E13, amendment 1).
- 8 `GENR` stores the combined plan hash over all space plans (E11, Scope 6); chunk cache keys use the plan hash.
- 9 Client re-pin: E11, validate-and-refuse only (Scope 7); the epoch race is gone with it.
- 10 Test setup: `test_valley` uses `test_terrain` (`plan()` branches on the space name; no listed spaces,
  spaces-less shrine sites); the hash file is per setup with plans keyed by `SpaceId` naming their generator (plan 2
  is `test_dungeon`'s); the shared `tests/content_test_setup.h` (E12) holds classes, templates, generators, data
  directory and test meshes.
- 11 `plan()` cannot call `height()`: `plan(version, WorldPlan&)` in/out with seed and space set before the call;
  `baseHeight` = natural height at the site origin (convention, used by the test generator).
- 12 Hashed conventions: `stampList` / `staticList` move to step 7; quarter-turn direction, inverse and yaw composition
  written (Scope 3); a `terrain_patch` in a space without terrain is a plan error; the digest takes each site chunk
  outside the `[0, 8)^2` block once; the shrine grid is 4 km (at most 64 sites), the digest about 64 + 120 chunks
  with the run time printed.
- 13 Validation order: plans and prefab checks run after class registration and `renderAssets`; `Parent`, `NetOwner`
  and `SpaceLink` refused in prefab components.
- 14 json11 `dump(double)`: a classic-locale stream with precision 17, byte-identical to `%.17g`; tested.
- 15 `GenerationState` only for worlds that write `GENR`: flat and hook-free test worlds keep their message counts.
- 16 Small: 64-bit seeds and hashes parsed with `std::from_chars`; trig bound derived (4 LSB) and tested at 8 LSB; the
  function is `NumberToStringRoundtrip` (no longer used); `ContentLint` fails with a message without the project
  directory; noise extremes sampled at 2^16 cell centres; amendment 1 names the §19 and §22 rows and marks the flight
  sentence for S23c.
- 17 UBSan: the `bounds` claim is limited to arrays of known size (no claim for `std::vector`), the guarantee for
  inline toolkit functions rests on `test_engine_content`, flags checked with `check_cxx_compiler_flag`.
- 18 Placement versions after a host load: moot with E11 (a client with other pins is refused); the loopback tests
  use separate caches where the identities differ.
- Also from E10: records exist only for stamped chunks (S23b adds tombstoned ones); edits and visits create none; the
  sequence stays in S22's store. From E12: `checkContentHashes` is an engine function (Scope 11); the
  `generated_site` golden moves to S23b.
- Rejected: none.

Round 3 (Opus 7.5, Fable 7; findings 1-13 of the S23a part of `review_r3.md`, plus the cross-plan decisions F1-F4 of
that file):

- 1 Constructor order: the pipeline is split into phase A (pure, no `World`: generators, templates, prefab files, space
  list, plans, validation, hashes), run first in the constructor so `createStateEntities` has the space list and
  `registerEngineSystems` appends `chunkStampSystem` (before `setupProfiler`) only when some plan has sites, and phase B
  (classes, components, meshes of prefab entities) after `renderAssets`; a load runs both before its commit (Scope 4).
- 2 One predicate `writesGenerationState()` (a non-legacy generator, or records) for `GENR`, `GenerationState` and the
  hash part; hook worlds are legacy, so "flat and hook worlds unchanged" holds; `legacyTerrain()` generators may have
  no slots or sites (registration check), so the predicate is fixed per setup (Scope 2, 7, 8).
- 3 Steps: the hook field stays until step 7, which moves its users and removes it together with `hasTerrain()`'s
  switch; step 6 uses an interim height source (pipeline for templates with a generator, the hook otherwise); stamping
  (9) precedes records and saves (10), whose step adds the content scenarios and `save_s23a.bin`; `GenerationState`
  is step 11. Each step builds and passes ctest.
- 4 `smoothQ16` is not monotone (1601 one-LSB decreases from `t = 48775`): the test checks range, `s(t + 1) >= s(t) - 1`
  and 2 LSB against the polynomial; the formula is unchanged (Scope 1, Tests).
- 5 `ContentHashes` and the section/total hash functions moved to the linted `gen/SectionHash.{h,cpp}`; plan hashes
  stay in `ContentHash.*` (Scope 6, Files).
- 6 Stamped records at the current version in `CREC` are logged as a deliberate extension of E10 (host-only `stamped`
  byte, never sent to clients; S23b) (Scope 8).
- 7 Rotated half-open rectangles: point tests inverse-rotate into the local frame; rotated areas (`sitesByChunk`, inside
  and overlap tests) use closed integer bounds, conservative by the boundary line; tests at the patch edges and a
  footprint ending on a chunk border (Scope 3, 4, Tests).
- 8 `checkContentHashes` runs phase A only with the file's pins (no `World`, no component checks); a missing file is an
  error (new files start as committed skeletons), the source-tree guard is in the callers (Scope 11).
- 9 Lint: missing scan entries are skipped until the step that adds them; CRLF and backslash-newline handled; fixture
  extended (Scope 10, Tests).
- 10 `mulShift` asserts the product, not each factor (the trig series' `x^2` of about `2.65 * 10^9`) (Scope 1, Tests).
- 11 Accessors S23b uses: `chunkBase(ChunkId)`, `chunkPlacement(ChunkId, placementVersion)` (synchronous) and
  `ContentSource::requestPlacement(pool, ...)` (pool form) (Scope 4).
- 12 Verification with `SAIGA_WARNINGS_AS_ERRORS=ON` and explicit narrowing casts (FixedMath rules); the UBSan design
  note limited to arrays of known size.
- 13 F3 `ScenarioWorld::setup()` (member renamed `worldSetup`); S22's lazy base wording (no materialized blocks; the
  re-pin test checks `terrainSample` against the v1 base); `SpaceKind` comes from S20's `SpaceTemplate.h`; the
  pipeline's `height` is a lock-free, allocation-free point query safe in `parallelEach` (Scope 4). Also from S23b's
  finding 7: `test_tree` is pinned at 16384 candidates per chunk and `test_rock` at 1024 (Scope 11).
- Rejected: none.

Implementation (step 1, reconciliation with the merged S20-S22 code; the names below are used from step 2 on):

- The templates are `WorldConfig::spaceTemplates` (`WorldSetup::config`), not a `WorldSetup` field; an empty list is
  one default template. Today `TerrainSystem` owns the templates and the per-space seeds (`TerrainSystem(templates,
  seeds)`), and `World::spaceTemplate` / `spaceCount` (returns `size_t`) / `spaceSeed` delegate to it; the constructor
  builds them first (before `extractor` and the registries). `TerrainEditStore(templates)` is built from the same list.
- S20's names match the plan: `SpaceTemplate::prototypeHeight` (`TerrainHeightHook`), `SpaceTemplate::hasTerrain()`,
  `TerrainHeightSource` (`core/math/TerrainSurface.h`, with `fromHook`, which clamps), `clampTerrainHeight`,
  `kTerrainHeightLimit` and `kHeightUnit` (`core/math/ChunkId.h`), `SpaceTemplate::enclosed(name, sizeChunks)` and
  `SpaceTemplate::enclosedOrigin(sizeChunks)` (E4). The engine wrapper of the grass rule is `generateGrassMask`
  (`engine/terrain/TerrainGeneration.h`, over `Vulkan::prototypeGrassMask`); the grass thresholds are
  `TerrainViewSettings::grassMinNormalY` / `grassMinAboveSea` (view settings, not the template).
  `TerrainGeneration::heightSource(template, seed)` builds a space's source once in `TerrainSystem`'s constructor.
- `TerrainSystem::sample(space, sx, sz)` (base), `editedSample` (base + delta, `World::terrainSample`),
  `readiness`, `completeView`, `setSink`; `clearTerrain` on the sink. `StreamingPool::submit(job, priority, done)`,
  `cancel`, `drain`, `runOneOnCaller` as D8.
- S22: `TerrainEditStore::sequence(ChunkId)`, `bumpSequence`, `chunks(space)`, `contentHash(chunk)`; the S22 part of
  `World::hash()` is the per-chunk loop at its end (S23a appends after it). `Replication::sendChunkSnapshot(peer)`;
  the join order in `hostReplicate` (pass 7) is `Resync` (epoch != 0), the spawn set, `sendChunkSnapshot`;
  `onHostLoad` sends `Resync`, then `sendChunkSnapshot`. Engine messages end with `TerrainChunkState` in
  `EngineMessage` (`net/NetIds.h`), so `GenerationState` is appended there.
- Saves: `kSaveSections` in `SaveFormat.h` (D7 order with `maxVersion`), `SectionWriter::beginOptional`; `SPCE` is
  written only with more than one space; `WorldLoader` (`WorldLoad.cpp`) runs `parseSave`, `checkSpaceTable`, the
  staging checks, `stageTerrain`, then `commit`, `commitTerrain` and `Replication::onHostLoad`.
- The world API's `checkComponentValues` / `addComponentValues` are free functions in `WorldApi.cpp`;
  `World::spawnByClass(info, space, position, initial)` asserts `!inStep()`; `beginSpawn`, `finishSpawn`,
  `startObject`, `abandonSpawn` exist.
- `ScenarioWorld` holds `WorldSetup setup` (renamed `worldSetup` in step 4, F3) and stores the data store into it
  after creating the host world (`setup.sharedData`); the content cache is stored the same way.
- The hook users of step 7: `samples/games/common/PrototypeSpace.h` (`prototypeSpaceTemplate`), `game_world`'s
  `islands` template (`islandsHeight`), `tests/terrain_test_hooks.h` (`testSpaceTemplate(size, spacing, hook, sea)`,
  `terraformTestTemplate`) and a few tests that set `prototypeHeight` inline (`test_engine_spaces`,
  `test_engine_terrain_edit`, `test_engine_world_api`).

Implementation (step 3, the toolkit):

- `smoothQ16`: the plan's Q16 Horner scheme does not have the properties the plan states for it (computed over all
  65537 inputs: 1601 decreases, the first at t = 48776, five of them by 2 LSB from t = 58555, and errors up to 3.6 LSB
  against the polynomial). Before anything was committed to the hashes it was replaced by the same Horner scheme with
  Q32 intermediates (`r = (6t - 15 * 2^16) * t + 10 * 2^32`, then `r * t >> 16` twice and `r * t >> 32`; every
  product below 2^53): exact endpoints, monotone (0 decreases) and within 1 LSB; `FixedMath.SmoothQ16` checks range,
  monotonicity and 2 LSB.
- The trig table rounds x to nearest (`(i * pi/2 * 2^45 + 2^29) >> 30`) before the series; the observed maximum error
  against libm is 2 LSB (bound 4, test 8). The `gradientNoise` extremes at the 2^16 cell centres are -64291 and 65534.
- The lint scanner lives in `tests/content_lint.h` (included by `test_engine_content.cpp`); include rules: a game's
  `content/` file may include its own headers by a bare name or `content/...`.
- `SAIGA_CONTENT_UBSAN`: the flags are set and checked (`check_cxx_compiler_flag`) in `cmake/Saiga_compiler_flags.cmake`
  (`SAIGA_CONTENT_UBSAN_FLAGS`), applied to `src/saiga/engine/content/**.cpp` and `test_engine_content`.

Implementation (steps 4-6):

- `GeneratorList` keeps each generator's slot table (`slots()` is called once at registration);
  `makeGeneratorList(games, templates)` registers `FlatGenerator` (name `flat`) as 0 and runs the checks.
  `GenerationTemplate` lives in `WorldPlan.h` (the plan holds it); `SpaceInstance::origin` is an enum (`Overworld`,
  `Listed`, `FromSite`, `Unreferenced`) with the plan's values.
- A site's `footprint` is the generator's reservation: the prefab's footprint must lie inside it ("fits"), and the
  rotated site footprint is what the inside, overlap and `sitesByChunk` rules use. `AuthoredSpaceGenerator` reserves
  the whole enclosed space (`[-origin, size - origin)`), so its site is listed in every chunk of the space.
- S22's `BrushKernel.h` had its own `floorDiv` / `ceilDiv`; it now uses the toolkit's (`gen/FixedMath.h`).
- The shared component path: `ComponentJson` holds `specialComponent`, `parentFromJson`, `checkRenderMeshValues`,
  `checkComponentValues` (on a `const World&`), `addComponentValues` and `mergeComponentValues` (the former body of
  `WorldApi::spawn`'s merge loop); the world API's behaviour is unchanged.
- Prefab parsing (`readPrefab`, `parsePrefab`, `prefabContentHash`, `prefabYaw`) is in `Prefab.{h,cpp}`; the
  content hash also covers the prefab's `spaces` (template names). Patch heights are limited to +-2 *
  `kTerrainHeightLimit` relative; the composed check (base + patch within the limit) is a plan error. `yaw_deg` is
  limited to +-3600.
- `ChunkBase` holds the heights and their hash; it has no `exclusion` member: candidates and grass cells test the
  sites' local exclusion rectangles directly (the point-in-local-rectangle rule of Scope 3), so no rotated rectangle is
  stored. `ChunkPlacement` carries all five section hashes and the total; `StampRecord` / `StaticPart` also carry the
  site index and (stamps) the prefab name, version and class name (the `stamps` section hashes them).
- `TerrainSystem` takes one `TerrainSpace {settings, seed, source, hasTerrain}` per space instance (built from the
  pipeline); `TerrainSystem::hasTerrain(space)` replaces its uses of `SpaceTemplate::hasTerrain()`, and
  `World::spaceHasTerrain` exists from step 6. Interim (until step 7): a template with generator 0 and a hook keeps
  the hook (`SpaceGeneration::legacyHook`, seeded with the instance seed as S20/S21).
- `World::generation()` gives the pipeline (`GenerationPipeline`), `World::generators()` its list. The pool form
  attaches to a running generation of the same key by waiting on its `shared_future` inside the pool job (a worker
  blocks; the main thread never does).
- Enclosed templates whose generator has terrain are refused in phase A (`SAIGA_EXIT_ERROR`).

Implementation (step 7, the switch):

- The engine tests name many hooks at run time (`testSpaceTemplate(size, spacing, hook, sea)`), while
  `WorldSetup::generators` is a plain function pointer. `tests/terrain_test_hooks.h` keeps a process-wide hook table:
  `hookGenerator(hook)` gives each distinct hook a fixed ID (1, 2, ... in the order of first use) and
  `registerTestHookGenerators` registers a `HookGenerator("test_hook_<id>", hook, S20's rule, true)` for every entry;
  the setups of those tests set `generators = registerTestHookGenerators`. The world API test's `hills` space
  registers its own `HookGenerator`. `heightSource(template, seed)` (TerrainGeneration) is gone: S20's tests that
  called it use `TerrainHeightSource::fromHook`.
- The games: `samples/games/common/PrototypeSpace.h` has `kPrototypeGenerator` (1), `prototypeGenerator()` and
  `registerPrototypeGenerator`; flight and racing set `generators = registerPrototypeGenerator`; `game_world` also
  registers `HookGenerator("islands", islandsHeight, ...)` as 2 for its second template.
- S21's enclosed-template death test keeps its message: phase A's refusal of an enclosed template whose generator has
  terrain names `SpaceTemplate::enclosed(name, sizeChunks)`.
- Checked: ctest 79/79 on the GPU, the Vulkan and engine tests 38/38 on lavapipe (goldens gate there, unchanged),
  `--scenarios` of flight (18), racing (18) and platformer (36) pass.

## Design changes

(Filled when the stage is done.)

## Results

(Filled when the stage is done.)
