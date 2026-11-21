# S20: Height tiles

Status: implemented (local branch `stage-s20`, rebased onto `master` b23e79ed with S19c; PR pending).

Sections follow the stage plan template of S05; a review log and the design-doc amendments are added at the end. The
cross-plan decisions D1 to D14 of review round 1 and E1 to E14 of review round 2 (shared by S20 to S23; E overrides
D where they differ) are referenced as "D<n>" / "E<n>"; the round-3 decisions F1 to F4 touch S20 only through F4 (the
foliage-texture check, see `FoliageClassNeedsTexture` in Tests).

## Goal

The open-world terrain becomes engine content: integer heights (`kHeightUnit` = 1/1024 m, bounded by
`kTerrainHeightLimit` = 2^23 units) from a per-space height source (today the hook `SpaceTemplate::prototypeHeight`,
wrapped as a `TerrainHeightSource`), one surface rule
shared by every consumer, multi-level tiles (257 x 257 at 1 m or 129 x 129 at 2 m, level L covering 2^L chunks) and an
always-resident 32 m pyramid. The tiles are streamed by camera distance on a new streaming pool and reach `RenderScene`
through the planned `RenderSink` terrain calls. The scene draws them as a CDLOD terrain with skirts, positioned from a
double camera through a per-frame anchor and integer node origins, with chunk-relative grass from per-chunk masks and the
sea-level water plane. `World::terrainHeight` is exact everywhere and drives the kinematic mover. `HeightField`,
`TerrainRenderer`, `GrassRenderer`, `WorldRenderer` and `WorldStatistics.cpp` are removed. `game_world` runs a 32 km
space at 1 m spacing in at most 768 MB of terrain memory (CPU + GPU). The flight demo flies over the height tiles.

S20 also fixes the interfaces S21 to S23 build on: `SpaceTemplate::hasTerrain()` (E6), `TerrainHeightSource` and
`clampTerrainHeight` (E7), `TerrainSystem` (streaming, `readiness`, `terrainUploadsPending`, the single send point for
tiles, pyramids and grass masks, E3), `TerrainStore` (base data only), `StreamingPool` (submit / drain), the GPU terrain
set with its rebuild and `clearTerrain` (E1), the sink calls `patchTerrainTile` / `patchTerrainPyramid` (implemented
and tested here, E2), the benchmark's complete-view mechanism (E9) and the shared test hooks
`tests/terrain_test_hooks.h` (§4, §5, Tests).

## Depends on

S19 (S19a: the prototype passes inside `RenderScene`, the pre-pass, the sections; S19b: `RenderSink`, `VulkanRenderSink`,
`RecordingRenderSink`, the app's view camera and `beforeRender`), S18 (the offset rule of §9.1, per-frame buffers,
`UploadBatch` and the transfer section, `chunkOrigin`/`regionOrigin`), S12 (scenarios and variants), S16 (host + client
variant), S06 (`JobPool`, `Hash`).

## Rules that apply

- R-MATH-1 (simulation heights are integers and double; float only in the render path, relative to the frame anchor or
  the camera), R-MATH-2 (tiny-eigen subset).
- R-DET-1 / §8.6: terrain heights are a pure function of (template, space seed, sample). `terrainHeight` gives the same
  value whether or not a tile is resident, so streaming timing never changes simulation results.
- R-CONTENT-1 starts in S23. The prototype hook keeps its float noise (a known deviation, §3). The surface rule and the
  tile and pyramid formats are already integer and final, because the §12.2 diagonal never changes once S20 fixes it.
- R-PAR-1: generation runs on the streaming pool and nowhere else (synchronous fallbacks: `terrainHeight` on a miss;
  `completeView` runs queued pool jobs on the calling thread through `StreamingPool::runOneOnCaller`). The prototype
  placement keeps its `#pragma omp` inside `vulkan/world/`, the accepted §3 deviation until S23.
- R-MECH-1: budgets, ranges, sea level, base spacing and the grass-mask thresholds are parameters. The engine decides
  none of them.
- R-REG-1 (spaces come from `WorldConfig::spaceTemplates`, an explicit list), R-FILE-1, R-DOC-1/2, R-TEST-1, R-TEST-2
  (terrain and `RenderScene`: benchmarks before and after), R-MCP-1 (changed tools keep their `handleMessage` tests).
- §5.2 / D2: `engine/terrain/` may use `saiga_vulkan` types; `saiga_vulkan` never includes engine headers. Vulkan 1.3
  core only. GCC/Clang/MSVC, export macros (`SAIGA_ENGINE_API`, `SAIGA_VULKAN_API`).
- D1: template validation uses `SAIGA_EXIT_ERROR` (every build), so its death tests run without `SAIGA_ASSERTS`.

## Scope

### 1. Integer heights and the surface rule (`saiga_core`)

- `src/saiga/core/math/ChunkId.h`: `constexpr double kHeightUnit = 1.0 / 1024.0;` (no offset; §12.2) and
  `constexpr int32_t kTerrainHeightLimit = 1 << 23;` (+-8192 m; moved here from S22, so S20's formats rely on it: every
  height difference fits 2^24 and converts to float exactly).
- New `src/saiga/core/math/TerrainSurface.h` (header only, namespace `Saiga`). It is used by the engine
  (`terrainHeight`), the pure tile generation (`vulkan/scene/TerrainTiles`), the prototype placement in `vulkan/world/`,
  and the render tests' CPU reference:

```cpp
// A height hook: pure, thread safe (called on the streaming pool); x, z in integer metres; the result in height units
using TerrainHeightHook = int32_t (*)(uint64_t seed, int64_t x, int64_t z);
inline int32_t clampTerrainHeight(int32_t h);   // to [-kTerrainHeightLimit, kTerrainHeightLimit]

// E7: what the generation functions and TerrainSystem::sample read: a callable over integer metres, the seed already
// bound. Copyable (pool jobs capture it by value; heavy state behind shared_ptr), thread safe, pure. Its results lie in
// +-kTerrainHeightLimit: fromHook clamps (every hook result passes the clamp), other sources (S23a's integer
// generators) guarantee it; the generation functions assert it (SAIGA_ASSERT). S23a builds its pipeline source with
// the same type and does not change the signatures below.
struct TerrainHeightSource
{
    std::function<int32_t(int64_t x, int64_t z)> height;   // empty: no terrain
    int32_t operator()(int64_t x, int64_t z) const { return height(x, z); }
    explicit operator bool() const { return bool(height); }
    static TerrainHeightSource fromHook(TerrainHeightHook hook, uint64_t seed);   // clampTerrainHeight(hook(seed, x, z))
};

// The cell (i, j)..(i+1, j+1) of a height grid is split along its diagonal (i, j)-(i+1, j+1) (§12.2, fixed forever):
// triangle A = (00, 10, 11) where fx >= fz, triangle B = (00, 11, 01) where fz > fx; fx, fz in [0, 1] of the cell.
// Integer form: fx, fz in [0, 2^fracBits] (units of 2^-fracBits of a cell), 0 <= fracBits <= 20, heights within
// +-kTerrainHeightLimit (asserted). The int64 numerator h00 * 2^f + dA * fx + dB * fz is below 2^46; the result in height
// units is rounded toward -infinity (floor division, not '>>' on a negative value: implementation defined in C++17).
int32_t terrainSurfaceUnits(int32_t h00, int32_t h10, int32_t h01, int32_t h11, int64_t fx, int64_t fz, int fracBits);
// Double form: the same triangles; for dyadic fx, fz (fracBits <= 20) every product and sum is below 2^53, so the result
// is exactly numerator / 2^fracBits (in height units, not rounded)
double terrainSurfaceDouble(int32_t h00, int32_t h10, int32_t h01, int32_t h11, double fx, double fz);
// Triangle indices of an n x n grid with this diagonal, counter-clockwise seen from above (the render grids use it)
void terrainGridTriangles(int n, std::vector<uint16_t>& indices);
```

- The prototype splits along (1,0)-(0,1) (`HeightField::height`, `heightmap.glsl`). S20 switches every consumer to the
  §12.2 diagonal, which changes the goldens (regenerated, step 7c).

### 2. Spaces and `terrainHeight` (`saiga_engine`)

- New `engine/world/SpaceTemplate.h` (one value type, R-FILE-1). These are the §7.9 fields S20 implements; the
  remaining fields come with their stages (see Design-doc amendments):

```cpp
enum class SpaceKind : uint8_t { OpenWorld, Enclosed };   // Enclosed: S21 (rejected before)
struct SpaceTemplate
{
    std::string name = "overworld";
    SpaceKind kind = SpaceKind::OpenWorld;
    int baseSpacing = 2;                         // 1 (257 x 257 tiles) or 2 (129 x 129)
    std::optional<int32_t> seaLevel = 0;         // height units; drawn only for a space with terrain (hasTerrain)
    int32_t sizeChunks = 128;                    // open world: multiple of 16 (4096 m root nodes), 16..512
    TerrainHeightHook prototypeHeight = nullptr; // null: flat, y = 0, no terrain drawn (D1)
    // E6: the one test every terrain rule keys on (terrain and sea drawn, tiles requested, readiness, the mover's
    // idle rule, S21's describe_world.has_terrain and enclosed validation). Today prototypeHeight != nullptr; S23a
    // reimplements it as Generator::hasTerrain(). Inline (SpaceTemplate is a header-only value type: no exported
    // symbol, so the MSVC DLL build needs no export macro)
    bool hasTerrain() const { return prototypeHeight != nullptr; }
};
```

- Validation (`validateSpaceTemplates`, in the `World` constructor, `SAIGA_EXIT_ERROR` with a message naming the field;
  D1): more than one entry (message naming S21), `Enclosed` (naming S21), `baseSpacing` not 1 or 2, `sizeChunks` of an
  open-world space not a multiple of 16 or outside 16..512, `seaLevel` beyond `kTerrainHeightLimit`. The size rule is an
  open-world rule (the root nodes); S21's enclosed spaces define their own chunk range (D12). The S21 templates fit:
  the cellar is `Enclosed`, the islands space is 128 chunks; S22's scenario templates use 16 chunks.
- `WorldConfig::spaceTemplates` (`std::vector<SpaceTemplate>`). An empty list means one default template (flat, no hook,
  D1), so existing games and scenarios are unchanged.
- Space seed: space 0 uses `WorldSetup::seed` (consistent with §12.1, where space 0's plan takes the world seed;
  amendment A3, D11). `World::spaceSeed(SpaceId)` and `World::spaceTemplate(SpaceId)` are new accessors.
- `double World::terrainHeight(SpaceId space, double x, double z) const` (§7.10, §12.2) = `terrain().height(...)`.
  x and z are clamped to the space `[0, sizeChunks * 256]` (D12). Sample coordinates are `x / baseSpacing` (exact for
  spacing 1 or 2). The cell index is `min(floor(x / baseSpacing), n - 1)` with `n = sizeChunks * 256 / baseSpacing`
  cells per side and `fx = x / baseSpacing - cell` in `[0, 1]`, so x exactly at the far edge uses the last cell with
  `fx = 1` (the sample at the edge exists: tiles share their borders). The three samples of the containing triangle come
  from `TerrainSystem::sample` (resident level-0 tile when there is one, else the space's `TerrainHeightSource`) and are
  interpolated with `terrainSurfaceDouble`; the result is multiplied by `kHeightUnit`. Both paths give bit-equal values;
  a test checks this. No memoization on a miss (the function is const and may run inside `parallelEach`). A space
  without terrain (`!hasTerrain()`) returns 0, and so does `TerrainSystem::sample` for it (S22's `terrainSample` and
  S23a's pipeline rely on a base of 0 there). Cost on a miss: 3 hook calls (about 2 µs with the prototype hook).
- `characterMoveSystem` (§8.2, §9.3): a moving character gets `y = world.terrainHeight(t.space(), x, z)` instead of 0.
  Idle characters (zero intent) in a space with terrain: their y is written only when it differs from `terrainHeight`
  at their x, z, so a spawned or teleported character is snapped in the next step and a resting one is never dirtied.
  In a space without terrain the idle rule is today's (not written), so every existing test keeps its expectations
  (e.g. `tests/test_engine_gameobject.cpp` `PlayerControlAndCharacterMove` spawns at y = 3 on the flat default; none
  of the current tests uses a hook). Cost: one `terrainHeight` per idle character per step in a terrain space (about
  2 µs each on the miss path of a sink-less host; `game_world` has one player). A character legitimately above the
  surface does not exist before collision (S28 replaces the mover). The system comment changes accordingly.
- Query counts (the cost of the miss path, measured in step 7b with the profiler): `game_world`'s player 1 per step
  (resident tiles: hits). The flight and racing contexts are shared, `World`-independent objects (`FlightContext` and
  `RacingContext` are created once and used by the system lambdas of both coop worlds, by `flightModelOf` / `Vehicle`
  through a captured context pointer and by the render side's chase cameras), so they do not call
  `World::terrainHeight`: they hold the space seed and base spacing and call the pure `prototypeSurfaceHeight` (§6),
  which equals `World::terrainHeight` for that template bit for bit (`Terrain.PrototypeSurfaceEqualsTerrainHeight`).
  It has no resident fast path: every query costs 3 hook calls (about 2 µs). Racing per vehicle 2 x 4 wheel mounts x
  4 sub-steps + 1 = 33 per step (about 70 µs); flight 2 per aircraft per step (ground and the autopilot's look-ahead)
  plus one per rendered frame for the chase camera; racing's render meshes (asphalt strips, curbs, cones, gates) query
  a few thousand points once at init (a few ms). Racing's site search (`Track::findSite` over the 8 km area) makes about
  1.1 M queries at startup: about 2 s instead of today's bilinear lookups, stated in the PR (the scenarios start that
  much later). No game-side cache in S20; if the racing host step spends more than 0.5 ms in terrain queries, the PR
  reports it and a per-context tile cache or S24's residency is the follow-up (no pin API in S20).
- `ScenarioWorld` uses the setup's templates (seeded terrain when the setup has a hook, §17.2). It never streams: every
  scenario world has a `RecordingRenderSink` (`ScenarioWorld::giveSink`), but only `EngineApp` (and tests that call it
  directly) calls `updateView` / `completeView`, and a sink alone requests nothing and starts no pool (§4.3).

### 3. Streaming pool (`engine/core/StreamingPool.{h,cpp}`, §8.5, D8)

```cpp
class SAIGA_ENGINE_API StreamingPool
{
   public:
    explicit StreamingPool(int workers);         // 2 by default (§8.5: 2-3); workers lower their priority (best effort:
                                                 // SetThreadPriority / setpriority, #ifdef per platform)
    ~StreamingPool();                            // cancels queued jobs, sets `cancelled` of running ones and joins them;
                                                 // completions and exceptions not drained are discarded
    using Job        = std::function<void(const std::atomic<bool>& cancelled)>;   // worker thread
    using Completion = std::function<void(bool cancelled)>;                      // main thread, in drain()
    uint64_t submit(Job job, int priority, Completion done);  // lower runs first; FIFO among equal priorities
    bool cancel(uint64_t id);                    // queued: dropped, `done` is never called, returns true; running: its
                                                 // `cancelled` flag is set (a job may stop early or ignore it and
                                                 // finish), `done(true)` is called at the next drain, returns false
    // Main thread: calls the completions of the finished jobs in finishing order. A job that threw: its exception is
    // captured per job and rethrown here (its completion is not called; the completions behind it stay queued for the
    // next drain)
    void drain();
    // The calling thread runs the first queued job itself (priority order), exactly as a worker would (its completion
    // or exception is queued for drain). Returns false if nothing was queued. completeView helps the workers with it
    bool runOneOnCaller();
    void waitIdle();                             // blocks until no job is queued or running; never throws (exceptions
                                                 // stay captured until drain)
    bool waitIdleFor(double seconds);            // the same with a timeout: false if jobs remain
    int queued() const; int running() const; int finished() const;
};
```

- Jobs capture copies (the `TerrainHeightSource`, the `TerrainLayout`, the `RegionId`), never the `World` or the store,
  and write only into their own result object (captured by `shared_ptr`), which the completion moves into the store.
- The pool keeps no registry of what a job generates: its own bookkeeping (queue entry, running count) ends when the
  job ends, whatever the outcome. Callers track their requests themselves. `TerrainSystem` records a request (key ->
  job id, counted against `maxInFlight`) at `submit`; the entry is removed by its completion (finished or
  `done(true)`, a cancelled result is dropped), or immediately when `cancel` of a queued job returns true. S23a's cache
  registers a key as in flight when its job starts executing; that is S23a's rule inside its own job code (D8), and its
  jobs ignore `cancelled` (S23a: running jobs always finish; the completion still reports the cancel).
- A generation exception is fatal for `TerrainSystem`. `drain` rethrows the captured exception without knowing the
  request, so `TerrainSystem`'s job lambdas name it themselves: each job body is wrapped in `try { ... } catch (const
  std::exception& e) { throw std::runtime_error(describe(request) + ": " + e.what()); } catch (...) { throw
  std::runtime_error(describe(request) + ": unknown exception"); }` (`describe`: "terrain tile space 0 region L2
  (4, 6)", "terrain pyramid space 0", "grass mask space 0 chunk (17, 9)"). `updateView` / `completeView` catch the
  rethrow of `drain` and stop with `SAIGA_EXIT_ERROR` and that message. A height source is pure and has no recoverable
  failure, so no request entry or `maxInFlight` slot outlives a failed job.
- Ownership: `TerrainSystem` owns the pool (`std::unique_ptr`, created lazily on the first submit from `updateView` /
  `completeView`; a world whose app never calls them, every scenario world and the headless hosts, starts no threads,
  with or without a sink). `~TerrainSystem` destroys the pool first (explicitly, before the stores), so no job outlives
  the data its completion would touch.

### 4. Tiles, pyramid, store and visual streaming

#### 4.1 Data types and pure generation (`vulkan/scene/`, below the engine, D2)

Data types live in `saiga/vulkan/scene/RenderSceneTypes.h`, because the sink and `RenderScene` consume them. The engine
store keeps the same objects by `shared_ptr` (no copies between store, sink and recording):

```cpp
struct TerrainLayout                     // fixed per GPU terrain set (D5): from the template + TerrainViewSettings
{
    int32_t baseSpacing = 2, sizeChunks = 128;                      // from the SpaceTemplate
    int32_t tilesPerLevel = 48, grassMaskLayers = 16;               // from TerrainViewSettings
    static constexpr int32_t pyramidSpacing = 32;                   // metres, for both base spacings
    int samplesPerSide() const { return 256 / baseSpacing + 1; }   // 257 / 129, shared borders
    int streamedLevels() const;          // levels with spacing < 32 m: 5 (1 m), 4 (2 m)
    int rootLevel() const;               // CDLOD level of the 4096 m root nodes: 7 (1 m), 6 (2 m)
    bool operator==(const TerrainLayout&) const;
};
struct TerrainTileData                   // RegionId level L < streamedLevels: 2^L chunks at spacing baseSpacing * 2^L
{
    int32_t minHeight = 0, maxHeight = 0;                          // height units, of these samples
    std::shared_ptr<const std::vector<int32_t>> heights;           // samplesPerSide^2, row j = z, index j * n + i,
                                                                   // sample (i, j) at regionOrigin + (i, 0, j) * spacing
};
struct TerrainPyramidData                // the whole space at 32 m: (8 * sizeChunks + 1)^2 samples (1025^2 for 32 km)
{
    TerrainLayout layout;
    std::optional<int32_t> seaLevel;     // the water plane until S39's setWater (amendment A9)
    int32_t minHeight = 0, maxHeight = 0;
    std::shared_ptr<const std::vector<int32_t>> heights;
};
struct GrassMaskData                     // one chunk: (256 / baseSpacing)^2 density texels, one per base cell
{
    int32_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> density;           // 0..255
};
```

New `vulkan/scene/TerrainTiles.{h,cpp}` (pure functions of their arguments, thread safe, no engine types; the engine's
`TerrainGeneration` wraps them, the renderer tests call them directly with the integer test hooks of
`tests/terrain_test_hooks.h` through `TerrainHeightSource::fromHook`):

- `TerrainTileData generateTerrainTile(const TerrainHeightSource&, const TerrainLayout&, RegionId)` evaluates the
  source at `regionOrigin + (i, j) * baseSpacing * 2^L` (integer metres). Coarser levels are point subsamples of
  the same function, so a level-L+1 sample equals the level-L sample at the same point; CDLOD morphing relies on this
  (and S22 assumes it).
- `TerrainPyramidData generateTerrainPyramid(const TerrainHeightSource&, const TerrainLayout&,
  std::optional<int32_t> seaLevel)` evaluates the source at 32 m steps (point samples).
- `GrassMaskData prototypeGrassMask(const TerrainTileData& level0, const TerrainLayout&, std::optional<int32_t>
  seaLevel, double minNormalY, double minAboveSea)` is the transitional rule (S23b takes generator masks): 255 where
  both triangles of the base cell have `normal.y >= minNormalY` and the cell's lowest sample is
  `>= seaLevel + minAboveSea` (`minAboveSea` in metres), else 0. It reads only `level0.heights`, so S22 calls it with a
  tile of merged heights (E3) and S23a's prototype `HookGenerator` with its chunk base.

#### 4.2 `TerrainStore` and `TerrainGeneration` (`engine/terrain/`)

- `engine/terrain/TerrainGeneration.{h,cpp}`: `TerrainHeightSource heightSource(const SpaceTemplate&, uint64_t
  seed)` (`fromHook(prototypeHeight, seed)`; empty without terrain), `generateTile(const TerrainHeightSource&, const
  TerrainLayout&, RegionId)`, `generatePyramid(...)`, `generateGrassMask(...)`, thin wrappers over `TerrainTiles`.
  `TerrainSystem` builds each space's source once; S23a replaces only `heightSource` by its pipeline source.
- `engine/terrain/TerrainStore.{h,cpp}` (one per space; owned by `TerrainSystem`): base data only (D3), never mutated
  by edits:
  - the resident tiles by `RegionId` (all levels) with the frame in which each was last desired (LRU);
  - the pyramid, generated lazily on the space's first view and kept CPU-cached while the world lives (D5);
  - the grass masks by `ChunkId`.
  It is mutated only on the main thread between steps (asserted with `SAIGA_ASSERTS`). `TerrainSystem::sample` reads
  its level-0 tiles. The tiles the sink holds are always a subset of the store's (removals are sent in the update that
  evicts, §4.3), so store, sink queue and recording share the same height vectors.

#### 4.3 `TerrainSystem` (`engine/terrain/TerrainSystem.{h,cpp}`, owned by the `World`, `World::terrain()`)

```cpp
struct TerrainViewSettings                 // game policy (R-MECH-1); EngineAppSetup / game code may change it
{
    double residencyRange0 = 400;          // level k tiles are desired within residencyRange0 * 2^k (3D distance from
                                           // the camera to the tile box; heights from the pyramid + 64 m margin)
    int tilesPerLevel = 48;                // budget per streamed level = GPU atlas layers (part of the layout)
    int grassMaskLayers = 16;              // budget of grass masks = GPU mask layers (part of the layout)
    int uploadsPerFrame = 8;               // tile sink calls per updateView (completeView sends everything)
    double grassRadius = 80;               // chunks within this radius get grass masks
    double grassMinNormalY = 0.82, grassMinAboveSea = 0.4;
    int workers = 2, maxInFlight = 8;
};
class SAIGA_ENGINE_API TerrainSystem
{
   public:
    double height(SpaceId space, double x, double z) const;          // World::terrainHeight
    int32_t sample(SpaceId space, int64_t sx, int64_t sz) const;     // base-grid sample of the BASE data: resident
                                                                     // level-0 tile or the space's height source; 0
                                                                     // for a space without terrain. A cheap point
                                                                     // query, const and thread safe (parallelEach).
                                                                     // S22's World::terrainSample = this + the edit
                                                                     // delta; S23a replaces only the source
    // Main thread, between steps (EngineApp, every rendered frame, with the view camera). In order:
    //  1. pool.drain() moves finished tiles / masks / pyramids into the store (an exception is fatal, §3);
    //  2. a space other than the current view space (or a changed layout) starts a new GPU terrain set (below); a
    //     view space without terrain sends clearTerrain() once (if anything was sent) and stops here: nothing is
    //     requested or sent for it;
    //  3. the desired set (it needs the pyramid's heights for the tile boxes): while the space's pyramid is not in the
    //     store, only the pyramid is requested and steps 3-6 are skipped (the desired set is empty, readiness 0).
    //     Otherwise per streamed level the tiles within the range, sorted by distance and truncated to tilesPerLevel
    //     (nearest first; the truncated count goes to Stats); grass masks for chunks within grassRadius, truncated to
    //     grassMaskLayers;
    //  4. requests: missing desired tiles by distance / range ratio (coarse levels first at equal ratio), then the
    //     masks of desired chunks whose level-0 tile is in the store (a mask job needs it), at most maxInFlight
    //     requests in flight; requests no longer desired are cancelled;
    //  5. evicts tiles that are not desired, LRU first, so that each level holds at most tilesPerLevel; a desired tile
    //     is never evicted (the truncation in 3 makes this always possible);
    //  6. sends every removal of 5 and of masks that left the radius (not budgeted: the sink never holds a tile the
    //     store dropped), then at most uploadsPerFrame tile additions by distance and the mask additions.
    void updateView(SpaceId space, const Vec3& camera);
    // Blocks until the desired set of this camera is resident and sent ("Complete-view loop" below: pyramid, then
    // tiles, then masks, repeated until nothing is missing; no maxInFlight or uploadsPerFrame limit). Returns false on
    // timeout (what is resident is still sent); the caller decides: tests, goldens, the first frame, the
    // SAIGA_SCREENSHOT capture and benchmark frames stop with SAIGA_EXIT_ERROR, an MCP screenshot answers with an error
    // (§4.3 EngineApp). The main thread (and with it the MCP poll) is blocked meanwhile. Ends a running load of S21's
    // loading screen (D6). A space without terrain returns true at once
    bool completeView(SpaceId space, const Vec3& camera, double timeoutSeconds = 120);
    // D6: the fraction of (the pyramid + the desired tiles of this camera whose box is within radius) that have been
    // sent to the sink; 0 while the pyramid is not sent; 1 for a space without terrain. Monotone for a fixed camera.
    // "Sent" is not "recorded": a load ends only when terrainUploadsPending() is 0 too (E8, S21)
    double readiness(SpaceId space, const Vec3& camera, double radius) const;
    size_t terrainUploadsPending() const;      // the sink's bytes not yet recorded (RenderSink::terrainUploadsPending);
                                               // 0 without a sink
    void setViewSettings(const TerrainViewSettings& settings);   // see "Settings changes" below
    TerrainLayout layout(SpaceId space) const;                   // template + current settings
    const TerrainPyramidData* pyramid(SpaceId space) const;      // CPU cache; null before the space's first view
                                                                 // (no updateView / completeView for it yet); the
                                                                 // same pointer while the world lives (D5)
    struct LevelStats { int desired = 0, truncated = 0, resident = 0, sent = 0; };
    // generated counts every finished generation (tiles, masks, pyramids); pyramidsGenerated only the pyramids (S21's
    // cache-reuse checks)
    struct Stats { std::array<LevelStats, 8> levels; int masksDesired, masksSent, pending, uploadsLastUpdate;
                   uint64_t generated, pyramidsGenerated, evicted; size_t cpuBytes; SpaceId viewSpace;
                   double lastCompleteMs; };
    const Stats& stats() const;
    bool poolCreated() const;                  // the lazy StreamingPool exists (tests: a sink alone starts none)
};
```

- **Complete-view loop.** A single pass cannot complete a first view: the desired set needs the pyramid, and a mask
  needs its chunk's level-0 tile in the store. `completeView` therefore runs, after `updateView`'s steps 1 and 2, with
  `deadline = now + timeoutSeconds`:
  - (a) the pyramid: if the space's pyramid is not in the store, it is submitted (unless already in flight), then
    "run" (below) and `drain`, so it is stored and sent;
  - (b) the desired set is computed (step 3), every missing desired tile is submitted (no `maxInFlight` limit), run,
    drained;
  - (c) the masks of desired chunks whose level-0 tile is now stored are submitted, run, drained;
  - (d) eviction and every send (steps 5 and 6 without the `uploadsPerFrame` limit); then the desired set is
    recomputed: if any desired tile or mask is still missing (a mask whose tile arrived in (c), a result that was
    cancelled), the loop goes back to (b); otherwise it returns true.
  "Run" = while the deadline is not reached and the queue is not empty, the calling thread executes one queued job
  (`runOneOnCaller`); then `waitIdleFor(deadline - now)` for the workers' running jobs. The deadline is checked
  **before** each caller job: no caller job is started when the remaining time is <= 0, and a caller job that is
  running when the deadline passes is finished, not preempted (a job cannot be interrupted), so the overrun is bounded
  by one job (46 ms for a 1 m prototype tile). On timeout, whatever finished is drained and sent, and it returns false.
- **Who streams.** Only `updateView` and `completeView` request anything, and only `EngineApp` calls them (for its view
  world; tests call them directly). `World::setRenderSink` alone requests nothing and creates no pool: a sink receives
  what the store already holds (nothing, in a world that never had a view). So every scenario world (each has a
  `RecordingRenderSink`), headless hosts and `--coop`'s hidden host never stream.
- **Single send point (D3, E3).** Every tile, pyramid and grass mask that reaches a sink passes
  `TerrainSystem::sendTile(RegionId, const TerrainTileData&)` / `sendPyramid(SpaceId, const TerrainPyramidData&)` /
  `sendGrassMask(ChunkId, const GrassMaskData&)` (private): pool results, a new sink, a GPU-set rebuild, a settings
  change. S22 merges its edits there (a merged copy for tiles that intersect edited chunks; for masks of edited chunks
  the mask it recomputed with `prototypeGrassMask` from merged heights, E3); the store stays base data.
- **GPU terrain set (D5, E1).** The sink holds terrain for one space at a time. Step 2 of `updateView` starts a new set
  when the view space changes or the layout changes: the sent state of every tile and mask is dropped (the sink drops its
  tiles and masks itself when it receives a pyramid of another space or layout, §5; no per-tile removal calls), the old
  space's streamed CPU tiles and masks are dropped (its pyramid stays CPU-cached), and `sendPyramid` sends the new
  space's pyramid (generated lazily on first view) before any of its tiles. When the new view space has no terrain
  (`!hasTerrain()`: hook-less, later S21's enclosed spaces), `TerrainSystem` sends `clearTerrain()` instead (only if
  a set was sent), the sink drops its set and draws no terrain, grass or water until the next pyramid. In S20 only the
  layout path occurs at run time (one space); `clearTerrain` is implemented and tested on `RenderScene` and
  `RecordingRenderSink` directly, and S21 drives the space path and the terrain-less target with this code (its
  1 m <-> 2 m switch is the "other layout" case).
- **Settings changes.** `tilesPerLevel` or `grassMaskLayers` change the layout: a new GPU terrain set as above (the
  resident CPU tiles stay and are re-sent through the send point, truncated to the new budget). `grassMinNormalY` /
  `grassMinAboveSea` change the mask rule: every mask is dropped (`removeGrassMask`) and regenerated. Other fields apply
  from the next update.
- **Where grass masks are generated.** On the pool, as a job submitted after the chunk's level-0 tile is in the store
  (the job captures the tile's `shared_ptr` and the thresholds). A level-0 tile covers one chunk, so one tile gives one
  mask.
- **Sink coupling (§10.1).** The four planned `RenderSink` virtuals plus `removeGrassMask` (A8), `clearTerrain` (E1),
  `terrainUploadsPending` (E8) and the patch calls (E2; implemented by `VulkanRenderSink` and `RecordingRenderSink`
  and tested in S20; S22 is the first caller):

```cpp
virtual void setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid) = 0;   // starts a GPU terrain set if
                                                                     // the space or layout differs from the current one
virtual void setTerrainTile(RegionId region, const TerrainTileData& tile) = 0;          // a set tile is replaced in place
virtual void removeTerrainTile(RegionId region) = 0;
virtual void setGrassMask(ChunkId chunk, const GrassMaskData& mask) = 0;                // a set mask is replaced in place
virtual void removeGrassMask(ChunkId chunk) = 0;
virtual void clearTerrain() = 0;                                     // E1: drop the set (camera space without terrain)
virtual void patchTerrainTile(RegionId region, int x0, int z0, int w, int h, const int32_t* heights) = 0;  // E2
virtual void patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights) = 0; // E2
virtual size_t terrainUploadsPending() const { return 0; }           // E8: bytes sent but not yet recorded
```

  Patch semantics (E2, the same in `RenderScene` and `RecordingRenderSink`): `(x0, z0)` is the first sample of the
  sub-rectangle in the tile's (pyramid's) own sample grid, `w x h` samples, `heights` row-major (row = z) with `w`
  values per row, absolute integer heights within `+-kTerrainHeightLimit`; the rectangle lies inside the tile
  (asserted). The data is copied at the call. A patch of a region that has no layer (not set, or removed) or of
  another space than the set's is ignored and counted (`TerrainAtlas::Stats::patchesIgnored`, the recording's
  counter); S22 sends patches only for sent tiles, so a non-zero count in its tests is a bug.

  `World::setRenderSink` hands the sink to the `TerrainSystem` too. A new sink receives the pyramid, the resident tiles
  and the masks the store holds (through the send point), the same way registrations are forwarded (S19b). A removed
  sink drops the sent state. Without a sink, `updateView` streams nothing; with a sink, nothing streams until
  `updateView` / `completeView` is called ("Who streams"). Order guaranteed to the sink: `setTerrainPyramid` before any
  tile or mask of the space, a tile is removed before its level would exceed `tilesPerLevel`, a mask is removed before
  the masks would exceed `grassMaskLayers`. `RecordingRenderSink` keeps the live tiles, pyramids and masks (and clears
  them on a pyramid of another space or layout or on `clearTerrain`, as `RenderScene` does), applies patches to its
  copies, counts calls and ignored patches, estimates the terrain memory with `TerrainAtlas::memoryEstimate(layout)`
  (§5; the memory test), and returns `size_t uploadsPendingForTests` (a public field, default 0) from
  `terrainUploadsPending()`, so tests (S20's `Readiness`, S21's load end rule) pin the forwarding.
  `VulkanRenderSink` forwards to `RenderScene`.
- **EngineApp.** `EngineApp::renderFrame` calls `world.terrain().updateView(view.space, view.position)` after
  `beforeRender` and before the sink's `setCamera`. It calls `completeView` instead at the "complete points":
  - the first rendered frame and the frame before a `--screenshot` / `SAIGA_SCREENSHOT` capture (a timeout is fatal);
  - every MCP `screenshot` request, with the current view camera (so `set_camera` + `screenshot` shows the streamed
    view). The handler computes the request's answer deadline with `answerDeadline(server)` (`WorldTools.cpp`:
    `callDeadline() - 0.1 * toolTimeoutSeconds`, valid only inside the handler) and stores it in a new field
    `ScreenshotRequest::deadline` (`AppTools.cpp`, next to `gui` and `answered`); the app keeps the pending requests
    reachable (`AppTools` registers them with the app, e.g. `EngineApp::pendingScreenshotDeadline()` returning the
    earliest deadline of the unanswered requests). The frame's `completeView` timeout is that earliest deadline minus
    now minus 50 ms (the margin for the answer). On timeout every request whose deadline has passed, or would pass
    before the next frame, is answered with the error "screenshot: terrain not complete before the deadline" (marked
    answered and dropped from the batch, as the deadline listener does; no frame is captured for it) before the poll
    runs again, so the existing "no frame was rendered" listener never answers it a second time. With the default
    `toolTimeoutSeconds` of 60 s the budget is 54 s; with 1 s (the test) it is 0.9 s minus the margin;
  - every frame while `EngineApp::setCompleteViewEachFrame(true)` (the benchmark mode, §6, E9; a timeout is fatal).
  After a complete point, a capture waits until `TerrainSystem::terrainUploadsPending() == 0`: each waited frame calls
  `completeView` again with the then current view camera (cheap when nothing is missing; a camera that moves keeps
  being completed) and captures the first frame after which nothing is pending, at most 16 extra frames, then it
  captures with a warning. The time of a frame's `completeView` calls is the frame's terrain stream time
  (`EngineApp::lastTerrainStreamMs()`, the last rendered frame's; 0 for a frame with `updateView`): like the render's
  waits it is not CPU work (`pendingWorkMs` excludes it), and `FrameStatistics` gets the frame's interval without it
  (`addFrame(interval - stream, ...)` plus `advance(stream)`, so the time base still covers it), so complete points are
  neither long frames nor stutter in the statistics window. The window's Terrain section shows it. **The stream time
  pauses the simulation clock:** `EngineApp::frame()` (the wall-clock form) subtracts the previous frame's stream time
  from the measured interval before it calls `frame(dt)` (`dt = max(0, interval - stream)`), so `advanceWorlds` sees
  no backlog from a complete point (no catch-up steps, no dropped backlog after a 2 s first view or a benchmark frame
  that generated tiles), and the statistics use the same `dt` (plus `advance(stream)`). `runFrames(n, dt)` (fixed dt)
  is unaffected. In `--coop` only the client world is the view world, so the hidden host never streams ("Who
  streams").
- The engine statistics window (F1) gets a "Terrain" section (desired / truncated / resident / sent per level, pending,
  uploads, CPU and GPU MB, upload bytes pending, the last stream time, nodes per level, ignored patches, readiness at
  256 m), and `app_status` gets a
  `terrain` block (`resident`, `pending`, `uploads_pending_bytes`, `cpu_mb`, `gpu_mb`, `readiness`). The block is checked
  in `tests/test_engine_app.cpp`.

### 5. Terrain, grass and water in `RenderScene` (`vulkan/scene/`)

`RenderScene` API (§10.1, as in `RenderSink`; main thread, effective in the next `renderFrame`):

```cpp
void setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid);   // another space or layout: a new GPU set
void setTerrainTile(RegionId region, const TerrainTileData& tile);           // new: a free layer of its level (asserted);
                                                                             // already set: replaced in the same layer
void removeTerrainTile(RegionId region);
void setGrassMask(ChunkId chunk, const GrassMaskData& mask);                 // a set mask: replaced in its layer
void removeGrassMask(ChunkId chunk);
void clearTerrain();                                                         // E1: drops the set; nothing drawn
void patchTerrainTile(RegionId region, int x0, int z0, int w, int h, const int32_t* heights);   // E2 (§4.3)
void patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights);  // E2 (§4.3)
size_t terrainUploadsPending() const;                                         // bytes not yet recorded
const TerrainAtlas::Stats& terrainAtlasStats() const;                         // step 4a: set, residency, bytes, patches
const TerrainPass::Stats& terrainPassStats() const;                           // step 4b: nodes, refusals, water
void setPrototypeEnvironment(const PrototypeSettings& settings, uint32_t foliageSeed);   // replaces
                          // setPrototypeTerrain: sky, sun, fog, clay shading, the foliage texture (until S36)
```

- **GPU terrain set** (`vulkan/scene/TerrainAtlas.{h,cpp}`, one per `RenderScene`, for the space of the last pyramid):
  - **Atlas**: one device-local storage buffer of `int32` heights, `streamedLevels * tilesPerLevel` layers of
    `samplesPerSide^2` values (240 layers x 264 196 B = 63.4 MB at 1 m; 192 x 66 564 B = 12.8 MB at 2 m). The values are
    the integer heights themselves, no per-tile reference: the shader subtracts a frame-global reference (below), and
    every difference is below 2^24 (`kTerrainHeightLimit`), so the conversion to float is exact. The page entry is the
    layer only. This replaces D4's "stored per-tile reference" (E2, amendment A21): patches write the integers into the
    layer, never a whole layer. Every buffer of the set is checked against the storage-buffer limit
    (`RenderSceneParameters::storageBufferLimit` if set, else the device's `maxStorageBufferRange`; Vulkan guarantees
    2^27 = 128 MiB, lavapipe has 128 MiB, desktop GPUs about 4 GiB); a larger layout is a fatal error naming the limit
    (`SAIGA_EXIT_ERROR`, every build).
  - **Pyramid**: a storage buffer of `int32`, `(8 * sizeChunks + 1)^2` values (4.2 MB for 32 km, 67 MB for 512 chunks,
    within 128 MiB, so `sizeChunks <= 512` needs no texture limit).
  - **Page table**: one storage buffer with a grid per streamed level (`sizeChunks / 2^L` squared; 21 824 entries for
    32 km), each entry `int32 layer` (-1: none).
  - **Grass masks**: a storage buffer of `grassMaskLayers` layers of `(256 / baseSpacing)^2` bytes (read as `uint`,
    4 texels per word; 1 MB at 1 m).
  - **Uploads**: buffer-to-buffer copies in the transfer section (no image copies, no image layouts) through the
    atlas's own `UploadBatch` and its own staging buffer per frame in flight (`VulkanRenderer::framesInFlight` = 2, as
    SceneCulling's frame resources), not SceneCulling's batch and `f.staging`: the terrain's staging is created once at
    `RenderSceneParameters::terrainUploadBytesPerFrame` (16 MiB) plus alignment and never grows, because each frame
    takes at most that many bytes from the queue and every queue item is at most that large (a tile is 264 196 B, the
    pyramid is queued in row ranges of at most the cap). Tiles, masks, patches, page-table ranges and pyramid row ranges
    are queued on the CPU (the queue holds the `shared_ptr` of tile and pyramid data, patches hold their copy) and
    recorded in queue order, so a patch of a tile whose full copy is still queued lands behind it; patches count
    against the per-frame cap like any copy. The page-table entry of a tile is uploaded in the same frame as its layer;
    `removeTerrainTile` drops a still-queued copy of that tile and its patches. A tile counts as GPU-resident
    (refinable, drawable) from the frame its copy is recorded; the pyramid only when all its rows are recorded (nothing
    is drawn before). The copies come after the barrier from the previous frames' reads (S18), so a layer is reused one
    frame after its removal without a race. `terrainUploadsPending()` is the queue's byte count.
  - **Rebuild (D5)**: a pyramid of another space or with another layout drops every tile and mask of the set and the
    upload queue, hands the old buffers to `VulkanBase::deferDelete` (frames in flight still read them) and allocates
    the new set. A pyramid of the same space and layout replaces the pyramid data in place. `clearTerrain()` drops the
    set the same way without allocating a new one.
  - **Test hooks**: `TerrainAtlas::readLayerForTests(RegionId)` and `readPyramidForTests()` copy a layer / the pyramid
    back to the host (after the queue is recorded), so the upload, patch and replace tests compare integers.
  - **Statistics** (step 4a, no drawing needed): `TerrainAtlas::Stats` (`RenderScene::terrainAtlasStats()`): the set's
    space and layout, set rebuilds, tiles set per level and GPU-resident per level, masks set, upload bytes queued and
    recorded this frame, patches applied and ignored (`patchesIgnored`), GPU bytes. The step-4a tests read only these.
  - `static TerrainMemory TerrainAtlas::memoryEstimate(const TerrainLayout&, const RenderSceneParameters& =
    RenderSceneParameters())` (the second argument defaults, so the recording sink and the memory test call
    `memoryEstimate(layout)` with the default staging cap): the GPU
    bytes (atlas, pyramid, page table, masks, node buffers, staging) and the CPU bytes of `RenderScene`'s side (node
    bounds, the upload batch's data vector of at most the cap) of a set; `RecordingRenderSink` uses it and
    `MemoryMatchesLayout` checks it against the allocated buffers.
  - **Node bounds** (CPU, per level, for every node of the hierarchy down to the leaf size): `own[k]` = min/max of the
    samples the node draws at level k (from its tile, or from the pyramid for levels >= `streamedLevels`), and
    `bound[k] = own[k] ∪ bound[k-1]` of its resident children. Coarse samples are point samples, so `own` underestimates
    the finer surface; the union propagates every resident finer tile up the hierarchy, so a parent box always contains
    what its descendants draw. Morphing interpolates between samples of the same node, so it stays inside `own`. Skirts
    extend the minimum by `skirtDepth`. `setTerrainTile`, `removeTerrainTile` and patches update `own` of the affected
    nodes and recompute `bound` up to the root (4 children per node; patches only widen, E2). Storage: four `int32`
    per node (own and bound min/max) for every node from the leaf size up: 1 398 080 nodes = 22.4 MB at 1 m, 349 504 =
    5.6 MB at 2 m for 32 km (in the memory budget).
- **`TerrainPass`** (`vulkan/scene/TerrainPass.{h,cpp}`, replacing `TerrainRenderer`):
  - CDLOD nodes of level k have vertex spacing `baseSpacing * 2^k` and size `32 * baseSpacing * 2^k`, with gridN 32,
    full and quarter grids. Level 0 has one vertex per sample. Levels below `streamedLevels` read their own tile level,
    coarser levels read the pyramid at stride `2^(k - streamedLevels)`. The roots are the 4096 m nodes
    (`(sizeChunks / 16)^2` of them, always considered: the view reaches the space edge whatever the load radius is).
  - Selection is in double: node boxes from `bound`, relative to the camera, camera-relative frustum planes from
    `SceneCamera`, ranges `terrainLodRange * 2^k`, the S19a morph minimum with leaf size `32 * baseSpacing`. A node is
    refined only if each child's tile is GPU-resident. A child that is out of range or not resident is drawn as a
    quarter at the parent's spacing (S19a's quarter mechanism). `terrainLodRange` does not scale with `baseSpacing`
    (1 m gives finer geometry at every level; the triangle count is reported, §6).
  - **Frame anchor.** Each frame `anchor` = the camera's xz rounded down to a multiple of 64 m (integer metres) and
    `refHeight` = the camera height in height units rounded down to a multiple of 65 536 (64 m) and clamped to
    `+-kTerrainHeightLimit`. `cameraRel = float(camera - (anchor.x, refHeight * kHeightUnit, anchor.z))` is computed once
    in double (one rounding, |cameraRel.xz| < 64 m). Both go into `TerrainFrame`.
  - Node instance (32 bytes, per-frame host-visible buffer, up to 8192 nodes, then clipped with a once-per-frame log):
    `ivec2 originAnchor` (node origin - anchor, integer metres, exact); `int sampleX, sampleZ` (global sample index of
    the node origin at its level); `uint levelFlags` (level, quarter, water); `float skirtDepth`; 2 words padding.
  - Vertex (shared function `terrainVertex` in `terrain_vertex.glsl`, `precise`, also used by the probe shader):
    - grid index `(i, j)`, morph target `(i & ~1, j & ~1)` (rounded down to even: collapsing odd vertices onto it turns
      the fine (0,0)-(1,1) grid into the coarse grid with the same diagonal, and the node origin is even, so neighbours
      agree); `k` = the morph factor from the camera-relative distance of the unmorphed vertex.
    - xz = `vec2(originAnchor + ivec2(i, j) * spacing) - cameraRel.xz` for the vertex and likewise for its target (the
      integer sum is exact below 2^24, the subtraction rounds once), then `mix(xz, xzTarget, k)`.
    - height = `mix(float(h(i,j) - refHeight), float(h(target) - refHeight), k) * kHeightUnit - cameraRel.y`, with `h`
      read through the page table (int32, the difference exact).
    - A vertex shared by two nodes (same level, a common edge) computes the same integers and the same `k`, so it is
      bit-identical: no sparkle with skirts off.
    - The normal uses central differences through the page table. Across a non-resident neighbour it falls back to the
      next coarser resident level (a possible shading seam, not a crack). Flat shading (the default) keeps `dFdx`.
  - Skirts: each grid gets a ring of skirt vertices (a flag in the vertex). They hang `skirtDepth` =
    `max(node max - node min, 2 * spacing)` metres below the edge vertex and morph with it. They close the T-junctions
    that residency-driven refinement creates; CDLOD morphing only rules those out when every level is present.
  - The water plane stays a mode of the terrain draw (level flag water): the 400 km plane centred on the camera's xz
    rounded down to a multiple of 4096 m (a power of two that divides the 28 672 m offset test and every root node), at
    height `float(seaLevel - refHeight) * kHeightUnit - cameraRel.y`. It is culled with its own plane box, not with
    terrain bounds. No water when `seaLevel` is empty. The S19a pass order and the deviations of §10.5 (water in the
    opaque MSAA pass, in-shader fog) stay. Depth pre-pass (`Mode::Depth` / `Equal`, invariant `gl_Position`) as in S19a.
  - Only the GPU set's space is drawn, and only when it is the camera's space. Statistics (step 4b):
    `RenderStatistics::terrainNodes` plus `TerrainPass::Stats` (`RenderScene::terrainPassStats()`: nodes and quarters
    per level, refinements refused for residency, the water node; residency, bytes and patches stay in
    `TerrainAtlas::Stats`).
  - GPU sections: the pass keeps S19a's section names, `depth.terrain` (pre-pass) and `scene.terrain` (terrain and
    water); `GrassPass` draws in `scene.grass` and the sky in `scene.sky`. `RenderScene::gpuSectionNames()` is
    unchanged, so `WorldBenchmark::gpuKeys()` and `game_world`'s `s03Passes` mapping keep working.
- **`GrassPass`** (`vulkan/scene/GrassPass.{h,cpp}`, replacing `GrassRenderer`): chunk-local cells, nothing depends on
  absolute world coordinates.
  - Each chunk has `N = 16 * round(256 / (16 * grassCellSize))` cells per side. The cell size is `256 / N` (0.3 m gives
    N = 848, 0.3019 m), and a chunk splits into 16 x 16 blocks of N/16 cells (16 m).
  - Per cell, from the chunk-local cell index `c` in `[0, N)^2` only: jitter, size and yaw hash `c` (integer hash,
    pattern repeats every 256 m: invisible within the 60 m radius).
  - Patches: value noise on an 8 m lattice (32 lattice cells per chunk), lattice indices wrapped modulo 32, so the
    noise is continuous across chunk borders and periodic in 256 m.
  - Wind phase `2 pi (22 * lx + 17 * lz) / 256` from the chunk-local position `(lx, lz)` in metres (continuous across
    borders because the wave numbers are integers; arguments below 250 rad, so `sin` is accurate), plus the CPU phases
    of S19a.
  - Draw: one instanced draw per visible block within the radius whose chunk has a mask and a GPU-resident level-0 tile.
    Blocks are culled on the CPU with the tile's node bounds. Per draw: the block origin relative to the anchor
    (integer metres, exact), the chunk-local block index, the mask layer; the position is `float(blockOriginAnchor) -
    cameraRel.xz + local offset`. The height is the surface rule over the four level-0 samples of the cell's base cell
    (relative to `refHeight`, like the terrain vertex).
  - Density: the mask texel times the patches and the density setting; slope and water no longer come from the shader,
    because the mask carries them.
- **`TerrainFrame`** (`shader/vulkan/scene/terrain_common.glsl`, replacing `WorldUniforms` / `world/common.glsl`) is
  owned by `TerrainPass` (a per-frame uniform buffer written every frame the new path draws), not by `PrototypeWorld`,
  which exists only after `setPrototypeTerrain` (`RenderScene.cpp`) until step 7d. It holds the camera-relative
  matrices, `cameraRel`, `refHeight`, and, as floats computed in double, the camera height above the sea level and
  above the fog base, plus the sun, fog, clay shading, LOD and grass parameters, filled from RenderScene's current
  `PrototypeSettings` (the defaults until `setPrototypeEnvironment`; then the terrain and grass use the default
  settings' light). `PrototypeSettings` gains `terrainSkirts = true` (debug and the crack test).
- **Environment** (`setPrototypeEnvironment`, step 7a): RenderScene keeps the sky pass (`scene/sky.*` on
  `TerrainFrame`), the sun, fog and clay parameters and the foliage texture of the `PrototypeFoliage` class. The new
  sky is drawn once `setPrototypeEnvironment` was called, independent of the terrain (no sky before: the steps 4-6
  tests clear to the background colour). In 7d `PrototypeWorld` keeps only this environment (or is folded into
  RenderScene; decided in the implementation).
- **Coexistence until 7d.** `setPrototypeTerrain` (old `PrototypeWorld`: `TerrainRenderer`, `GrassRenderer`, old sky
  on `WorldUniforms`) and the new path (`setTerrainPyramid` / `setPrototypeEnvironment`) are exclusive on one scene
  (`SAIGA_ASSERT` in both directions); no test or game uses both. The old shaders keep working because the new ones are
  **copies**, not moves: `scene/terrain_shading.glsl` (copy of `world/shading.glsl` on `TerrainFrame`, including
  `../world/prototype_shading.glsl`), `scene/grass.frag` (copy of `world/foliage.frag`), `scene/sky.vert` /
  `sky.frag` (copies of `world/sky.*`; `sky.frag` includes `terrain_common.glsl` and `terrain_shading.glsl`). The
  `world/` originals are deleted in 7d.
- Shaders: new `scene/terrain.vert`, `terrain.frag`, `terrain_vertex.glsl`, `terrain_heights.glsl` (page table, atlas,
  pyramid, masks), `terrain_common.glsl`, `terrain_shading.glsl`, `terrain_probe.comp` (tests), `grass.vert`,
  `grass.frag` (including `terrain_common.glsl` and `terrain_shading.glsl`), `sky.vert`, `sky.frag`. Removed in 7d:
  `world/terrain.*`, `world/grass.vert`, `world/heightmap.glsl`, `world/common.glsl`, `world/shading.glsl`,
  `world/foliage.frag`, `world/sky.*`. `world/prototype_shading.glsl` stays (included by `scene/flat.frag`,
  `scene/prototype_foliage.frag` and `scene/terrain_shading.glsl`, prototype shading until S36).
- Test hooks (not called by games):
  - `TerrainPass::selectedNodes() -> const std::vector<TerrainNodeInfo>&`: the last frame's node list in instance
    order (`{level, origin, sampleX, sampleZ, quarter, water}`), so a test picks the node index of a probe.
  - `TerrainPass::probeVertices(const std::vector<TerrainProbe>&) -> std::vector<vec4>` runs `terrainVertex` in a
    compute dispatch with the frame's node data (`{node index, grid x, z}` in, camera-relative position out).
  - `RenderScene::readDepthForTests()` copies the last frame's single-sampled depth (S19a's `SAMPLE_ZERO` resolve
    target) to the host, so one test checks the real vertex pipeline against the probe and the CPU surface.

### 6. The world prototype (`vulkan/world/`), the games and the benchmarks

- New `vulkan/world/PrototypeTerrain.{h,cpp}` (D2: no engine types):
  - `int32_t prototypeHeight(uint64_t seed, int64_t x, int64_t z)` (a `TerrainHeightHook`) is the S01 noise of
    `HeightField::generate`, evaluated per point. The noise seed is `uint32_t(seed) * 7717`. The border falloff is no
    longer 10 % of the world size: it is the prototype's absolute 819.2 m at the edges of the 32 km space. So the area
    `[0, 8192]^2` is unchanged except where its old east and north coasts become land. The result is
    `clampTerrainHeight(lround(h * 1024))`.
  - `kPrototypeSeaLevel` = 6144 (6 m), `kPrototypeMountainHeight` = 520, `kPrototypeSpaceChunks` = 128,
    `kPrototypeArea` = 8192 m.
  - `double prototypeSurfaceHeight(uint64_t seed, int baseSpacing, double x, double z)`: the surface rule over the
    clamped hook with `World::terrainHeight`'s clamp and cell rule, equal to it bit for bit for such a space, for code
    without a `World` (the flight and racing contexts, racing's site search, placement, tests). About 2 µs per call.
  - `PrototypeAreaStats prototypeAreaStats(seed, x0, z0, size, step)`: the highest hook sample (`Vec3`) and the min /
    max height on a grid (`game_world`'s start camera and `game_get_state`; 32 m over 8 km = 66 k samples, about
    50 ms).
- `samples/games/common/PrototypeSpace.h` (new, header only): `Engine::SpaceTemplate prototypeSpaceTemplate()` returns
  the template of `game_world`, flight and racing (32 km, 1 m, `kPrototypeSeaLevel`, `prototypeHeight`). It lives in the
  games (D2); the tests build the same template inline from the `PrototypeTerrain` constants.
- `ForestRenderer::placeTrees(const PrototypeArea& area, int count, uint32_t seed, species)` with `PrototypeArea {x0,
  z0, size, seed, baseSpacing}`. The probabilities come from a 4 m grid of hook samples over the area (2049^2 for 8 km,
  OpenMP, the §3 deviation; a quarter of the old `HeightField::generate` cost). The tree y is
  `float(prototypeSurfaceHeight(...)) - 0.25f` (sunk as before). New overloads take the placed trees instead of a
  `HeightField`: `prototypeTreeSets(const std::vector<TreeInstance>& trees, const std::vector<TreeSpecies>&, const
  std::vector<MeshId>&)` and `PrototypeForest::init(RenderScene&, std::vector<TreeInstance> trees, const
  WorldSettings&)` (its foliage texture then comes from `setPrototypeEnvironment`). They are added in 7a next to the
  `HeightField` overloads, which are removed in 7d with the old `placeTrees`.
- `WorldRenderer` and `WorldStatistics.cpp` are removed (§19: "replaced in S20"). `PrototypeScene`
  (`vulkan/world/PrototypeScene.{h,cpp}`) owns `WorldSettings`, the settings GUI, the mapping to `PrototypeSettings`
  and the `PrototypeForest`, with `attach(RenderScene&, trees)` and `update()`. The statistics window's terrain and
  timing parts move to the engine statistics window (§4); the tree counts stay in `PrototypeScene::imgui`.
  `HeightField` is removed. `WorldCommon.h` keeps the frustum helpers and gains `viewRotation` (from `WorldRenderer.h`;
  the platformer uses it). `WorldParameters` (in `WorldRenderer.h`) goes with it (7d); its fields move as follows:
  `terrain` (`HeightField::Parameters`: size, spacing, noise) becomes the `SpaceTemplate` (`sizeChunks`, `baseSpacing`,
  the hook) and with it the `TerrainLayout`; `treeCount` and `seed` become the arguments of `placeTrees(PrototypeArea,
  count, seed, species)` (`PrototypeArea::seed` is the hook's seed); the foliage seed becomes the second argument of
  `setPrototypeEnvironment(settings, foliageSeed)`.
- Every `HeightField` / `WorldRenderer` consumer and its replacement:

| Consumer | Today | After S20 |
|---|---|---|
| `game_world` `main.cpp` start camera | `hf.highestPoint()`, `hf.worldSize()`, `hf.height` | `prototypeAreaStats` over the 8 km area, centre (4096, 4096), `world.terrainHeight` |
| `game_world` walk mode, altitude display | `heightField().height` | `world.terrainHeight(0, x, z)` |
| `game_world` benchmark | `WorldBenchmark(name, heightField(), gpu)` | `WorldBenchmark(name, ground, gpu)`, ground = max(terrainHeight, sea level) |
| `WorldMcp` `game_set_camera` | `hf.height` | `WorldMcpContext::ground` |
| `WorldMcp` `game_get_state` | `size_m`, `min/max_height`, `water_level`, `highest_point`, `center` from `hf` | context fields: `worldSize` (space size, 32 768), min/max from `TerrainSystem::pyramid` (the whole space), or while it is null (before the first view, no sink) the game's `prototypeAreaStats` min/max of the 8 km area; `seaLevel`, `highestPoint` (computed once by the game), centre ground from `ground`; `cpu_terrain_select_ms` removed, `terrain` stats added |
| `WorldMcp` camera limit | `maxWorldSizes * hf.worldSize()` | `maxWorldSizes * worldSize` |
| `WorldMcp` `game_terrain_query` | `hf.height`, `hf.normal(x, z, 2)`, `waterLevel()` | `ground`; normal by central differences of `ground` at 2 m (`groundNormal`, `WorldMcp.cpp`); `under_water` against `seaLevel` |
| `samples/games/common/WorldLighting.h` `worldGround(WorldRenderer&)` | unused | removed with the file |
| Flight `FlightContext` (`heightField`; `ground` used by `flightModelOf`'s `FlightModel::ground` in the system lambdas of `FlightWorld.cpp` and by `FlightRender`'s chase camera) | `heightField.height`, `waterLevel()` | the context holds `seed` (the setup's `WorldSetup::seed`, passed to `FlightContext::create(seed)`) and `baseSpacing`; `ground = max(prototypeSurfaceHeight(seed, baseSpacing, x, z), kPrototypeSeaLevel * kHeightUnit)`; circuit centre (4096, 4096), radius 2458 m (0.3 x 8192, as today) |
| Racing `RacingContext::terrain` (`carGround` of `Vehicle` through the captured context, `RacingRender`'s chase camera and the init-time asphalt, curb, cone and gate meshes, `RacingRender.cpp` around lines 45, 78, 105, 122) | `heightField.height` | `prototypeSurfaceHeight(seed, baseSpacing, x, z)` (seed from `RacingContext::create(seed)`) |
| Racing site search `findSite(ground, hf.worldSize(), r, hf.waterLevel() + 3, ...)`, tree grid `hf.worldSize() / cell` | `heightField` | `findSite(ground, kPrototypeArea, r, 6 + 3, ...)` over `prototypeSurfaceHeight` (about 1.1 M queries, about 2 s at startup, §2); grid over 8192 m |
| Flight / racing `*Render.cpp` | `worldRenderer.init(params, heightField)` | `PrototypeScene` |
| Platformer `PlatformerRender.cpp` | includes `WorldRenderer.h` (for `viewRotation`) | includes `WorldCommon.h` |
| `sample_vulkan_scene_benchmark` | `HeightField::generate` for the trees | `placeTrees` over the 8 km area |
| `tests/test_vulkan_scene.cpp` `DepthPrePassMatches` | `HeightField` + `setPrototypeTerrain` | test tiles (`generateTerrainTile` with an integer hook) + `setTerrainPyramid` / `setTerrainTile` |
| `tests/test_vulkan_world.cpp` | `HeightField` in 10 tests | see Tests |

  Changed tool outputs (S14 `handleMessage` tests follow): `size_m` 32 768, `min/max_height` from the pyramid (32 m),
  `highest_point` from the 32 m grid of the 8 km area, `game_terrain_query` heights on the §12.2 diagonal.
- `WorldBenchmark::pose(name, ground, frame)` takes `std::function<double(double, double)> ground` (terrain or water,
  whichever is higher). The paths keep the centre (4096, 4096) until S27 (§17.7).
- **Benchmark mode (D9, E9).** Terrain residency is deterministic: `game_world --benchmark` calls
  `EngineApp::setCompleteViewEachFrame(true)`, so `completeView` runs in every rendered frame (warm-up and measured),
  and the set of resident and sent tiles depends only on the pose sequence; what is recorded per frame follows the
  upload queue and the byte cap, a function of the frame number. No measured value depends on worker scheduling. The
  stream time reaches `WorldBenchmark` (saiga_vulkan, no engine types) as a field of its add-times call:
  `WorldBenchmark::Sample` gets `double terrainStreamMs = 0` (appended after `waitPartsMs`, so the existing aggregate
  initializers keep their meaning). `game_world`'s `beforeRender` fills it, when it builds frame k's sample in frame k+1
  (as for `intervalMs` today), from `app.lastTerrainStreamMs()`, which at that point is frame k's (its `completeView`
  ran after frame k's `beforeRender`). `addSample` stores `intervalMs - terrainStreamMs` as the frame interval, so
  `cpu_ms`, `frame_interval_ms`, `fps`, `long_frames`, `pacing_p99_ms` and `stalled_frames` (`countStalls`: > 4 x the
  median and > 50 ms, which a 46 ms tile on top of a 3 ms frame would otherwise trip) all exclude it; `frameWorkMs`
  (`frame_work_ms`, "CPU busy") already excludes it because `EngineApp`'s work does (§4.3); `cpu_work_ms` (the
  recording) never contained it. `result()` adds `terrain_stream_ms` (mean, median, p95, p99, max of the measured
  frames), `wait_ms.terrain_stream` (mean, next to `frame_slot`, `acquire`, `present`) and the block `terrain: {uploads,
  upload_mb, resident_tiles, cpu_mb, gpu_mb}` (from `TerrainSystem::Stats` and `TerrainAtlas::Stats`, filled by
  `game_world` through a new `WorldBenchmark::setTerrainInfo(json)`); `summaryText` shows a "terrain stream" row under
  "CPU busy". (`game_world` measures `intervalMs` itself between `beforeRender` calls, wall clock, so the subtraction in
  `addSample` is needed even though `EngineApp` pauses the simulation clock during the stream time, §4.3: the benchmark
  frames run no catch-up steps after a frame that generated tiles.) `game_world --base-spacing 2` (a benchmark-only
  flag) runs the same benchmark on a 2 m template, to separate a code regression from the content change of 1 m.
- `game_world`: `spaceTemplates = {prototypeSpaceTemplate()}`, 200k trees in `[0, 8192]^2`, `PrototypeScene`. The
  `--benchmark` runs on this space. `residencyRange0 = max(400, 1.3 * terrainLodRange)` when the setting changes (400 for
  the default 300 m; the truncation of §4.3 bounds larger values).
- Platformer: default flat template, no terrain drawn; only the include changes.

## Out of scope

Terraforming, edit deltas and the chunk record (S22; S20 only provides the patch calls and the send point). Phase-7
residency from player positions, simulation radii and arrivals (S24). Far content (S27). Collision (S28). The generation
pipeline, integer noise and generated grass masks (S23). Further spaces, `Enclosed`, the space switch and the loading
screen (S21; S20 provides the GPU-set rebuild and `readiness`). Environment, fog and ambient as `SpaceTemplate` data
(S36). Real water (S39). Disk caching of tiles. A `terrain_height` world tool. A move of the benchmarks to the 32 km
centre (S27). Pinned (always-resident) tiles.

## Design notes

- **Same sample count per tile at every level.** Level L doubles the spacing and covers 2^L chunks, so every tile is one
  atlas layer of the same size, and CDLOD level k reads exactly tile level k. The 32 m pyramid replaces streamed tiles
  for the coarse levels; at 1 m those would be 32 to 128 m spacing tiles covering 8 to 32 km.
- **Storage buffers instead of images.** The vertex shader reads exact samples (no filtering), so buffers lose nothing.
  They upload through the existing buffer-to-buffer `UploadBatch` (no image layouts, no per-layer transitions of a
  `Texture2DArray`, no blocking `uploadLayer` with mipmaps) and drop the 4096-texel limit of the pyramid. int32 values
  keep the CPU and GPU data identical (no conversion on the main thread) and make patches plain copies.
- **Memory (the 768 MB criterion)** for 32 km at 1 m with the defaults:
  - CPU, engine: 240 tiles x 264 196 B = 63.4 MB (the per-level budget bounds the store; the desired set is truncated
    to it; the sink's tiles and the upload queue share these vectors); the pyramid 4.2 MB; 16 grass masks 1 MB; up to
    `maxInFlight` finished results not yet drained 2.1 MB.
  - CPU, `RenderScene`: node bounds 22.4 MB (1 398 080 nodes x 16 B); the upload batch's data vector at most 16.8 MB
    (the cap); patch copies in the queue (bounded by the cap per frame times the frames they wait; S22 caps its patch
    bytes per frame, counted in `uploads_pending_bytes`).
  - GPU: atlas 63.4 MB, pyramid 4.2 MB, page table 0.09 MB, grass masks 1 MB, node instance buffers 0.5 MB
    (8192 x 32 B x 2 frames in flight), staging 2 x 16.8 MB = 33.6 MB (fixed, never grows, §5).
  - Total about 213 MB. The bound is structural (budgets times the layout, the node count of the space and the
    staging cap), so the test checks both the formula (`TerrainAtlas::memoryEstimate` + the store's budget) and a run.
  - The desired set per level at `residencyRange0` R is the tiles whose box lies within R * 2^k: for R = 400 m about 16
    to 25 per level (geometric LOD, independent of the level); R = 650 m reaches 48. Beyond that the set is truncated
    nearest-first and the CDLOD selection draws the cut-off area from coarser levels (no holes; `Stats::truncated`).
- **Precision without hi + lo.** A terrain vertex is an exact integer position relative to the frame anchor minus one
  per-frame `cameraRel` (|cameraRel| < 64 m, rounded once in double), plus exact height terms. Near vertices have small
  integers, so their error is the rounding of a value below a few hundred metres: about 1e-5 m. A far vertex's absolute
  error grows with its distance but stays at about 1e-4 px, so §9.1's split offset is unnecessary. Shared vertices are
  bit-identical because they are computed from the same integers.
- **Skirts are required, not cosmetic.** CDLOD's crack-free morph assumes every level is available. Refinement gated by
  residency (§12.2) leaves level-k nodes next to level-(k+1) nodes inside the range of k.
- **Residency is separate from the render range.** The engine streams by `residencyRange0`, the scene refines by
  `terrainLodRange` where tiles exist. A render range larger than the residency range shows coarser terrain; it never
  shows holes (§12.2: the view distance does not depend on the load radius).
- **Bounds are hierarchical.** A coarse box from point samples can be lower than the fine surface; propagating resident
  descendants' bounds up keeps culling conservative for everything drawn (§5).
- **Determinism.** The store only caches what the hook returns, so simulation results never depend on the camera or the
  pool. The engine scenario and the host + client variant prove it for the mover.
- **Prototype cost.** The S01 noise is roughly 27 gradient evaluations per sample, about 0.7 µs. A 1 m level-0 tile is
  about 46 ms per worker, the pyramid about 0.7 s. Pyramid first, then coarse levels first, gives a full view in a few
  seconds on two workers. The complete points hide this from screenshots and benchmarks.

## Tests

CI (lavapipe for Vulkan; all compilers). Test registration in `tests/CMakeLists.txt`: `test_core_terrain_surface`
(core), `test_vulkan_terrain` (`MODULE_VULKAN`, `"saiga_vulkan"`), `test_engine_terrain` and
`test_engine_terrain_render` (`MODULE_ENGINE`, `"saiga_engine"`; the engine module implies Vulkan).

Shared test headers (header only):

- `tests/terrain_test_hooks.h` (the `test_vulkan_*` tests include it too, so engine types appear only in its
  engine-only part below; S22 and S23 add their helpers to it): cheap integer `TerrainHeightHook`s (`flatHook`,
  `slopeRidgeHook` (a slope with ridges, `walker_follows_terrain`), `periodic4096Hook` (the origin-offset test),
  `highFrequencyHook` (the crack test), `spikeHook(x, z, height)` via a small table, `integerTestHook` (a cheap plane
  with a seed-dependent offset, for S22's render and edit tests through its `terraformTestTemplate`)), the gated hook `latchedHook` (file-static state: it blocks on a latch
  only when called off the thread recorded by `armLatch()`, i.e. on the pool's workers, so the main thread's
  synchronous paths and its share of `completeView` never block; `openLatch()` before the world is destroyed;
  `sleepAtOrigin` mode: sleeps 100 ms when called at sample (0, 0), once per pyramid and per tile containing the
  origin), `testLayout(sizeChunks, baseSpacing)`, `testTile` / `testPyramid` (the pure generation over
  `TerrainHeightSource::fromHook`), `renderUntilUploaded(RenderScene&, maxFrames)` and `fillTestTerrain(RenderScene&,
  const TerrainLayout&, const TerrainHeightSource&, std::optional<int32_t> seaLevel, const Vec3& camera, bool
  withMasks, SpaceId space = 0)`: sets the pyramid, the tiles of every streamed level whose box lies within
  `400 * 2^L` m of the camera (nearest `tilesPerLevel` per level, the default rule of §4.3 with heights from the
  pyramid), with `withMasks` the `prototypeGrassMask`s (default thresholds) of the level-0 chunks within 80 m, then
  `renderUntilUploaded`. It feeds the renderer tests and the ported `test_vulkan_world` tests without an engine.
  **Engine-only part**, inside `#ifdef SAIGA_TEST_ENGINE` (includes `saiga/engine/world/SpaceTemplate.h`):
  `testSpaceTemplate(sizeChunks, baseSpacing, hook, seaLevel)` (S22 adds `terraformTestTemplate` here). The gate is
  not the engine module's configured macro (it is defined in the `test_vulkan_*` binaries too, which do not link
  `saiga_engine`) but `SAIGA_TEST_ENGINE=1`, which `saiga_test` in `tests/CMakeLists.txt` defines for every test whose
  libraries include `saiga_engine` (`if ("saiga_engine" IN_LIST ARGN) target_compile_definitions(${TARGET_NAME}
  PRIVATE SAIGA_TEST_ENGINE=1)`).
- `tests/render_scene_test_context.h`: the fixture `RenderSceneTest` (the `VulkanTest` fixture cannot be used: it
  holds one static `HeadlessWindow` with a `VulkanForwardRenderer`, while a `RenderScene` is its own renderer). Per
  test a `HeadlessWindow` and a `RenderScene` (`Create(width, height, samples, RenderSceneParameters)`, imgui off,
  `nearPlane` 0.1), skipped with the same message as `VulkanTest` without a Vulkan 1.3 device, and without
  `drawIndirectFirstInstance` (as today's `VulkanWorldScene`); `TearDown` waits idle and destroys the scene before the
  window. Helpers: `SetCamera(space, eye, rotation, fovy)`, `RenderFrame()`, `Capture(n)` (renders n frames, waits,
  returns the last), `deviceName()`. Death tests in it (`AtlasLimits`) set `GTEST_FLAG_SET(death_test_style,
  "threadsafe")` as `FoliageClassNeedsTexture` does. `test_vulkan_terrain` uses it directly;
  `test_engine_terrain_render` derives `TerrainRenderTest` from it with a `World` whose sink is a
  `VulkanRenderSink(*scene)` (constructed after the scene, destroyed before it).
- `tests/forest_reference.h` (7b): `ForestReference`, `cameraPlanes`, `forestReference(trees, species, settings, eye,
  rotation, aspect)` and `expectForest`, moved out of `test_vulkan_world.cpp` (taking the placed trees instead of a
  `WorldRenderer`), so the goldens in `test_engine_terrain_render` keep the §17.5 CPU check.

- `tests/test_core_terrain_surface.cpp`:
  - `TerrainSurface.IntegerRule` (both triangles, the diagonal, edges, vertices, negative heights rounding toward
    -infinity, heights at `+-kTerrainHeightLimit` with fracBits 20 without overflow).
  - `.DoubleEqualsIntegerForDyadic`: at 10^5 seeded points with fracBits in 0..20: `terrainSurfaceDouble(fx / 2^f, ...)
    * 2^f == numerator` (exact, as double) and `floor(terrainSurfaceDouble(...)) == terrainSurfaceUnits(...)`.
  - `.ClampToLimit` (`clampTerrainHeight`).
  - `.GridTrianglesUseTheDiagonal` (every cell shares (i, j)-(i+1, j+1), counter-clockwise from above).
- `tests/test_engine_terrain.cpp` (new, no Vulkan device):
  - `Terrain.TerrainHeightEqualsTheIntegerRule`: §22 acceptance. 10^4 seeded points on the 1/256 m grid incl. points on
    and next to cell diagonals, cell edges, vertices, near 30 km, below sea level and exactly on the far edges
    (x or z = `sizeChunks * 256`: the last cell with `fx = 1`), for spacing 1 and 2. In cell units a 1/256 m point has
    `fracBits = 8 + log2(baseSpacing)` (8 at 1 m, 9 at 2 m: the fraction is a multiple of 1/512 cell):
    `floor(terrainHeight / kHeightUnit) == terrainSurfaceUnits(hook samples, fracBits)` and
    `terrainHeight / kHeightUnit * 2^fracBits == numerator` (same formula as S23a); at vertices equality with the
    sample.
  - `.PrototypeSurfaceEqualsTerrainHeight`: `prototypeSurfaceHeight` equals `World::terrainHeight` bit for bit for
    `prototypeSpaceTemplate`'s settings (built inline from the `PrototypeTerrain` constants) at 10^3 points incl. the
    edges, spacing 1 and 2 (the flight and racing contexts rely on it).
  - `.ResidentTileEqualsHook`: bitwise, at random points, with and without resident tiles.
  - `.FlatWithoutHook` (`hasTerrain()` false, `terrainHeight` 0, `TerrainSystem::sample` 0 at seeded samples incl.
    negative and far ones, nothing requested, readiness 1), `.HookResultsClamped` (a hook
    returning `INT32_MAX`, through `fromHook`).
  - `.TemplateValidation`: death tests (no `SAIGA_ASSERTS` needed) for two templates, `Enclosed`, `sizeChunks` 8 and 520
    and 100, spacing 3, a sea level beyond the limit.
  - `.TilesShareBordersAndSubsample` (neighbour borders equal; a level-L+1 sample equals the level-L one), `.Pyramid`
    (point samples of the hook), `.GrassMaskRule` (slope and sea thresholds).
  - `StreamingPool.*`: priority order and FIFO; cancel queued (returns true, no completion) and running (returns
    false, the job sees `cancelled`, `done(true)`); `queued()` / `running()` / `finished()` with jobs held on a latch
    (2 running, the rest queued; after a queued cancel the counts drop at once); `runOneOnCaller` runs the first queued
    job on the calling thread (thread id) and queues its completion; `waitIdleFor` returns false while a held job
    runs; an exception rethrown at `drain` while the other completions stay queued for the next drain, and the pool's
    counts are freed; `waitIdle` does not throw; destruction with queued and running jobs.
  - `TerrainStreaming.GenerationExceptionIsFatal` (death test: a hook that throws in a pool job; `SAIGA_EXIT_ERROR`,
    no `SAIGA_ASSERTS` needed; the message names the request, e.g. "terrain pyramid space 0", and the hook's text).
  - `TerrainStreaming.*` with a `RecordingRenderSink`:
    - the desired set per level against a CPU reference of the rule;
    - `DesiredSetTruncatedToBudget`: `residencyRange0` = 2000 m: at most `tilesPerLevel` desired per level, the nearest
      ones, `Stats::truncated` > 0, no assertion, eviction keeps the budget; the same for masks with `grassRadius`
      = 600 m and 16 layers;
    - LRU eviction never removes a desired tile and keeps `tilesPerLevel`;
    - at most `uploadsPerFrame` tile additions per update; every removal of an evicted tile in the same update (the
      recording's tiles stay a subset of the store's);
    - the pyramid comes before the tiles and masks; masks reach the sink only through `sendGrassMask` (a new sink
      receives them; a layout change re-sends them);
    - `CompleteViewFirstView`: one `completeView` on a fresh world (no pyramid, no tiles) returns true with the
      pyramid, every desired tile and every desired mask sent (the loop of §4.3: masks need the tiles of the same
      call), readiness 1, nothing pending in the store, `Stats::pyramidsGenerated` 1; a second call generates nothing;
    - `CompleteViewTimeout`: with `latchedHook` armed on the test thread and a 0.1 s timeout it returns false within
      0.1 s plus one caller job (the caller ran the queued jobs, the workers' jobs block; no fatal error inside
      `completeView`; no caller job starts after the deadline: a counter in the hook); the latch is opened before the
      world is destroyed;
    - `UpdateViewRequestsThePyramidFirst`: before the pyramid is stored, `updateView` requests only the pyramid (no
      tile request, `Stats::levels[*].desired` 0, readiness 0);
    - masks for chunks within the grass radius, their removal, and their regeneration after a threshold change;
    - `LayoutChangeStartsNewSet`: changing `tilesPerLevel` sends a pyramid with the new layout, the recording sink holds
      only tiles sent after it, within the new budget;
    - `Readiness`: 0 before the pyramid, rises monotonically to 1 for a fixed camera, 1 after `completeView`, 1 for a
      space without terrain; `terrainUploadsPending` forwards the sink's value
      (`RecordingRenderSink::uploadsPendingForTests` set to 4096 is returned; 0 without a sink);
    - `ClearTerrainOnRecording`: `clearTerrain` drops the recording's tiles, masks and pyramid; a later pyramid starts
      a new set (the `TerrainSystem` trigger itself is first driven by S21's second space);
    - `SinkAloneStreamsNothing`: a world with a `RecordingRenderSink` (as every scenario world) and no `updateView` /
      `completeView` call runs 100 steps with a walking character: no job submitted, no pool created (`Stats::pending`
      0, `poolCreated()` false), the recording holds no terrain; without a sink `updateView` sends nothing; a sink set
      after a `completeView` receives the pyramid, tiles and masks through the send point;
    - streaming changes neither `world.hash()` nor `terrainHeight`;
    - `PatchesReachTheRecording`: `patchTerrainTile` / `patchTerrainPyramid` through the sink change the recorded
      copies (S22's entry point); a patch of an unsent region or of another space is ignored and counted.
  - `TerrainStreaming.MemoryBudget32km`: §22 acceptance, the `game_world` layout (128 chunks, 1 m, default budgets) with
    a cheap integer test hook. A `completeView` at every 20th pose of the flyover circle, a 30 km straight line at
    300 m and at 2 m, and a 90 degree turn. Asserts `Stats::cpuBytes` (store, pyramid, masks, results not drained) +
    `TerrainAtlas::memoryEstimate(layout)` (GPU incl. the fixed staging, and RenderScene's CPU side: node bounds and the
    upload data vector) `<= 768 MB` at every pose, and the static formula bound.
  - `CharacterMove.FollowsTerrain`, `CharacterMove.IdleSnapsOnlyWhenOff` (in a terrain space a teleported idle
    character is snapped in the next step; a resting one is not written: no dirty mark),
    `CharacterMove.IdleUnchangedWithoutTerrain` (the flat default: an idle character at y = 3 stays there, as today).
- `tests/test_engine_scenarios.cpp`: `walker_follows_terrain`, `slopeRidgeHook` of `terrain_test_hooks.h`. A player walks
  500 m; y equals `terrainHeight` every step. It passes in the plain, repeat, save/load and host + client variants
  (delay 0 and 2).
- `tests/test_vulkan_terrain.cpp` (new, `RenderSceneTest` fixture; `RenderScene` driven with test tiles from
  `generateTerrainTile` / `generateTerrainPyramid` and the hooks of `terrain_test_hooks.h`; `renderUntilUploaded`
  renders frames until `terrainUploadsPending() == 0`; probes pick their node indices from
  `TerrainPass::selectedNodes()`). Step 4a (atlas, no drawing; assertions on `TerrainAtlas::Stats` and the read-back
  hooks only):
  - `VulkanTerrain.UploadBudgetSpreadsBursts`: 150 tiles set at once, 30 in each of the 5 streamed levels of a 1 m
    layout (within `tilesPerLevel` 48); each frame records at most `terrainUploadBytesPerFrame`, all are GPU-resident
    after the expected number of frames (`ceil(150 * 264 196 / cap)` plus the page-table bytes), the staging buffers
    keep their initial size.
  - `.UploadsMatchData` / `.ReplaceInPlace`: `readLayerForTests` equals the tile's heights; setting a set tile keeps
    its layer and the read-back changes.
  - `.PatchTileAndPyramid`: a patch sub-rectangle read back equals the patched data (`readLayerForTests`,
    `readPyramidForTests`), the rest of the layer is unchanged; a patch issued while the tile's full copy is still
    queued lands after it (read back after the queue drains); patches count against the byte cap; the node bounds of
    every ancestor widen; patches of an unset region and of another space are ignored and counted.
  - `.NewSetOnOtherSpaceOrLayout`: a pyramid of SpaceId 1 drops the tiles, masks and queue of space 0
    (`TerrainAtlas::Stats`), validation clean with frames in flight; a pyramid of the same space and layout keeps them
    (S21 merges its set-rebuild test into this one, E1).
  - `.ClearTerrain`: `clearTerrain` drops the set (deferDelete, validation clean with frames in flight); nothing is
    drawn; a later pyramid allocates a new set.
  - `.AtlasLimits`: with `RenderSceneParameters::storageBufferLimit` = 32 MiB (test-only lower limit), a 1 m layout
    whose atlas exceeds it stops with `SAIGA_EXIT_ERROR` naming the limit (every build); a new tile on a full level is
    a `SAIGA_ASSERT` death test inside `#ifdef SAIGA_ASSERTS`.
  - `.MemoryMatchesLayout` (`TerrainAtlas::Stats` GPU bytes and RenderScene's CPU-side bytes = `memoryEstimate`).
  - `.BoundsIncludeFinerTilesCpu`: the CPU bound of every ancestor contains a resident level-0 tile's 300 m spike;
    after `removeTerrainTile` the bounds shrink again.

  Step 4b (drawing):
  - `VulkanTerrain.OneVertexPerSampleAtLevelZero`: §22 acceptance, spacing 1 and 2. Probes of level-0 nodes have xz on
    the sample grid (exact) and their height equals `terrainHeight`'s CPU reference within 1e-6 * distance + 1e-4 m.
  - `.MorphTargetsAndSharedVertices`: probes of partly morphed vertices equal `mix(h(i,j), h(i&~1, j&~1), k)`; the edge
    vertices of two adjacent same-level nodes are bit-identical.
  - `.RenderedDepthMatchesSurface`: an oblique view (40 degrees down) of a level-0 area; `readDepthForTests`, unproject
    50 pixel centres, their heights equal the CPU surface rule at their xz within 1e-3 m x distance (checks the real
    vertex pipeline and the diagonal, not only the probe).
  - `.CameraMoveAt30km`: §22 acceptance. The camera at 28 672 m, probes from 0.5 m to 50 m at levels 0 to 3, a
    0.25 mm/px footprint at the nearest probe. Before and after a 1 mm move, each GPU probe projects within 0.25 px of
    the double projection, so the move error is <= 0.5 px. Also checked against an expected failure with a
    float-absolute camera (the test computes the bug's error: > 4 px).
  - `.NoCrackAtLevelBorders`: §22 acceptance. A high-frequency test hook; a level-0 tile next to an area only level 1
    covers; the camera 40 m above a point 30 m from the border, looking across it 35 degrees down, the border at a
    third of the image width (off-centre); sky off, magenta clear colour: 0 magenta pixels. Negative control: with
    `terrainSkirts = false` there are more than 10 magenta pixels, so the view can see cracks.
  - `.BoundsIncludeFinerTiles`: a flat coarse level with a resident level-0 tile carrying a 300 m spike; the camera
    places the spike at the top image edge where the coarse box alone is outside the frustum; the spike is drawn
    (pixel count).
  - `.RefinesOnlyResidentTiles` (nodes per level with and without a tile; a tile queued but not yet recorded is not
    refined).
  - `.PyramidOnly` (the whole space drawn from the pyramid).
  - `.TileRemoveAndLayerReuseInFlight` (remove and set in consecutive frames, validation clean).
  - `.PatchedImageEqualsFullTile`: a patched tile renders pixel-equal to a full `setTerrainTile` with the patched data.
  - `.WaterAtSeaLevel` / `.NoWaterWithoutSeaLevel`.
  - `.GrassNeedsMask` (no clumps without a mask or with a zero mask).
  - `.GrassChunkRelative` (the image is invariant under a 256 m camera + content offset).
- `tests/test_vulkan_scene.cpp`: `DepthPrePassMatches` on test tiles (table in §6), same assertions.
- `tests/test_engine_terrain_render.cpp` (new, fixture `TerrainRenderTest` derived from `RenderSceneTest`: a `World`
  with `testSpaceTemplate` (`terrain_test_hooks.h`) and a `VulkanRenderSink` on the fixture's scene,
  `TerrainSystem::completeView` before each capture and `renderUntilUploaded`; D2):
  - `TerrainRender.OriginOffsetWithinGoldenTolerance`: §22 acceptance. The 32 km space with a test hook periodic in
    4096 m (so periodic in 28 672 m) and grass masks. Two compositions at 512 x 512 with `fovy` = 40 degrees (square
    image, so 40 degrees horizontally too), each at pose A and at A + (28 672, 0, 28 672): (1) terrain + water: A =
    (1500, h + 150, 1500), looking along +x 35 degrees down: the top edge looks 15 degrees down and the top corners
    about 14.2 degrees, so the frustum meets flat ground within 600 m (a ridge ahead only shortens that) and no sky or
    space edge is visible (at 60 degrees the top edge would look 5 degrees down and reach 1.7 km); (2) grass: A =
    (1700, h + 2, 1500), 30 degrees down (the top corners about 9.5 degrees down: ground within 12 m). The image
    pairs pass the golden metrics of §17.6 against each other. Fog depends only on the camera-relative distance and
    height, and the water centre snaps to 4096 m, so both poses produce the same inputs.
  - Goldens `VulkanWorldGolden.*` (moved from `test_vulkan_world.cpp`; §22 acceptance; step 7c):
    - Setup: the prototype hook, base spacing 2 (CI cost; 1 m would generate 4x more samples for the same image),
      `sizeChunks` 128, the sea level 6 m, 5000 trees by `placeTrees` in the 1 km square `[1024, 2048]^2` (seed 1),
      `completeView` and the upload wait before the capture, the rest as §17.6.
    - `world_overview`: 300 m above the ground 100 m inside the square's west edge at its centre z, 45 degrees down
      along +x.
    - `world_forest`: 2 m above max(ground, water) at a clearing near the square's centre, looking along +x and slightly
      down. The pose is chosen in the implementation (nearest tree >= 15 m) and recorded in the test comment.
    - The counts (trees per LOD, terrain nodes) are committed with them in `ExpectCounts`. The CPU reference of §17.5
      still checks the tree counts.
- `tests/test_vulkan_world.cpp` (no engine; the scene tests' fixture `VulkanWorldScene` is built on `WorldRenderer`,
  `HeightField` and `WorldParameters`). From 7b a new fixture `VulkanPrototypeScene : RenderSceneTest` (96 x 64,
  4x MSAA, as today) with `Attach(PrototypeScene&, treeCount, seed)` (trees by `placeTrees` over the 1 km area
  `[1024, 2048]^2`, the prototype hook at spacing 2, `sizeChunks` 16, the sea level 6 m), `Terrain(camera, withMasks)`
  (`fillTestTerrain` with that template's layout and `TerrainHeightSource::fromHook(prototypeHeight, seed)`) and
  `Reference()` (`forest_reference.h`). Every test of the file and its disposition:
  - Unchanged: `FrustumPlanes`, `SettingsJson`, `TreeLodEnds`, `FoliageColorWord`, `WorldBenchmark.Summarize`,
    `.SummaryText`, `.CountStalls`, `.ResultJson`, `.IncompleteRun`.
  - Adapted in 7a / 7b: `TreePlacement` (new API; `posYaw(1) == float(prototypeSurfaceHeight(x, z)) - 0.25f` exactly,
    trees above the sea), `PrototypeTreeSets` (placed trees), `WorldBenchmark.Paths` (ground callback).
  - Ported to `VulkanPrototypeScene` in 7b (same assertions unless noted): `RendersTerrainTreesAndSky` (the sky vs the
    clear colour, terrain nodes > 0, the forest against the reference), `FlatShadingIsStableFarFromTheOrigin` (at
    30 km on `fillTestTerrain` with `slopeRidgeHook` in a 128-chunk test layout; the reference frame without terrain
    and water), `ForestFollowsSettings`, `GrassToggle` (grass from masks: `Terrain(camera, true)`; the clump count
    from `GrassPass`'s statistics instead of `PrototypeWorld::stats()`), `TimeIndependentOfWallClock`,
    `DepthResolveAndPrePass` (4 and 1 samples), `Sections` (the expected name list is unchanged: §5 GPU sections),
    `ReattachReusesTreeMeshes` (a second `PrototypeScene::attach`), `ResizeWithPrototype`, `TreesToggleInFlight`,
    `PrototypeSettingsSnapshotPerFrame` (`setPrototypeSettings` stays the per-frame settings setter),
    `ReattachStartsFreshStatistics` (the forest part on `PrototypeScene`; the GPU-times part through
    `RenderScene::takeGpuFrameTimes`, `WorldRenderer::stats().gpuMs` is gone), `GameTools` (ground callback, the
    changed `game_get_state` / `game_terrain_query` fields), and `ReplacePrototypeTerrain` as
    `SetEnvironmentAgain` (`setPrototypeEnvironment` again with an attached forest rebinds the foliage texture: same
    image and counts).
  - `FoliageClassNeedsTexture`: moved to `VulkanPrototypeScene` in 7b; from 7a the death message is "needs the
    foliage texture of setPrototypeEnvironment" (`setPrototypeTerrain` also provides the texture until 7d). The check
    stays in `addMaterial` / `setMaterial` in S20; S23c moves it into the frame (F4) and changes this test.
  - Removed in 7d: `HeightFieldInterpolation`, `SharesAnExternalHeightField` (no `HeightField`), and the
    `VulkanWorldScene` fixture with its `WorldRenderer` helpers (from 7b it serves only the goldens until 7c).
  - Moved in 7c: `VulkanWorldGolden.*` (to `test_engine_terrain_render`, above).
  - New: `PrototypeTerrain.SurfaceHeightMatchesRule` (`prototypeSurfaceHeight` = the surface rule over the hook),
    `PrototypeTerrain.HighestPoint`.
- `tests/test_engine_app.cpp`: the `terrain` block of `app_status`; an MCP `set_camera` far away followed by
  `screenshot` returns a frame with every desired tile sent (`readiness` 1) and no pending uploads. With `latchedHook`,
  armed on the main thread only **after the first frame** (the first frame is a complete point whose timeout is fatal),
  and `toolTimeoutSeconds` = 1: a `set_camera` to an unstreamed area plus `screenshot` answers "terrain not complete
  before the deadline" within the 0.9 s budget (`answerDeadline`), exactly once, and the app keeps running (the latch
  is opened in `TearDown`). With the `sleepAtOrigin` mode (100 ms per generation that touches the origin),
  `setCompleteViewEachFrame(true)` completes every frame; the frame that generated the pyramid has
  `lastTerrainStreamMs()` >= 100, the frame statistics count no long frame, and the next frame runs no catch-up step
  (`World::stepIndex` advances by the steps of the frame's own `dt`: the stream time paused the clock).
- `tests/test_vulkan_world.cpp` benchmark tests: `WorldBenchmark` subtracts `terrainStreamMs` from the interval
  (`cpu_ms`, `frame_interval_ms`, `stalled_frames`: a 60 ms stream on a 3 ms frame is no stall), reports
  `terrain_stream_ms` and `wait_ms.terrain_stream`, and old aggregate initializers of `Sample` keep their meaning.

Local (GPU):

- Every `test_vulkan_*`, `test_engine_terrain_render` and `test_engine_app` test on the RTX 5080 with validation.
- `game_world` flight over the 32 km space (MCP `set_camera` path to all four coasts, 2 m to 3 km altitude): the
  terrain block of `app_status` stays <= 768 MB, and there are no holes or cracks in screenshots.
- `game_world --benchmark flyover` and `forest` before (S19b tip) and after, 5 interleaved runs each (S19a's method);
  after also with `--base-spacing 2`.
- `sample_vulkan_scene_benchmark` unchanged within noise.
- Flight, racing and platformer `--demo` and `--scenarios`, plus a TSAN build of `test_engine_terrain`
  (`SAIGA_DEBUG_TSAN`).

## Done when

The S20 row of §22:

- terrain CPU+GPU memory <= 768 MB for 32 km at 1 m (`MemoryBudget32km` with the `game_world` layout, CI; `game_world`'s
  `app_status`, local);
- origin offset 0 vs 30 km within golden tolerance (`OriginOffsetWithinGoldenTolerance`, at 28 672 m, amendment A11);
- a 1 mm camera move moves no vertex > 0.5 px (`CameraMoveAt30km`, A12);
- `terrainHeight` equals the integer rule incl. diagonals (`TerrainHeightEqualsTheIntegerRule`) and level 0 has one
  vertex per sample with matching height (`OneVertexPerSampleAtLevelZero`, `RenderedDepthMatchesSurface`);
- no background pixel between levels (`NoCrackAtLevelBorders`) (CI);
- the S02 goldens regenerated in a dedicated commit after review (step 7c);
- the flight demo screenshots reviewed (local).

Also:

- All CI tests pass on GCC, Clang and MSVC.
- The demos' `--scenarios` pass. Re-recorded values (racing lap time, flight autopilot positions) are listed with the
  reason in the PR.
- The benchmarks are reported and within §17.10. A regression above 5 % is explained (R-TEST-2): the 2 m run separates
  code changes from the content change of 1 m (about 4x the terrain triangles at every level, since `terrainLodRange`
  does not scale with the spacing). A new RTX 5080 baseline is committed when the change is accepted.
- `docs/engine_design.md` and `AGENTS.md` are updated as in "Design-doc amendments" and "Files".

## Implementation steps

Each step builds and passes ctest (CI recipe, lavapipe); no step leaves a failing test.

1. **Surface rule.** `kHeightUnit`, `kTerrainHeightLimit`, `TerrainSurface.h` (hook type, `TerrainHeightSource`,
   clamp, rule), `test_core_terrain_surface.cpp` (registered in `tests/CMakeLists.txt`).
2. **Spaces.** `SpaceTemplate` (incl. `hasTerrain`), `validateSpaceTemplates`, `WorldConfig::spaceTemplates`,
   `spaceSeed` / `spaceTemplate`, a minimal `TerrainSystem` with `height` / `sample` (source path only),
   `terrainHeight`, `characterMoveSystem` with the idle rule, `tests/terrain_test_hooks.h` (the integer hooks) and its
   engine-only part (`testSpaceTemplate`, `SAIGA_TEST_ENGINE` in `saiga_test`), `walker_follows_terrain`, the first
   `test_engine_terrain.cpp` cases (registered in `tests/CMakeLists.txt`). Engine scenarios without a template are
   unchanged.
3. **Streaming pool.** `StreamingPool` (incl. `runOneOnCaller`, `waitIdleFor`) + tests (TSAN locally); the
   `latchedHook` of `terrain_test_hooks.h`.
4. **Tile types, pure generation and `RenderScene` terrain**, in two commits:
   - 4a **Atlas.** The `RenderSceneTypes.h` types, `TerrainTiles`, `TerrainAtlas` (buffers and limits, the own upload
     batch and staging, upload queue and byte cap, page table, node bounds, rebuild, `clearTerrain`, patches with the
     ignored count, `readLayerForTests` / `readPyramidForTests`, `memoryEstimate`, `TerrainAtlas::Stats`),
     `RenderScene::setTerrainPyramid` / `setTerrainTile` / `removeTerrainTile` / `patchTerrain*` / `clearTerrain` /
     `terrainUploadsPending` / `terrainAtlasStats` (nothing drawn yet), the test helpers of `terrain_test_hooks.h`
     (`testTile`, `testPyramid`, `renderUntilUploaded`), `tests/render_scene_test_context.h` (`RenderSceneTest`), and
     the step-4a cases of `test_vulkan_terrain.cpp` (registered in `tests/CMakeLists.txt`).
   - 4b **Pass.** `TerrainPass` (selection, anchor, `TerrainFrame`, skirts, water, `TerrainPass::Stats`, the S19a
     section names), the new shaders (`terrain_shading.glsl` as a copy), the probe and `selectedNodes`,
     `readDepthForTests`, `fillTestTerrain` (without masks), and the step-4b cases without grass. The old
     `TerrainRenderer` still serves `setPrototypeTerrain`; the new pass draws only when a pyramid is set; the two paths
     are exclusive (asserted).
5. **Grass.** `GrassPass` (chunk-local hashes, lattice and wind), masks, `setGrassMask` / `removeGrassMask`,
   `grass.vert` / `grass.frag` (a copy of `world/foliage.frag`), `fillTestTerrain`'s masks, the grass tests.
6. **Engine streaming**, in two commits:
   - 6a **Store, system and sinks.** `TerrainGeneration`, `TerrainStore`, the full `TerrainSystem` (desired set with
     truncation, LRU, the send point incl. `sendGrassMask`, GPU-set rebuild on a layout change, the `clearTerrain`
     trigger, masks on the pool, the job lambdas that name their request, `completeView` with its loop and deadline
     rule, `readiness`, `terrainUploadsPending`, stats incl. `pyramidsGenerated`, `poolCreated`), the resident path of
     `terrainHeight`, the `RenderSink` virtuals incl. `clearTerrain`, the patches and `terrainUploadsPending`
     (`RecordingRenderSink` incl. `uploadsPendingForTests`, `VulkanRenderSink`), `World::setRenderSink` forwarding,
     the streaming and memory tests of `test_engine_terrain.cpp`.
   - 6b **App.** `EngineApp` (`updateView`, the complete points incl. the MCP screenshot with
     `ScreenshotRequest::deadline` and the earliest pending deadline, the upload wait, `setCompleteViewEachFrame`,
     `lastTerrainStreamMs` and its exclusion from the work, `FrameStatistics` and the simulation `dt`, the statistics
     section, `app_status`), the `test_engine_app.cpp` cases, `test_engine_terrain_render.cpp` with
     `TerrainRenderTest` and `OriginOffsetWithinGoldenTolerance` (registered in `tests/CMakeLists.txt`).
7. **Prototype and games**, in four commits:
   - 7a **New path next to the old.** `PrototypeTerrain`, the new `placeTrees`, `prototypeTreeSets` and
     `PrototypeForest::init` overloads (the `HeightField` ones stay), `PrototypeScene`, `setPrototypeEnvironment` with
     `scene/sky.*` (copies; exclusive with `setPrototypeTerrain`), `WorldCommon.h::viewRotation`, `WorldBenchmark`
     (ground callback overload, `Sample::terrainStreamMs`, `setTerrainInfo`, the new JSON keys and summary row),
     `WorldMcpContext` (the new fields, used when set; the `WorldRenderer` path stays), the `FoliageClassNeedsTexture`
     message naming `setPrototypeEnvironment`, the new `PrototypeTerrain` tests and the benchmark tests.
   - 7b **Migrate.** `samples/games/common/PrototypeSpace.h`; `game_world` (incl. `--base-spacing` and
     `setCompleteViewEachFrame` in `--benchmark`), flight and racing (contexts on `prototypeSurfaceHeight` with the
     setup's seed), platformer include, `sample_vulkan_scene_benchmark`; `test_vulkan_scene` `DepthPrePassMatches`;
     `tests/forest_reference.h`, the `VulkanPrototypeScene` fixture and every port of the `test_vulkan_world`
     disposition list (Tests) except the goldens (they still run on `VulkanWorldScene` and `WorldRenderer`); measure
     the racing terrain query time (§2).
   - 7c **Goldens (dedicated commit, after the images were reviewed).** The `VulkanWorldGolden.*` tests move to
     `test_engine_terrain_render.cpp` with the new compositions, the regenerated `world_overview.png` /
     `world_forest.png` (written with the AGENTS.md recipe, on lavapipe) and the committed counts; the old golden tests
     leave `test_vulkan_world.cpp`. No other change in this commit.
   - 7d **Remove the old path.** `HeightField`, `TerrainRenderer`, `GrassRenderer`, `WorldRenderer`,
     `WorldStatistics.cpp`, `setPrototypeTerrain`, the old `placeTrees`, the `WorldRenderer` path of `WorldMcp` and
     `WorldBenchmark`, the `HeightField` overloads of `prototypeTreeSets` / `PrototypeForest::init`,
     `samples/games/common/WorldLighting.h`, the old shaders (`world/terrain.*`, `grass.vert`, `heightmap.glsl`,
     `common.glsl`, `shading.glsl`, `foliage.frag`, `sky.*`), `WorldParameters`, the `VulkanWorldScene` fixture and the
     remaining `HeightField` tests; `PrototypeWorld` shrinks to the environment.
8. **Measure and document.** Benchmarks (before and after, and the 2 m run), demo screenshots and scenario
   re-recordings, design-doc and AGENTS.md updates, then this file's Results and Design changes.

## Verification

```shell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSAIGA_BUILD_SAMPLES=ON -DSAIGA_ARCHNATIVE=OFF \
  -DSAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON          # + the CMake 4 / GCC 15 flags of AGENTS.md where needed
cmake --build build --parallel -- -k 0
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest --test-dir build --output-on-failure --parallel 4
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json SAIGA_UPDATE_GOLDEN=1 ./build/tests/test_engine_terrain_render \
  --gtest_filter='VulkanWorldGolden.*'            # step 7c only
# the benchmark recipe of AGENTS.md (build-bench), flyover and forest x 5 interleaved, before and after;
# after also with --base-spacing 2
for g in flight racing platformer; do ./build/samples/games/game_$g --scenarios; done
SAIGA_SCREENSHOT=/tmp/flight.png ./build/samples/games/game_flight --demo
```

## Files

- New:
  - `src/saiga/core/math/TerrainSurface.h`
  - `src/saiga/engine/world/SpaceTemplate.h`
  - `src/saiga/engine/core/StreamingPool.{h,cpp}`
  - `src/saiga/engine/terrain/{TerrainGeneration,TerrainStore,TerrainSystem}.{h,cpp}`
  - `src/saiga/vulkan/scene/{TerrainTiles,TerrainAtlas,TerrainPass,GrassPass}.{h,cpp}`
  - `src/saiga/vulkan/world/{PrototypeTerrain,PrototypeScene}.{h,cpp}`
  - `samples/games/common/PrototypeSpace.h`
  - `shader/vulkan/scene/{terrain.vert,terrain.frag,terrain_vertex.glsl,terrain_heights.glsl,terrain_common.glsl,terrain_shading.glsl,terrain_probe.comp,grass.vert,grass.frag,sky.vert,sky.frag}`
    (`terrain_shading.glsl`, `grass.frag` and `sky.*` start as copies of `world/shading.glsl`, `world/foliage.frag` and
    `world/sky.*`)
  - `tests/test_core_terrain_surface.cpp`, `tests/test_engine_terrain.cpp`, `tests/test_vulkan_terrain.cpp`,
    `tests/test_engine_terrain_render.cpp`, `tests/terrain_test_hooks.h`,
    `tests/render_scene_test_context.h`, `tests/forest_reference.h`
- Changed:
  - `src/saiga/core/math/ChunkId.h`
  - `src/saiga/engine/world/World.{h,cpp}`
  - `src/saiga/engine/gameplay/CharacterMoveSystem.cpp`
  - `src/saiga/engine/sync/{RenderSink,RecordingRenderSink,VulkanRenderSink}.*`
  - `src/saiga/engine/app/{EngineApp,AppTools,EngineAppSetup}.*` (`FrameStatistics` itself is unchanged: the app
    passes the interval without the stream time and calls `advance`; `AppTools.cpp`: `ScreenshotRequest::deadline`
    and the pending deadlines registered with the app; `answerDeadline` in `WorldTools.cpp` is used as it is)
  - `src/saiga/engine/tools/ScenarioWorld.*` (comment)
  - `src/saiga/vulkan/scene/{RenderScene,RenderSceneTypes,PrototypeWorld}.*`
  - `src/saiga/vulkan/world/{ForestRenderer,PrototypeForest,WorldBenchmark,WorldMcp,WorldCommon,WorldSettings}.*`
  - `samples/games/{world,flight,racing}/` (incl. `FlightWorld.{h,cpp}`, `RacingWorld.{h,cpp}`, `FlightRender.cpp`,
    `RacingRender.cpp`), `samples/games/platformer/PlatformerRender.cpp`
  - `samples/vulkan/sample_vulkan_scene_benchmark.cpp`
  - `tests/{CMakeLists.txt,test_vulkan_world.cpp,test_vulkan_scene.cpp,test_engine_scenarios.cpp,test_engine_app.cpp,test_engine_gameobject.cpp}`
    (the last only if the idle rule needs a comment; its expectations hold on the flat default)
  - `data/tests/golden/world_{overview,forest}.png` (7c)
  - `data/tests/benchmarks.json` (on acceptance)
  - `docs/engine_design.md`; `AGENTS.md` (layout: `engine/terrain/`, `TerrainTiles`, the shader moves; the golden
    recipe's binary `test_engine_terrain_render`; the test count; the `--base-spacing` benchmark flag), this file
- Removed:
  - `src/saiga/vulkan/world/{HeightField,WorldRenderer}.*`, `WorldStatistics.cpp`
  - `src/saiga/vulkan/scene/{TerrainRenderer,GrassRenderer}.*`
  - `samples/games/common/WorldLighting.h`
  - `shader/vulkan/world/{terrain.vert,terrain.frag,grass.vert,heightmap.glsl,common.glsl,shading.glsl,foliage.frag,sky.vert,sky.frag}`
    (7d)

## Risks and open questions

- **Generation cost of the float noise at 1 m.** First full view in seconds, with streaming visible during fast flight.
  Mitigation: pyramid and coarse levels first, two workers, the complete points. If the flyover shows upload stalls,
  `uploadsPerFrame` / `terrainUploadBytesPerFrame` are tuned and the numbers are reported.
- **CI time.** The goldens generate about 2.7 M samples with the expensive hook (about 2 s each). The other Vulkan
  terrain tests use cheap integer hooks. If CI time grows noticeably, the golden pyramid could be shared per process
  through a test-local cache.
- **Float noise across compilers** (until S23). As today, the golden counts gate only on lavapipe; FMA contraction can
  change single samples by one unit, which is invisible.
- **Skirt artifacts.** They can show at silhouettes and the space edge. Their depth is the node's height range; the
  space edge lies under the sea for the prototype hook.
- **Benchmarks.** At 1 m every level has about 4x the terrain triangles (terrain + water up from about 0.09 ms). With
  `completeView` in every frame the streaming cost is excluded from the frame interval (`cpu_ms`,
  `frame_interval_ms`, `stalled_frames`) and from CPU busy, and reported in `terrain_stream_ms` /
  `wait_ms.terrain_stream`; the upload copies stay in the GPU `transfer` section and are explained. The flyover's
  per-frame `completeView` makes a run take longer in wall time (new 1 m tiles at 46 ms each), not its numbers.
- **Racing startup.** The site search over `prototypeSurfaceHeight` costs about 2 s (1.1 M queries) instead of
  bilinear lookups; acceptable for a tech demo and stated. If the scenario runs become too slow locally, a coarser site
  grid is the fix (the site must stay the same, checked by the scenarios).
- **Racing site and the demos' recorded values** can shift slightly with the new surface (1 m sampling, the diagonal).
  They are re-recorded with a note. The site is expected to stay; if it moves, the old site is pinned in
  `RacingContext`.
- **Storage-buffer reads in the vertex shader** instead of a texture: same access pattern (one fetch per vertex plus the
  normal's four); measured in the benchmark. If lavapipe or the RTX path is slower, the atlas can become an R32I
  texture array later without changing the sink interface.
- **Open question for review: base spacing of the goldens.** Spacing 2 keeps CI fast, but `game_world` uses 1 m. A
  local 1 m screenshot check covers the 1 m path, and `OneVertexPerSampleAtLevelZero` runs at both spacings.

## Design-doc amendments

To be applied to `docs/engine_design.md` in step 8, together with the usual status updates of §3, §7.9, §7.10, §8.5,
§9.3, §10.1, §10.5, §12.2, §17.1, §17.2, §17.5 to §17.7, §18.1, §18.4, §19 and §22:

- **A1 (§12.2, skirts):** "tile meshes have skirts (as the prototype CDLOD)" is wrong: the prototype has none. New
  text: the node grids have skirts, because refinement depends on residency. Depth = the node's height range, at least
  two spacings.
- **A2 (§12.2 / §19, diagonal):** record that S20 moves every consumer from the prototype's (1,0)-(0,1) split to the
  fixed (0,0)-(1,1) diagonal (goldens regenerated).
- **A3 (§12.4, space seed):** §12.4 gives every template `hashCombine(world seed, index)`, while §12.1 has space 0's
  plan take the world seed. Resolution (D11): space 0 = the world seed, `hashCombine(world seed, index)` for index >= 1.
- **A4 (§17.7, benchmark world):** "default `WorldParameters` (8 km, 200k trees)" becomes `game_world`'s 32 km, 1 m
  space with the 200k trees in the 8 km prototype area `[0, 8192]^2`. The paths keep the centre (4096, 4096) until
  S27, and S27 moves them to the space centre with far content. Terrain residency is completed in every frame
  (`EngineApp::setCompleteViewEachFrame`, E9); its time is excluded from the frame interval, CPU busy and the stall
  count and reported as `terrain_stream_ms` / `wait_ms.terrain_stream` with a `terrain` block; `--base-spacing 2` for
  comparisons.
- **A5 (§12.2, the prototype hook):** it continues the S01 noise beyond 8 km, with the border falloff at the edges of
  the 32 km space and its absolute width (819.2 m), since the hook signature carries no world size.
- **A6 (§12.2, the hook's location, D2):** the hook and `prototypeSurfaceHeight` live in
  `vulkan/world/PrototypeTerrain` (the prototype's game side, like `PrototypeForest`, used by the tests); the
  `SpaceTemplate` that uses it is built in `samples/games/common/PrototypeSpace.h`. Tile, pyramid and grass-mask
  generation are pure functions in `vulkan/scene/TerrainTiles`, wrapped by the engine's `TerrainGeneration`.
- **A7 (§7.9):** S20 implements `name`, `kind` (OpenWorld only; Enclosed rejected until S21), `baseSpacing`, `seaLevel`,
  `sizeChunks` (open world: multiple of 16, 16..512) and `prototypeHeight`. Validation uses `SAIGA_EXIT_ERROR`.
  `ambient` and `fog` are marked [S36]: until then the game's `PrototypeSettings` carry them as run-time settings. An
  empty `spaceTemplates` means one flat default template. A space without a hook is flat and draws no terrain (D1).
- **A8 (§10.1, `RenderSink` / `RenderScene`):** add `removeGrassMask(ChunkId)`, `clearTerrain()` (E1),
  `terrainUploadsPending()` (E8) and the calls `patchTerrainTile`, `patchTerrainPyramid` (E2: absolute int32 heights of
  a sub-rectangle, copied at the call, queued behind a pending full copy, ignored and counted without a layer). Masks
  leave with the grass radius. `setTerrainPyramid` of another space or layout starts a new GPU terrain set (D5) and the
  sink drops the old set's tiles and masks itself (no removal calls); `setTerrainTile` / `setGrassMask` of a set
  entry replace it in place.
- **A9 (§10.1, sea level):** until S39's `setWater`, the sea level travels in `TerrainPyramidData` together with the
  space's `TerrainLayout`.
- **A10 (§12.2, GPU format):** replaces "R32F tiles": the GPU terrain set is int32 storage buffers (atlas with
  `streamedLevels * tilesPerLevel` layers, page entry = layer only, pyramid, page table, grass masks) filled by buffer
  copies through the scene's terrain upload queue with a per-frame byte cap; the shader
  subtracts a frame-global reference height and scales by `kHeightUnit` (exact, differences below 2^24 by
  `kTerrainHeightLimit`). The pyramid is fixed at 32 m for both spacings, CDLOD root nodes are 4096 m, node origins are
  integers relative to a per-frame 64 m anchor, and node bounds are hierarchical.
- **A11 (§22, S20 row):** "origin offset 0 vs 30 km" is tested at 28 672 m (112 chunks), a multiple of the 4096 m root
  node, of every tile level and of the 4096 m water snap, so both images have the same LOD structure; the poses look
  down so that neither sky nor the space edge is in view.
- **A12 (§22, S20 row):** "a 1 mm camera move moves no vertex > 0.5 px" means the vertex's displacement differs from
  the double-precision displacement by at most 0.5 px (near vertices legitimately move more than 0.5 px). Tested on the
  GPU vertex function with probes.
- **A13 (§12.2 / §12.3, request point):** in S20 the visual requests come from the view camera in `EngineApp` between
  steps (`TerrainSystem::updateView`), not from phase 7. Phase-7 requests from players come with S24.
- **A14 (§8.5):** the streaming pool is per `World` (owned by `TerrainSystem`), created on the first request, with
  best-effort low priority; `submit(job, priority, completion)` and `drain()` on the main thread, exceptions rethrown at
  the drain (D8).
- **A15 (§19 / §18.1):** `WorldRenderer` and `WorldStatistics.cpp` are replaced by `PrototypeScene` (settings, GUI,
  forest) plus the engine statistics window's terrain section. `HeightField` is removed and `WorldCommon.h` keeps the
  frustum helpers and `viewRotation`. `setPrototypeTerrain` becomes `setPrototypeEnvironment`.
- **A16 (§12.2):** `terrainHeight` clamps x and z to the space (D12).
- **A17 (§17.2):** `ScenarioWorld` uses the setup's space templates (flat without a hook).
- **A18 (§12.2, height range):** `kTerrainHeightLimit` = 2^23 units (+-8192 m) is introduced by S20; every hook result
  is clamped to it (S22 clamps edits, S23 generators to the same limit).
- **A19 (§12.2 / §12.4, readiness and the GPU terrain set, D5, D6):** `TerrainSystem::readiness(space, camera, radius)`
  is the fraction of the pyramid plus the desired tiles within the radius sent to the sink; the sink holds one GPU
  terrain set (the camera's space), rebuilt on a space or layout change; other spaces keep only their CPU pyramid.
- **A20 (§8.2, mover):** in a space with terrain, idle characters are written only when their y differs from
  `terrainHeight`; without terrain the idle rule is unchanged (not written).
- **A21 (§10.1 / §12.2, D4 amended by E2):** D4's "the atlas converts with its stored per-tile reference" is replaced:
  there is no per-tile reference, tiles and patches are absolute int32 heights, the shader subtracts one frame-global
  reference. S20 implements and tests both patch calls; S22 only calls them.
- **A22 (§7.9 / §12.1, E6 / E7):** `SpaceTemplate::hasTerrain()` is the single "has terrain" test (S23a:
  `Generator::hasTerrain()`). Generation and `TerrainSystem::sample` read a `TerrainHeightSource` (a callable over
  integer metres, the seed bound); the hook is wrapped by `fromHook`, which clamps with `clampTerrainHeight`.
- **A23 (§17.1, statistics; §8.1, the clock):** the time of a frame's `completeView` (complete points, benchmark mode)
  is the frame's terrain stream time: not CPU work, not part of the frame interval in the statistics window (the time
  base still advances), shown in the window's Terrain section. It pauses the simulation clock: `EngineApp::frame()`
  subtracts it from the next frame's `dt`, so a complete point causes no catch-up steps.
- **A24 (§12.3, who streams):** only `TerrainSystem::updateView` / `completeView` request terrain, and only the app
  calls them for its view world; a render sink alone (every scenario world has a `RecordingRenderSink`) requests nothing
  and starts no pool.

## Results

Measured on the RTX 5080 Laptop GPU (driver with validation for the tests) and on lavapipe (Mesa with LLVM 21.1.8).

- **Tests.** ctest 73/73 on the GPU (local build with samples; with the CI options 71 tests: 67 + the four new
  `test_core_terrain_surface`, `test_engine_terrain`, `test_vulkan_terrain`, `test_engine_terrain_render`); every
  `test_vulkan_*` / `test_engine_*` on lavapipe 33/33.
- **Done when.** `MemoryBudget32km` (the `game_world` layout at 1 m): worst CPU + GPU 198.1 MB, static bound 202.9 MB
  (<= 768 MB). `OriginOffsetWithinGoldenTolerance` at 28 672 m: identical images (mean 0). `CameraMoveAt30km`,
  `TerrainHeightEqualsTheIntegerRule`, `OneVertexPerSampleAtLevelZero` (both spacings), `RenderedDepthMatchesSurface`,
  `NoCrackAtLevelBorders` pass on both devices. The S02 goldens were regenerated on lavapipe in step 7c (counts:
  `world_overview` 0 / 2 / 3937 trees per LOD, 72 terrain nodes; `world_forest` 27 / 338 / 506, 131 nodes); on the RTX
  (not gating) `world_forest` differs by mean 1.02/255. After the implementation review's relief-independent LOD metric
  `world_overview` (a high camera: coarser) has 65 terrain nodes and was regenerated (it had differed by mean 0.17/255).
- **Scenarios.** `--scenarios` of flight, racing and platformer: 18 runs passed each (scenarios x variants incl. host + client), no value
  re-recorded (the flight autopilot positions and the racing lap keep their committed tolerances on the new surface).
  Racing startup: the site search over `prototypeSurfaceHeight` takes 1.33 s.
- **Screenshots** (`SAIGA_SCREENSHOT`, RTX): `game_world` (the default camera 167 m above the 8 km area), flight,
  racing and platformer `--demo`, reviewed: terrain, trees, grass near the ground, sky and water as before; no holes
  or background between levels. At steep near silhouettes the skirts of finer nodes show as short saw teeth
  (the risk "Skirt artifacts" below; accepted for the prototype).
- **Benchmarks** (build-bench, Release without asserts and `-march=native`; 5 interleaved runs, medians of the run
  medians; before = the S19b tip `c990775` with the plan, after = this branch at 1 m and with `--base-spacing 2`):

  | | before | after 1 m | after 2 m |
  |---|---|---|---|
  | flyover `cpu_ms` (interval without stream time) | 2.47 | 3.83 | 3.37 |
  | flyover GPU busy / `scene` | 2.45 / 2.36 | 2.51 / 2.43 | 1.87 / 1.76 |
  | flyover trees / terrain + water / sky | 1.440 / 0.094 / 0.011 | 1.406 / 0.191 / 0.013 | 1.401 / 0.123 / 0.011 |
  | flyover upload + cull | 0.075 | 0.106 | 0.091 |
  | flyover `terrain_stream_ms` median / mean | - | 22.0 / 13.1 | 6.4 / 3.9 |
  | forest `cpu_ms` | 2.08 | 2.19 | 2.14 |
  | forest GPU busy / `scene` | 2.06 / 2.01 | 2.19 / 2.14 | 2.14 / 2.09 |
  | forest trees / terrain + water / grass | 0.469 / 0.049 / 0.688 | 0.555 / 0.114 / 0.681 | 0.555 / 0.063 / 0.681 |
  | terrain CPU / GPU MB (flyover) | - | 88.7 / 98.1 | 36.3 / 49.0 |

  `sample_vulkan_scene_benchmark` (2 runs): cull 0.068 -> 0.058 ms, instance draw 1.28 -> 1.27 ms; 126 572 -> 122 517
  LOD-2 instances (the trees are placed on the new surface).
  Explanations (R-TEST-2): terrain + water at 1 m is about 2x (4x the triangles per level, as expected); the 2 m run
  shows the code's own part (+31 % flyover, +29 % forest, 0.03 ms / 0.014 ms: skirts, storage-buffer reads, the
  quarter grids). The forest's trees +18 % (equal at 1 m and 2 m, while the flyover's trees are -2.6 %, the same
  pipeline and shaders) is content: `placeTrees` now samples the hook on the integer surface instead of the bilinear
  2 m height field (199 769 trees in 976 sets instead of 199 466 in 928, and the scene benchmark's visible count
  changes too), so the forest path near the ground sees a different tree distribution. The flyover's frame interval
  is not comparable: at 1 m the path streams in most frames (median 22 ms), and before such a frame the benchmark lets
  the GPU drain (see deviations), which stays in the interval; its GPU sections are the comparison. No new baseline is
  committed (on acceptance, Done when).
- TSAN: `test_engine_terrain` built with `-fsanitize=thread` (Core + Vulkan + Engine) passes 36/36 without a
  ThreadSanitizer warning. `SAIGA_DEBUG_TSAN=ON` itself does not compile (also on `master`):
  `cmake/Saiga_compiler_flags.cmake` appends `"-fsanitize=thread -static-libasan ..."` as one list element, so the
  flags were passed through `CMAKE_CXX_FLAGS` / the linker flags instead (a separate fix).
- Not done locally: the MCP flight to the four coasts of the 32 km space (the instructions allowed only the
  `SAIGA_SCREENSHOT` demo runs; the memory bound is covered by `MemoryBudget32km`).

## Review log

Plan review rounds (local Opus) and later the PR reviews are recorded here: date, reviewer, findings, resolution.

- **Review round 1** (2026-11-21, Opus and Fable, 6/10 each, 21 merged findings; cross-plan decisions D1 to D14):
  - Layering (1, D2): pure generation moved to `vulkan/scene/TerrainTiles`; `prototypeSpaceTemplate` moved to
    `samples/games/common/PrototypeSpace.h`; engine-path render tests (goldens, origin offset) in the new
    `test_engine_terrain_render.cpp`; `test_vulkan_terrain` uses the pure functions available in step 4.
  - Atlas upload (2): storage buffers (int32) instead of a `Texture2DArray`, buffer copies through `UploadBatch`, a
    per-frame byte cap with GPU residency from the recording frame, staging high-water in stats and the budget;
    `UploadBudgetSpreadsBursts`.
  - Grass invariance (3): chunk-local hashes, an 8 m lattice wrapped per chunk, integer wave numbers for the wind.
  - Desired set over budget (4): truncated nearest-first per level and for masks, `Stats::truncated`, a test with an
    oversized range.
  - Layout (5): `TerrainLayout` = template + view settings, fixed per GPU set; a change starts a new set (test).
  - Bounds (6): hierarchical node bounds incl. skirts; water culled separately; `BoundsIncludeFinerTiles`.
  - Shared vertices (7): per-frame 64 m anchor, integer node origins, one frame-global reference height.
  - Origin offset test (8): downward poses (no sky, no space edge, ground within 700 m), a separate grass pose, a
    4096 m-periodic hook, water snapped to 4096 m.
  - Crack test (9): oblique 35 degree view across an off-centre border, negative control > 10 pixels.
  - Benchmarks (10, D9): `completeView` per measured frame, streaming time separate, `--base-spacing 2` run;
    `terrainLodRange` does not scale with the spacing (stated).
  - StreamingPool (11, D8): `submit(job, priority, completion)`, `drain`, per-job exceptions, `waitIdle` never throws,
    destruction order, in-flight registration at job start.
  - `sizeChunks <= 512` (12): no 4096-texel limit any more (pyramid in a storage buffer, checked against
    `maxStorageBufferRange`); the multiple-of-16 rule is an open-world rule and S21's/S22's templates fit.
  - Surface preconditions (13): fracBits <= 20, `kTerrainHeightLimit` in S20 with clamping, corrected test text.
  - HeightField consumers (14): table with every consumer and its replacement; changed tool outputs listed.
  - Omissions (15): `test_vulkan_scene` `DepthPrePassMatches`, the platformer include, the shader include chain
    (`terrain_shading.glsl`, `grass.frag`), `TreePlacement` compared against `float(prototypeSurfaceHeight)`.
  - Step 7 (16): split into 7a to 7d, all green; readiness and desired-vs-resident exposed (D6); masks generated on the
    pool, regenerated on a threshold change.
  - Sink coupling (17, D4, D5): replace-in-place, the patch calls (implemented and tested in S20), the single send
    point for S22's merged copies, GPU-set rebuild per space.
  - EngineApp (18): the MCP screenshot is a complete point with the upload wait; `app_status` gets pending and
    readiness.
  - Mover (19): idle characters snapped when off the surface; query counts stated, no cache or pin API in S20 (S24's
    residency is the fix if the measurement in 7b says so).
  - Spec gaps (20): morph target `(i & ~1, j & ~1)` and the morphed height formula; `RenderedDepthMatchesSurface`
    checks the real vertex pipeline.
  - Validation (21, D1): `SAIGA_EXIT_ERROR`.
  - Rejected: none. Not adopted as suggested: the image-copy path of finding 2 (storage buffers make it unnecessary)
    and a 64 km test space for finding 8 (the downward poses keep A11's 28 672 m in the 32 km space).
- **Review round 2** (2026-11-21, Opus and Fable, 7/10 each, 18 merged findings; cross-plan decisions E1 to E14; the
  worktree at master incl. #66, benchmark and statistics text based on its code):
  - Atlas format vs D4/S22 (1, E2): D4's per-tile reference replaced (A21); patch semantics (sub-rectangle of absolute
    heights, copied at the call, queued behind a pending full copy, counted against the byte cap, ignored and counted
    without a layer or for another space, bounds widened at every level); `readLayerForTests` / `readPyramidForTests`;
    `PatchTileAndPyramid` on read-back integers.
  - Space switch (2, E1): implicit drop on a pyramid of another space or layout stays; `clearTerrain()` in sink and
    scene for a terrain-less view space; tests on `RenderScene` and the recording; the trigger is first driven by S21.
  - Benchmark mode (3, E9): `EngineApp::setCompleteViewEachFrame`; `WorldBenchmark::Sample::terrainStreamMs` (appended
    field of `addSample`) filled from `EngineApp::lastTerrainStreamMs`; subtracted from the interval before `cpu_ms`,
    `frame_interval_ms` and `countStalls`; CPU busy excludes it in `EngineApp`; `terrain_stream_ms`,
    `wait_ms.terrain_stream`, `terrain` block, summary row; the F1 statistics exclude it too (A23).
  - StreamingPool (4): the in-flight-at-start rule moved to S23a's own job code; `TerrainSystem` records requests at
    submit and frees them on completion or queued cancel; `runOneOnCaller` and `waitIdleFor` for `completeView`;
    generation exceptions are fatal (no leaked request); cancel wording aligned with S23a (running jobs may ignore the
    flag); the impossible test replaced by `queued()` / `running()` with held jobs.
  - Flight/racing (5): the contexts are `World`-independent, so they call the pure `prototypeSurfaceHeight` with the
    setup's seed (bit-equal to `terrainHeight`, new test); per-call-site table incl. the racing render rows; costs
    stated (70 µs per racing step, about 2 s site search at startup).
  - Coexistence 4-7c (6): `TerrainFrame` owned by `TerrainPass`; the moved shaders are copies until 7d; `scene/sky.frag`
    includes `terrain_shading.glsl`; the new sky is drawn after `setPrototypeEnvironment` (none in steps 4-6); the two
    paths are exclusive (asserted); the 7a overloads of `prototypeTreeSets` / `PrototypeForest::init` listed.
  - Grass masks in the send point (7, E3): `sendGrassMask`; `setGrassMask` replaces in place.
  - `fracBits` (8): `8 + log2(baseSpacing)` and `* 2^fracBits`.
  - `AtlasLimits` (9): `RenderSceneParameters::storageBufferLimit` = 32 MiB; the full-level case inside
    `#ifdef SAIGA_ASSERTS`.
  - Memory (10): a dedicated terrain upload batch and fixed staging (never grows) per frame in flight; the upload data
    vector, the queue (shared vectors, patch copies) and the node bounds (22.4 MB at 1 m) added; total about 213 MB;
    `TerrainAtlas::memoryEstimate` used by the recording and `MemoryBudget32km`.
  - OriginOffset (11): `fovy` 40 degrees, corner rays recomputed (ground within 600 m / 12 m).
  - Test hooks (12): `tests/terrain_test_hooks.h` shared with S22/S23; `prototypeGrassMask` signature confirmed (E3).
  - Complete points (13): `completeView` returns false on timeout, callers decide; the MCP screenshot uses its answer
    deadline and answers an error; each waited frame completes again; `terrainUploadsPending` (E8);
    `TerrainPass::selectedNodes()` for the probes.
  - Mover (14): the idle rule applies only in spaces with terrain (existing tests unchanged), cost stated.
  - Edge cases (15): far edge = last cell with `fx = 1` (tested); `game_get_state` min/max fall back to the game's
    area stats while the pyramid is null.
  - Hook type (16, E7): `TerrainHeightSource` with `fromHook` (clamps); `hasTerrain()` (E6).
  - Step size (17, E14): steps 4 and 6 split into 4a/4b and 6a/6b.
  - `sample` comment (18): the base read; S22's `terrainSample` adds the delta.
  - Rejected: from 10, "staging slots = `VulkanBase::numSwapchainFrames`": the scene's per-frame resources are per
    frame in flight (`VulkanRenderer::framesInFlight` = 2, static_assert in SceneCulling), not per swap chain image.
    Not adopted as suggested: 14's "snap on spawn/teleport only" (gating the existing rule on `hasTerrain()` keeps all
    tests and needs no teleport counter); 5's "World& per call site" (the contexts have no `World`; the pure function is
    exact and simpler); 3's "game calls `completeView` in `beforeRender`" (E9 allows either; the `EngineApp` switch keeps
    the stream-time accounting in one place).
- **Review round 3** (2026-11-21, Opus and Fable, 8/10 each, 15 findings; cross-plan decisions F1 to F4, of which
  only F4 touches S20):
  - Fixture (1): `tests/render_scene_test_context.h` with `RenderSceneTest` (per-test `HeadlessWindow` + `RenderScene`,
    the `VulkanTest` skip behaviour plus the `drawIndirectFirstInstance` skip); `test_vulkan_terrain` uses it,
    `test_engine_terrain_render` derives `TerrainRenderTest` with a `VulkanRenderSink` on the scene; `AtlasLimits`
    keeps `GTEST_FLAG_SET(death_test_style, "threadsafe")`.
  - `test_vulkan_world` (2): a disposition for every test (unchanged, adapted, ported to `VulkanPrototypeScene` in 7b,
    moved in 7c, removed in 7d; `ReplacePrototypeTerrain` becomes `SetEnvironmentAgain`); `fillTestTerrain` in
    `terrain_test_hooks.h`; `forest_reference.h` for the goldens; the S19a section names kept (`gpuKeys`,
    `s03Passes`); where the `WorldParameters` fields go; `FoliageClassNeedsTexture` names `setPrototypeEnvironment`
    (F4: S23c moves the check into the frame).
  - Complete-view loop (3): pyramid, then the desired set and tiles, then masks, repeated until nothing is missing;
    `updateView` requests only the pyramid until it is stored; tests `CompleteViewFirstView`,
    `UpdateViewRequestsThePyramidFirst`.
  - Benchmark clock (4): the stream time pauses the simulation clock (`EngineApp::frame()` subtracts it from the next
    `dt`), no catch-up steps; A23 extended.
  - Timeout (5): the deadline is checked before each caller job, a running caller job is not preempted (overrun bounded
    by one job); `latchedHook` blocks only off the arming thread and is armed after the first frame.
  - MCP deadline (6): `ScreenshotRequest::deadline` from `answerDeadline` in the handler, the earliest pending deadline
    bounds the frame's `completeView` (minus a 50 ms margin), the timeout answers once before the poll; 0.9 s budget in
    the 1 s test.
  - `sample` on a terrain-less space returns 0 (7), in `FlatWithoutHook`.
  - Exceptions (8): the job lambdas rethrow with their request in the message.
  - `UploadBudgetSpreadsBursts` (9): 30 tiles in each of 5 levels.
  - Streaming gate (10): "no sink" replaced by "only `EngineApp` calls `updateView` / `completeView`; a sink alone
    requests nothing and starts no pool" (A24, `SinkAloneStreamsNothing`, `poolCreated()`).
  - Step 4a stats (11): `TerrainAtlas::Stats` (4a) separate from `TerrainPass::Stats` (4b).
  - `memoryEstimate` (12): the parameters argument defaults.
  - `hasTerrain()` inline (13); the engine-only part of `terrain_test_hooks.h` is gated on `SAIGA_TEST_ENGINE`, which
    `saiga_test` defines for every test linking `saiga_engine`.
  - `integerTestHook` (14): kept and described; S22 uses it through `testSpaceTemplate` in its
    `terraformTestTemplate` (S22's round 3), so the claim is now true on both sides.
  - Pyramid reuse (15): `Stats::pyramidsGenerated`; `pyramid(space)` keeps its pointer.
  - Also: `RecordingRenderSink::uploadsPendingForTests` (S21's load end rule pins the forwarding with it).
  - Rejected: none.

### Implementation (2026-11-21, Claude Opus 5.5)

Deviations from the plan, each the smallest sound fix:

- `setPrototypeEnvironment` and the `FoliageClassNeedsTexture` message came in step 5 instead of 7a (the grass needs
  the foliage texture); the sky followed in 7a.
- `WorldNoise` moved out of `HeightField` into its own file in 7a (the hook needs it after 7d removes `HeightField`);
  `PrototypeForest::init` and `PrototypeScene::attach` take the species seed.
- Rebase onto `master` with S19c: the Terrain statistics are a section of the debug UI's Engine tab
  (`EngineAppDebugUi.cpp`, "Terrain") instead of the statistics window; `PrototypeScene::imgui` draws widgets only and
  the games add it as the Engine tab's "World" section; `app_status` has both `ui_tab` and the terrain block.
- Benchmark mode: before a frame that has to stream (`TerrainSystem::viewComplete` is false), `EngineApp` waits for
  the GPU to finish the frames in flight. Without it the GPU ran ahead during the stream and the interval minus the
  stream time came out at about 0.7 ms (1414 fps). The wait is frame wait (not work), pauses the simulation clock and
  the statistics' interval like the stream time, but stays in the benchmark's interval (`lastTerrainStreamMs` does not
  contain it). The flyover at 1 m streams in most frames, so its interval is an upper bound; §17.7 says so.
- `CameraMoveAt30km`: the probes are filtered to a cone in front of the camera, the focal length comes from the
  nearest probe, the camera sits at half-ulp offsets of 30 km, yaw 45 degrees and pitch -20 degrees.
- `StreamTimePausesTheClock` (lavapipe): `residencyRange0 = 48`, no grass, only the samples after the move frame are
  checked, the longest interval < 50 ms.
- `FramesStepsScreenshotsAndProfiles`: the profile has the scope `frame/render/terrain_stream`; the identity is work
  + wait + stream == frame.
- `TerrainAtlas`'s minimum-capacity assertion uses the 16-byte aligned formula of `memoryEstimate`.
