# S23c: Overworld generator (tech demo)

Status: done (implemented on `stage-s23c`; deviations in the review log's implementation entry and in Design changes).

Sections follow the stage plan template of S05; "Implementation steps", "Tech demo", "Risks and open questions",
"Design-doc amendments" and "Review log" are added for this stage. Third of the three S23 stages (D13 of review
round 1; the §22 rows are amended in `S23a_generation_pipeline.plan.md`, amendment 1). Scope after review round 2
(E12): the flight demo only, no cave, the `generated_site` golden moved to S23b.

## Goal

The flight demo becomes the first game on an integer generator (§12.1, §18.4, R-CONTENT-1): a game-owned
`OverworldGenerator` with land around the demo's circuit, tier-1 slots (forests and rocks), grass and ruin sites (one
fixed ruin in front of the plane at the demo screenshot, more on a grid), and the game's own committed content hashes
checked by a game scenario through S23a's `checkContentHashes`.

## Depends on

- S23a: the pipeline, `Generator` (in/out `plan`, `hasTerrain`) / `GenerationTemplate`, the toolkit
  (`gen/FixedMath.h`, `TrigTable`, `IntNoise`), prefabs (quarter-turn conventions), stamping (around `Player`
  entities), the hash file format and `checkContentHashes(setup, file, update)` (engine function, E12; phase A only; a
  missing file is an error, new files start as committed skeletons; the caller guards updates with
  `SAIGA_PROJECT_SOURCE_DIR`), `ScenarioWorld::setup()` (F3 of review round 3), the lint (which scans
  `samples/games/*/content/`; a directory that does not exist yet is skipped), the prototype `HookGenerator`
  registered by `samples/games/common/`.
- S23b: content render sets (`centres = View` in the app, `Players` in scenarios), grass from the generator,
  `completeView` waiting for content sets.
- S20: flight's migration (S20's table of game call sites): the 32 km template at base spacing 1 with
  `kPrototypeSeaLevel` (6 m), the circuit centre (4096, 4096) with radius 2458 m (in the corner of the space),
  `FlightContext::ground` = `max(prototypeSurfaceHeight(seed, baseSpacing, x, z), sea level)`, `PrototypeScene`
  (terrain, water, grass, sky, the prototype forest; `attach(RenderScene&, trees)`, its `WorldSettings` with the
  foliage seed it passes to `RenderScene::setPrototypeEnvironment(settings, foliageSeed)`, which replaces
  `setPrototypeTerrain` and creates the foliage texture), `prototypeTreeMesh` / `prototypeTreeLodEnds` /
  `prototypeTreePadding` of `vulkan/world/PrototypeForest`, `FoliageGenerator::treeSpecies(seed)` (spruce, broadleaf,
  birch; meshes 1 unit high, scaled to the tree height by the instance).
- S13b / S16b / S19b: the flight demo (`samples/games/flight/`), `--demo` (720 steps before the first frame,
  `main.cpp`), `--scenarios` with `localOnly` scenarios, `SAIGA_SCREENSHOT` (frame 60), the meshes registered by name
  in `WorldSetup::renderAssets` (`registerDemoMesh`, `demoMaterial`), the driver entities (`spawnDriven`: `Player` on
  the driver, which follows its plane through `driverPose`).

## Rules that apply

R-CONTENT-1 (the generator is integer only; the lint covers it), R-MECH-1 (the generator, its slots, prefabs and
thresholds are game policy), R-TEST-1 (the games are samples: their hash check is a game scenario, run locally),
R-TEST-2 (benchmarks), R-FILE-1, R-DATA-1 (prefab files), R-DOC-1/2, §18.4 (the tech demo shows the stage's feature).

## Scope

### 1. `OverworldGenerator` (`samples/games/flight/content/OverworldGenerator.{h,cpp}`, linted)

- Name `flight_overworld`, version 1. An integer overworld in the spirit of the prototype's look (not bit-equal to
  the float hook). Constants of the generator (content, R-CONTENT-1; in height units): sea level `6 * 1024` (equal to
  S20's `kPrototypeSeaLevel`; the generator cannot include `vulkan/world`, so `FlightWorld.cpp` has
  `static_assert(kFlightSeaLevel == kPrototypeSeaLevel)` and `PrototypeScene`'s water plane stays at that level),
  mountain height up to `520 * 1024` (as `kPrototypeMountainHeight`).
- Template `flight_overworld` (built in `FlightWorld.cpp`, replacing `prototypeSpaceTemplate()` in flight's setup):
  open world, 128 chunks (32 km), base spacing 1, sea level 6 m, `generator` = the `OverworldGenerator`'s ID,
  `fullRadius` 1000 m (the default; see Scope 3; 3000 m after the live demo, L2).
- **Layout after the live demo** (L1 of the review log; replaces the land disc below, which stays here as the planned
  and first implemented layout). Land over the whole 32 km space; noise inputs integer metres, every `fbm` / `ridged`
  satisfies `octaves <= cellShift`:
  - land mask `m` (Q16, land where `m > 0`): 12000 + `fbm(cellShift 13 (8 km), 3 octaves)`; around the circuit centre
    `m = max(m, floor)` with floor 30000 within 3 km, fading linearly to -65536 at 7 km (L4: 2.9 km and 4.5 km, the
    circuit in the centre of the space) (full land under the circuit, the respawn point and the fixed ruin; the floor
    comes before the coast detail, so a coast near the circuit is not round); + 9000 * `fbm(cellShift 10, 3 octaves)`
    (coast detail, 1 km - 256 m); the space edge lowers `m` by up to 80000 over its last 1200 m (the circuit lies 1638 m
    inside; L4: 13.9 km). About 66 % of the space is land (continent, islands, bays, inland seas of up to about 10 km);
  - mountain ranges: the range weight is the zero band `|fbm(cellShift 13, 2 octaves)| < 20000` (smoothed), faded along
    its length by a 4 km gradient noise, inland only (rising over `m` in [0, 24000]) and none within 4 km of the circuit
    centre (full beyond 6 km; L4: 3.3 km, full beyond 4.3 km); per sample, so the ranges are long winding lines;
  - `biomes` plan grid (256 m): the vegetation value `min(fbm(cellShift 11, 2 octaves), 12000)` (meadow below -9000,
    else forest); `biomeAt` raises it towards 65536 by the range weight (mountain from 22000) for slots and grass;
  - `rivers` plan grid (64 m over the space, 512 x 512): the valley weight (Q16): 65536 on a river or lake cell, falling
    linearly to 0 over 7 cells (a chamfer distance transform). Up to 24 rivers start in the foothills (range weight
    0.1-0.27, `m >= 20000`, at least 6 km (L4: 4.3 km) from the circuit centre, 2.5 km apart, in a hash order) and walk
    in 32 m steps down the gradient of the continent's two coarsest octaves (+-35 degrees by a noise) until the sea, the
    space edge or another river; they stop before a range (weight > 0.3) and 4 km (L4: 3.3 km) from the circuit centre
    and end in a lake (320 m, noisy) where the guide stops falling for 40 steps; lowland lakes (160-640 m) on a 4 km
    grid (45 %) where `m >= 16000`, the range weight is below 0.25 and at least 5.5 km (L4: 4.2 km; a lake and its
    valley reach 1.3 km) from the circuit centre;
  - **height, in this order**: `m <= 0`: the sea floor, -4 m at the coast to -100 m at `m = -20000` (smoothed); land:
    12 m + the inland plateau `min(m, 40000) * 70 m / 65536` + hills `fbm(cellShift 10, 6 octaves)` +-60 m + the
    range weight `w` times (120 m + `ridged(cellShift 11, 5 octaves)^2` * 900 m), clamped to [sea level + 1 m, 1200 m],
    blended over the shore band `m < 5000` to the coast floor -4 m, then the valleys towards 3 m below the sea level
    with the weight `valley * (1 - w)` (no gorges through the ranges); the result lies in [-100 m, 1200 m] (asserted;
    measured -100 m .. 1039 m). `kFlightMountainHeight` is 1200 m (no longer `kPrototypeMountainHeight`: the terrain is
    clay shaded, nothing reads the prototype's value in flight; the `static_assert` is removed).
- **Layout as planned and first implemented** (superseded by L1; the circuit is fixed by S20 in the corner of the
  space, so the land is placed where the demo flies).
  Noise inputs are integer metres; every `fbm` / `ridged` call satisfies S23a's `octaves <= cellShift`:
  - land disc: `d` = the distance from the circuit centre (4096, 4096) in metres (`isqrt64`); the coast radius
    `R(x, z) = 5500 m + fbm(cellShift 11 (2048 m), 3 octaves) * 1000 m / 65536`, so `R` lies in [4500, 6500] m; land
    where `d < R`. The circuit (S20: radius 2458 m; the autopilot flies about 2498 m, measured from the step-600
    position), its respawn point (6554, 4096) and the 300 m holding altitude's ground lie at `d <= 2560 m < 4500 m`:
    land everywhere under the circuit;
  - `biomes` plan grid (256 m cells): meadow, forest, mountain from `fbm(cellShift 11, 2 octaves)` (filled first in
    `plan()`, which then places the sites with natural heights, S23a in/out plan); `rivers` plan grid (64 m cells): two
    channels from the mountains to the coast;
  - **height, in this order** (fixed before step 4, because it enters the committed hashes):
    1. land base: 20 m + hills `fbm(cellShift 10 (1024 m), 6 octaves)` scaled to +-60 m + mountains
       `ridged(cellShift 12 (4096 m), 5 octaves)` mapped to 0..400 m and weighted by the `biomes` grid's mountain
       value (Q16);
    2. clamp to [sea level + 1 m, 520 m];
    3. coastal falloff: for `R - 500 m <= d < R` a `lerpQ16` from the clamped land height to the coast floor -40 m;
       for `d >= R` the sea floor from -40 m at the coast to -100 m at 1 km beyond it, -100 m farther out (the rest of
       the 32 km space is ocean; the pyramid shows it; far content is S27's);
    4. river carving: inside a channel of the `rivers` grid and `d < R - 500 m`, `min(h, sea level - 2 m)`;
    5. range check: the result lies in the generator's own range [-100 m, 520 m] (`SAIGA_ASSERT` in `height()`, a
       generator bug otherwise), well inside the engine limit `±kTerrainHeightLimit` (±8192 m, asserted by the
       pipeline for every integer generator).
- **Slots** (mesh tables in Scope 2; `Candidate::scale` is in 1/1024 of the mesh's unit, uint16, at most 64):
  `spruce` and `broadleaf` (`TombstoneOnEdit`), an 8 m jittered grid (1024 candidates per chunk each), accepted by
  biome density (forest high, meadow low, mountain spruce only), slope and at least 1 m above sea level; their meshes
  are 1 unit high (`TreeSpecies`), so the scale is the tree height: spruce 10-20 m (`10240`-`20480`), broadleaf 8-16 m
  (`8192`-`16384`), drawn by `contentHash64`; `rock` (`ReSnapOnEdit`), a 32 m grid (64 per chunk), mesh radius 1 m,
  scale 1-3 m (`1024`-`3072`). `keptCount` halves per level (for S27). About 30 000 instances within 1 km of the
  camera (S18 culls 200 000).
- Grass masks by height and slope (constants of the generator, R-CONTENT-1); no `grassEditRule` (no terraforming in
  flight).
- **Sites** (all inside the space, D12; footprints validated by the pipeline; prefab `ruin`, Scope 2):
  - the **fixed ruin** (key 1) at (6272, 5760), the centre of chunk (24, 22) (I2: (6256, 5856); L4: (18544, 18144) in
    chunk (72, 70), the same offset from the circuit centre), no quarter turn, `baseHeight` = the
    natural height at the origin. `--demo` simulates 720 steps before the first frame and the screenshot is frame 60,
    so the plane is about 780 steps (13 s at 65 m/s, about 850 m of the 2498 m circle) past its start (6554, 4096),
    near (6448, 4937) heading north-west; the ruin lies about 840 m ahead of it, about 7 degrees left of its heading
    (outside the turn), inside the 1 km content radius and below the chase camera's horizon. Step 3 prints the plane's
    position at the screenshot frame and moves the point if needed (before the hash file is committed in step 4; the
    final value is named in the PR). `plan()` checks the fixed ruin's ground with `SAIGA_EXIT_ERROR` (every build):
    natural height at the origin >= sea level + 2 m and slope over the footprint (max - min of the natural heights at
    its centre and four corners) <= 8 m ("fixed ruin site on unsuitable ground: move it"). It is **exempt from the
    circuit rule** below: it lies 2739 m from the circuit centre, about 280 m outside S20's 2458 m line and 240 m
    outside the flown circle, and its plateau (40 m around the origin) is far from the ground under the plane;
  - **grid ruins** (keys 100 + cell index): a 2 km grid over `[0, 10240)^2` (25 cells; L1: over the whole space,
    16 x 16 cells, keys 100 + `gz * 16 + gx`, 67 grid ruins), one candidate per cell
    jittered by `contentHash64` (+-600 m) with a random quarter turn; skipped when the natural height at the origin is
    below sea level + 2 m, the slope over the footprint (as above) exceeds 8 m, the rotated footprint leaves the space
    or overlaps another site (incl. the fixed ruin; closed bounds, S23a), or the origin lies within 300 m of the
    circuit line (the autopilot's ground profile stays the natural one). At most 26 sites in total.
- `FlightWorld.cpp` registers the generator (`WorldSetup::generators`) and the template; the prototype `HookGenerator`
  is no longer used by flight.
- `FlightContext::ground` (S20: `prototypeSurfaceHeight`, the hook without a `World`) becomes `flightGround(const
  World&, x, z) = max(world.terrainHeight(0, x, z), sea level)`: `flightModelOf` and the chase camera already have the
  world (`FlightRender` through `app->world()`); the context keeps the circuit only.

### 2. Prefab and meshes (`data/flight/prefabs/ruin.json`, `FlightWorld.cpp`, `samples/games/common/Mesh.{h,cpp}`)

- `ruin.json` (format 1 of S23a; released prefab files are never edited, §15.3). Every position is an integer number
  of metres (exact in 1/256 m and 1/1024 m) and lies on the full-weight plateau:
  - footprint `[-40, -40, 40, 40]` (80 m);
  - `terrain_patch`: a 73 x 73 plateau at relative height 0 (origin `[-36, -36]`, samples -36..36), weight 256 in the
    inner 57 x 57 samples (-28..28) and an integer linear ramp to 0 over the outer 8 samples;
  - exclusion `[[-36, -36, 36, 36]]` (no trees or rocks on the plateau);
  - `static_geometry` (12 parts, yaw in integer degrees):
    - 8 pillars `flight/ruin_pillar` (cylinder, centred on its axis, half length 6 m, so `y = 6` puts the base on the
      plateau) at `[24, 6, 0]`, `[17, 6, 17]`, `[0, 6, 24]`, `[-17, 6, 17]`, `[-24, 6, 0]`, `[-17, 6, -17]`,
      `[0, 6, -24]`, `[17, 6, -17]` (a ring of about 24 m), yaw 0;
    - 4 walls `flight/ruin_wall` (box 16 x 6 x 2 m, centred, so `y = 3`) inside the ring at `[0, 3, 16]` and
      `[0, 3, -16]` (yaw 0) and `[16, 3, 0]` and `[-16, 3, 0]` (yaw 90); they do not touch the pillars;
  - `entities`: one `FlightChest` at `[0, 0, 0]` (a game class in `FlightWorld.cpp`: `RenderInstance` mesh
    `flight/chest`, adds `Health` in `onSpawn`; the prefab sets its value). Its mesh is built with its base at the
    origin (`MeshBuilder::box(Vec3(0.6, 0.4, 0.4), color, Vec3(0, 0.4, 0))`), so the entity's `y = 0` stands on the
    plateau (0.4 m is not a multiple of 1/1024 m and could not be a prefab y).
  - Readable at the screenshot: the 80 m plateau spans about 5 degrees (about 100 px of 1920) at 840 m; the 12 m
    pillars stand on it.
- Meshes, registered with `World::registerMesh` in flight's `renderAssets` (presentation; the generator's slot mesh
  tables name them):
  - `flight/spruce`, `flight/broadleaf`: `prototypeTreeMesh(species, prototypeTreeLodEnds(WorldSettings()),
    prototypeTreePadding(WorldSettings()))` for species 0 (spruce) and 1 (broadleaf) of
    `FoliageGenerator::treeSpecies(foliageSeed)`, with the material `flight/foliage` (`PipelineClass::PrototypeFoliage`,
    two-sided). `foliageSeed` is one constant of `FlightWorld.h` used by both sides: `renderAssets` builds the species
    with it, and `FlightRender` sets it as the foliage seed of its `PrototypeScene`'s `WorldSettings`, which passes it
    to `setPrototypeEnvironment`, so the meshes' leaf layers match the foliage texture. The two `MeshLods` are built
    once per process and seed (a function-local cache `std::map<uint32_t, std::array<std::shared_ptr<const MeshLods>,
    2>>` under a mutex) and shared by every world: `renderAssets` runs per world, and `--scenarios` builds several
    worlds per scenario. Registering the `PrototypeFoliage` material before the foliage texture exists is legal from
    this stage on (Scope 3b, F4); scenario worlds have a `RecordingRenderSink` (no GPU) and are not affected;
  - `flight/rock`: `MeshBuilder::rock(radius, color)` (new in `samples/games/common/Mesh.{h,cpp}`: a low-poly
    icosahedron with a fixed vertex jitter table, flat colors), radius 1 m;
  - `flight/ruin_pillar` (`MeshBuilder::cylinder(1.5, 6, 12, color)`), `flight/ruin_wall` (`MeshBuilder::box(Vec3(8, 3,
    1), color)`), `flight/chest` (above) with the flat demo material (`registerDemoMesh`).

### 3. Rendering (`samples/games/flight/FlightRender.{h,cpp}`)

- `FlightRender` keeps `PrototypeScene` for terrain, water, grass and sky and drops its prototype forest: it calls
  `PrototypeScene::attach(scene, {})` with an empty tree list (no `placeTrees`; with no trees `PrototypeScene` creates
  no `PrototypeForest` sets: if S20's `attach` builds an empty forest anyway, S23c adds that branch in
  `vulkan/world/PrototypeScene.cpp`, listed in Files). The trees are the generator's tier-1 sets (S23b). Grass masks
  come from the generator (S23b).
- **Content radius**: flight keeps `fullRadius` = 1000 m until S27: at 300 m altitude trees and rocks are drawn up to
  1 km, bare terrain beyond (S27's far content brings them to the horizon). The cost of a larger radius (2 km: about 4x
  the sets and the generation per second of flight) is measured and reported, not adopted. §18.4 is amended.
  L2 (after the live demo): `fullRadius` = 3000 m; `completeView` generates the missing sets on the streaming pool
  (its workers and the caller) instead of one by one on the caller, and `FlightRender` raises the pool's workers to a
  third of the hardware threads (2 to 6).

### 3b. Foliage material check in the frame (F4 of review round 3; `src/saiga/vulkan/scene/RenderScene.cpp`)

- Today `RenderScene::addMaterial` / `setMaterial` assert that a `PipelineClass::PrototypeFoliage` material has the
  foliage texture (`RenderScene.cpp` ~723 / ~730). `EngineApp` sets the world's sink at `EngineApp.cpp:194`
  (`World::setRenderSink` replays the registered materials, `World.cpp` ~1577) before `renderHooks.init` (`:357`),
  where `FlightRender` attaches `PrototypeScene` and calls `setPrototypeEnvironment`: flight's `flight/foliage`
  material would abort at start.
- The check moves to the frame: materials of any class can be added and set at any time; the frame asserts
  (`SAIGA_ASSERT`) when the `PrototypeFoliage` pipeline class has at least one instance to draw (its per-class instance
  count in `SceneCulling`'s offset table, known on the CPU before the cull pass) and the foliage texture does not exist:
  "PrototypeFoliage instances need the foliage texture of setPrototypeEnvironment". Flight's first frame runs after
  `init`, so it passes; `game_world` and racing are unchanged.
- The death test `FoliageClassNeedsTexture` (today `tests/test_vulkan_world.cpp:1218`, in the place S20's port gives
  it) changes: `addMaterial` and `setMaterial` with the class pass without the environment; a frame with one instance
  of a mesh using that material dies with the new message; the same frame after `setPrototypeEnvironment` renders.
- A per-frame integer test; RenderScene changes are performance relevant (R-TEST-2), so `game_world --benchmark
  flyover` and `forest` (which draw `PrototypeFoliage`) run before and after (Benchmarks).

### 4. Game hashes (`data/flight/content_hashes.json`, scenario `flight_content_hashes`)

- `flight_content_hashes` (`ScenarioList::add(..., localOnly = true)`: it generates content, no networking to test)
  calls `checkContentHashes(w.setup(), w.setup().dataDirectory + "/content_hashes.json", update)` (`ScenarioWorld::
  setup()`, F3: the scenario function receives only the `ScenarioWorld`) with `update` from
  `SAIGA_UPDATE_CONTENT_HASHES`, and fails with the report on any mismatch. The data directory is engineMain's (the
  source tree's `data/flight` with `SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON`, else `data/flight` below the working
  directory, or `--data`). The update guard is the scenario's: with `update` set it requires a non-empty
  `SAIGA_PROJECT_SOURCE_DIR` and passes `SAIGA_PROJECT_SOURCE_DIR "/data/flight/content_hashes.json"` instead of the
  data directory's path, else it fails with a message (no copy written outside the source tree).
- The file (S23a format, no `toolkit` block: the engine's file has it) is committed in step 4 first as a skeleton
  (the entry, its chunk IDs and digest block, every value `"0x0"`; S23a: a missing file is an error, the update fills
  an existing one), then filled by the update run: one setup entry (flight's seed as a string,
  `flight_overworld` plan and placement version 1), the plan hash of space 0 (flight has one template) and the
  combined plan hash, and the chunks `0:0:0` (sea), `0:16:16` (circuit centre), `0:25:16` (the start point),
  `0:24:22` (the fixed ruin), `0:127:127` (the last chunk), `0:100:60` (far sea); `digest_block` `[12, 12, 28, 28]`
  (16 x 16 chunks, a 4 km block over the circuit's east half incl. the start and the fixed ruin; L4, the circuit in the
  centre: `0:64:64`, `0:73:64`, `0:72:70` and `[60, 60, 76, 76]`) plus every site chunk
  outside it (at most 26 sites, about 100 chunks). The scenario prints its run time (about 350 chunks at <= 50 ms on a
  `JobPool(3)`: about 6 s).
- Update procedure (AGENTS.md, next to the engine's): `SAIGA_UPDATE_CONTENT_HASHES=1
  ./build/samples/games/game_flight --scenarios flight_content_hashes` rewrites the file and fails; in a dedicated
  commit whose message gives the reason; a released version's entry is never regenerated.
- It runs locally (samples are not run by ctest, R-TEST-1). The cross-compiler guarantee for the game rests on the
  toolkit and the engine hashes (all CI compilers), the lint (all CI jobs scan `samples/games/flight/content/`), and
  the Clang CI job compiling the sample.

### 5. Flight scenarios (`samples/games/flight/FlightWorld.cpp`)

- `autopilot_flies_the_circuit`: its positions at steps 600 (`Vec3(6506.433, 402.839, 4752.384)`) and 1800
  (`Vec3(5858.335, 362.351, 5875.344)`) follow the ground at 300 m, so the y values (and through the altitude hold
  slightly x and z) change with the terrain: re-recorded, old and new values in the PR. The 300 +- 30 m altitude check
  and "did not crash" stay.
- `crashes_into_terrain` and the respawn checks need ground under the circuit and at the respawn point (6554, 4096;
  L4: (18842, 16384)):
  guaranteed by the land disc (Scope 1); their relative expectations (altitude > 250 m after the respawn, flies away
  and back) are unchanged. The co-op and host + client variants are unchanged.
- New `flight_ruin_is_stamped` (all variants): a driven plane (`spawnDriven<Aircraft>(w.world(), w.defaultPeer(),
  0)`; the driver is the `Player` and follows its plane through `driverPose`) is **held** above the fixed ruin: the
  scenario steps in its own loop (at most 300 steps, 5 s) and before every step teleports the host's plane to 200 m
  above (6272, 5760) with level attitude and zero velocity (on the host, as the respawn does), so neither the circuit
  respawn (281 m away) nor the 65 m/s cruise moves it off chunk (24, 22) (L4: (72, 70); the scenario reads the
  site). The driver's square then always intersects
  the ruin chunk, so stamping works with the scenario worlds' 128 m `fullRadius` (S23a) and the replication delays of
  the host + client variants. The loop ends when all checks hold:
  - the host has exactly one `FlightChest`, at the prefab position converted exactly (`(6272, baseHeight / 1024,
    5760)`, `==` on doubles), with the prefab's `Health` value;
  - every client has exactly one replicated chest (no stamping on clients);
  - `terrainHeight(0, 6272, 5760)` equals `baseHeight / 1024` (full patch weight at the centre) on every peer;
  - every peer's recording sink has `instanceSetAt(RegionId{0, 0, 24, 22})` whose last 12 instances are the ruin's
    static parts (slot 255, in `staticList` order) after the tier-1 instances.
  The scenario fails if the loop ends at 300 steps with a check still open (naming it).
- New `flight_slot_meshes_resolve` (`localOnly`): every slot mesh of the generator and every static and
  `RenderInstance` mesh of `ruin.json` resolves through `World::findMesh` (no "unknown mesh" log).
- `flight_content_hashes` (Scope 4).

### 6. Demo (`--demo`)

- `--demo` spawns the `Autopilot` plane only (no driver, no `Player`; `main.cpp`), and S23a stamps only around
  `Player` entities: the demo shows the plateau patch, the ruins' static pillars and walls, the tier-1 forests and
  rocks and the grass, but no chest. The chest appears when flying yourself (the driver is a `Player`) and is checked
  by `flight_ruin_is_stamped`.
- The screenshot (`SAIGA_SCREENSHOT`, frame 60) shows the fixed ruin ahead of the plane.

## Out of scope

- `game_world` and racing stay on the prototype `HookGenerator` (float hook, `placeTrees`): moving them would regenerate
  the goldens and the `flyover` / `forest` baselines and move racing's track without showing anything new. They move
  with S27, together with the removal of the prototype fixture (S23a amendment 13). The platformer is unchanged.
- A cave or other instance spaces in flight (E12, D14: the plane cannot enter them; the engine's
  `site_entrance_links_instance_space` scenario covers instance spaces).
- The `generated_site` golden (S23b, E12).
- Far content to the horizon (S27); colliders (S28); terraforming in flight (none).

## Design notes

- **Flight shows the pipeline** in the demo that already shows terrain (§18.4): the `--demo` circuit passes a ruin,
  so its screenshot shows a site, tier-1 sets and the flattened patch.
- **The land follows the circuit**: S20 fixed the circuit in the corner of the 32 km space; a land disc around it
  guarantees ground under the autopilot and the respawn point without touching the flight code's circuit. L1 keeps
  the guarantee as a floor of the land mask around the circuit and puts land over the whole space; L4 moves the circuit
  to the centre of the space (`kFlightCircuitCentre` 16384, read by `FlightContext`), so the view from every part of
  the turn shows land, and narrows the protected zone, so mountains and lakes rise in view of the circuit.
- **Game hashes are local, engine hashes are CI**: the games are not built on the GCC and MSVC jobs and not run by
  ctest; the engine generator exercises the same toolkit and pipeline on all three compilers. The game reuses the
  engine's `checkContentHashes`, so the file format and the update rule are the same.
- **No cave**: the plane cannot enter it, a walking pilot would be a new game mode (D14), and the engine scenario
  covers site instance spaces.
- **1 km of tier 1** until S27 (L2: 3 km): the demo's look ends in bare terrain beyond 1 km; S27's far content is the planned fix
  (S27 depends on S23c, E13).

## Implementation steps

Each step builds (samples ON, warnings as errors) and passes ctest, and `game_flight --scenarios` passes locally at
the end of every step (round 3: the template switch, the render migration and the re-recorded values are
one step, so `--scenarios` never runs on a half-migrated demo).

1. **Generator and lint**: `OverworldGenerator` (layout, the height order, rivers, biomes, slots with their scales,
   grass, the in/out `plan` with the fixed ruin's checks and the grid ruins), compiled into `game_flight` but not
   registered yet; the lint scans `samples/games/flight/content/` and passes on all CI jobs.
2. **Foliage check, prefab and meshes** (no behaviour change for the demo): the F4 change in `RenderScene.cpp` and the
   changed `FoliageClassNeedsTexture` death test (ctest), `game_world --benchmark flyover` / `forest` before and after;
   `MeshBuilder::rock`; `data/flight/prefabs/ruin.json`; `FlightChest`; the mesh registration in `renderAssets`
   (`flight/spruce`, `flight/broadleaf` with the process cache, `flight/rock`, the ruin meshes; registered but not yet
   drawn).
3. **The switch**: the `flight_overworld` template and the generator registration in `FlightWorld.cpp`, `flightGround`
   (the context without the hook), `FlightRender` without the prototype forest (`attach(scene, {})`, the foliage seed),
   the screenshot point measured and the fixed ruin placed (its ground checks pass), the `autopilot_flies_the_circuit`
   values re-recorded (old and new in the PR), `crashes_into_terrain` checked.
4. **New scenarios and hashes**: `flight_ruin_is_stamped`, `flight_slot_meshes_resolve`, the skeleton
   `data/flight/content_hashes.json` committed and filled by the update run (in its own commit),
   `flight_content_hashes`.
5. **Benchmarks and docs** (`docs/engine_design.md` §18.4 / §3 / §19, `AGENTS.md`: the game hash file, scenario and
   update recipe).

## Tests

- `game_flight --scenarios` (local): `flight_content_hashes`, `flight_ruin_is_stamped`, `flight_slot_meshes_resolve`
  and the existing flight scenarios (with the re-recorded values), in their variants.
- The changed death test `FoliageClassNeedsTexture` (Scope 3b; CI, Vulkan tests on lavapipe): registration of a
  `PrototypeFoliage` material without the environment passes, a frame drawing it dies with the new message, the frame
  after `setPrototypeEnvironment` renders.
- The `ContentLint` test of S23a scans `samples/games/flight/content/` on all CI jobs.
- The Clang CI job builds `game_flight`.

Benchmarks (R-TEST-2; release build of AGENTS.md; reported as information, D9: the demo's frame time depends on
worker scheduling and is not compared):

- chunk generation per section for `OverworldGenerator` at base spacing 1 (report target <= 50 ms per 256 m chunk,
  single-threaded, desktop reference, S23a amendment 11);
- the pyramid's generation time (32 km at 32 m steps) and the first `completeView` of the demo start;
- content set streaming during the demo: sets generated per second, pool time per set, sets pending per frame;
- `game_world --benchmark flyover` and `forest` before and after the F4 change (R-TEST-2: RenderScene; within 5 %).

## Tech demo

`game_flight --demo`: the flight over the generated overworld with forests, rocks, grass and ruins (screenshot of the
ruin pass with `SAIGA_SCREENSHOT`). `game_world` and racing unchanged (screenshots compared).

## Done when

The S23c row of §22 (S23a amendment 1):

- **`flight_content_hashes`** passes and the flight `--demo` screenshot over the fixed ruin is reviewed (local,
  reported in the PR with the benchmark numbers).

Also: `flight_ruin_is_stamped`, `flight_slot_meshes_resolve` and the existing flight scenarios pass (local, re-recorded
values in the PR); `game_world` and racing screenshots unchanged (local); docs updated.

Met (see Results): `flight_content_hashes` reproduces with GCC 15 (RelWithDebInfo with asserts, Release without) and
Clang 19; the `--demo` screenshot shows the generated terrain, forests and the fixed ruin ahead of the plane; all flight
scenarios pass in their variants; the `game_world` and racing screenshots show their prototype worlds as before.

## Verification

- The CI recipe of AGENTS.md with samples ON.
- `./build/samples/games/game_flight --scenarios`; `SAIGA_SCREENSHOT=/tmp/flight.png ./build/samples/games/game_flight
  --demo`; the same for `game_world` and `game_racing` (unchanged).

## Files

- New: `samples/games/flight/content/OverworldGenerator.{h,cpp}`, `data/flight/prefabs/ruin.json`,
  `data/flight/content_hashes.json`.
- Changed: `samples/games/flight/FlightWorld.{h,cpp}` (template, generator registration, `FlightChest`, mesh
  registration in `renderAssets` with the tree mesh cache, the foliage seed constant, `flightGround`, the scenarios
  and re-recorded values), `samples/games/flight/FlightRender.{h,cpp}` (no prototype forest, the foliage seed, the
  chase camera's ground from the world), `src/saiga/vulkan/scene/RenderScene.cpp` (F4: the `PrototypeFoliage` check in
  the frame instead of `addMaterial` / `setMaterial`), the test holding `FoliageClassNeedsTexture` (today
  `tests/test_vulkan_world.cpp`; where S20 ports it), `src/saiga/vulkan/world/PrototypeScene.cpp` (only if S20's
  `attach` builds a forest for an empty tree list),
  `samples/games/flight/main.cpp` (its description: no "8 km world prototype"), `samples/games/common/Mesh.{h,cpp}`
  (`MeshBuilder::rock`), `docs/engine_design.md`, `AGENTS.md`, this file.

## Risks and open questions

1. **The demo's look changes** (integer terrain, tier 1 up to 1 km instead of the prototype forest): reviewed through
   the screenshots; the flight demo has no golden or benchmark baseline.
2. **Generation time at base spacing 1** before S24: the render path is asynchronous (S23b), stamping uses the plan
   (S23a); chunk generation and streaming are reported.
3. **The screenshot point** depends on the autopilot over the new terrain; step 3 measures it before the hash file is
   committed in step 4 (moving the fixed ruin later changes the committed plan).
4. **Trees end at 1 km** until S27 (accepted, amendment 1; L2: at 3 km).
5. **F4 weakens the early check**: a `PrototypeFoliage` material without the texture is now caught at the first frame
   that draws it instead of at registration; the assert names the cause, and the death test covers it.

## Design-doc amendments

1. §18.4 flight row: "a small airplane over a 32 km generated overworld (the integer `OverworldGenerator`: land
   around the circuit, ruin sites, forests and rocks as tier-1 sets within 1 km until S27's far content, CDLOD
   terrain)" replaces "over the 8 km world prototype (CDLOD terrain, 200k GPU culled trees)"; the game's hash file,
   `flight_content_hashes` and its update recipe.
2. §3 / §19: the flight demo is the first game on an integer generator; the float fixture remains for `game_world` and
   racing until S27 (S23a amendments 1 and 13: the `placeTrees` row names S23c for flight and S27 for the others).
3. `AGENTS.md`: the game hash update recipe next to the engine's.

## Review log

Round 1 (Opus, Fable; `review_r1_S23.md`; decisions D1-D14). This plan was split off the S23 plan (D13); the S23a and
S23b plans log the findings they own.

- 0 Split: this is S23c (the game port, the game hashes and the golden, separable from the engine stages).
- 3 (sites at the origin, chunks outside the space): the flight sites lie at interior points and the hash chunks
  inside the space (D12), as the engine's.
- 8 (layering): the golden lives in D2's engine-linked render target, not in `test_vulkan_world`; `game_world` and
  racing stay on the prototype `HookGenerator` (`engine/terrain/`, registered by `samples/games/common/`).
- 18 (old saves): the tech demos keep no save files; a save of the flight demo from before this stage would be refused
  by S23a's `GENR` rule (generator name changed for the template), not re-based.
- Rejected: none.

Round 2 (Opus 6, Fable 5; merged findings in `review_r2_S23.md`; cross-plan decisions E1-E14, which override D1-D14
where they differ). Findings 1-9 of the S23c part:

- 1 Scope: E12 applied: flight only; the cave dropped (D14); one fixed ruin at the screenshot point with coordinates;
  grid ruins skip overlaps, the space edge, water, steep ground and the circuit; no `placeTrees` forest in flight;
  `fullRadius` stays 1 km until S27 (decided, §18.4 amended); meshes registered through `World::registerMesh`;
  autopilot values re-recorded; the hash scenario is `localOnly` and calls the engine's `checkContentHashes`; the
  golden moved to S23b, so S20's render target is no longer a dependency here (S23b holds it as a hard dependency).
- 2 Flight layout: a land disc around S20's circuit centre (4096, 4096) with a coast radius of at least 4500 m, so the
  circuit (r 2458 m), its slots and the respawn point are over land; the fixed ruin at (6272, 5760) about 850 m ahead
  of the plane at the screenshot frame; an 80 m plateau with 12 m pillars, readable at that distance (Scope 1, 2).
- 3 Slot meshes: `prototypeTreeMesh` for `flight/pine` and `flight/broadleaf` with a `PrototypeFoliage` material
  registered on the world, its order against the foliage texture (`PrototypeScene` attached in `init`, drawn from the
  first frame); `MeshBuilder::rock`; all mesh names listed; `flight_slot_meshes_resolve` checks them.
- 4 Prefab entities in `--demo`: the demo claim is scoped to patch, statics, tier 1 and grass; the chest (mesh
  `flight/chest`) is covered by `flight_ruin_is_stamped` with a driver (`Player`) and appears when flying yourself.
- 5 Plan-time heights: S23a's in/out `plan()` lets the generator call `height()`; `baseHeight` is the natural height
  at the site origin; the slope test uses the natural heights at the footprint's centre and corners.
- 6 Hash scenario: `localOnly`; the data directory is engineMain's (source tree with
  `SAIGA_WRITE_PROJECT_DIR_TO_HEADER`, an update without it fails); explicit chunk IDs; the digest block is the 4 km
  block over the circuit's east half plus the site chunks (at most 26 sites), run time printed; no toolkit block.
- 7 Benchmarks: chunk generation against <= 50 ms per 256 m chunk at base spacing 1; pyramid time and content
  streaming reported as information; no frame-time comparison (D9).
- 8 Files and amendments: `FlightRender.*` (no forest), `FlightWorld.*` (template, registration, classes, meshes,
  re-recorded values, `flightGround`, which replaces S20's hook-based `FlightContext::ground`), `main.cpp`,
  `Mesh.{h,cpp}`; AGENTS.md's game hash recipe; the §19 `placeTrees` row and the §18.4 "8 km world prototype" text;
  step 1 split into generator + lint, prefab + template, render migration; the sea level tied to
  `kPrototypeSeaLevel` by a `static_assert` and the mountain height mirrored.
- 9 Existing flight scenarios: `autopilot_flies_the_circuit` re-recorded (values in the PR); `crashes_into_terrain`
  and the respawn rely on the land disc (respawn point over land, stated).
- Rejected: none.

Round 3 (Opus 6, Fable 6; findings 1-9 of the S23c part of `review_r3.md`, plus the cross-plan decisions F1-F4 of
that file):

- 1 Blocker, foliage material at start: F4: the `PrototypeFoliage` check moves from `RenderScene::addMaterial` /
  `setMaterial` to the frame (a class with instances to draw and no foliage texture asserts); `RenderScene.cpp` and the
  changed `FoliageClassNeedsTexture` death test are listed, the `game_world` benchmarks run before and after (Scope 3b).
  S20's `setPrototypeEnvironment(settings, foliageSeed)` is named; one foliage seed constant feeds `treeSpecies` in
  `renderAssets` and `PrototypeScene`'s environment; the slots are `spruce` (species 0) and `broadleaf` (species 1).
- 2 Hash scenario: `ScenarioWorld::setup()` (F3, added by S23a) gives the setup; the file is committed as a skeleton
  and filled by the update run (S23a: a missing file is an error); the scenario guards updates with
  `SAIGA_PROJECT_SOURCE_DIR` (Scope 4).
- 3 Scales: `Candidate::scale` is in 1/1024 of the mesh unit; the tree meshes are 1 unit high, so the scale is the
  height (spruce 10-20 m, broadleaf 8-16 m); rocks have a 1 m mesh and scale 1-3 m (Scope 1).
- 4 `ruin.json`: the 12 static parts with explicit integer positions on the full-weight plateau (pillars on a ring of
  about 24 m, walls inside it at 16 m), y from the centred cylinder (6) and box (3); the chest mesh is built with its
  base at the origin so its entity y is 0 (Scope 2).
- 5 `flight_ruin_is_stamped`: the plane is held 200 m above the ruin by a per-step teleport on the host, so the 128 m
  scenario `fullRadius`, the circuit respawn and the cruise speed cannot move the driver off chunk (24, 22); a bounded
  loop waits for the chest and the set on every peer (Scope 5).
- 6 Height order fixed (land base, clamp to sea + 1 m, coastal falloff, river carving, range check); -100..520 m is
  the generator's own range, the engine limit is ±8192 m (Scope 1).
- 7 No forest: `PrototypeScene::attach(scene, {})`; `PrototypeScene.cpp` listed in case S20's `attach` needs the
  empty-list branch (Scope 3, Files).
- 8 Steps: the template switch, `flightGround`, the render migration and the re-recorded autopilot values are one step
  (3), so `--scenarios` passes after every step; the F4 change and the meshes come before it (step 2).
- 9 The fixed ruin is exempt from the circuit rule (it lies about 280 m outside S20's circuit line) and `plan()` checks
  its ground (height >= sea + 2 m, slope <= 8 m) with `SAIGA_EXIT_ERROR`; noise parameters as cell shifts (11 / 10 /
  12, i.e. 2048 / 1024 / 4096 m) with octaves (3 / 6 / 5) <= cell shift; the screenshot point is about step 780 near
  (6448, 4937) on the flown 2498 m circle with the ruin about 840 m ahead, 7 degrees left; the two tree `MeshLods` are
  cached per process and seed (`renderAssets` runs per world).
- Rejected: none.

Implementation (against the merged S20-S23b code; each the smallest sound fix):

- I1 The real APIs: `PrototypeScene::attach(scene, trees, seed)` (the seed is the foliage seed; flight passes
  `kFlightFoliageSeed` = 1, as before); `checkContentHashes(setup, file, update)` and `ScenarioWorld::setup()` as
  planned; the hash file format has no `toolkit` block for the game; `World::generators().find(id)->slots` for the
  slot meshes; `readPrefab` for `flight_slot_meshes_resolve`; `RecordingRenderSink::instanceSetAt`.
- I2 Fixed ruin: the planned point (6272, 5760) fails the plan's own ground check (footprint slope 8.2 m > 8 m on the
  fixed terrain). Measured at the screenshot (step 3, a temporary print): frame 60 is about step 890 (not 780: the
  first frames step at real time while the start content is built), the plane at (6417, 5043) heading (-0.34, 0.94).
  The ruin moved to (6256, 5856): slope 4.2 m, still in chunk (24, 22) (the hash file's chunk and the scenario's region
  are unchanged), about 830 m ahead of the plane near the image centre below the horizon.
- I3 River carving applies wherever the `rivers` weight is non-zero, not only for `d < R - 500 m`: the condition would
  leave a vertical wall at `R - 500 m` (the coastal falloff starts there from the land height); `min(h, sea - 2 m)` is a
  no-op in the sea, which lies below. The channel cells are 0 / 1; the weight is their bilinear interpolation between
  the 64 m cell centres, doubled and smoothed (`smoothQ16`), so banks are continuous. The rivers start at the two highest
  biome cells within 3.5 km of the circuit centre (at least 2.5 km apart) and walk outward in 32 m steps, turned by up to
  +-40 degrees by a gradient noise (trig table), until 64 m past the coast or the space edge (the land disc is clipped by
  the space edge at x = 0 and z = 0, because S20's circuit lies in the corner; one river ends there).
- I4 The `biomes` grid stores the fbm value (Q16) per 256 m cell; heights and slots use its bilinear interpolation
  between the cell centres (a per-cell weight would make heights discontinuous at cell borders). Mountain weight:
  `smoothQ16` over the biome values [12000, 40000]; the ridged value is squared before the 0..400 m mapping (peaks).
  Biomes: meadow below -9000, mountain from 22000, forest between; the slots dither the biome lookup by +-64 m.
- I5 Tree density: spruce / broadleaf probabilities per biome (meadow 900 / 2600, forest 12000 / 16000, mountain
  12000 / 0 in Q16, the tree line at 380 m) scaled by a 128 m clump noise (glades and groves); the first try (73 % of
  the forest cells) covered the ruin in the screenshot. About 343 instances per land chunk, about 22 000 in the 63 sets
  within 1 km at the start (the plan estimated 30 000).
- I6 Grass masks from the natural heights on a 4 m lattice (65 x 65 heights per chunk, bilinear per cell): one height
  per cell corner would double the chunk time (S23a's note on the test generator). Rules: at least 0.5 m above the sea,
  below 400 m, at most 2.5 m rise over a 4 m lattice edge; density by biome (230 / 150 / 80) +-50 % by a value noise.
- I7 `OverworldGenerator::planGrids(plan)` is public (`plan()` calls it, then places the sites): the layout can be
  inspected without the fixed-ruin check (used while tuning).
- I8 F4: SceneCulling's offset table has no per-class instance counts on the CPU. The frame check
  (`RenderScene::checkFoliageInstances`, only when the foliage texture is missing and a mesh of the class is registered)
  scans the instance sets' distinct mesh lists and the live dynamic instances; zero cost once the texture exists.
- I9 `PrototypeScene::attach(scene, {}, seed)`: S20's attach builds a `PrototypeForest` that registers the prototype's
  tree meshes and material but creates no instance sets for an empty list; `PrototypeScene.cpp` is unchanged.
- I10 The game sources are outside `SAIGA_CONTENT_UBSAN`'s scope (engine content and `test_engine_content`): the
  generator was built and its hashes reproduced with Clang 19 without UBSan; the lint covers it on every CI job.
- I11 Constants mirrored by `static_assert` in `FlightWorld.cpp`: the sea level, the mountain height and the circuit
  centre (the generator cannot include `vulkan/world`).

Implementation review (2026-11-22, rounds 1 and 2, one reviewer over the S20-S23c stack):

- Round 1 (13, minor): `checkFoliageInstances` walked every instance set each frame in every build but only
  `SAIGA_ASSERT` used it, and a release build drew the foliage class with an unwritten descriptor -> fixed (`S23c
  review`): the walk only with `SAIGA_ASSERTS`; in every build `SceneCulling::recordDraw` skips the PrototypeFoliage
  class while there is no foliage texture.
- Round 1 (14, minor): `OverworldGenerator`'s outward vector divided signed values with a truncating `/` -> `floorDiv`;
  the content hashes and the recorded autopilot values were unchanged.
- Round 2 (14, nit): the skip's comment now says that only the draw is skipped: the GPU culling still processes those
  instances and its statistics (`instancesPerLod`, the tracked meshes) count them.
- The S20 fixes below this branch were rebased in without conflicts.

User feedback after the live demo (2026-11-22: "looks smaller than a 32km world"; the land disc of 4.5-6.5 km in one
corner and trees ending at 1 km):

- L1 Layout: land over the whole space (Scope 1, "Layout after the live demo"): a continent mask with coasts, islands
  and inland seas (about 66 % land), mountain ranges along the zero band of an 8 km noise up to about 1040 m
  (`kFlightMountainHeight` 1200 m, the prototype mirror removed), rivers from the foothills to the sea or a lake,
  lowland lakes, grid ruins on a 2 km grid over the whole space (67 instead of 11). Kept: land under the whole
  circuit (a mask floor within 3 km of its centre; 0 of 2160 samples on circles of 2300, 2458 and 2600 m below sea
  level + 1 m), the respawn point (47.5 m) and the fixed ruin at (6256, 5856) (natural height 50.9 m, slope 4.6 m over
  its footprint; in front of the plane at the screenshot), heights within [-100 m, 1200 m], integer only (the lint
  passes: 22 files, 0 findings; three `size_t` loop counters it refused became `uint32_t`). Tree line 380 -> 700 m and
  grass up to 700 m (was 400 m) for the higher mountains. Generator name and version stay (`flight_overworld` v1):
  the stage is not released, the hash file is regenerated.
- L2 Content radius 1000 -> 3000 m (Scope 3). The engine has no cap on `fullRadius` and no content levels yet (S27:
  `keptCount` is declared, `ContentRenderSystem` builds level 0 only), so the radius is the only lever. Measured with
  the new layout (radius sweep below): the frame stays far within budget up to 3 km; the start was the limit:
  `completeView` built the sets one by one on the caller (35 s to the first frame at 3 km). Engine change:
  `ContentRenderSystem::viewUpdate` with `complete` (`completeSets`) requests every missing set on the streaming pool
  without the per-frame budget and runs the pool with the caller until they are installed or the deadline passed (no
  request after the deadline; the deadline-0 test is unchanged); `FlightRender::init` raises the terrain streaming
  pool's workers (`TerrainViewSettings::workers`, game policy) to a third of the hardware threads, 2 to 6.
- L3 Recorded values: `autopilot_flies_the_circuit` re-recorded (the ground under the circuit is about 25 m higher:
  the inland plateau): step 600 (6505.742, 325.853, 4741.635) -> (6505.847, 351.931, 4741.826), step 1800 (5879.619,
  330.511, 5857.057) -> (5878.704, 352.505, 5858.160). `data/flight/content_hashes.json` regenerated (every plan and
  chunk hash except the far-sea chunk 0:127:127; the chunk 0:100:60 is land now; 368 chunks in the digest, was 277:
  more site chunks). `crashes_into_terrain`, `flight_ruin_is_stamped` and the other scenarios pass unchanged.

User feedback after the second live demo (2026-11-22: "put the plane in the center of the terrain. right now half the
turn just shows a tiny terrain with mostly water"; S20's circuit at (4096, 4096) lies 1.6 km from the space edge, so
half of each turn looks over the edge ocean):

- L4 The circuit in the centre of the space: `kFlightCircuitCentre` 4096 -> 16384 (`OverworldGenerator.h`), and
  `FlightContext::create` reads it (centre (16384, 16384), S20's radius 0.3 * 8192 m kept; the `static_assert` tying
  the centre to `kPrototypeArea / 2` removed). Everything else follows the context: the autopilot circuit, the respawn
  point and the slots (18842, 16384) of `AircraftBase::respawn`, the `--demo` start, the co-op client's demo pilot and
  the respawn checks of `crashes_into_terrain`. The fixed ruin keeps its offset from the circuit centre:
  (6256, 5856) -> (18544, 18144) (chunk (72, 70); natural height 39.9 m, footprint slope 6.0 m <= 8 m; in the
  `--demo` screenshot in the same place ahead of the plane). The protected zone narrowed (the generator's other layout
  rules are unchanged, so the terrain beyond about 4.3 km is L1's): the land floor within 2.9 km fading to -65536 at
  4.5 km (was 3 / 7 km), no ranges within 3.3 km, full beyond 4.3 km (was 4 / 6 km; rivers start beyond 4.3 km and stop
  at 3.3 km, their 448 m valleys end 2.85 km from the centre), lowland lake centres at least 4.2 km away (was 5.5 km; a
  lake of up to 864 m with its valley reaches 1.3 km). Layout map (whole space at 32 m): an inland sea about 3.4 km
  from the circuit centre towards -z, mountain ranges 3.5-6 km away towards -x, +x -z and +z (up to about 1000 m; the
  +z range fills the `--demo` screenshot ahead of the plane), a large lake about 4.5 km towards +x, forests and meadows
  under the circuit and towards -x +z; 0 of 2160 samples on circles
  of 2300, 2458 and 2600 m below sea level + 1 m; the respawn point at 57.9 m; 65 % land, heights -100 m .. 1039 m,
  71 sites (was 68), plan 17 ms, land chunk heights 50-58 ms (unchanged).
- Recorded values: `autopilot_flies_the_circuit` step 600 (6505.847, 351.931, 4741.826) -> (18794.478, 369.778,
  17027.615), step 1800 (5878.704, 352.505, 5858.160) -> (18163.513, 351.898, 18147.840) (the same circuit 12 288 m
  further, over different ground). `data/flight/content_hashes.json`: the circuit chunks renamed with the circuit
  (`0:16:16` -> `0:64:64` centre, `0:25:16` -> `0:73:64` start, `0:24:22` -> `0:72:70` fixed ruin, `digest_block`
  `[12, 12, 28, 28]` -> `[60, 60, 76, 76]`), then regenerated by the update run in its own commit: the plan hash, the
  combined plan hash, the three renamed chunks and the digest change; `0:0:0`, `0:100:60` and `0:127:127` are
  unchanged (368 chunks in the digest, 5.4 s). The other scenarios pass unchanged.

## Design changes

The design-doc amendments 1-3 are made in `docs/engine_design.md` (§3 rows and the known deviations, §10.1 (the F4
check), §12.1 (exists: S23c; the first game generator; the game hash file and its update), §18.4 (the flight row and
the S23c paragraph), §19 (`placeTrees` row), §22 (S23c done)) and in `AGENTS.md` (the game hash recipe, the layout).
Changes against the plan (details in the implementation entry of the review log): the fixed ruin at (6256, 5856) (I2),
river carving without the `d < R - 500 m` limit and with smooth banks (I3), bilinear biome values (I4), clumped tree
densities (I5), grass from a 4 m lattice (I6), the public `planGrids` (I7), the F4 check over the sets' mesh lists
(I8). After the live demo: the layout over the whole space (L1, §12.1 and §18.4 updated), `fullRadius` 3000 m and
`completeView` generating the content sets on the streaming pool (L2, §10.1 and §18.4 updated). After the second live
demo: the circuit in the centre of the space with a narrower protected zone (L4, §12.1 and §18.4 updated).

## Results

Measured on the RTX 5080 Laptop GPU, lavapipe (Mesa with LLVM 21.1.8), GCC 15.2 and Clang 19.

- **Done when.** `flight_content_hashes` passes (1 entry: plan hash of space 0 and the combined plan hash, the 6
  listed chunks, the digest over the 16 x 16 block [12, 12, 28, 28] and the site chunks outside it: 277 chunks, 2.6 s on
  the `JobPool(3)` in RelWithDebInfo, 2.4 s in Release, 2.5 s with Clang 19). The file was committed as a skeleton and
  filled by the update run in its own commit. The `--demo` screenshot (frame 60, about step 890) shows the generated
  land, clumped forests and meadows, the bare terrain beyond the 1 km content radius, mountains, and the fixed ruin
  (plateau, pillars, walls) about 830 m ahead of the plane near the image centre.
- **Plan.** 12 sites: the fixed ruin (6256, 5856) (base 27 m, slope 4.2 m) and 11 grid ruins (14 of the 25 cells
  skipped: water, slope, space edge or the circuit rule); two rivers. The land: 61 % of [0, 11 km]^2, of it 28 %
  meadow, 63 % forest, 9 % mountain; heights -100 m .. 369 m; the ground under the flown circle 4-126 m.
- **Scenarios** (GCC and Clang builds): flight 30 (the 18 existing with the re-recorded `autopilot_flies_the_circuit`
  values, `flight_ruin_is_stamped` in all six variants (plain, repeat, save/load, `HostClient`, `HostClientDelay`,
  `HostClientAsync`; done after 1, 1, 1, 2, 8, 5 steps), `flight_slot_meshes_resolve` and `flight_content_hashes`
  (localOnly: plain, repeat, save/load)); racing 18, platformer 36. `autopilot_flies_the_circuit`: step 600 (6506.433,
  402.839, 4752.384) -> (6505.742, 325.853, 4741.635), step 1800 (5858.335, 362.351, 5875.344) -> (5879.619,
  330.511, 5857.057); the 300 +- 30 m altitude check and `crashes_into_terrain` (respawn over land) pass unchanged.
- **Tests.** ctest 80/80 on the GPU (the local build with samples); the Vulkan and engine tests 39/39 on lavapipe
  (goldens unchanged); the changed `VulkanPrototypeScene.FoliageClassNeedsTexture` passes on both (registration passes,
  the frame with a foliage instance dies with the new message, the frame after `setPrototypeEnvironment` draws the
  tree). `ContentLint.Sources` scans 22 files (the two generator files added), 0 findings. The Clang 19 build of the
  games (`build-clang`, samples ON, warnings as errors) has no warnings.
- **Benchmarks** (`build-bench`: Release, no asserts, no `-march=native`; before = step 1, after = the F4 change; two
  runs each):

  | | before | after |
  |---|---|---|
  | flyover `cpu_ms` | 3.735, 3.850 | 3.700, 3.863 |
  | flyover trees / terrain + water / sky / upload + cull (GPU ms) | 1.401, 1.399 / 0.189 / 0.012, 0.013 / 0.099, 0.103 | 1.392, 1.400 / 0.189, 0.191 / 0.012, 0.013 / 0.100, 0.108 |
  | forest `cpu_ms` | 2.181, 2.192 | 2.179, 2.209 |
  | forest trees / terrain + water / grass / upload + cull | 0.555 / 0.114 / 0.679, 0.678 / 0.036, 0.037 | 0.552 / 0.114, 0.115 / 0.679, 0.694 / 0.037, 0.036 |

  Within the run-to-run variation (R-TEST-2; the check runs only without the foliage texture, which `game_world` has).
- **Generation** (Release, single-threaded, `OverworldGenerator` at base spacing 1; information): plan 4.2 ms; a land
  chunk 35.0 ms (heights 31.1, tier 1 0.64, grass 3.3; 343 instances), a sea chunk 3.7 ms (heights), within the 50 ms
  report target; the pyramid's 1025^2 heights at 32 m 104 ms. Demo start (`--demo`, Release): the terrain's first
  `completeView` 750 ms, the first frame after 3.1 s with 63 content sets built; then 2.3 content sets per second at
  65 m/s (one pool job each, about 35 ms), none left pending at the 2 s samples, 0 unknown meshes. The 2 km content
  radius (measured, not adopted): 222 sets at the start (3.5x), the first frame after 8.9 s, 4.2 sets per second.
- Not done locally: the MSVC build (CI); `SAIGA_CONTENT_UBSAN` does not cover game sources (I10).

After the live demo (L1-L3; RTX 5080 Laptop GPU, GCC 15.2 RelWithDebInfo with asserts; the demo headless at
1920 x 1080 through MCP: `measure_performance` over 2 x 300 frames from about step 1100 of `--demo`, the first frame
timed from the start to `app_status` reporting a rendered frame, incl. the 720 steps of `--demo`):

- **Layout** (whole space at 32 m): 66 % land (17 % meadow, 65 % forest, 17 % mountain), heights -100 m .. 1039 m,
  68 sites (the fixed ruin + 67 grid ruins), plan 17.5 ms (was 5 ms), the pyramid's 1025^2 heights 678 ms (was
  104 ms), the heights of a land chunk 50-58 ms (was 32 ms; single-threaded, -O2): above the 50 ms report target,
  information only; the content sets are generated on the pool workers.
- **Radius sweep** (new layout, 2 workers; drawn instances after culling, per-frame values averaged; the GPU ran at a
  low clock in these runs, its totals vary by up to 2x between runs at this load):

  | radius | live sets | drawn instances | GPU total ms | scene.instances ms | cull ms | CPU work ms (p95) | first frame, sets one by one |
  |---|---|---|---|---|---|---|---|
  | 1000 (before: land disc) | 75-78 | 2 300 | 3.40 (0.96 at a high clock) | 0.09-0.19 | 0.03-0.06 | 0.53-0.56 (0.87) | 6.1 s |
  | 1000 | 75-78 | 2 300 | 3.69-3.77 | 0.18-0.25 | 0.06 | 0.49-0.52 (0.88) | 9.4 s |
  | 2000 | 241-249 | 12 000-13 500 | 3.89-4.00 | 0.33-0.37 | 0.09 | 0.54-0.60 (0.94) | 17.9 s |
  | 2500 | 366 | 23 400-23 800 | 3.97-4.18 | 0.51-0.53 | 0.10 | 0.58-0.60 (1.14) | 25.9 s |
  | 3000 | 508-514 | 38 800-39 700 | 4.31-4.32 | 0.79-0.80 | 0.13 | 0.68-0.71 (1.23) | 34.9 s |

  The frame stays far within the 16.7 ms budget at every radius; the start decided: with `completeView` on the pool,
  2500 m starts after 9.8 s with 2 workers, 5.0 s with 6 and 3.8 s with 10; 3000 m after 5.9 s with 6 workers (1000 m:
  2.2 s), about the land disc's 6.1 s at 1 km. Adopted: 3000 m with 2-6 workers (6 here). The streaming keeps up at
  cruise speed (the live set count stays at 505-514 while flying; about 6 new sets per second at 65 m/s, one pool job
  of about 55 ms each).
- **Final** (3000 m, 6 workers): the first frame after 5.9 s (`--demo` and without), 505-511 live sets, about 39 000
  drawn instances, GPU 4.33 ms (a run at a higher clock: 2.80 ms), `scene.instances` 0.58-0.79 ms, cull 0.09-0.14 ms,
  CPU work 0.61-0.67 ms (p95 1.0 ms).
- **Scenarios** (GCC build): flight 30 passed, 0 failed (all variants incl. `HostClient`, `HostClientDelay`,
  `HostClientAsync`); `flight_content_hashes` reproduces with GCC 15 and Clang 19 (368 chunks, 5.5 s).
- **Tests.** ctest 80/80 on lavapipe (goldens unchanged; `ContentRender.*` incl. the deadline-0 case of
  `ViewBuildsSetsAndReadiness` pass with the pool build); `ContentLint.Sources` 22 files, 0 findings; the Clang 19
  build of `game_flight` and `test_engine_content` without warnings. The `game_world` benchmarks were not run (they
  open a window): `game_world`'s prototype space has no content sets, so `completeSets` returns before requesting
  anything and its frames are unchanged.
- **Screenshots**: the `--demo` frame (forests to 3 km, the ranges and lakes on the horizon, the fixed ruin ahead of the
  plane) and MCP views from 2.5 km above (9000, 9000), over the central range, the eastern inland sea and the
  south-western coast.

After the second live demo (L4; RTX 5080 Laptop GPU, GCC 15.2 RelWithDebInfo with asserts):

- **Scenarios**: flight 30 passed, 0 failed (all variants); `flight_content_hashes` reproduces (368 chunks, 5.4 s).
- **Tests.** ctest 80/80 on lavapipe; `ContentLint.Sources` 22 files, 0 findings.
- **Start**: `--demo` with `SAIGA_SCREENSHOT` (720 steps, then 60 frames at 3 km content radius) 6.3 s, as before the
  move (6.5 s); the land chunk heights are unchanged (50-58 ms).
- **Screenshot**: the `--demo` frame shows the forests to 3 km, the +z range ahead of the plane, the lake towards +x on
  the horizon and the fixed ruin ahead of the plane, where it was before the move.
