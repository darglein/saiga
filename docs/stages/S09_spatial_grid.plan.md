# S09: Spatial grid

Status: done.

Sections follow the stage plan template of S05.

## Goal

The world has a uniform 32 m grid per space (docs/engine_design.md §9.5), updated in phase 6 for created, destroyed
and changed entities, and `World::queryRadius(space, center, radius, mask)` returns the entities within a radius
sorted by entity index (deterministic). Ticking (S10), gameplay queries and replication interest (S16) build on it.

## Depends on

S08 (World, phase 6, the structural log, the changed-transform list).

## Rules that apply

R-FILE-1 (the grid class and its system in their own files), R-SYS-1 (`spatialGridSystem` is a free function
registered in phase 6), R-DET-1 (results sorted by entity index; the grid's internal order never leaks), R-PAR-1
(queries only read, allowed in `parallelEach`), R-REG-1/2, R-DOC-1/2, R-TEST-1, R-TEST-2 (grid benchmark).

## Scope

`src/saiga/engine/world/` (namespace `Saiga::Engine`):

- `SpatialGrid.h/.cpp` (`SAIGA_ENGINE_API`): square cells of `kCellSize = 32` m in x and z (columns); per space an
  `unordered_map` from the cell key to its items `{Entity, Vec3 position}`; per entity index the cell (a stable pointer
  into the map), its coordinates and the slot in the cell. `update(e, space, position)` (insert, move, or rewrite in
  place when the cell is unchanged; a newer handle of the same index replaces the older one), `remove(index)`,
  `clear()`, `contains`, `entityAt`, `size`, `forEachWithin(space, center, radius, fn)` (cells of the radius' square,
  or all occupied cells of the space if those are fewer). Non-finite coordinates map to cell 0 of that axis, far ones
  are clamped to +-2^30 cells.
- `SpatialGridSystem.h/.cpp`: `spatialGridSystem` (phase 6, after `transformSystem`, `RunOn::Both`): consumes the
  world's grid consumer of the structural log (`Transform` added: the alive entity enters; removed: it leaves, in log
  order, so a reused slot leaves before it enters again), then updates every entity of the changed list.
  `rebuildSpatialGrid(world)`: clears, skips the log, inserts every alive entity (between steps; S11 `load`).
- `World`: owns the `SpatialGrid` and its structural log consumer (registered in the constructor, before any entity);
  `queryRadius(space, center, radius, mask = {})` and the variant into `std::vector<Entity>& out`; engine-internal
  `spatialGrid()`, `spatialGridConsumer()`.

Tests and docs:

- `tests/test_engine_spatial_grid.cpp` (new), `tests/test_engine_ecs_benchmark.cpp` (grid benchmark).
- `docs/engine_design.md` §3, §7.10 (`queryRadius`), §8.2 (the system), §9.5, the S09 row; `AGENTS.md`.

## Out of scope

- World-state and space-state entities and their exclusion from the grid (S10), ticking from the grid (S10), interest
  (S16), `moveToSpace` (S21; the grid already follows `Transform::space()` on the next update), chunk membership (S24).

## Design notes

- Positions as of the last phase 6: queries read the position stored in the grid, never the live `Transform`. A query
  from a `parallelEach` body therefore never reads a `Transform` another body writes (no data race), and every query of
  a phase sees the same snapshot. The cost: an entity created or moved after phase 6 is found by its old state until
  the next phase 6 (documented in §9.5).
- Destroyed entities are filtered at query time (`alive`, generation), so a destroy never has to touch the grid before
  phase 6; the log removes the item there.
- Created entities come from the structural log, not from the changed list: a new entity at the origin with an
  unchanged `Transform` is not in the changed list.
- 2D columns, 3D distance: worlds are wide, not tall; a column holds the few entities above each other.
- The cell pointer in each entry: an unchanged cell (the common case) costs no hash lookup; `unordered_map` elements
  keep their address until erased (cells are erased only when empty), and the maps are heap-allocated per space so
  growing the space vector cannot move them.

## Tests

`tests/test_engine_spatial_grid.cpp`:

- `SpatialGrid`: within the radius across cell borders, negative cells, spaces, the y axis; move, remove (also twice),
  a newer handle of the same index; huge, infinite, negative and NaN radii, NaN and far positions.
- `queryRadius`: sorted by index with and without a mask, the boundary included, the out variant; positions of the last
  phase 6 (created and moved between steps), destroyed entities never returned (also with a reused slot); created
  during a step (found in phase 8 of the same step and in the next step); children moved by their parent; destroying a
  parent removes the children; two worlds with the same history return the same results; brute-force comparison;
  `rebuildSpatialGrid`; idle entities leave the changed list empty; queries from `parallelEach` with 1 and 4 threads
  give the same results; an alive entity missing from the grid enters it when it moves; an aborted step (phase 7 throws
  after a phase-4 create) keeps the grid consistent.

Benchmark (non-gating): `World::step` with 10^5 entities spread over 2 km x 2 km moving at 20 m/s, and 10^4
`queryRadius` calls of 50 m.

## Done when

All tests pass in CI on GCC, Clang and MSVC; the benchmark prints its numbers; the design describes the implemented
API.

## Verification

- The CI recipe of AGENTS.md, `ctest -R engine`, full ctest.
- AddressSanitizer + UBSan and ThreadSanitizer builds of the grid tests (standalone compile, as in S06-S08).

## Design changes

- §7.10: `queryRadius` takes `mask = {}` and has a variant into an output vector (no allocation).
- §9.5: the S09 implementation paragraph (columns, positions of the last phase 6, stale handles filtered, large radii,
  NaN, `rebuildSpatialGrid`).
- §8.2: `spatial_grid` registered after `transform`. §3: a row for the grid; the S09 row is done.
- Review round 1: `SpatialGrid` is not copyable (MSVC instantiates the copy operations of an exported class, its
  `unique_ptr` members do not copy); the changed list takes the handle from the registry (`Registry::aliveAt(index)`),
  so an alive entity missing from the grid enters it when it moves; `rebuildSpatialGrid` skips the log after the
  inserts; a radius whose square overflows compares the scaled distance (an entity at 1e300 was found within 1e160),
  non-finite positions are never found; tests for both and for an aborted step.

## Results

- `tests/test_engine_spatial_grid.cpp`: 15 tests pass (GCC 15); the S08 world and ECS tests are unchanged.
- Benchmark (GCC 15, RelWithDebInfo, laptop): 10^5 entities moving at 20 m/s over 2 km x 2 km: `World::step` 2.1-2.7 ms;
  10^4 `queryRadius(50 m)` 70 ms (7 us per query, about 190 hits each). The S08 step benchmark with 10^5 moving
  entities rises from 0.97 ms to 1.9 ms (R-TEST-2: the grid rewrites the stored position of every changed entity,
  about 9 ns each, mostly cache misses on the entry and the item; idle entities: 0.51 ms, unchanged).
- AddressSanitizer + UBSan and ThreadSanitizer (`parallelEach` queries, 3 repeats): no reports.
- Full ctest (CI options, GCC 15): 55/55 pass.

## Files

`src/saiga/engine/world/{SpatialGrid.h,SpatialGrid.cpp,SpatialGridSystem.h,SpatialGridSystem.cpp,World.h,World.cpp,
TransformSystem.cpp}`, `tests/test_engine_spatial_grid.cpp`, `tests/test_engine_ecs_benchmark.cpp`,
`tests/CMakeLists.txt`, `docs/engine_design.md`, `AGENTS.md`, this file.
