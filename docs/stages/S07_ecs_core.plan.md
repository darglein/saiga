# S07: ECS core

Status: done.

Sections follow the stage plan template of S05.

## Goal

The engine has its entity-component store (`docs/engine_design.md` §7.1-§7.5): entities with stale-handle detection,
components registered by explicit enum IDs, pooled storage, deterministic views, and `visit` reflection that later
stages use for saves (S11), hashing (S08), JSON (S11), MCP (S14) and replication (S16). `World` (S08) will own a
`Registry` and forward these calls.

## Depends on

S04, S05, S06.

## Rules that apply

- R-MATH-1 (double simulation state; floats only in render components), R-STATE-2 (references by `Entity`, checked
  with `alive`), R-REG-1/2 (registration by enum ID, no string lookups in C++), R-FILE-1, R-DOC-1/2, R-TEST-1,
  R-TEST-2 (the ECS becomes performance relevant: `tests/test_engine_ecs_benchmark.cpp`).
- AGENTS.md: `SAIGA_ENGINE_API` on exported non-template symbols, portability (GCC, Clang, MSVC).

## Scope

`src/saiga/core/math/ChunkId.h` (namespace `Saiga`, header only, usable by `saiga_vulkan`):

- `using SpaceId = uint16_t;` `struct ChunkId { SpaceId space; int32_t x, z; }`, `struct RegionId { SpaceId space;
  uint8_t level; int32_t x, z; }`, `struct InstanceKey { uint32_t generatorSlot, counter; }`: `==`, `!=`, `<`
  (lexicographic), `name`, `version`, `visit`.

`src/saiga/engine/ecs/` (namespace `Saiga::Engine`):

- `Ids.h`: `ComponentId` (`uint8_t`), `EventId`, `ClassId` (`uint16_t`), `kMaxComponents = 256`,
  `kFirstGameComponent = 64`; `enum class Replicate { Never, Host, HostReliable, OwnerAuthoritative }`,
  `enum class Saved { Yes, No }`, `enum class RunOn { Host, Client, Both }` (stored, evaluated from S08/S16).
- `Entity.h`: `struct Entity { uint32_t index = 0, generation = 0; }` (`valid()`, `==`, `!=`, `<`); `{0, 0}` invalid.
- `PersistentId.h`: `struct PersistentId { uint64_t value = 0; }` (comparison operators; the mechanism is S10).
- `ComponentMask.h`: 256 bits (`std::array<uint64_t, 4>`): `set`, `reset`, `test`, `containsAll`, `any`, `==`.
- `Visit.h`: the field-type rules of §7.2 as compile-time checks (`static_assert` with a message naming the field type)
  and the visitor options `EnumNames` (enum value names; the count must equal `E::Count`, checked at compile time) and
  `Quantize{step}` (stored for S16). Supported: `bool`, all fixed-size integers, `double`, enums (only with
  `EnumNames`), `std::string`, `Entity`, `PersistentId`, `Vec2/3/4`, `Quat`, `Mat3/4`, nested structs with `name`,
  `version` and `visit` (incl. `ChunkId`, `RegionId`, `InstanceKey`), `std::array` and `std::vector` of supported types;
  `float`, `vec*`, `quat`, `mat*` only in a `Render*` component that declares `static constexpr bool allowFloat = true`
  (render components). Anything else is a compile error.
- `VisitBinary.h`: `BinaryWriter` / `BinaryReader` visitors (little-endian, the double bit pattern, strings and vectors
  with a u64 length, enums as `int64`); `toBytes(const T&)`, `fromBytes(T&, bytes)` (on truncated input: an assert,
  changed in review round 1 to replaced in review round 1 by `fromBytes` returning false with a message in every build).
  The base for `world.save()` (S11) and `world.hash()` (S08).
- `FieldList.h`: `describeFields<T>()` -> `std::vector<FieldInfo{name, type}>` (type names such as `"double"`,
  `"vec3d"`, `"vector<ItemStack>"`, `"enum"` with the value names): the field list of the schema tables (S11) and
  `inspect` (S14).
- `ComponentPool.h`: `PoolBase` (type-erased: `id`, `has(index)`, `remove(index)`, `size()`, `denseEntities()`) and
  `ComponentPool<T>` (dense `std::vector<T>` and `std::vector<Entity>`, sparse pages of 4096 entries from the entity
  index to the dense index; swap-and-pop removal).
- `Registry.h/.cpp` (`SAIGA_ENGINE_API`):
  - `registerComponent<T>(Replicate = Never, Saved = Yes)`: asserts the ID is unused, `T::id`/`name`/`version`/`visit`
    exist and the fields pass the type check; at 200 registered types a warning is logged.
  - `validateRegistration()`: the engine IDs 0..n and the game IDs 64..m are contiguous (every enum value up to the
    last registered one is registered); `componentInfo(id)` (`name`, `version`, `replicate`, `saved`), `registered(id)`.
  - `create()`, `destroy(e)` (removes all components; the generation increments; the slot goes to the back of a FIFO
    free list), `alive(e)`, `add<T>(e, args...)` (asserts if present), `remove<T>(e)`, `get/tryGet/read/tryRead`,
    `has<T>`, `mask<C...>()` (static), `entityMask(e)`, `aliveCount()`, `flags(e)` / `setFlags` (the `Simulated`
    flag of §12.3 later).
  - Slots: generation, state (`Free`, `Reserved` (used from S08), `Alive`), flags, mask. Slot 0 is never used.
  - `Transform` (ID 0) is built in: every alive entity has one, stored in an array by entity index; `remove<Transform>`
    is rejected at compile time (`static_assert`).
  - Asserts name the entity (`index:generation`) and the component (`name`).
- `View.h`: `View<C...>` (`const C` for read access): `each(fn(Entity, C&...))` iterates the dense entity array of the
  smallest non-`Transform` pool, or the slots for a `Transform`-only view, and tests the masks; skips slots that are
  not `Alive`. Structural changes (`create`, `destroy`, `add`, `remove`) during an `each` assert. `count()`.
- `EntityLog.h/.cpp`: `SAIGA_LOG_E(category, entity, format, ...)`: a log line prefixed with `e<index>:<generation>`.

`src/saiga/engine/components/`:

- `EngineIds.h`: `enum class EngineComponent : uint8_t { Transform = 0, Count }` (grows per stage).
- `Transform.h`: the class of §7.4 (`space()`, getters, setters that compare bitwise and set `changedThisStep`,
  `teleport`, `visit` of space, position, rotation and scale; the flags are not visited). The hierarchy, the
  `changedLastStep` shift and `moveToSpace` are S08/S21.

Tests and docs:

- `tests/test_engine_ecs.cpp`, `tests/test_engine_ecs_benchmark.cpp` (non-gating timings printed), registered in
  `tests/CMakeLists.txt`.
- `docs/engine_design.md`: §3, §7 (implemented signatures, design changes below), the S07 row; `AGENTS.md`.

## Out of scope

- `World`, commands, the structural log, events, phases, `parallelEach`, `Transform` hierarchy (`Parent`,
  `LocalTransform`), flag shifting (S08); `world.hash()` (S08); saves, schema tables, migrations, `dumpJson` (S11);
  `Persistent`, `makePersistent`, `resolve` (S10); replication (S16).
- Reserved entities as an API (`Reserved` exists as a slot state; `reserve`/`activate` come with the commands in S08).

## Design notes

- `T::id` is used directly (no type map). The pools live in `std::array<std::unique_ptr<PoolBase>, 256>`; a typed access
  casts the pool after asserting that it is registered (and, with asserts, that the registered type has the same name: a
  copy-pasted component that kept another ID is caught).
- Deterministic order: pools are dense arrays whose order depends only on the sequence of structural changes; free
  slots are reused FIFO. No hash maps anywhere in the store.
- The field-type check is a visitor that only does `static_assert`s; `registerComponent<T>` instantiates `T::visit`
  with it, so every registered component is checked at compile time, also when no test visits it.
- Float fields: only in a component that declares `allowFloat` and whose name starts with `Render`; both are checked at
  compile time by `registerComponent` (`checkVisitFields`).
- The binary visitors are deliberately simple (no schema); S11 adds the schema and migrations on top.

## Tests

`tests/test_engine_ecs.cpp`:

- Entities: `create` returns increasing indices from 1; `destroy` makes the handle stale (`alive` false, a new entity
  in the slot has a higher generation); FIFO reuse of free slots; `{0, 0}` is never alive; 10^4 create/destroy cycles.
- Components: `add`/`has`/`get`/`read`/`tryGet`/`tryRead`/`remove`; masks; `destroy` removes every component;
  references stay valid until a structural change (documented, not tested beyond the swap-and-pop case); death tests
  (with `SAIGA_ASSERTS`): add twice, get missing, get on a stale handle, unregistered component, ID registered twice,
  structural change during `each` (`remove<Transform>` is a `static_assert`).
- Registration: `validateRegistration` accepts contiguous engine and game IDs and rejects a gap; `componentInfo`.
- Views: one, two and three components, `const` access, `Transform`-only view over the slots, iteration order is the
  dense order of the smallest pool and is identical for two registries with the same history; removed and destroyed
  entities are not visited.
- `visit`: the binary round trip of a component with every supported field type (incl. a nested struct in a vector,
  `std::array`, an enum with names, `Entity`, `ChunkId`, `Quat`, `Mat4`); malformed input is rejected (`fromBytes`
  returns false); `describeFields` names and types; `Transform` round trip and its bitwise setters (`changedThisStep`
  only on a real change).
- `ChunkId`/`RegionId`/`InstanceKey` comparisons; `SAIGA_LOG_E` prefixes the entity.
- Compile-time checks: ctest cases built from `tests/compile_fail/ecs_field_rules.cpp` (a control case that must
  compile and nine cases that must fail with their `static_assert` message, checked by
  `tests/compile_fail/expect_build_failure.cmake`); run them after the normal build with
  `ctest -R '^compile_fail_ecs_field_' --output-on-failure`.

`tests/test_engine_ecs_benchmark.cpp` (non-gating, prints times): create 10^5 entities with 3 components, iterate one,
two and three component views, random get, remove and re-add a component on half of them, destroy all.

## Done when

All tests pass in CI on GCC, Clang and MSVC; the benchmark runs in CI and its line is printed in the step "Golden image
metrics" of the Ubuntu jobs; the design describes the
implemented API.

## Verification

- The CI recipe of AGENTS.md, `ctest -R test_engine`, full ctest.
- AddressSanitizer build of the ECS tests (standalone compile with `-fsanitize=address,undefined`).

## Design changes

- §7.2: float fields need `static constexpr bool allowFloat = true` on a `Render*` component (both checked at compile
  time); enums inside containers are not supported; `Quantize` only on double, `Vec`, `Mat` and `Quat` fields; the
  binary layout of `VisitBinary.h` is written down.
- §7 intro: the S07 file list and API.
- `Replicate` has one enum for components and events (`Cosmetic` is for events only, §13.4).
- `saiga_assert_fail` prints to stderr instead of stdout (assertion diagnostics; gtest death tests match stderr, so
  they can check the messages that name the entity and the component). AGENTS.md updated.
- §3: a row for the ECS core; the S07 row is done.
- Review round 1: `BinaryReader`/`fromBytes` validate in every build and report errors (`fromBytes` returns false; the
  checks were asserts, compiled out in release builds, for untrusted saves and packets); vector lengths are bounded by
  the remaining input divided by the element's minimum size; only fixed-width integers; `std::vector<bool>` rejected;
  `allowFloat` requires the `Render` name prefix (checked at compile time, R-MATH-1 updated); the typed pool access
  compares the registered name; `has<T>` on an unregistered type asserts; `describeFields` gives enum value names;
  compile-fail tests of the field rules (`tests/compile_fail/ecs_field_rules.cpp`, 10 ctest cases); `printCurrentStack`
  writes to stderr like the assert message.
- Review round 2: `has<T>` checks the registered name too; `ComponentPool::emplace` and `Registry::create` allocate
  before the first change (consistent state after `bad_alloc`); the reader assigns integers only after a successful
  read and accepts only 0 and 1 for `bool`; the Windows stack trace goes to stderr as well; a test registers a
  `Render*` component with floats (the accepted side of the rule); §7.2: the gap check is `validateRegistration()`;
  AGENTS.md: build before ctest (compile-fail cases rebuild a stale build).
- Review round 3: `pool<T>()` also checks the element size; component names are unique at registration (saves
  load by name); the aggregate `add` documents exact argument types (narrowing is an error on Clang and MSVC); the
  remove-during-iteration death test; doc fixes (§9.6 file, §17.7 benchmark text, Visit.h comments).

- CodeRabbit (PR #48): components need noexcept moves (`static_assert`), so removal and `destroy` cannot throw
  halfway; `destroy` enqueues the free slot first; `emplace` reserves `data` too; registration allocates the pool
  before publishing it; `Transform` assignment is private (only the registry resets a reused slot, R-STATE-3);
  `std::vector<bool>` with any allocator is rejected; the field check stops at a struct that contains itself through
  a container (a tree); vectors of zero-byte elements round-trip (their count is capped at 2^16); the compile-fail
  cases check the failed build and the message (`expect_build_failure.cmake`); the CI metrics step prints the ECS
  benchmark line; tests assert that positive round trips decode.

- CodeRabbit, second pass: `emplace` no longer reserves `data` ahead (an argument may refer to a component of the same
  pool: `add<Health>(b, read<Health>(a))`; the append is the commit point); structural changes from component
  destructors during `destroy` assert (they could get the slot being freed); the writer refuses more zero-byte elements
  than the reader accepts (`kMaxEmptyElements`); the reader limits struct nesting to 64 (deep trees in untrusted input);
  `Transform::visit` is documented as detached-only.

- CodeRabbit, third pass: the writer's zero-byte element limit is an error in every build (`toBytes(value, out,
  &error)` returns false, `toBytes(value)` stops via `SAIGA_EXIT_ERROR`); the ECS tests also pass with `SAIGA_ASSERTS`
  undefined.

- CodeRabbit, fourth pass: `destroy` clears each mask bit with its component (a later destructor sees `has() == false`);
  the writer enforces the reader's limits (`kMaxStructDepth` = 64, enum values without a name) with an error instead of
  writing an unreadable buffer; `toBytes` encodes into a separate buffer (the output may be a field of the value).

## Notes for S08

From review round 3, for the World built on this registry:

- `Transform&` references from a view point into `std::vector<Transform>`; `create()` grows it. When `create`/`spawn`
  during a step reserve entities (§7.6), reserving must not reallocate `transforms`/`slots` while a view runs: reserve
  ahead outside of views, or store transforms in stable pages like the sparse pages.
- `hash()`, `save()` and `dumpJson()` need a type-erased path to the component data: a per-pool binary write/visit
  (a virtual in `PoolBase` or a function pointer filled by `registerComponent`), plus `Transform`; hash in entity
  index order (dense order depends on history).
- Const iteration (`hash() const`) and an atomic or worker-forbidden `iterationDepth` for nested views inside
  `parallelEach`; a flag predicate in `View` for `Simulated`; access to slot generations, states and the free list
  for saves (§15.1); the structural log hook in `destroy` (decide whether `Transform` is logged).
- `Transform::visit` writes the fields without the change flags: decode into a temporary and apply position, rotation
  and scale through the setters or `teleport`, and a different space through `World::moveToSpace` (the setters do not
  change the space) when loading or replicating.

## Results

- `tests/test_engine_ecs.cpp`: 31 tests pass (GCC 15), including 3 death tests that match the assert messages; the
  binary layout is pinned (`VisitBinary.h`).
- `tests/test_engine_ecs_benchmark.cpp` (GCC 15, RelWithDebInfo, i9 laptop; 10^5 entities with 3 components, ms):
  create 16.5, view<1> 0.47, view<Transform, 1> 0.86, view<3> 0.66, random read 1.56, remove + re-add on half 0.92,
  destroy 13.2.
- AddressSanitizer + UBSan (Clang 19, standalone build of the ECS tests): no reports.
- Eigen backend: `tests/test_engine_ecs.cpp` compiles with `SAIGA_USE_EIGEN` (syntax check against `submodules/eigen`).
- Compile-fail cases: 10/10 (each fails with its own `static_assert` message, the control case builds).
- Full ctest (CI options + samples, GCC 15): 53/53 pass.

## Files

`src/saiga/core/math/ChunkId.h`, `src/saiga/engine/ecs/{Ids.h,Entity.h,PersistentId.h,ComponentMask.h,Visit.h,
VisitBinary.h,FieldList.h,ComponentPool.h,Registry.h,Registry.cpp,View.h,EntityLog.h,EntityLog.cpp}`,
`src/saiga/engine/components/{EngineIds.h,Transform.h}`, `tests/test_engine_ecs.cpp`,
`tests/test_engine_ecs_benchmark.cpp`, `tests/CMakeLists.txt`, `docs/engine_design.md`, `AGENTS.md`, this file.
