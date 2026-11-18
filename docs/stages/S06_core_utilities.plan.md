# S06: Core utilities

Status: done.

Sections follow the stage plan template of S05.

## Goal

The engine has the small core services every later stage builds on (`docs/engine_design.md` §5.1 `core/`): a job pool
for parallel loops where the calling thread participates (§8.5, R-PAR-1), a seedable PCG32 random number generator
(§7.10, §8.6), FNV-1a 64 hashing that is identical on every machine and compiler (§7.10), and a logging ring buffer
with `SAIGA_LOG` (§17.1) that the MCP `logs` tool (S14) will read.

## Depends on

S04 (the engine module).

## Rules that apply

- R-PAR-1 (parallelism only through the engine's pools; `JobPool` is that mechanism), R-DET-1 (`Rng`, never `rand()`).
- R-FILE-1 (one class per file), R-DOC-1/2, R-TEST-1 (a unit test per feature).
- AGENTS.md: `SAIGA_ENGINE_API` on exported non-template symbols, portability (GCC, Clang, MSVC), tests registered in
  `tests/CMakeLists.txt` inside `if (MODULE_ENGINE)`.

## Scope

All in `src/saiga/engine/core/`, namespace `Saiga::Engine`:

- `JobPool.h/.cpp` (`SAIGA_ENGINE_API`):
  - `explicit JobPool(int workerThreads = 0)`, `~JobPool()` (joins the workers), not copyable or movable;
  - `void setWorkerThreads(int count)` (stops and restarts the workers; waits for a `parallelFor` of another thread;
    not from inside a chunk, asserted),
    `int workerThreads() const`;
  - `void parallelFor(int64_t begin, int64_t end, int64_t chunk, const std::function<void(int64_t, int64_t)>& fn)`:
    calls `fn(chunkBegin, chunkEnd)` for the fixed chunks `[begin + k * chunk, min(begin + (k + 1) * chunk, end))`
    (chunk > 0, asserted); the calling thread takes chunks too and the call returns when every chunk is done. With 0
    workers, or when called from inside a chunk (nested), the chunks run serially on the calling thread in order.
    Calls from different threads are serialized. An exception thrown by `fn` is rethrown in the caller after all
    started chunks have finished (the remaining chunks are skipped from the moment a thread sees the failure, best
    effort).
- `Rng.h` (header only): PCG32 (`pcg32` of pcg-random.org: XSH RR output, 64-bit LCG state, odd increment from a stream
  id):
  - `explicit Rng(uint64_t seed = 0, uint64_t stream = 0)` (pcg32 seeding: the reference values of the PCG paper's
    demo hold);
  - `uint32_t nextU32()`; `double uniform()` in [0, 1) from 53 bits (two outputs); `double uniform(double a, double b)`
    in [a, b) (a result rounded up to b is replaced by the largest double below b); `int64_t range(int64_t a, int64_t b)` uniform in [a, b] inclusive without modulo bias (asserts a <= b;
    one output for spans up to 2^32, two above);
  - `struct State { uint64_t state, increment; }`, `State state() const`, `void setState(const State&)` (saved by the
    world from S08/S11), `operator==`.
- `Hash.h` (header only): `constexpr uint64_t kFnvOffset`, `kFnvPrime`; `uint64_t fnv1a64Bytes(const void*,
  size_t, uint64_t h = kFnvOffset)` and `constexpr uint64_t fnv1a64(std::string_view, uint64_t h = kFnvOffset)`
  (separate names: with one overload set `fnv1a64(str.c_str(), seed)` would hash `seed` bytes); class `Hasher`
  (`add(const void*, size_t)`, `addU8/U32/U64/I32/I64` as little-endian bytes independent of the host, `addDouble`
  (the IEEE bit pattern, -0.0 and NaN payloads kept), `addString` (length then bytes), `uint64_t value() const`);
  `uint64_t hashCombine(uint64_t seed, uint64_t value)` (FNV-1a over the 16 little-endian bytes, for derived seeds).
  Never `std::hash` (its values differ between standard libraries).
- `LogRing.h/.cpp` (`SAIGA_ENGINE_API`): a thread-safe ring buffer of `Line { uint64_t sequence; std::string category;
  std::string message; }` with `explicit LogRing(size_t capacity = 4096)`, `uint64_t add(category, message)` (returns
  the sequence number, starting at 1), `std::vector<Line> since(uint64_t sequence) const` (the kept lines with a
  larger sequence number, oldest first), `uint64_t lastSequence() const`, `size_t capacity() const`, `void clear()`
  (keeps the sequence counter).
- `Log.h/.cpp` (`SAIGA_ENGINE_API`): `LogRing& engineLog()` (the process-wide ring), `void setLogEcho(bool)` (stdout,
  default on), `void logPrintf(const char* category, const char* format, ...)` (printf-style, format checked by
  GCC/Clang; any length; one stdout line `[category] message`) and the macro `SAIGA_LOG(category, format, ...)`.
- `tests/test_engine_core.cpp` in `tests/CMakeLists.txt` (`if (MODULE_ENGINE)`).
- `docs/engine_design.md`: §3 (exists), §7.10 (Rng/Hash signatures), §8.5 (JobPool signature and semantics), §17.1,
  the S06 row (done).

## Out of scope

- `parallelEach` and its asserts (S07/S08: needs views and the world), `world.random()` (S08), `SAIGA_LOG_E` (S07),
  the MCP `logs` tool (S14), the streaming pool (S20), the Jolt job system adapter (S28).
- Log levels and log files (not in the design; a category is enough until a stage needs more).

## Design notes

- `JobPool` workers sleep on a condition variable between jobs. A job is a stack object of `parallelFor`; workers
  register on it under the pool mutex and `parallelFor` returns only after every registered worker has left it, so no
  worker touches a finished job. Chunks are claimed with an atomic counter (fixed boundaries, dynamic assignment): the
  chunk boundaries never depend on the thread count, the assignment of chunks to threads does.
- Nested calls run serially (a thread-local flag), so a chunk may call code that uses the pool without deadlock.
- `Rng::range` uses Lemire's multiply-shift with rejection for spans up to 2^32 and 64-bit rejection sampling above;
  `uniform()` uses the standard 53-bit construction `((a >> 5) * 2^26 + (b >> 6)) / 2^53`. No `std::*_distribution`:
  their results differ between standard libraries (the cause of the S04 Clang/GCC golden mismatch was a related
  unspecified-order problem).
- `Hasher` hashes integers as little-endian bytes and doubles as their bit pattern, so a hash does not depend on the
  host byte order and (with R-CONTENT-1 integer inputs) not on the compiler.
- The log ring is process-wide (one engine instance per process); `ScenarioWorld` and MCP read it by sequence number.
  stdout echo can be turned off (tests, benchmarks).

## Tests

`tests/test_engine_core.cpp`:

- `JobPool`: every index covered exactly once for several worker counts (0, 1, 3, 8), chunk sizes (1, 7, larger than
  the range) and ranges (empty, negative begin); the caller participates (N workers, N + 1 chunks, every chunk not on
  the caller blocks until a caller chunk ran: without participation it would time out); serial order with 0 workers;
  nested `parallelFor` runs serially; an exception reaches the caller; `setWorkerThreads` between jobs; many
  consecutive small jobs (worker hand-over between jobs); calls from two threads are serialized.
- `Rng`: the pcg32 reference outputs (seed 42, stream 54: `0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293,
  0xbfa4784b, 0xcbed606e`); same seed same sequence, different stream different sequence; `uniform()` in [0, 1) and
  its mean; `range` bounds inclusive (incl. single value, negative, full int64 range) and roughly uniform; state
  save/restore continues the sequence.
- `Hash`: FNV-1a 64 reference values (`""` = `0xcbf29ce484222325`, `"a"` = `0xaf63dc4c8601ec8c`, `"foobar"` =
  `0x85944171f73967e8`); `constexpr` evaluation; `Hasher` integer encoding is little-endian (equal to hashing the byte
  array), `addDouble(-0.0) != addDouble(0.0)`, `addString` distinguishes `"ab"+"c"` from `"a"+"bc"`; `hashCombine` is
  order dependent.
- `Log`: `SAIGA_LOG` formats (also longer than any internal buffer) and lands in the ring with increasing sequence
  numbers; `since` returns only newer lines; the ring keeps the newest `capacity` lines; concurrent logging from
  several threads loses no line and keeps sequence numbers unique.

## Done when

All the tests above pass in CI on GCC, Clang and MSVC; `docs/engine_design.md` describes the implemented API (§7.10,
§8.5, §17.1) and marks S06 done.

## Verification

- The CI recipe of AGENTS.md, `ctest -R test_engine`, full ctest.
- `JobPool` and the log ring under ThreadSanitizer: the test binary built locally with `-fsanitize=thread` (a separate
  build directory with `SAIGA_DEBUG_TSAN=ON`, or the test and the core sources compiled directly).

## Design changes

- §7.10: `Rng` gains `nextU64`, the stream argument and `State`/`state()`/`setState()` (the world saves the state,
  §8.6); the rule "no `std::*_distribution`" is written down; `Hasher` and `hashCombine` are named. Reason: the
  signatures were only sketched.
- §8.5: `JobPool` semantics written down (fixed chunk boundaries, serial with 0 workers and when nested, serialized
  calls, exceptions reach the caller).
- §17.1: the ring's interface (`engineLog()`, `since`, `setLogEcho`).
- §3: a row for the core utilities; the S06 row is done.
- Review round 1: `fnv1a64Bytes` instead of a `fnv1a64(const void*, size_t)` overload (a `const char*` plus a seed
  picked the byte overload); `uniform(a, b)` excludes `b` also when the formula rounds up; exceptions skip the
  remaining chunks only best effort (documented; the test checks the skipping only serially); ring insertion and the
  stdout echo under one lock (stdout in sequence order). §7.10 and §8.5 updated.
- Review round 2: `uniform(a, b)` with `std::fma` (the same rounding with and without FMA contraction, e.g.
  `-march=native`, MSVC, aarch64); §8.5: `parallelFor` cannot host Jolt's job system, S28 adds a submit/barrier API or
  uses Jolt's thread pool; §8.4, §12.1, §12.4 use `hashCombine` instead of a non-existent `fnv1a(a, b)`.

## Results

- `tests/test_engine_core.cpp`: 22 tests pass (GCC 15, RelWithDebInfo); the PCG32 reference values of the PCG demo and
  the FNV-1a reference values hold; the pinned `range`/`uniform`/`Hasher` values are equal with GCC 15 and Clang 19.
- ThreadSanitizer (Clang 19, the engine core sources and the test compiled with `-fsanitize=thread`): no reports in 5
  runs of the JobPool and Log tests and 20 repeats of the JobPool tests.
- Full ctest (CI options + samples, GCC 15): 41/41 pass.

## Files

`src/saiga/engine/core/{JobPool.h,JobPool.cpp,Rng.h,Hash.h,LogRing.h,LogRing.cpp,Log.h,Log.cpp}`,
`tests/test_engine_core.cpp`, `tests/CMakeLists.txt`, `docs/engine_design.md`, `AGENTS.md`, this file.
