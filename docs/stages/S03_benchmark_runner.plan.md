# S03: Benchmark runner

Status: done.

Sections follow the stage plan template; the template is added to §22 by S05 (PR #42).

## Goal

Rendering performance of the world prototype is measured reproducibly: `sample_vulkan_world --benchmark <name>` flies a
fixed camera path, collects unsmoothed CPU and GPU frame times, writes `benchmark_<name>.json` and exits; a baseline for
the development GPU is committed in `data/tests/benchmarks.json`, so later stages (S19, S20, S27, S37, ...) can compare
against it (R-TEST-2: a regression of more than 5% must be explained).

## Depends on

S01 (§22).

## Rules that apply

- R-TEST-2 (benchmarks before/after for performance-relevant changes), §17.7 (the benchmark spec), §17.10 (budgets).
- R-DOC-1/2 (signatures from §17.7; design changes update the document).
- AGENTS.md conventions (export macros, formatting of new code, tests registered in `tests/CMakeLists.txt`).

## Scope

- `WorldRenderer` (`src/saiga/vulkan/world/WorldRenderer.h`): `struct FrameTimes` with the unsmoothed CPU frame time and
  the GPU time per pass, each tagged with the frame number it belongs to (GPU times arrive a few frames late), and
  `const FrameTimes& lastFrameTimes() const`. The per-frame callbacks receive the same values.
- `src/saiga/vulkan/world/WorldBenchmark.h/.cpp` (prototype, moved to `engine/app/` in S13):
  - `WorldBenchmark::names()` (`flyover`, `forest`), `warmupFrames = 60`, `measuredFrames = 600`;
  - `static Pose pose(const std::string& name, const HeightField& hf, int measuredFrame)`: the camera paths of §17.7
    (flyover: one revolution in 600 frames on a 3 km circle around (4096, 4096), ground height below the camera + 300 m,
    looking at the center; forest: 2 m above max(terrain, water level) from center - 300 m to center + 300 m along x,
    looking along +x);
  - `static Stat summarize(std::vector<double>)`: median and p95 (nearest rank);
  - a runner that registers a frame callback, poses the camera per frame, collects the samples of the measured frames
    (60 warm-up frames excluded) and builds the result JSON
    `{gpu, quality: "default", frames, cpu_ms: {median, p95}, gpu_ms: {<pass>: {median, p95}}}` (pass keys from
    `WorldRenderer::gpuPassName`).
- `samples/vulkan/sample_vulkan_world.cpp`: `--benchmark <name>`: default `WorldParameters` (8 km, 200k trees),
  1920 x 1080 window, MSAA 4, validation, vsync, ImGui and defragmentation off, `freezeTime`, no MCP; writes the JSON to
  the working directory and exits (exit code 1 for an unknown name).
- `VulkanExampleBase` gets a constructor with explicit window, Vulkan and renderer parameters (the benchmark overrides
  the `config.ini` values).
- `data/tests/benchmarks.json`: GPU name -> benchmark -> result, for the development GPU.

## Out of scope

- Headless benchmarks and `EngineApp` (S13); `step`, `recording`, `extract` CPU times (S13, S19).
- Quality tiers (S37), the 32 km world (S27), `host_step` and the terraforming benchmark (S22, S28, S29, S32).
- A CI job: benchmarks need a GPU and are local only (§17.7).

## Design notes

- The CPU time of frame N is the time from its `transfer()` to the next one, measured when frame N + 1 is prepared. The
  GPU times of frame N are read back when the frame has completed (2-3 frames later). Samples are therefore matched by
  frame number, and the runner waits until the GPU times of the last measured frame arrived (or a few frames more).
- Every completed frame delivers its GPU times: `transfer()` reads all completed timestamp slots into a queue (oldest
  first) and passes one per call. One frame is written per call, so the queue keeps up. The result records the number
  of GPU samples (`gpu_frames`) as a check.
- No GUI during the benchmark: ImGui is disabled (not only hidden), it would add CPU and GPU time that is not part of the
  world and make the result depend on `config.ini`.

## Tests

- `tests/test_vulkan_world.cpp` (headless): `WorldBenchmark.Summarize` (median and p95 of known lists, empty input),
  `WorldBenchmark.Paths` (flyover: radius 3 km around the center, 300 m above the ground below the camera, looking at the
  center, start == end after one revolution; forest: 2 m above max(terrain, water), x from center - 300 to center + 300,
  looking along +x), `WorldBenchmark.ResultJson` (keys and pass names from fake samples; warm-up frames excluded).
- `VulkanWorldRender.FrameTimes` (headless render): `lastFrameTimes()` delivers per-pass GPU times whose sum is the world
  GPU time, with GPU frame numbers lower than the CPU frame number.

## Done when

- `sample_vulkan_world --benchmark flyover` and `--benchmark forest` run on the development GPU and write their JSON.
- `data/tests/benchmarks.json` holds the baseline of both benchmarks for the development GPU (local).
- `docs/engine_design.md` §17.7 and the S03 row are updated; AGENTS.md says how to run the benchmarks.

## Verification

- CI configuration: the CI recipe of AGENTS.md ("Build (Linux)"), then `cd build && ctest --output-on-failure`.
- Local (GPU, display): the benchmark build of AGENTS.md ("Benchmarks": release, asserts and `-march=native` off,
  `build-bench/`), then from `build-bench/samples/vulkan/` run `./sample_vulkan_world --benchmark flyover` and
  `--benchmark forest` (twice each, to judge the noise).

## Design changes

- `FrameTimes` carries the frame numbers of its CPU and GPU values (`cpuFrame`, `gpuFrame`) and the total GPU time
  (`gpuMs`): the CPU and GPU values of one call belong to different frames (§17.7 lists only `cpuMs` and `gpuPassMs`).
  The former `WorldRenderer::FrameSample` (used by `measure_performance`) is replaced by `FrameTimes`. §17.7 updated.
- Timestamp readback: all completed slots are queued and delivered one per frame. Before, only the newest completed
  slot was read, and about half of the frames had no GPU times (first measurement: 304-366 of 600). §17.7 updated.
- The result JSON has more keys: `gpu_frames`, `width`, `height` (the rendered framebuffer), `build` (`type`,
  `asserts`, `archnative`: the baseline must name its build, §17.10 measures release without asserts) and
  `cpu_work_ms` (the CPU work of `transfer()` + `render()`; `cpu_ms` is the frame interval, bounded below by the GPU
  frame time). `gpu_ms` is null without GPU times. `saiga_buildconfig.h` gains `SAIGA_ARCHNATIVE`,
  `SAIGA_ARCHNATIVE_ACTIVE` (set where `Saiga_compiler_flags.cmake` adds `-march=native`) and `SAIGA_BUILD_TYPE` (the
  CMake build type). §17.7 updated.
- Exit code non-zero if the benchmark did not finish or the result was not written; the old result is deleted at the
  start; a bare `--benchmark` is an error; `SAIGA_WORLD_TREES` is ignored in benchmark mode.
- `VulkanExampleBase` has a constructor with explicit window, Vulkan and renderer parameters (the interactive sample
  still reads `config.ini`; the benchmark overrides resolution, vsync, validation, ImGui and defragmentation).
- Baseline GPU: the S03 row asked for "the reference GPU". The baseline is for the development GPU (RTX 5080 Laptop);
  the §17.10 budget reference (RTX 3060 / RX 6600 class) is not available here. The S03 row now says "development GPU",
  and the reference-class baseline moved to S37 (which checks the budgets); §17.7 and the S37 row updated.
- The rendered size is the swap chain extent (`VulkanRenderer::surfaceSize()`, new accessor), recorded for the measured
  frames (a resize during the warm-up does not count); a run whose size changed while measuring exits non-zero. A
  failed write prints its own error.
- `build.archnative` is true only when `-march=native` was really added (decided in CMake), `build.type` is the CMake
  build type.
- `SAIGA_SCREENSHOT` is ignored in benchmark mode (`VulkanForwardRenderer::disableAutoScreenshot()`, new): its
  automatic screenshot closes the window at frame 60, which made the run fail without a hint.

## Results

Baseline GPU: NVIDIA GeForce RTX 5080 Laptop GPU, the development GPU (not the §17.10 budget reference; the baseline on
an RTX 3060 / RX 6600 class card follows in S37), 1920 x 1080 rendered, MSAA 4, default `WorldParameters` (8 km, 200k
trees), all 600 measured frames with CPU and GPU times. Build: the AGENTS.md benchmark build (Release,
`SAIGA_ASSERTS=OFF`, `SAIGA_ARCHNATIVE=OFF`; JSON `build: {type: Release, asserts: false, archnative: false}`). ImGui and
defragmentation off. Two runs each (median / p95 in ms; frame = `cpu_ms`, work = `cpu_work_ms`):

| | frame | work | GPU trees | GPU terrain + water | GPU grass | GPU upload + cull | GPU sky |
|---|---|---|---|---|---|---|---|
| flyover run 1 | 2.371 / 6.595 | 0.034 | 1.493 | 0.094 | 0 | 0.023 | 0.011 |
| flyover run 2 | 2.358 / 6.494 | 0.035 | 1.478 | 0.092 | 0 | 0.023 | 0.011 |
| forest run 1 | 1.988 / 3.978 | 0.034 | 0.475 | 0.050 | 0.627 | 0.021 | 0.006 |
| forest run 2 | 1.974 / 3.905 | 0.033 | 0.476 | 0.050 | 0.627 | 0.021 | 0.006 |

Run-to-run noise: the frame interval median below 1% (0.5-0.7%), its p95 1.5-1.9% (frame pacing), the
GPU pass medians about 1%, passes below 0.1 ms up to 5% (timer resolution). Run 2 of each is the committed baseline in
`data/tests/benchmarks.json`. The world's CPU work is small (0.03 ms); the frame interval is bound by the GPU and
presenting. The flyover (300 m above the ground) draws no grass (its radius is 60 m).

Tests: `WorldBenchmark.Summarize`, `WorldBenchmark.Paths`, `WorldBenchmark.ResultJson` (incl. `cpu_work_ms`, `build`,
null `gpu_ms`), `VulkanWorldRender.FrameTimes` (GPU frames consecutive, no gaps) pass on the GPU and on lavapipe; all
ctest tests pass.

## Files

`src/saiga/vulkan/world/{WorldRenderer.h,WorldRenderer.cpp,WorldBenchmark.h,WorldBenchmark.cpp,WorldMcp.cpp}`,
`src/saiga/vulkan/VulkanForwardRenderer.h` (`disableAutoScreenshot`), `cmake/Saiga_compiler_flags.cmake`,
`src/saiga/vulkan/window/{GLFWSample.h,GLFWSample.cpp}`, `src/saiga/vulkan/Renderer.h`, `samples/vulkan/sample_vulkan_world.cpp`,
`tests/test_vulkan_world.cpp`, `data/tests/benchmarks.json`, `src/saiga/saiga_buildconfig.h.in`, `docs/engine_design.md`,
`AGENTS.md`, this file.
