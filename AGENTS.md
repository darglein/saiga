# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project Overview

**Saiga** is a C++ utility, rendering and computer-vision framework (OpenGL and Vulkan
renderers, a CUDA module, and a vision module for SLAM, reconstruction, registration and
optimization).

- Language: C++17 (`cxx_std_17` on `saiga_core`), CUDA 14 for the CUDA module
- Build system: CMake (>= 3.16), version 1.3.2 (root `CMakeLists.txt`)
- Namespace: `Saiga`
- Libraries: one shared library per module — `saiga_core`, `saiga_opengl`, `saiga_vulkan`,
  `saiga_vision`, `saiga_cuda` (exported as `Saiga::<name>`)
- Tests: googletest (`submodules/googletest`), one executable per test file, run with `ctest`
- License: MIT

## Repository Layout

```
CMakeLists.txt          Root build: all SAIGA_* options, dependency includes, configured headers, subdirs
cmake/
  Dependencies_<module>.cmake   Finds each module's dependencies; sets MODULE_<NAME> if the module can be built
  Module_CMakeLists.txt         Shared module recipe: GLOB_RECURSE *.cpp/*.cu/*.cc + headers into the module lib
  Saiga_compiler_flags.cmake    Warning / optimization / sanitizer flags
  select_compute_arch.cmake     CUDA arch selection (SAIGA_CUDA_ARCH, else autodetect)
  Find*.cmake                   Find modules for the optional dependencies
src/
  saiga/
    core/               saiga_core: math, image (load/store/decode), util (threads, ini, file io), geometry,
                        camera, imgui, window, sound, model, time, sophus, ...
    vision/             saiga_vision: ba, pgo, imu, features, reconstruction, slam, recursive solvers, ceres/g2o glue
    cuda/               saiga_cuda: device helpers, reduce/scan/shfl, memory, streams, imageProcessing, thrust helpers
    opengl/             saiga_opengl: buffers, shaders, textures, rendering, window, imgui, ffmpeg, vr
    vulkan/             saiga_vulkan: instance, swapchain, pipelines, memory, renderer
    export.h            SAIGA_<MODULE>_API dll export/import macros
    *.h.in              Configured into build/include/saiga/ (saiga_buildconfig.h, saiga_modules.h, git sha)
  internal/             stb_image wrappers, precompiled-header lists
shader/                 GLSL shaders (OpenGL / Vulkan)
data/                   Sample assets (textures, models)
samples/<module>/       One sample executable per sample_<module>_*.cpp/.cu
tests/                  test_<module>_*.cpp/.cu + CMakeLists.txt (explicit saiga_test(...) list)
submodules/             Vendored git submodules: eigen, glog, googletest, glfw, assimp, zlib, zstd, libpng,
                        libtiff, libzip, libavif, aom, SVT-AV1, TinyTIFF (built from submodules/CMakeLists.txt)
scripts/setup_worktree.sh   Submodule init for a git worktree (borrows objects from the main checkout)
.github/workflows/tests.yml The CI: Ubuntu 24.04, CPU-only build (OpenGL on Mesa llvmpipe) + ctest
travis/, .travis.yml    Legacy Travis CI — unused
```

## Setup

```shell
git submodule update --init --recursive      # in a git worktree: scripts/setup_worktree.sh
```

System packages (what CI installs on Ubuntu 24.04):

```shell
sudo apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build ccache \
  xorg-dev libwayland-dev libxkbcommon-dev wayland-protocols extra-cmake-modules \
  libgl-dev libegl-dev libegl-mesa0 libgl1-mesa-dri
```

The dependency list in `README.md` is partly outdated (e.g. GLM is no longer used — Eigen is the
math backend); most image/compression libraries come from `submodules/`.

## Build (Linux)

The canonical build is the CI recipe — no GPU, no samples (the OpenGL module is built and tested on
Mesa's software renderer). Use it unless the change needs CUDA or Vulkan:

```shell
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSAIGA_MODULE_CUDA=OFF \
  -DSAIGA_MODULE_OPENGL=ON \
  -DSAIGA_MODULE_VULKAN=OFF \
  -DSAIGA_BUILD_SAMPLES=OFF \
  -DSAIGA_ARCHNATIVE=OFF \
  -DSAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON
cmake --build build --parallel -- -k 0
```

- `SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON` writes the source dir into the generated headers so the
  tests find `shader/` and `data/` from the build dir.
- Without Ninja, drop `-G Ninja` and `-- -k 0` (Makefiles: `cmake --build build --parallel $(nproc)`).
- **CMake >= 4.0** refuses the `cmake_minimum_required(VERSION 3.0)` of `submodules/glog`; add
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the configure. (CI uses CMake 3.28 and does not need it.)
- The CUDA and Vulkan modules need a GPU / CUDA toolkit and are **not covered by CI**. The OpenGL
  module is, but only through its headless tests (no window, renderers or samples). For a CUDA build leave `SAIGA_MODULE_CUDA=ON` and pin the archs with
  `-DSAIGA_CUDA_ARCH="86;89"` (dotless, e.g. `75;80;86;89+PTX`), otherwise the installed GPU is
  autodetected.
- The first build also compiles the submodules (their tests are off) and takes a while; later builds
  are incremental.

### Build options (CMake)

| Option | Default | Effect |
|--------|---------|--------|
| `SAIGA_MODULE_CORE` / `_VISION` / `_CUDA` / `_OPENGL` / `_VULKAN` | ON | Build the module *if its dependencies are found* (silently skipped otherwise) |
| `SAIGA_BUILD_TESTS` | ON | Build the gtest executables + register them with ctest |
| `SAIGA_BUILD_SAMPLES` | ON | Build `samples/` |
| `SAIGA_BUILD_SHARED` | ON | Shared libraries (the export macros in `export.h` matter) |
| `SAIGA_ARCHNATIVE` | ON | `-march=native` (CI turns it OFF) |
| `SAIGA_ASSERTS` | ON | Enable `SAIGA_ASSERT` |
| `SAIGA_DEBUG_ASAN` / `_MSAN` / `_TSAN` | OFF | Sanitizers (TSAN excludes ASAN/MSAN) |
| `SAIGA_CUDA_ARCH` | "" | CUDA archs; empty = autodetect |
| `SAIGA_WITH_<LIB>` | mostly ON | Use an optional library if found (ASSIMP, LIBZIP, FFMPEG, FREETYPE, LIBTIFF, YAMLCPP, OPENAL, OPENMESH, G2O, CHOLMOD, ...) |
| `SAIGA_WITH_TINY_EIGEN` / `_TINY_GLOG` | OFF | Use the tiny replacements instead of Eigen / glog (disables vision + some tests) |

The configure prints `Module enabled: <Name>` and `Test enabled: <name>` lines — check them to see
what is actually built. With the CI options on a plain Ubuntu 24.04, that is **Core + Vision + Opengl** and
39 tests; the Ceres-based vision tests (`test_vision_bundle_adjustment*`, `test_vision_pose_*`,
`test_vision_imu_decoupled_solver`) are only built when Ceres is installed.

## Tests

```shell
cd build && ctest --output-on-failure --parallel 4     # what CI runs
ctest -R test_core_ini --output-on-failure             # one test
./build/tests/test_core_ini --gtest_filter='Ini.*'    # the binary directly (binaries are in build/tests/)
```

- Each `tests/test_*.cpp/.cu` is its own executable, registered in `tests/CMakeLists.txt` with
  `saiga_test(<file> [extra libs])` (links `saiga_core` + gtest; vision tests pass `"saiga_vision"`,
  CUDA tests `"saiga_cuda"`). **Tests are not globbed** — a new test file must be added there,
  inside the matching `MODULE_*` / dependency guard (e.g. the bundle-adjustment tests need
  `CERES_FOUND`).
- Test style: `using namespace Saiga;`, plain `TEST(Suite, Name)`, `EXPECT_*`/`ASSERT_*`; a
  regression test carries a short comment describing the bug it guards against.
- Shared helpers: `tests/compare_numbers.h`, `tests/numeric_derivative.h` (numeric Jacobians for the
  vision derivative tests), `tests/opengl_test_context.h` (the `OpenGLTest` fixture).
- **OpenGL tests** (`test_opengl_*`, built when OpenGL + EGL are found) need no window or X server: the
  `OpenGLTest` fixture creates a headless EGL context (`CreateEGLContext()`, from the first EGL device —
  e.g. the NVIDIA GPU — else Mesa) and **skips** every test if there is no EGL display. Derive the
  suite from `OpenGLTest` and use `TEST_F`; it fails a test that leaves a GL error behind. The core
  profile needs a bound VAO to draw. To run them on Mesa's software renderer as CI does:
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 ctest -R test_opengl`.

## Code Conventions

- Formatting: `.clang-format` (Google base): 4-space indent, **Allman braces**, 120-column limit,
  left-aligned pointers (`int* x`), aligned consecutive assignments, `#  ` indented preprocessor
  directives. Include groups in order: `"saiga/..."`, `"internal/..."`, other `"..."`, `<...>`.
  Run `clang-format -i <file>` on the files you change (only those).
- Every source file starts with the MIT header block:
  ```cpp
  /**
   * Copyright (c) 2021 Darius Rückert
   * Licensed under the MIT License.
   * See LICENSE file for more information.
   */
  ```
- `#pragma once` in headers; everything in `namespace Saiga { ... }  // namespace Saiga`.
- Includes use the full path from `src/`: `#include "saiga/core/util/ini/ini.h"`.
- Exported (non-template, non-inline) symbols need the module's export macro from
  `src/saiga/export.h` — `SAIGA_CORE_API`, `SAIGA_VISION_API`, `SAIGA_CUDA_API`, `SAIGA_OPENGL_API`,
  `SAIGA_VULKAN_API` — or the Windows (DLL) build fails to link.
- Error checks: `SAIGA_ASSERT(cond)` / `SAIGA_ASSERT(cond, "msg")` (`saiga/core/util/assert.h`) —
  the norm across the library (glog `CHECK*` is used only in a few places).
- Keep the code portable: it must build with GCC, Clang and MSVC (Linux and Windows).

## Gotchas

- Module sources are **globbed** (`cmake/Module_CMakeLists.txt`): after adding or removing a
  `.cpp`/`.cu` file, re-run the CMake configure.
- A missing optional dependency silently disables a module, a feature or a test rather than failing
  the configure — a change can "build" while its code was never compiled. Check the
  `Module enabled` / `Test enabled` lines.
- Generated headers live in `build/include/saiga/` (`saiga_buildconfig.h`, `saiga_modules.h`) — do not
  edit them; change the `src/saiga/*.h.in` templates.
- Do not modify anything under `submodules/` (third-party code pinned by commit).
- `build*/` and `cmake-build-*/` are git-ignored; build in-tree in `build/`.

## Verification

```shell
cmake -S . -B build <CI options above> && cmake --build build --parallel   # must compile
cd build && ctest --output-on-failure --parallel 4                         # all tests must pass
```

For changes in the CUDA / Vulkan modules, which CI does not build, also build that module locally (with
a GPU) and say so if it could not be built. CI covers the OpenGL module only through the
`test_opengl_*` tests; for rendering changes also run a sample on a GPU.
