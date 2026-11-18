# S02: Golden image tests

Status: done.

Sections follow the stage plan template of S05 (PR #42). Goal to Verification and Files are written
before the implementation, Design changes and Results after it.

## Goal

Render changes are checked against committed reference images of the prototype world, so later stages (the
`RenderScene` rewrite in S17-S19 above all) can show that the image did not change unintentionally.

## Depends on

S01 (the prototype, `WorldShading::Clay`, `WorldSettings::freezeTime`).

## Rules that apply

- R-TEST-1 (a golden image for every render feature that changes the image, from S02).
- R-DOC-1/2 (signatures from §17.6; design changes update the document).
- AGENTS.md: test registration, Vulkan tests on lavapipe, clang-format on changed C++ files.

## Scope

- `tests/golden_image.h` (header only): `bool compareGolden(const TemplatedImage<ucvec4>& image, const std::string&
  name, const std::string& deviceName)`:
  - golden file `data/tests/golden/<name>.png` (found through `SAIGA_PROJECT_SOURCE_DIR`);
  - pass: mean absolute difference over all pixels and RGB channels < 1/255, at most 0.5% of the pixels differ by more
    than 16/255 in a channel;
  - gates (gtest failure) only if `deviceName` contains `llvmpipe`, otherwise prints the metrics;
  - on a mismatch writes `golden_failed/<name>_actual.png` and `_diff.png` into the working directory;
  - `SAIGA_UPDATE_GOLDEN=1` writes the golden instead of comparing, only on lavapipe.
- `tests/test_vulkan_world.cpp`: the render fixture gets a configurable image size; fixture `VulkanWorldGolden`
  (512 x 512) with the tests `WorldOverview` and `WorldForest` (§17.6 setup: `terrain.size = 513`, 5000 trees, seed 1,
  clay, `freezeTime`, 5 frames); both `EXPECT_EQ` the drawn tree instances per LOD and the terrain nodes.
- `data/tests/golden/world_overview.png`, `world_forest.png`, generated on lavapipe.
- `.github/workflows/tests.yml`: upload `build/tests/golden_failed/` with `actions/upload-artifact` (`if: failure()`,
  pinned by commit SHA like the other actions).

## Out of scope

- Golden mode for incremental effects (fixed exposure, full updates): added by the stage of each effect (§17.6).
- Goldens of other features (tone map in S17, lighting in S36, ...): added by their stages.

## Design notes

- The device name is a parameter (not read inside `compareGolden`), so the helper needs no Vulkan include and works for
  any renderer.
- The test working directory of ctest is `build/tests/` (no `WORKING_DIRECTORY` in `saiga_test`), which is the
  artifact path in CI.
- The callbacks of the test app reference the world: each test waits for the device and clears them before the world is
  destroyed.

## Tests

- `VulkanWorldGolden.WorldOverview`, `VulkanWorldGolden.WorldForest` (`tests/test_vulkan_world.cpp`).

## Done when

- The goldens pass in CI (lavapipe).
- Five local lavapipe runs in a row pass (deterministic); a GPU run prints the metrics without gating.
- `docs/engine_design.md` §3, §17.6 and the S02 row, and AGENTS.md (how to update goldens) are updated.

## Verification

- Build with the CI recipe of AGENTS.md, then `cd build && ctest --output-on-failure --parallel 4`.
- Lavapipe: `cd build/tests && VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ./test_vulkan_world
  --gtest_filter='VulkanWorldGolden.*'` (repeat 5 times).
- GPU: the same without `VK_DRIVER_FILES` (prints the metrics).

## Design changes

- `compareGolden` takes the device name as a third parameter (§17.6 had two): the gating needs it, and the helper stays
  independent of Vulkan. §17.6 updated.
- `world_overview` stands 300 m above the ground 100 m from the west edge and looks 45 degrees down along +x, so
  terrain and trees fill the image (§17.6 had the center and 30 degrees: two thirds of that image were sky, and a
  terrain-only regression could stay under the 0.5% limit); `world_forest` stands 2 m above max(ground, water) (§17.6
  gave no heading and no water rule). The forest camera stands at the edge of a clearing 28 m west and 80 m north of the
  center and looks along +x slightly down: a close tree with alpha-to-coverage foliage, grass, trunks, the forest at
  mid distance and the sky (at the center it stood under a crown, which covered 40% of the image). §17.6 updated.
- The tests also check the terrain node count (used by S19 like the tree counts).
- The instance and terrain node counts gate only on lavapipe, like the images: the placement uses std random
  distributions and float comparisons, so other standard libraries, compilers or drivers may differ by a few trees.
- Failure images are written only where the comparison gates (GPUs always differ a little); `golden_failed/` is
  git-ignored.
- Gray goldens are stored as 8-bit grayscale PNGs (`saveGolden` / `loadGolden`), about half the size of RGBA; no PNG
  optimizer is installed.
- The committed goldens come from a local lavapipe (Mesa with LLVM 21.1.8); CI's lavapipe (Mesa 25.2.8, LLVM 20.1.2)
  passes within the tolerance. After a Mesa update that fails CI, the goldens and counts are taken from the failed run
  (exact commands in AGENTS.md). CI prints the golden metrics of every run from ctest's log (step "Golden image
  metrics"), so nothing is rendered twice.
- Failure images are saved like goldens (gray if possible), so they can be copied as they are.
- `loadGolden` also reads gray + alpha and RGB PNGs (e.g. after a PNG optimizer) and reports an unsupported format
  separately from a missing golden.

## Results

- Lavapipe (Mesa with LLVM 21.1.8, 256-bit): the goldens were generated there; with the final goldens 5 runs in a
  row match them exactly (mean absolute difference 0, max 0) with the committed counts.
- NVIDIA RTX 5080 Laptop GPU (not gating, final cameras): `world_overview` mean 0.14/255, 0.02% of the pixels
  > 16/255; `world_forest` mean 0.60/255, 0.59% > 16/255 (alpha-to-coverage foliage differs between the drivers). The
  instance and terrain node counts are equal to lavapipe's.
- Instance counts: `world_overview` trees per LOD 0 / 7 / 3575, 45 terrain nodes; `world_forest` (clearing camera)
  43 / 281 / 677, 38 terrain nodes.
- Golden sizes: `world_overview.png` 145 KB, `world_forest.png` 82 KB (8-bit grayscale).
- CI (Mesa 25.2.8, LLVM 20.1.2), run 43754502053 (the final cameras and goldens): `world_overview` mean 0.0053/255,
  0.011% of the pixels > 16/255, max 32/255; `world_forest` mean 0.14/255, 0.19% > 16/255, max 37/255 (limits 1/255
  and 0.5%); counts 0 / 7 / 3575, 45 nodes and 43 / 281 / 677, 38 nodes, equal to the committed ones. This is the
  reference margin; a Mesa update on the CI runner may need the regeneration from the CI artifact (AGENTS.md).

## Files

`tests/golden_image.h`, `tests/test_vulkan_world.cpp`, `data/tests/golden/world_overview.png`,
`data/tests/golden/world_forest.png`, `.github/workflows/tests.yml`, `docs/engine_design.md`, `AGENTS.md`, this file.
