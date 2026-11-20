# S18: Instances and culling

Status: done.

Sections follow the stage plan template of S05.

## Goal

`RenderScene` draws scene content of its own: meshes and materials registered by ID and name, static instance sets per
region and dynamic instances with GPU interpolation, in one instance buffer, positioned through a per-frame offset table
relative to a double camera, culled on the GPU (count pass with stored slots, prefix sum, compacted write pass) and
drawn with indexed indirect draws per (view, pipeline class, mesh, LOD) bucket. Static meshes are imported from glTF
(assimp) into `MeshLods` with the flat-color model (faces duplicated, palette indices). The culling and LOD rule
reproduce the prototype forest's counts, so S19 can move the trees onto this path.

## Depends on

S17 (the HDR frame, its formats and output contract, SPIR-V 1.6 shaders), S13 (`RenderScene`, `EngineApp`), S07
(`ChunkId`, `RegionId`, `InstanceKey`), the prototype's `ForestRenderer` (`tree_cull.comp`: the culling rule this stage
generalizes; a reference for the counts).

## Rules that apply

R-MATH-1 (double positions on the CPU; float only relative to a region or chunk origin and in the offset table),
R-TEST-1 (headless Vulkan tests; §17.5 counts against a CPU reference), R-TEST-2 (`RenderScene` and culling are
performance relevant: culling timings here, the forest comparison in S19), R-MECH-1 (LOD distances, bounds padding and
materials are game data), R-PAR-1, §5.2 (the scene types live in `saiga_vulkan`, no engine headers), AGENTS.md (Vulkan
1.3 core only: `drawIndirectFirstInstance` required and asserted, `multiDrawIndirect` optional; VMA not in headers).

## Scope

### 1. Types (`vulkan/scene/RenderSceneTypes.h`, `core/math/ChunkId.h`)

```cpp
using MeshId = uint16_t; using MaterialId = uint16_t;   // 0 = none
struct MeshVertex { vec3 position; uint32_t normal; /* oct snorm16x2 */ uint32_t color; /* sRGB8 RGB + linear alpha,
                    or a palette index */ vec2 uv; };   // 28 bytes; color is an R32_UINT attribute, decoded in the shader
struct MeshLod { std::vector<MeshVertex> vertices; std::vector<uint32_t> indices; float endDistance; };  // LOD i is used
                                                         // up to endDistance; beyond the last LOD the instance is culled
struct MeshLods { std::vector<MeshLod> lods; /* 1..4, non-decreasing endDistance (equal: an empty LOD), asserted */
                  vec3 sphereCenter;
                  float sphereRadius;
                  float boundsPadding = 0; /* added to the radius: wind sway etc. (R-MECH-1) */ };
enum class ShadingModel : uint8_t { FlatColor, SmoothTextured };
struct MaterialData   // §10.8 data model; textures, normal maps and wind in S43
{
    ShadingModel model = ShadingModel::FlatColor;
    bool usePalette = false;            // the vertex color word is a palette index (else RGBA8, already containing any
                                        // base color factor of an import)
    vec4 baseColor = vec4(1, 1, 1, 1);  // multiplied with the vertex color (1 for imported meshes)
    float roughness = 0.8f, metalness = 0; vec3 emissive = vec3(0, 0, 0);
    bool alphaMask = false, twoSided = false;
    uint8_t pipelineClass = 0;          // 0: derived from alphaMask / twoSided; S19 adds PrototypeFoliage (the class
                                        // also selects how the vertex color word is interpreted)
};
struct StaticInstance   // 32 bytes, region-relative (static_assert on size and offsets)
{ float position[3]; uint32_t rot01, rot23; /* quaternion as snorm16 pairs */ float scale; uint16_t mesh, flags;
  uint32_t key; /* InstanceKey: generator slot in the upper 8 bits, counter in the lower 24 */ };  // flags: 0, reserved
// GLSL (std430): float px, py, pz; uint rot01, rot23; float scale; uint meshFlags; uint key; (no vec3 alignment,
// no 16-bit storage feature; unpackSnorm2x16 + normalize)
struct DynamicPose { Vec3 position = Vec3::Zero(); Quat rotation = Quat::Identity(); double scale = 1; };  // chunk-relative
struct InstanceSetHandle { uint32_t id = 0; };  struct InstanceHandle { uint32_t id = 0; };
struct RenderStatistics { uint64_t frame = 0; std::array<uint32_t, 4> instancesPerLod{}; uint32_t terrainNodes = 0,
                          drawCommands = 0, nonEmptyBuckets = 0; };
// ChunkId.h: the origins in double, consistent with §12's RegionId (x, z in region units of 2^level chunks):
// chunk: (x, 0, z) * 256 m; region of level L: (x, 0, z) * 256 m * 2^L
Vec3 chunkOrigin(const ChunkId&);  Vec3 regionOrigin(const RegionId&);
```

A palette is one per game: `RenderScene::setPalette(std::vector<PaletteEntry>)` with `struct PaletteEntry { ucvec4 color;
/* sRGB8 + alpha */ uint8_t roughness, metalness; }`, at most 4096 entries, a storage buffer read by index. Materials are
a std430 buffer indexed by `MaterialId` (model, `usePalette`, base color, roughness, metalness, emissive, flags);
the mesh record holds its material. The shader chain: `instance.vert` reads `compacted[gl_InstanceIndex]` → the
instance → the mesh record → the material, and passes `flat uint` material and color word to `flat.frag`. The alpha
test of an `alphaMask` material: vertex alpha · `baseColor.a` < 0.5. Palette and materials may change after the first
frame (rewritten through the per-frame descriptor sets, like growth).

### 2. Scene content (`vulkan/scene/RenderScene.{h,cpp}`, `MeshPool.*`, `InstanceBuffer.*`, `SceneCulling.*`)

The §10.1 API for this stage: `addMaterial(MaterialId, name, data)`, `addMesh(MeshId, name, lods, material)` (explicit
IDs, asserted free and names unique: the `World`'s name tables assign them in S19's `RenderSink`; S30's generated
meshes take IDs from the top of the range), `addInstanceSet`, `updateInstances`, `removeInstance`,
`removeInstanceSet`, `addDynamic`, `setDynamic`, `removeDynamic`, `setCamera`, `setInterpolation`, `statistics`,
`gpuSectionNames` (the existing section list, with `cull` added; no separate pass-name API; S19 follows),
`setMaterial(MaterialId, data)` (a change of `alphaMask`, `twoSided` or `pipelineClass` rebuilds the bucket layout).
`RenderSceneParameters` gains `nearPlane` and the initial capacities (vertices, indices, static instances, dynamics, in elements) and
`allowMultiDrawIndirect` (true; tests turn the path off without a new device).

- **Mesh pool**: one vertex and one index buffer, each sub-allocated with its own VMA virtual block in element units
  (vertices, indices: VMA alignments are powers of two, a 28-byte vertex is not; `VmaVirtualBlock` forward declared,
  not in a header); a GPU table `MeshId → record` sized to the highest registered ID (a zeroed record, `lodCount` 0,
  is rejected by the count pass: mesh 0 and unregistered IDs); per record: material, per LOD first index, index count,
  vertex offset, end distance, the sphere and padding, and `bucketBase`.
- **Buckets** are dense: one per (view, existing LOD of a registered mesh), sorted by (pipeline class, mesh
  registration order, LOD), with a `[first, count)` range per class (a mesh has one material, so one class); `addMesh`
  rebuilds the layout, the commands' static fields (`indexCount`, `firstIndex`, `vertexOffset`) and every
  `bucketBase`. The counts are a separate buffer (cleared by `vkCmdFillBuffer`; the commands keep their static fields),
  copied for the statistics readback.
- **Instance buffer**: the same scheme for `StaticInstance`s; a set is a contiguous range of `capacity` instances with
  its region and bounds (an AABB around its instances' bounding spheres, radius `(sphereRadius + boundsPadding) *
  scale`, from the decoded quaternions, computed on the CPU at `addInstanceSet` and `updateInstances`; the meshes must be
  registered then, asserted; the GPU's zeroed-record reject is a guard only) and a stable offset-table slot, assigned at `addInstanceSet` and freed at
  `removeInstanceSet`. `removeInstance` writes scale 0 (a tombstone). Dynamic instances live in a second buffer with prev
  and curr pose (float, chunk-relative), their chunk's offset-table slot, mesh and `boneOffset` (unused before S34).
- **Growth**: a virtual block has a fixed size and offsets cannot be chosen, so growing creates a new 2x buffer and
  block, re-allocates every live range in it, copies the ranges whose data is already on the GPU with one
  `vkCmdCopyBuffer` (a region per range), updates the CPU records (set first index; mesh first index and vertex
  offset), the GPU mesh records and the commands' static fields, and releases the old buffer and block through
  `deferDelete`. Pending uploads are queued by set or mesh (not by buffer offset) and recorded after the growth copy
  against the new offsets. Sizes grow to `max(2x, required)`. Per-frame host-visible buffers (offset table, work chunks,
  dynamics, readback) recreate that frame slot's buffer when too small; shared buffers (mesh record table, material
  buffer, palette) get a new buffer plus `deferDelete`; mesh records, `bucketBase` and the commands' static fields are
  written only by in-order transfer commands, never through a host mapping, since frames in flight read them. The
  scratch (>= 256 · Σ⌈countᵢ/256⌉ entries), compacted (>= static + dynamic capacity), count
  and command buffers are recreated at frame start when too small (`deferDelete`, no copy). Descriptors reach the buffers through per-frame descriptor sets (rewritten for the
  frame being recorded), never a set used by a frame in flight.
- **Per-frame resources** (one per frame in flight, host-visible): the offset table (rewritten each frame: per slot a
  `vec4`, xyz = `origin - cameraPosition` computed in double, w = 1 valid; slots of other spaces and free slots are
  written with w = 0, which the count pass rejects), the work chunk list, the dynamics (the CPU array is the source of
  truth; every live dynamic is rewritten into the frame slot's buffer each frame: mesh, chunk slot, prev and curr
  position as float, rotations as float quaternions, scale, `boneOffset`; free or unset entries carry mesh 0), the
  statistics readback. Chunk slots of dynamics: a reference-counted `ChunkId → slot` map, updated when `setDynamic`
  changes the chunk. Uploads into the instance buffer and the mesh pool go through staging in
  the frame's transfer callback, after a barrier from the previous reads (vertex and compute) to the transfer write, so
  a range freed and reused in one frame is safe.
- **Camera**: `setCamera(space, position, rotation, fovy)` in double, the aspect from `size()`, `nearPlane` from the
  parameters (reversed-Z, infinite far); `RenderScene::viewProjection()` exposes the camera-relative matrices for the
  game's render callback (one camera for the shared depth buffer); nothing is culled or drawn before the first
  `setCamera`. `setInterpolation(alpha)`. Each offset slot is two `vec4`s (`hi.xyz` with the valid flag in w, `lo.xyz`;
  `origin - cameraPosition` in double split into float hi and lo); a vertex is placed at
  `((instance.position + hi) + lo) + rotate(vertex * scale)`, the count pass places the sphere centre with the same
  function, declared `precise` (no reassociation); dynamics use their chunk slot the same way. The view matrix holds
  only the camera rotation.
- **Culling** (§10.2; one view; buffers carry a view dimension), with the prototype's rule generalized:
  1. CPU: set bounds against the frustum → visible sets; their ranges are split into work chunks of at most 256
     instances `{first, count, slot}` over each set's used size (the highest written index; the tail beyond it is never
     visited); the dynamics are chunked with `slot = 0xFFFFFFFF` (each instance carries its own chunk slot); a direct
     dispatch, one workgroup per chunk (asserted `<= maxComputeWorkGroupCount[0]`, at least 65535);
  2. count pass (invocation i of chunk c handles `chunk.first + i` if `i < chunk.count`; its scratch entry is
     `c · 256 + i`, set to "not visible" when rejected): reject tombstones (`scale == 0`), zeroed mesh records, invalid
     offset entries (w = 0) and instances beyond the mesh's last `endDistance`; distance = |camera − the transformed
     sphere centre|, `endDistance` not scaled by the instance scale (the prototype's `tree_cull.comp`); invocations
     with `i >= chunk.count` write "not visible" too (the scratch buffer is reused across frames); frustum test of
     the bounding sphere (mesh sphere rotated and scaled, radius + `boundsPadding * scale`) against 5 planes (reversed-Z
     with an infinite far plane: no far plane), LOD i = the first with distance <= `endDistance[i]`; dynamic instances
     use the interpolated pose with the frame's `alpha` (the vertex shader's: the same GLSL function, negating curr
     when `dot < 0`, nlerp near `dot = 1`; the CPU reference uses the same formula in float: `interpolatePose` in
     `RenderSceneTypes.h`, templated on the scalar, which S19b's extract uses in double); `atomicAdd`
     into the count of the bucket (view, pipeline class, mesh, LOD) and store the returned slot and the bucket per
     visible instance in a scratch buffer;
  3. scan: bucket offsets (one workgroup looping over the dense bucket array), writing `instanceCount` and
     `firstInstance` of each bucket's indirect command;
  4. write pass: every invocation reads its scratch entry (no second visibility decision) and, if visible, writes
     `uvec2(instance index with a high bit for dynamics, offset slot in the low 24 bits | LOD << 24)` at bucket offset +
     slot (the LOD for S19's foliage tint);
  5. draw: buckets ordered (view, pipeline class, mesh, LOD), so each pipeline class's buckets are contiguous; one
     `vkCmdDrawIndexedIndirect` per pipeline class and view with `drawCount` = its buckets, or one per bucket without
     `multiDrawIndirect` (empty ones too). Pipeline classes: opaque, alpha-masked, two-sided, alpha-masked two-sided.
  The counts are cleared with `vkCmdFillBuffer` each frame; barriers (sync2): fill → count → scan → write → draw
  (`DRAW_INDIRECT` / `INDIRECT_COMMAND_READ`, vertex and storage reads).
  Statistics: the scan writes `instancesPerLod[4]`, `nonEmptyBuckets` and the per-LOD counts of a short list of
  tracked mesh IDs (`RenderScene::trackMeshStatistics`) into the frame's small readback buffer (the CPU
  never decodes buckets with a layout that `addMesh` may have changed since), read from a completed frame and tagged
  with its `RenderScene` frame number; `drawCommands` = the indirect draw calls issued (comparable to
  `ForestRenderer::drawCalls()`); a class with more buckets than `maxDrawIndirectCount` is split into several calls.
- **Drawing**: the instances are drawn in the scene pass after the game's render callback; shading is the flat-color
  model with a fixed sun and ambient (S36 adds lighting) and the material's emissive; `SmoothTextured` without textures
  renders its base color (textures in S43). The pipelines use `RenderScene::formats()`; the instance shaders follow
  S17's output contract (clay mode: display values, clamped, own sRGB encoding; otherwise linear, multiplied by
  `preExposure()`). GPU sections: `cull` (count, scan, write) between `transfer` and `scene`, written every frame (also
  without instances, S17's rule); the app's section list and `game_world`'s `gpu_frame_ms` gain the key.

### 3. Mesh import (`vulkan/scene/MeshImport.{h,cpp}`)

One material for the mesh (`addMesh` takes one): `usePalette` when a palette is given, `baseColor` 1 (the base color
factors are baked into the color words), `alphaMask` from `alphaMode MASK`, `twoSided` from `doubleSided`, emissive,
roughness and metalness from the glTF material; merged primitives whose materials differ in any of these are an error
naming both. The sphere is computed over all LODs after baking; `TEXCOORD_0` is imported into `uv` (zero without);
zero-area faces are dropped (no NaN normal).

Color word: sRGB8 RGB plus linear alpha; the import computes `COLOR_0 · baseColorFactor` in linear (both are linear in
glTF), converts to sRGB, rounds to nearest; the palette match is on the 8-bit RGB. Alpha is forced to 255 unless the
material's `alphaMode` is `MASK` (assimp 5.2.5 delivers alpha 0 for `VEC3` colors).

```cpp
struct MeshImportOptions { const std::vector<ucvec4>* palette = nullptr; /* exact match required, else an error */ };
bool importMeshLods(const std::string& file, MeshLods& out, MaterialData* material, std::string& error,
                    const MeshImportOptions& = {});
```

glTF 2.0 (`.gltf`, `.glb`) through assimp (the submodule's glTF2 importer; assimp reaches `saiga_vulkan` through
`saiga_core`'s public link, to be confirmed in the MSVC job; without `SAIGA_USE_ASSIMP` the function returns an error
and the import tests are compiled out with `#ifdef SAIGA_USE_ASSIMP`, not skipped: the Ubuntu jobs fail on skips). LODs are the nodes whose names end in `LOD0`, `LOD1`, ... (e.g. `Tree_LOD0`), or the whole
scene as one LOD; all primitives under a LOD node are merged (assimp splits them per material) with the node
transforms baked (normals with the inverse transpose; a negative determinant flips the winding); LOD suffixes are
contiguous from `LOD0`, at most 4; once a LOD node exists, mesh nodes outside the LOD nodes are an error.
Post-processing: no `Triangulate`, no `JoinIdenticalVertices`; every `aiMesh` must have `mPrimitiveTypes ==
aiPrimitiveType_TRIANGLE` (strips and fans are accepted: assimp converts them; points and lines are errors).
Flat colors: vertices duplicated per face with the face normal computed from the baked positions; the color word as
above, or with a palette the index of the exact entry (a miss is an error naming the color). `endDistance`s are not in glTF: the game sets them (R-MECH-1). Errors (no mesh, too many vertices,
non-triangles, a palette miss) are messages, not asserts.

## Out of scope

Shadow views (S36), Hi-Z occlusion (§21), projected-size LOD and impostors (S27), textures, normal maps and wind (S43),
skinning (S34), generated meshes (S30), the engine side (`RenderInstance`, `RenderSink`, extract: S19), the prototype
trees on this path with their textured, wind-animated foliage shading (S19 ports `tree.vert` / `foliage.frag` onto the
instance layout as a transitional pipeline class; recorded in S19's plan) and the forest benchmark comparison (S19).

## Design notes

- The instance path is built and tested standalone in `saiga_vulkan` (no engine dependency), so S19 only wires the
  engine's extract to it.
- The count pass stores each visible instance's slot, so the write pass cannot disagree with it (no second visibility
  decision, no bucket overflow).
- The culling rule (5 planes, distance to the sphere centre, end distances with culling beyond the last LOD, bounds
  padding) is the prototype's `tree_cull.comp` generalized, so S19 can reach equal counts per LOD; the prototype's
  `freezeCulling` (a separate cull camera) is dropped in S19 (a debugging aid without a user).
- Precision: offsets are computed in double and stored as float relative to the camera; a vertex's error is half an
  ulp of its local position plus half an ulp of the offset (both up to about 1 mm at 16-32 km). The offset part moves
  with the camera, so near content of a large region would step in about 2 mm increments; the offset is therefore
  stored as hi + lo floats and applied as `(local + hi) + lo`, which leaves only the local error (stated in §9.1 /
  §10.1).

## Tests

`tests/test_vulkan_scene.cpp` (lavapipe in CI):
- Partial chunks: a 300-instance set, then a frame where the same chunk index covers 10 instances: counts and pixels
  match the reference (no stale scratch entries).
- Statistics across layout changes: `addMesh` right after a frame, then that frame's statistics are still correct.
- Rejections: a dynamic in another space, a removed dynamic, an unset dynamic and an instance of an unregistered mesh
  are not counted.
- Culling counts: 10^4 seeded instances of 3 meshes x 3 LODs in several regions against a CPU reference of the same
  rule (decoded quaternions, 5 planes, end distances, padding; §17.5 tolerance); after `updateInstances`,
  `removeInstance` (a tombstone inside the frustum is not counted) and `removeInstanceSet`; sets of another space are not
  drawn; the read statistics carry the frame they come from (the test waits idle and reads a known frame).
- Growth: capacity exceeded with live sets and meshes, also with uploads pending in the same frame; counts and pixels
  unchanged afterwards; validation clean; the old buffers released after their frames (the `VulkanMemory` allocation
  count before and after `waitIdle`).
- Dynamic instances: prev/curr poses drawn at `alpha` 0, 0.5, 1 (a marker mesh's rendered pixel position matches the
  interpolated projection within a pixel; culling and LOD from the same interpolated pose); chunk-relative poses in two
  chunks; a 270° rotation about one axis passes through −45° at `alpha` 0.5 (the shorter arc); `q` vs `−q` does not
  move.
- Precision: a camera at 30 km (dyadic coordinates) looking at a level-0 chunk renders the same image as at the
  origin; a set of a level-6 region whose origin is more than 8 km from its instances near the camera: a marker at 0.1 m
  with a narrow `fovy` (a pixel footprint of at most 0.25 mm) has its sub-pixel centroid within 1 px of the
  double-precision projection, and the centroid moves monotonically over 0.25 mm camera steps (fails without hi/lo).
- Mesh import: a small committed glTF (`data/tests/scene/flat_box.gltf`, two LODs as `Box_LOD0`/`Box_LOD1`, two
  materials, a `VEC3` `COLOR_0`) imports with merged primitives, duplicated faces (vertex count = 3 * triangles), face
  normals and opaque alpha; with a palette the indices; a palette miss, a `LINES` primitive and a non-contiguous LOD
  suffix are errors; rendered (one instance) it matches a new golden
  `scene_flat_box`.
- Draw paths with and without `multiDrawIndirect` (`allowMultiDrawIndirect = false`) give equal counts and images;
  `drawCommands` and `nonEmptyBuckets`.

Benchmark (local, R-TEST-2, an S19 input): `samples/vulkan/sample_vulkan_scene_benchmark.cpp` (built by the AGENTS.md
bench recipe, samples ON, tests OFF): 200k instances of flat meshes with the prototype trees' triangle counts and LOD
distances over the flyover path; writes `benchmark_scene_cull.json` with the `cull` section's GPU time and the instance
draw time; reported in the PR; the S03 benchmarks re-run unchanged within 5 %. The AGENTS.md bench recipe and
Verification name the new target (`--target sample_vulkan_scene_benchmark`).

## Done when

- The S18 row of §22: culling counts per LOD equal a CPU reference (§17.5 tolerance); the dynamic instance
  interpolation test; an imported glTF mesh renders with face-duplicated flat normals (CI).
- `docs/engine_design.md` updated: §10.1 (the S18 API [exists], explicit IDs, the palette, capacities, the offset formula
  with rotation, scale, per-set slots, hi/lo and the valid flag, `RenderStatistics` fields), §10.2 (as implemented:
  work chunks with a direct dispatch, stored slots, `uvec2` entries, buckets (view, class, mesh, LOD), the rule), §10.8
  (the data model [exists: S18], the material buffer, the palette entry, the color word), §9.1 (the precision note),
  §14 (mesh import), §17.5 [exists], §17.6 (the new golden), §17.7 (the culling benchmark), §18.1 (files), §19 (the
  `ForestRenderer` migration row: S18/S19 split), §4/§12 (`regionOrigin`'s convention: x, z in units of 2^L chunks), §3, §22 (S18 done; S19 ports the tree shading); `AGENTS.md` (the scene files, the test
  data).

## Verification

The CI recipe of AGENTS.md; `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R test_vulkan_scene`; the
benchmark recipe of AGENTS.md (flyover, forest) and the culling benchmark.

## Files

`src/saiga/vulkan/scene/{RenderScene.{h,cpp},RenderSceneTypes.h,MeshPool.{h,cpp},InstanceBuffer.{h,cpp},
SceneCulling.{h,cpp},MeshImport.{h,cpp}}`, `src/saiga/core/math/ChunkId.h`, `shader/vulkan/scene/{instance_count.comp,
bucket_scan.comp,instance_write.comp,instance.vert,flat.frag}`, `tests/test_vulkan_scene.cpp`,
`data/tests/scene/flat_box.gltf`, `data/tests/golden/scene_flat_box.png`, `samples/vulkan/sample_vulkan_scene_benchmark.cpp`,
`samples/vulkan/CMakeLists.txt`, `tests/test_engine_app.cpp` and `samples/games/world/main.cpp` (the `cull` section), `docs/engine_design.md`, `AGENTS.md`, this
file.

## Design changes

- **Additional files**: `vulkan/scene/SubAllocatedBuffer.{h,cpp}` (the VMA virtual block, the growth with its pending
  copy and the residency of each range, shared by the mesh pool's vertex and index buffers and the instance buffer),
  `UploadBatch.{h,cpp}` (the frame's uploads through the frame slot's staging buffer, one `vkCmdCopyBuffer` per
  destination), `RenderSceneTypes.cpp` (the encoders), `shader/vulkan/scene/instance_common.glsl` (the GLSL structs,
  `placePoint`, the rotation decode and the dynamic interpolation shared by the count pass, the write pass and
  `instance.vert`), `data/tests/scene/flat_box_lines.gltf`, `flat_box_lod_gap.gltf` (the import error cases) and
  `make_gltf.py` (the generator of the three glTF files).
- **One descriptor set layout** with all 13 storage-buffer bindings for the three compute pipelines and the four draw
  pipelines (identically defined layouts are compatible), one set per frame slot, rewritten for every recorded frame.
- **The CPU set test also uses the end distances**: a set whose AABB is farther from the camera than the largest last
  end distance of its meshes gets no work chunks (equivalent to the per-instance rule, which the count pass still
  applies; it keeps the far sets of the culling benchmark out of the dispatch).
- **Sets keep a CPU copy** of their instances: it is the source of the uploads, queued as ranges per set and merged per
  frame (one copy command must not write a range twice); `updateInstances` beyond the used size also uploads the
  tombstones of the gap, which the culling now visits. Bounds are recomputed over the used size at
  `addInstanceSet` and `updateInstances`; an instance scale must be >= 0 (asserted there).
- **Statistics**: the scan writes `instancesPerLod` and `nonEmptyBuckets` directly into the frame slot's host-visible
  statistics buffer (no copy of the counts buffer); a frame without a camera or instances gets zeros from the CPU.
  `RenderScene::statistics()` (non-const) reads the completed slots and keeps the newest.
- **API additions**: `PipelineClass` (1..4 explicit classes, 0 derived; asserted), the encoders `packNormalOct` /
  `unpackNormalOct`, `packColor`, `packSnorm2x16` / `unpackSnorm2x16`, `StaticInstance::setRotation` / `getRotation` /
  `setPosition` / `getPosition`, `RenderScene::viewMatrix()` / `projectionMatrix()` (with `viewProjection()`),
  `hasCamera()`, `multiDrawIndirect()`; `setDynamic`'s `boneOffset` defaults to 0; `chunkSize` (256 m) next to
  `chunkOrigin` / `regionOrigin`; `VulkanMemory::allocationCount()` (the growth test's released buffers) and
  `WorldBenchmark::buildInfo()` (public, for the culling benchmark).
- **Frame barriers**: the transfer section's barriers include the compute and indirect stages (the scene's buffers
  written there are read by the culling; the culling's writes of the previous frame precede this frame's transfer
  writes); the cull section starts with a barrier from the previous frame's draws and passes.
- **glTF `VEC3` colors**: a `COLOR_0` whose alphas are all 0 is treated as `VEC3` (alpha 1) also for a `MASK` material;
  the plan's "alpha forced to 255 unless MASK" alone would make every MASK mesh with a VEC3 color invisible. The import
  sets placeholder end distances (LOD i: (i + 1) * 1000 m) and limits a LOD to 2^24 vertices.
- **Fixed light**: sun `normalize(0.4, 1, 0.7)` (0.8) and ambient 0.35; a symmetric direction lit the box's +x and +z
  faces equally, so the golden would not show three distinct face shades.
- **Tests**: the rejection of "an instance of an unregistered mesh" is a dynamic of an unregistered mesh (a static set
  asserts registered meshes at `addInstanceSet`, as the plan requires; the zeroed record is the guard for dynamics).
  The scene tests are the fixture `VulkanSceneContent` (derived from `VulkanScene`), the import test
  `VulkanMeshImport.FlatBox` (no device). The CPU reference's tolerance is the plan's: per LOD and in total the GPU
  count differs by at most the number of instances within 1e-3 relative of a plane or an end distance.
- **Local review** (after the results below): the draws use a snapshot of the frame's transfer (each pipeline class's
  bucket range, the vertex, index and command buffers), so `addMesh` or a class change of `setMaterial` in the render
  callback takes effect in the next frame (before, the draws read the new layout against the old commands buffer:
  `VUID-vkCmdDrawIndexedIndirect-drawCount-00540`, and bound a grown vertex buffer before its copy). The storage
  buffers bound whole (static instances, dynamics, compacted, scratch, work chunks, offset table, counts, commands,
  bucket LODs, statistics) grow at most to `maxStorageBufferRange` (128 MiB on lavapipe); more is a fatal error
  (`SAIGA_EXIT_ERROR`) in every build; the tables of mesh records and materials are bounded by the 16-bit IDs (7.3 MB,
  below the minimum 2^27 of the limit). The count and write passes dispatch in 2D beyond
  `maxComputeWorkGroupCount[0]` (before: an assert only). `RenderSceneParameters::storageBufferLimit` and
  `workGroupLimit` lower the limits for tests. New tests: `ContentChangesInRenderCallback`, `MaterialClassChange`,
  `PaletteColors`, `AlphaMaskDiscard`, `StorageBufferLimit`, `WorkGroupRows` (`test_vulkan_scene`: 24 tests). The
  culling benchmark records the
  statistics of each frame once.
- **Plan amendment** (after the implementation: the plan's later commits for S19a and S19b, implemented here):
  `interpolatePose` in `RenderSceneTypes.h`, templated on the scalar (`InstancePose<T>`; `DynamicPose` is
  `InstancePose<double>`), the CPU reference of the shaders' `dynamicPose` (the tests' culling reference in float,
  S19b's extract in double); `MeshLods` end distances non-decreasing and >= 0 (an equal one is an empty LOD: the
  first LOD with distance <= endDistance is never the later of two equal ones; the count pass, the set test and the
  CPU reference needed no change); `RenderScene::trackMeshStatistics` with at most `maxTrackedMeshes` = 16 IDs (S19a
  tracks its tree species): the host writes the frame's list into the statistics buffer (352 bytes), the scan writes
  4 counts per tracked mesh from its mesh record's `bucketBase` and LOD count (0 for an unregistered ID or a missing
  LOD), `RenderStatistics::trackedMeshes` reports them with the list of that frame. Tests:
  `InterpolatePoseMatchesShader`, `EqualDistanceLods`, `TrackedMeshStatistics` (`test_vulkan_scene`: 27 tests).
- **The culling benchmark runs headless** (`HeadlessWindow`, no present): it needs no window and isolates the scene's
  sections; 120 frames without a camera give the empty scene section, `draw_ms` = `scene_ms` - its median. The trees
  are `ForestRenderer::placeTrees` on the default height field and `FoliageGenerator`'s meshes made flat (faces
  duplicated, no textures, a two-sided material: the leaf cards are single quads), one set per chunk.

## Results

Tests (`VK_DRIVER_FILES=.../lvp_icd.json ctest --parallel 8`, RelWithDebInfo, GCC 15): 65 of 66 pass; only
`test_opengl_window` fails locally (no display, as before). `test_vulkan_*` and `test_engine_app` (14 tests) also pass
on the RTX 5080 Laptop GPU with the validation layer (no validation message). `test_vulkan_scene` has 18 tests (8 of
S17, 9 `VulkanSceneContent`, `VulkanMeshImport.FlatBox`). `VulkanSceneContent.CullingCountsMatchReference`: 1345 of
10^4 instances drawn in the initial frame, 62 within the 1e-3 tolerance band; the GPU counts are within it after every
change. `VulkanSceneContent.Precision` fails without the lo part of the offsets (1.2-1.8 px off the double projection)
and passes with it. The demos' `--scenarios` pass (9/9 each); the `--demo` screenshots of the three demos and
`game_world` (`SAIGA_SCREENSHOT`, RTX 5080) look as before.

Golden (lavapipe, Mesa with LLVM 21.1.8): `scene_flat_box` new (RTX 5080, not gating: mean 0/255, max 0/255); the other
goldens unchanged.

Benchmarks (release build of AGENTS.md, RTX 5080 Laptop GPU, 1920 x 1080; before = the S17 tip (the plan commit),
after = this stage; 3 runs each; medians of the run medians, ms):

| | flyover before | flyover after | forest before | forest after |
|---|---|---|---|---|
| `cpu_ms` (frame interval) | 3.79 (3.77-3.80) | 3.75 (3.03-3.83) | 2.14 (2.03-2.17) | 2.08 (2.04-2.13) |
| terrain + water | 0.105 | 0.107 (+2.1 %) | 0.063 | 0.064 (+1.7 %) |
| trees | 1.449 | 1.451 | 0.470 | 0.472 (+0.6 %) |
| grass | 0 | 0 | 0.671 | 0.675 (+0.6 %) |
| sky | 0.011 | 0.011 | 0.008 | 0.008 |
| upload + cull | 0.025 | 0.025 | 0.023 | 0.023 |
| `gpu_frame_ms.gpu` (whole frame) | 2.312 | 2.316 | 1.468 (1.45-1.76) | 1.448 |
| sections | transfer 0.028, scene 2.273, post 0.010 | transfer 0.029, cull 0.0001, scene 2.275, post 0.010 | transfer 0.026, scene 1.426, post 0.010 | transfer 0.028, cull 0.0001, scene 1.400, post 0.010 |

All passes and the frame interval are within 5 % (R-TEST-2). The `transfer` section grew by 1.3-1.7 µs (+4.7 % /
+6.6 %): the stronger barriers of the frame (compute and indirect stages) and the per-frame offset and descriptor
writes; below the timer noise of passes under 0.1 ms (§17.7). The empty `cull` section costs 0.1 µs. The flyover
interval is in the slow mode of S17's bimodal flyover (3.8 ms) in the before runs and in two of three after runs.
`data/tests/benchmarks.json` is not updated (S17's baseline is not accepted yet).

Culling benchmark (`sample_vulkan_scene_benchmark`, same build, 4 runs; 199,466 instances in 928 sets, triangles per LOD
(148, 90, 18), (516, 140, 28), (460, 132, 28); multi-draw: 1 indirect draw call per frame): `cull_ms` 0.066-0.068
(p95 0.070-0.072), `draw_ms` 1.27-1.32 (p95 1.36-1.40; `scene_ms` 1.32-1.36, `scene_empty_ms` 0.035-0.047),
`transfer_ms` 0.002, CPU `recording_ms` 0.11-0.16 (the work chunks of 928 sets and the offset table); 126.6k instances
drawn per frame (median), all in LOD 2 (the flyover camera is 300 m above the ground). For orientation only, the
prototype's flyover `trees` pass is 1.45 ms and its `upload + cull` 0.025 ms (alpha-tested foliage with alpha to
coverage, a different shading); S19 compares the forest on equal terms.

Left open: CI has not run on this branch (lavapipe on the runner: the `scene_flat_box` margin on CI's Mesa; the MSVC job:
assimp reaching `saiga_vulkan` through `saiga_core` and the new code on MSVC); the culling benchmark has no committed
baseline (an S19 input); the flyover interval stays bimodal (S17).

