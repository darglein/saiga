# S18: Instances and culling

Status: planned.

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
struct MeshLods { std::vector<MeshLod> lods; /* 1..4, ascending endDistance, asserted */ vec3 sphereCenter;
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
     when `dot < 0`, nlerp near `dot = 1`; the CPU reference uses the same formula in float); `atomicAdd`
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
  Statistics: the scan writes `instancesPerLod[4]` and `nonEmptyBuckets` into the frame's small readback buffer (the CPU
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
