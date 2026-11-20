# Copyright (c) 2021 Darius Rückert
# Licensed under the MIT License.
# See LICENSE file for more information.
#
# Generates the glTF files of the mesh import tests (tests/test_vulkan_scene.cpp, S18):
#   python3 data/tests/scene/make_gltf.py data/tests/scene
# flat_box.gltf: Box_LOD0 (a unit box, two primitives: top/bottom with material BoxTop, the sides with BoxSide, whose
# baseColorFactor is 0.5) and Box_LOD1 (a tetrahedron), both translated by (0, 0.5, 0), VEC3 COLOR_0;
# flat_box_lines.gltf: a LINES primitive; flat_box_lod_gap.gltf: LOD0 and LOD2 without LOD1.
import base64, json, struct, sys

def build(primitives_per_node, nodes, materials, mode=4):
    # primitives_per_node: {mesh name: [(positions, indices, colors, material)]}
    data = bytearray()
    accessors, views, meshes = [], [], []
    def add(arr, fmt, comp, typ, count, target, minmax=None):
        nonlocal data
        while len(data) % 4: data.append(0)
        off = len(data)
        for v in arr: data.extend(struct.pack(fmt, *v) if isinstance(v, tuple) else struct.pack(fmt, v))
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data) - off, "target": target})
        a = {"bufferView": len(views) - 1, "componentType": comp, "count": count, "type": typ}
        if minmax: a["min"], a["max"] = minmax
        accessors.append(a)
        return len(accessors) - 1
    for name, prims in primitives_per_node:
        ps = []
        for pos, idx, col, mat in prims:
            mn = [min(p[i] for p in pos) for i in range(3)]
            mx = [max(p[i] for p in pos) for i in range(3)]
            pa = add(pos, '<3f', 5126, "VEC3", len(pos), 34962, (mn, mx))
            ca = add(col, '<3f', 5126, "VEC3", len(col), 34962)
            ia = add(idx, '<H', 5123, "SCALAR", len(idx), 34963)
            ps.append({"attributes": {"POSITION": pa, "COLOR_0": ca}, "indices": ia, "material": mat, "mode": mode})
        meshes.append({"name": name, "primitives": ps})
    gltf = {"asset": {"version": "2.0", "generator": "saiga tests (S18)"},
            "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
            "nodes": nodes, "meshes": meshes, "materials": materials,
            "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(data)).decode()}],
            "bufferViews": views, "accessors": accessors}
    return json.dumps(gltf, indent=1)

# The unit box [-0.5, 0.5]^3: per face 4 corners (CCW from outside)
def quad(c, col):
    return c, [0, 1, 2, 0, 2, 3], [col] * 4
faces = {
    "+y": [(-.5, .5, .5), (.5, .5, .5), (.5, .5, -.5), (-.5, .5, -.5)],
    "-y": [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)],
    "+x": [(.5, -.5, .5), (.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5)],
    "-x": [(-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5), (-.5, .5, -.5)],
    "+z": [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)],
    "-z": [(.5, -.5, -.5), (-.5, -.5, -.5), (-.5, .5, -.5), (.5, .5, -.5)],
}
def merge(keys, col):
    pos, idx, cols = [], [], []
    for k in keys:
        base = len(pos)
        pos += faces[k]; idx += [base + i for i in [0, 1, 2, 0, 2, 3]]; cols += [col] * 4
    return pos, idx, cols
topCol = (0.8, 0.2, 0.1)
sideCol = (0.1, 0.6, 0.3)
tetraCol = (0.2, 0.2, 0.9)
pa, ia, ca = merge(["+y", "-y"], topCol)
pb, ib, cb = merge(["+x", "-x", "+z", "-z"], sideCol)
# LOD1: a tetrahedron
tp = [(-.5, -.5, -.5), (.5, -.5, -.5), (0, -.5, .5), (0, .5, 0)]
ti = [0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0]
tp2, ti2, tc2 = [], [], []
for t in range(4):
    for k in range(3):
        tp2.append(tp[ti[3 * t + k]]); ti2.append(len(ti2)); tc2.append(tetraCol)
mats = [{"name": "BoxTop", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.7}},
        {"name": "BoxSide", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1], "metallicFactor": 0, "roughnessFactor": 0.7}}]
lod0 = ("BoxMesh0", [(pa, ia, ca, 0), (pb, ib, cb, 1)])
lod1 = ("BoxMesh1", [(tp2, ti2, tc2, 0)])
out = sys.argv[1]
nodes = [{"name": "Box_LOD0", "mesh": 0, "translation": [0, 0.5, 0]}, {"name": "Box_LOD1", "mesh": 1, "translation": [0, 0.5, 0]}]
open(out + "/flat_box.gltf", "w").write(build([lod0, lod1], nodes, mats) + "\n")
# Error cases: a LINES primitive, a LOD gap (LOD0, LOD2)
nodes = [{"name": "Box_LOD0", "mesh": 0}]
open(out + "/flat_box_lines.gltf", "w").write(build([("Lines", [(pa, ia, ca, 0)])], nodes, mats[:1], mode=1) + "\n")
nodes = [{"name": "Box_LOD0", "mesh": 0}, {"name": "Box_LOD2", "mesh": 1}]
open(out + "/flat_box_lod_gap.gltf", "w").write(build([lod0, lod1], nodes, mats) + "\n")
