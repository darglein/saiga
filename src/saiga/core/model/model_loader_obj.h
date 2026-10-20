/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/tostring.h"

#include "UnifiedModel.h"

namespace Saiga
{
SAIGA_CORE_API std::vector<UnifiedMaterial> LoadMTL(const std::string& file);


// Loads a Wavefront OBJ file into a UnifiedModel.
//
// Supported: v, vt, vn, f (polygons are triangulated as a fan, negative indices are relative to the end of the
// current list), mtllib and usemtl. Each material group becomes one UnifiedMesh with that material. Vertices are
// shared within a mesh if they have the same position, texture and normal index. Missing normals are computed from
// the faces. Other statements (groups, objects, smoothing groups, lines, ...) are ignored.
class SAIGA_CORE_API ObjModelLoader
{
   public:
    std::string file;

   public:
    ObjModelLoader() {}
    ObjModelLoader(const std::string& file);

    // Throws std::runtime_error for invalid face indices.
    bool loadFile(const std::string& file);

    UnifiedModel out_model;

    static constexpr int INVALID_VERTEX_ID = -911365965;
    struct SAIGA_CORE_API IndexedVertex2
    {
        int v = INVALID_VERTEX_ID;
        int n = INVALID_VERTEX_ID;
        int t = INVALID_VERTEX_ID;
    };

   private:
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> texCoords;

    std::vector<std::vector<IndexedVertex2>> faces;
    std::vector<UnifiedMaterialGroup> material_groups;

    void createMeshes();

    std::vector<std::vector<IndexedVertex2>> triangulateFace(const std::vector<IndexedVertex2>& face);

    void parseLine(std::string_view line);
    // Resolves relative indices and adds the triangulated polygon.
    void addFace(std::vector<IndexedVertex2> ivs);
};

}  // namespace Saiga
