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


class SAIGA_CORE_API ObjModelLoader
{
   public:
    std::string file;
    bool verbose = false;

   public:
    ObjModelLoader() {}
    ObjModelLoader(const std::string& file);



    bool loadFile(const std::string& file);

    //    AlignedVector<vec4> vertexData;    // x: specular
    //    AlignedVector<vec4> vertexColors;  // only when given by the material. Otherwise: white!
    std::vector<VertexNT> outVertices;
    // std::vector<ivec3> outTriangles;

    UnifiedModel out_model;
    //    std::vector<UnifiedMaterialGroup> triangleGroups;
    //    std::vector<UnifiedMaterial> materials;
    void separateVerticesByGroup();
    void calculateMissingNormals();
    void computeVertexColorAndData();

    //    void toTriangleMesh(TriangleMesh<VertexNC, uint32_t>& mesh);
    //    void toTriangleMesh(TriangleMesh<VertexNTD, uint32_t>& mesh);


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


    void createVertexIndexList();

    std::vector<std::vector<IndexedVertex2>> triangulateFace(const std::vector<IndexedVertex2>& face);


    // ================== Parser Tmp Variables ================

    std::vector<IndexedVertex2> ivs;
    void parseLine();

    void parseF();
};

}  // namespace Saiga
