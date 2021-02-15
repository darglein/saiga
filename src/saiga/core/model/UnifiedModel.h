/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/vertex.h"

#include <vector>


namespace Saiga
{
struct UnifiedMaterial
{
    std::string name;
    vec4 color_diffuse  = vec4(0, 1, 0, 0);
    vec4 color_ambient  = vec4(0, 1, 0, 0);
    vec4 color_specular = vec4(0, 1, 0, 0);
    vec4 color_emissive = vec4(0, 1, 0, 0);
    std::string texture_diffuse;
    std::string texture_normal;
    std::string texture_alpha;

    UnifiedMaterial() {}
    UnifiedMaterial(const std::string& name) : name(name) {}
};

struct UnifiedMaterialGroup
{
    int startFace  = 0;
    int numFaces   = 0;
    int materialId = -1;
};

class SAIGA_CORE_API UnifiedModel
{
   public:
    UnifiedModel() {}
    UnifiedModel(const std::string& file_name);

    int NumVertices() const { return position.size(); }
    int NumFaces() const { return triangles.size(); }

    // Vertex Data
    std::vector<vec3> position;
    std::vector<vec3> normal;
    std::vector<vec4> color;
    std::vector<vec2> texture_coordinates;

    // Face data for surface meshes stored as index-face set
    std::vector<ivec3> triangles;

    // The material is given on a per face basis.
    // The material group defines which faces have which material.
    std::vector<UnifiedMaterial> materials;
    std::vector<UnifiedMaterialGroup> material_groups;

    // Check status
    bool HasPosition() const { return !position.empty(); }
    bool HasNormal() const { return !normal.empty(); }
    bool HasColor() const { return !color.empty(); }
    bool HasTC() const { return !texture_coordinates.empty(); }
    bool HasMaterials() const { return !materials.empty(); }
};


}  // namespace Saiga
