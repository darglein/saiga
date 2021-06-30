/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/LineMesh.h"
#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/geometry/vertex.h"
// #include "saiga/core/model/
#include "saiga/core/image/managedImage.h"
#include "saiga/core/model/UnifiedMesh.h"
#include "saiga/core/model/animation.h"
#include "saiga/core/util/Range.h"

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
    std::string texture_bump;
    std::string texture_alpha;
    std::string texture_emissive;

    UnifiedMaterial() {}
    UnifiedMaterial(const std::string& name) : name(name) {}

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& strm, const UnifiedMaterial& material);
};


struct UnifiedMaterialGroup
{
    int startFace  = 0;
    int numFaces   = 0;
    int materialId = -1;

    Range<int> range() const { return Range<int>(startFace, startFace + numFaces); }
};

class SAIGA_CORE_API UnifiedModel
{
   public:
    UnifiedModel() {}
    UnifiedModel(const UnifiedMesh& mesh) { this->mesh.push_back(mesh); }
    UnifiedModel(const std::string& file_name);
    ~UnifiedModel();


    void Save(const std::string& file_name);


    UnifiedModel& ComputeColor();
    UnifiedModel& SetVertexColor(const vec4& color);

    UnifiedModel& Normalize(float dimensions = 2.0f);
    UnifiedModel& AddMissingDummyTextures();

    AABB BoundingBox() const;

    std::string name;

    std::vector<UnifiedMesh> mesh;


    // The material is given on a per face basis.
    // The material group defines which faces have which material.
    std::vector<UnifiedMaterial> materials;
    std::vector<UnifiedMaterialGroup> material_groups;
    std::vector<Image> textures;
    std::map<std::string, int> texture_name_to_id;

    // Bone Data (only used for animated models)
    AnimationSystem animation_system;


    int TotalTriangles() const
    {
        int n = 0;
        for (auto& m : mesh) n += m.NumFaces();
        return n;
    }

    int TotalVertices() const
    {
        int n = 0;
        for (auto& m : mesh) n += m.NumVertices();
        return n;
    }

    std::pair<UnifiedMesh, std::vector<UnifiedMaterialGroup>> CombinedMesh(int vertex_flags = VERTEX_POSITION) const;

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& strm, const UnifiedModel& model);


   private:
    void LocateTextures(const std::string& base);
};


}  // namespace Saiga

