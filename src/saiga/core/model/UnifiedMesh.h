/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/LineMesh.h"
#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/geometry/vertex.h"
#include "saiga/core/image/managedImage.h"
#include "saiga/core/model/animation.h"
#include "saiga/core/util/Range.h"

#include <vector>


namespace Saiga
{
enum VertexDataFlags : int
{
    VERTEX_POSITION            = 1 << 0,
    VERTEX_NORMAL              = 1 << 1,
    VERTEX_COLOR               = 1 << 2,
    VERTEX_TEXTURE_COORDINATES = 1 << 3,
    VERTEX_EXTRA_DATA          = 1 << 4,
    VERTEX_BONE_INFO           = 1 << 5,
};

class SAIGA_CORE_API UnifiedMesh
{
   public:
    int NumVertices() const { return position.size(); }
    int NumFaces() const { return triangles.size(); }


    std::string name;

    UnifiedMesh() {}
    UnifiedMesh(const UnifiedMesh& a, const UnifiedMesh& b);

    // Vertex Data
    std::vector<vec3> position;
    std::vector<vec3> normal;
    std::vector<vec4> color;
    std::vector<vec2> texture_coordinates;
    std::vector<vec4> data;
    std::vector<BoneInfo> bone_info;

    // Face data for surface meshes stored as index-face set
    std::vector<ivec3> triangles;

    // Line indices for line meshes
    std::vector<ivec2> lines;

    int material_id = 0;


    // Transforms this model inplace
    // returns a reference to this
    UnifiedMesh& transform(const mat4& T);

    // Overwrite the color of every vertex
    // returns a reference to this
    UnifiedMesh& SetVertexColor(const vec4& color);

    // Use barycentric interpolation to compute the color at a given triangle.
    // Vertex colors must be available!
    vec4 InterpolatedColorOnTriangle(int triangle_id, vec3 bary) const ;


    UnifiedMesh& SmoothVertexColors(int iterations, float self_weight);

    // Set n = -n
    UnifiedMesh& FlipNormals();

    // Reorder the triangle indices
    UnifiedMesh& InvertTriangleOrder();


    // Computes the per vertex normal by weighting each face normal by its surface area.
    UnifiedMesh& CalculateVertexNormals();



    // Duplicates vertices so that each vertex is used exactly in one face.
    UnifiedMesh& FlatShading();


    // Removes all vertices with the given indices
    // Faces are currently not updated (maybe todo in the future)
    UnifiedMesh& EraseVertices(ArrayView<int> vertices);

    // Merge vertices that are closer than 'distance' apart
    UnifiedMesh& RemoveDoubles(float distance);


    // Remove triangles that reference the same vertex twice
    UnifiedMesh& RemoveDegenerateTriangles();

    //
    // gather == true:
    //    vertex_new[i] = vertex_old[idx[i]]
    //
    // gather == false:
    //    vertex_new[idx[i]] = vertex_old[i]
    UnifiedMesh& ReorderVertices(ArrayView<int> idx, bool gather = true);
    UnifiedMesh& RandomShuffle();
    UnifiedMesh& RandomBlockShuffle(int block_size);
    UnifiedMesh& ReorderMorton64();


    UnifiedMesh& Normalize(float dimensions = 2.0f);

    AABB BoundingBox() const;



    std::vector<Triangle> TriangleSoup() const;


    // Check status
    bool HasPosition() const { return !position.empty(); }
    bool HasNormal() const { return !normal.empty(); }
    bool HasColor() const { return !color.empty(); }
    bool HasTC() const { return !texture_coordinates.empty(); }
    bool HasData() const { return !data.empty(); }
    bool HasBones() const { return !bone_info.empty(); }

    VertexDataFlags Flags() const;


    // Conversion Functions from unified model -> Triangle mesh
    // The basic conversion functions for the saiga vertices are defined below,
    // however you can also define conversions for custom vertex types.
    template <typename VertexType, typename IndexType>
    TriangleMesh<VertexType, IndexType> Mesh() const
    {
        TriangleMesh<VertexType, IndexType> mesh;
        mesh.vertices = VertexList<VertexType>();
        mesh.faces    = TriangleIndexList<IndexType>();
        return mesh;
    }


    // Conversion Functions from unified model -> Line mesh
    // The basic conversion functions for the saiga vertices are defined below,
    // however you can also define conversions for custom vertex types.
    template <typename VertexType, typename IndexType>
    Saiga::LineMesh<VertexType, IndexType> LineMesh() const
    {
        Saiga::LineMesh<VertexType, IndexType> mesh;
        mesh.vertices = VertexList<VertexType>();
        mesh.lines    = LineIndexList<IndexType>();
        return mesh;
    }

    template <typename VertexType>
    std::vector<VertexType> VertexList() const;

    template <typename IndexType>
    std::vector<Vector<IndexType, 3>> TriangleIndexList() const;


    template <typename IndexType>
    std::vector<Vector<IndexType, 2>> LineIndexList() const;


    void SaveCompressed(const std::string& file);
    void LoadCompressed(const std::string& file);

   private:
    void LocateTextures(const std::string& base);
};

}  // namespace Saiga


#include "UnifiedModel.hpp"
