/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UnifiedMesh.h"

#include "saiga/core/geometry/kdtree.h"
#include "saiga/core/math/Morton.h"
#include "saiga/core/math/random.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"
#include "saiga/core/util/zlib.h"

#include "internal/noGraphicsAPI.h"

#include "model_loader_obj.h"
#include "model_loader_ply.h"

namespace Saiga
{
UnifiedMesh::UnifiedMesh(const UnifiedMesh& a, const UnifiedMesh& b)
{
    auto combine = [&](auto v1, auto v2) {
        decltype(v1) flat;

        if (!v1.empty() && v2.empty())
        {
            v2.resize(b.NumVertices());
        }

        if (v1.empty() && !v2.empty())
        {
            v1.resize(a.NumVertices());
        }

        flat = v1;
        flat.insert(flat.end(), v2.begin(), v2.end());

        return flat;
    };

    position            = combine(a.position, b.position);
    normal              = combine(a.normal, b.normal);
    color               = combine(a.color, b.color);
    texture_coordinates = combine(a.texture_coordinates, b.texture_coordinates);
    data                = combine(a.data, b.data);
    bone_info           = combine(a.bone_info, b.bone_info);

    triangles = a.triangles;
    lines     = a.lines;

    for (auto t : b.triangles)
    {
        t(0) += a.NumVertices();
        t(1) += a.NumVertices();
        t(2) += a.NumVertices();
        triangles.push_back(t);
    }

    for (auto t : b.lines)
    {
        t(0) += a.NumVertices();
        t(1) += a.NumVertices();
        lines.push_back(t);
    }
}


UnifiedMesh& UnifiedMesh::transform(const mat4& T)
{
    if (HasPosition())
    {
        for (auto& p : position)
        {
            p = (T * make_vec4(p, 1)).head<3>();
        }
    }
    if (HasNormal())
    {
        for (auto& n : normal)
        {
            n = (T * make_vec4(n, 0)).head<3>();
        }
    }
    return *this;
}

UnifiedMesh& UnifiedMesh::SetVertexColor(const vec4& c)
{
    color.resize(position.size());
    for (auto& co : color)
    {
        co = c;
    }
    return *this;
}

UnifiedMesh& UnifiedMesh::FlipNormals()
{
    for (auto& n : normal)
    {
        n = -n;
    }
    return *this;
}

UnifiedMesh& UnifiedMesh::FlatShading()
{
    auto flatten = [this](auto old) {
        decltype(old) flat;
        for (auto& tri : triangles)
        {
            flat.push_back(old[tri(0)]);
            flat.push_back(old[tri(1)]);
            flat.push_back(old[tri(2)]);
        }
        return flat;
    };

    if (!position.empty()) position = flatten(position);
    if (!normal.empty()) normal = flatten(normal);
    if (!color.empty()) color = flatten(color);
    if (!texture_coordinates.empty()) texture_coordinates = flatten(texture_coordinates);
    if (!data.empty()) data = flatten(data);
    if (!bone_info.empty()) bone_info = flatten(bone_info);

    std::vector<ivec3> flat_triangles;
    for (int i = 0; i < triangles.size(); ++i)
    {
        flat_triangles.push_back(ivec3(i * 3, i * 3 + 1, i * 3 + 2));
    }
    triangles = flat_triangles;

    CalculateVertexNormals();

    return *this;
}

UnifiedMesh& UnifiedMesh::EraseVertices(ArrayView<int> vertices)
{
    for (auto i : vertices)
    {
        SAIGA_ASSERT(i >= 0 && i < NumVertices());
    }



    std::vector<int> valid_vertex(NumVertices(), 1);
    for (auto v : vertices)
    {
        SAIGA_ASSERT(valid_vertex[v] == 1);
        valid_vertex[v] = 0;
    }

    int valid_count = 0;
    std::vector<int> new_indices(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        new_indices[i] = valid_count;
        if (valid_vertex[i]) valid_count++;
    }

    int old_size = NumVertices();

    auto erase = [&](auto old) {
        SAIGA_ASSERT(old.size() == old_size);
        decltype(old) flat;
        for (int i = 0; i < old_size; ++i)
        {
            if (valid_vertex[i])
            {
                flat.push_back(old[i]);
            }
        }
        return flat;
    };

    if (!position.empty()) position = erase(position);
    if (!normal.empty()) normal = erase(normal);
    if (!color.empty()) color = erase(color);
    if (!texture_coordinates.empty()) texture_coordinates = erase(texture_coordinates);
    if (!data.empty()) data = erase(data);
    if (!bone_info.empty()) bone_info = erase(bone_info);

    SAIGA_ASSERT(valid_count == position.size());
    SAIGA_ASSERT(old_size - vertices.size() == position.size());

    // update triangle id + remove triangles with an invalid vertex
    for (auto& t : triangles)
    {
        for (int i = 0; i < 3; ++i)
        {
            int vid = t(i);
            if (valid_vertex[vid])
            {
                t(i) = new_indices[t(i)];
            }
            else
            {
                t(0) = -1;
            }
        }
    }

    triangles.erase(std::remove_if(triangles.begin(), triangles.end(), [](ivec3 t) { return t(0) == -1; }),
                    triangles.end());


    return *this;
}


UnifiedMesh& UnifiedMesh::ReorderVertices(ArrayView<int> idx, bool gather)
{
    SAIGA_ASSERT(idx.size() == NumVertices());
    for (auto i : idx)
    {
        SAIGA_ASSERT(i >= 0 && i < NumVertices());
    }

    auto reorder = [&](auto old) {
        decltype(old) new_vert(old.size());
        for (int i = 0; i < old.size(); ++i)
        {
            if (gather)
            {
                new_vert[i] = old[idx[i]];
            }
            else
            {
                new_vert[idx[i]] = old[i];
            }
        }
        return new_vert;
    };
    position            = reorder(position);
    normal              = reorder(normal);
    color               = reorder(color);
    texture_coordinates = reorder(texture_coordinates);
    data                = reorder(data);
    bone_info           = reorder(bone_info);

    // TODO
    SAIGA_ASSERT(triangles.empty());
    SAIGA_ASSERT(lines.empty());

    return *this;
}

UnifiedMesh& UnifiedMesh::RandomShuffle()
{
    auto sequence = Random::shuffleSequence(NumVertices());
    return ReorderVertices(sequence);
}
UnifiedMesh& UnifiedMesh::RandomBlockShuffle(int block_size)
{
    int n_blocks  = NumVertices() / block_size;
    auto sequence = Random::shuffleSequence(n_blocks);

    std::vector<int> indices(NumVertices());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < n_blocks; ++i)
    {
        int offset = sequence[i] * block_size;


        for (int j = 0; j < block_size; ++j)
        {
            int linear_offset  = i * block_size + j;
            int shuffle_offset = offset + j;

            indices[linear_offset] = shuffle_offset;
        }
    }

    return ReorderVertices(indices);
}
UnifiedMesh& UnifiedMesh::ReorderMorton64()
{
    auto bb = BoundingBox();

    vec3 offset = bb.min;
    vec3 scale  = float(1 << 20) / (bb.max - bb.min).array();

    std::vector<std::pair<uint64_t, int>> morton_list;
    morton_list.reserve(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        vec3 p  = (position[i] + offset).array() * scale.array();
        auto mc = Morton3D(p.cast<int>());
        morton_list.emplace_back(mc, i);
    }


    std::sort(morton_list.begin(), morton_list.end(), [](auto a, auto b) { return a.first < b.first; });

    std::vector<int> sequence;
    sequence.reserve(morton_list.size());
    for (auto i : morton_list)
    {
        sequence.push_back(i.second);
    }


    ReorderVertices(sequence, true);

#if 0
    {
        std::vector<std::pair<uint64_t, int>> morton_list;
        for (int i = 0; i < NumVertices(); ++i)
        {
            vec3 p  = (position[i] + offset).array() * scale.array();
            auto mc = Morton3D(p.cast<int>());
            morton_list.emplace_back(mc, i);
        }
        SAIGA_ASSERT(std::is_sorted(morton_list.begin(), morton_list.end()));
    }
#endif

    return *this;
}

UnifiedMesh& UnifiedMesh::Normalize(float dimensions)
{
    auto box = BoundingBox();
    float s  = dimensions / box.maxSize();
    vec3 p   = box.getPosition();

    mat4 S = scale(vec3(s, s, s));
    mat4 T = translate(-p);



    return transform(S * T);
}

AABB UnifiedMesh::BoundingBox() const
{
    AABB box;
    box.makeNegative();
    for (auto& p : position)
    {
        box.growBox(p);
    }
    return box;
}


std::vector<Triangle> UnifiedMesh::TriangleSoup() const
{
    std::vector<Triangle> result;
    for (auto t : triangles)
    {
        Triangle tri;
        tri.a = position[t(0)];
        tri.b = position[t(1)];
        tri.c = position[t(2)];
        result.push_back(tri);
    }
    return result;
}


UnifiedMesh& UnifiedMesh::CalculateVertexNormals()
{
    normal.resize(position.size());
    std::fill(normal.begin(), normal.end(), vec3(0, 0, 0));

    for (auto& tri : triangles)
    {
        vec3 n = cross(position[tri(1)] - position[tri(0)], position[tri(2)] - position[tri(0)]);
        normal[tri(0)] += n;
        normal[tri(1)] += n;
        normal[tri(2)] += n;
    }

    for (auto& n : normal)
    {
        n.normalize();
    }

    return *this;
}
VertexDataFlags UnifiedMesh::Flags() const
{
    return VertexDataFlags(HasPosition() * VERTEX_POSITION | HasNormal() * VERTEX_NORMAL | HasColor() * VERTEX_COLOR |
                           HasTC() * VERTEX_TEXTURE_COORDINATES | HasBones() * VERTEX_BONE_INFO |
                           HasData() * VERTEX_EXTRA_DATA);
}
vec4 UnifiedMesh::InterpolatedColorOnTriangle(int triangle_id, vec3 bary) const
{
    SAIGA_ASSERT(triangle_id >= 0 && triangle_id < triangles.size());
    SAIGA_ASSERT(HasColor());
    auto tri = triangles[triangle_id];

    vec4 c1 = color[tri(0)];
    vec4 c2 = color[tri(1)];
    vec4 c3 = color[tri(2)];

    return bary(0) * c1 + bary(1) * c2 + bary(2) * c3;
}
UnifiedMesh& UnifiedMesh::SmoothVertexColors(int iterations, float self_weight)
{
    SAIGA_ASSERT(HasColor());
    for (int it = 0; it < iterations; ++it)
    {
        std::vector<float> count(NumVertices(), 0);
        std::vector<vec4> colors_new(NumVertices(), vec4::Zero());

        for (auto t : triangles)
        {
            for (int i = 0; i < 3; ++i)
            {
                int v1 = t(i);
                for (int j = 0; j < 3; ++j)
                {
                    float w = i == j ? self_weight : 1;
                    int v2  = t(j);
                    colors_new[v2] += color[v1] * w;
                    count[v2] += w;
                }
            }
        }

        for (int i = 0; i < NumVertices(); ++i)
        {
            colors_new[i] = colors_new[i] / count[i];
        }

        color = colors_new;
    }
    return *this;
}
UnifiedMesh& UnifiedMesh::RemoveDoubles(float distance)
{
    std::vector<int> to_merge(NumVertices());

    std::vector<int> to_erase;
    std::vector<int> valid(NumVertices(), 0);
    KDTree<3, vec3> tree(position);
    for (int i = 0; i < position.size(); ++i)
    {
        auto ps     = tree.RadiusSearch(position[i], distance);
        bool found  = false;
        to_merge[i] = i;
        for (auto pi : ps)
        {
            if (valid[pi])
            {
                to_erase.push_back(i);
                to_merge[i] = pi;
                found       = true;
                break;
            }
        }
        if (!found)
        {
            valid[i] = true;
        }
    }

    for (auto& t : triangles)
    {
        for (int i = 0; i < 3; ++i)
        {
            t(i) = to_merge[t(i)];
        }
    }

    for (auto& t : lines)
    {
        for (int i = 0; i < 2; ++i)
        {
            t(i) = to_merge[t(i)];
        }
    }

    return EraseVertices(to_erase);
}

void UnifiedMesh::SaveCompressed(const std::string& file)
{
    BinaryOutputVector strm;
    strm << position << normal << color << texture_coordinates << data << bone_info;
    strm << triangles << lines;
    strm << material_id;
    auto compressed = compress(strm.data.data(), strm.data.size());
    File::saveFileBinary(file, compressed.data(), compressed.size());
}

void UnifiedMesh::LoadCompressed(const std::string& file)
{
    *this = {};

    auto compressed_data = File::loadFileBinary(file);
    auto data_raw        = uncompress(compressed_data.data());
    BinaryInputVector strm(data_raw.data(), data_raw.size());
    strm >> position >> normal >> color >> texture_coordinates >> data >> bone_info;
    strm >> triangles >> lines;
    strm >> material_id;
}

template <>
std::vector<Vertex> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());



    std::vector<Vertex> mesh;

    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }


    return mesh;
}

template <>
std::vector<VertexC> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());


    std::vector<VertexC> mesh;


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }

    if (HasColor())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = color[i];
        }
    }
    else
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = vec4(1, 1, 1, 1);
        }
    }
    return mesh;
}



template <>
std::vector<VertexNC> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());


    std::vector<VertexNC> mesh;


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }

    if (HasColor())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = color[i];
        }
    }
    else
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = vec4(1, 1, 1, 1);
        }
    }

    if (HasNormal())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].normal = make_vec4(normal[i], 0);
        }
    }
    return mesh;
}


template <>
std::vector<VertexNT> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());
    SAIGA_ASSERT(HasTC());

    std::vector<VertexNT> mesh;


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
        mesh[i].texture  = texture_coordinates[i];
    }

    if (HasNormal())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].normal = make_vec4(normal[i], 0);
        }
    }

    return mesh;
}

}  // namespace Saiga
