/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UnifiedModel.h"

#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include "model_loader_obj.h"
#include "model_loader_ply.h"

#ifdef SAIGA_USE_ASSIMP
#    include "saiga/core/model/model_loader_assimp.h"
#endif

namespace Saiga
{
std::ostream& operator<<(std::ostream& strm, const UnifiedMaterial& material)
{
    std::cout << "[Mat] " << std::setw(20) << material.name << material.color_diffuse.transpose()
              << ", tex: " << material.texture_diffuse << ", " << material.texture_normal << ", "
              << material.texture_bump << ", " << material.texture_alpha << ", " << material.texture_emissive;
    return strm;
}

UnifiedModel::UnifiedModel(const std::string& file_name)
{
    auto full_file = SearchPathes::model(file_name);
    if (full_file.empty())
    {
        throw std::runtime_error("Could not open file " + file_name);
    }

    std::string type = fileEnding(file_name);

    if (type == "obj")
    {
        ObjModelLoader loader(full_file);
        *this = loader.out_model;
        for (auto& v : loader.outVertices)
        {
            position.push_back(v.position.head<3>());
            normal.push_back(v.normal.head<3>());
            texture_coordinates.push_back(v.texture);
        }
        LocateTextures(full_file);
    }
#ifdef SAIGA_USE_ASSIMP
    else
    {
        AssimpLoader al(full_file);
        *this = al.Model();
        LocateTextures(full_file);
    }
#else
    else
    {
        throw std::runtime_error(
            "Unknown model file format " + to_string(type) +
            "\n You can compile saiga with Assimp to increase the number of supported file formats.");
    }
#endif
}


UnifiedModel::~UnifiedModel() {}

void UnifiedModel::Save(const std::string& file_name)
{
#ifndef SAIGA_USE_ASSIMP
    throw std::runtime_error("UnifiedModel::Save requires ASSIMP");
#else
    AssimpLoader al;
    al.SaveModel(*this, file_name);
#endif
}


UnifiedModel& UnifiedModel::transform(const mat4& T)
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

UnifiedModel& UnifiedModel::SetVertexColor(const vec4& c)
{
    color.resize(position.size());
    for (auto& co : color)
    {
        co = c;
    }
    return *this;
}

UnifiedModel& UnifiedModel::FlipNormals()
{
    for (auto& n : normal)
    {
        n = -n;
    }
    return *this;
}

UnifiedModel& UnifiedModel::FlatShading()
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
UnifiedModel& UnifiedModel::Normalize(float dimensions)
{
    auto box = BoundingBox();
    float s  = dimensions / box.maxSize();
    vec3 p   = box.getPosition();

    mat4 S = scale(vec3(s, s, s));
    mat4 T = translate(-p);



    return transform(S * T);
}

AABB UnifiedModel::BoundingBox() const
{
    AABB box;
    box.makeNegative();
    for (auto& p : position)
    {
        box.growBox(p);
    }
    return box;
}

std::vector<vec4> UnifiedModel::ComputeVertexColorFromMaterial() const
{
    std::vector<vec4> color;
    color.resize(position.size());

    for (auto& mg : material_groups)
    {
        for (auto i : mg.range())
        {
            for (auto k : triangles[i])
            {
                color[k] = materials[mg.materialId].color_diffuse;
            }
        }
    }
    return color;
}

std::vector<Triangle> UnifiedModel::TriangleSoup() const
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

void UnifiedModel::LocateTextures(const std::string& base)
{
    auto search = [base](std::string str) -> std::string {
        if (str.empty()) return "";
        std::string result;



        std::replace(str.begin(), str.end(), '\\', '/');


        // first search relative to the parent
        result = SearchPathes::model.getRelative(base, str);
        if (!result.empty()) return result;


        // no search in the image dir
        result = SearchPathes::image.getRelative(base, str);
        if (!result.empty()) return result;

        if (result.empty())
        {
            std::cout << "Could not find image " << str << std::endl;
            // throw std::runtime_error("File not found!");
        }
        return result;
    };

    for (auto& mat : materials)
    {
        mat.texture_diffuse  = search(mat.texture_diffuse);
        mat.texture_normal   = search(mat.texture_normal);
        mat.texture_bump     = search(mat.texture_bump);
        mat.texture_alpha    = search(mat.texture_alpha);
        mat.texture_emissive = search(mat.texture_emissive);
    }
}


UnifiedModel& UnifiedModel::CalculateVertexNormals()
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


template <>
std::vector<Vertex> UnifiedModel::VertexList() const
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
std::vector<VertexC> UnifiedModel::VertexList() const
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
    else if (HasMaterials())
    {
        auto color = ComputeVertexColorFromMaterial();
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
std::vector<VertexNC> UnifiedModel::VertexList() const
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
    else if (HasMaterials())
    {
        auto color = ComputeVertexColorFromMaterial();
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
std::vector<VertexNT> UnifiedModel::VertexList() const
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

template <>
std::vector<VertexNTD> UnifiedModel::VertexList() const
{
    SAIGA_ASSERT(HasPosition());
    SAIGA_ASSERT(HasTC());


    std::vector<VertexNTD> mesh;

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

    if (HasData())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].data = data[i];
        }
    }

    return mesh;
}


template <>
std::vector<BoneVertexCD> UnifiedModel::VertexList() const
{
    std::vector<BoneVertexCD> mesh;

    SAIGA_ASSERT(HasPosition());
    SAIGA_ASSERT(HasBones());


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }

    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].bone_info = bone_info[i];
        mesh[i].bone_info.normalizeWeights();
    }

    if (HasColor())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = color[i];
        }
    }
    else if (HasMaterials())
    {
        auto color = ComputeVertexColorFromMaterial();
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


std::ostream& operator<<(std::ostream& strm, const UnifiedModel& model)
{
    std::cout << "[UnifiedModel] " << model.name << "\n";
    std::cout << "  Vertices " << model.position.size() << " Triangles " << model.triangles.size() << "\n";
    std::cout << "  Bounding Box " << model.BoundingBox() << "\n";
    std::cout << "Materials\n";
    for (auto& m : model.materials)
    {
        strm << " " << m << std::endl;
    }

    return strm;
}


}  // namespace Saiga
