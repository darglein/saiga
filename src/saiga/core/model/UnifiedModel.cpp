/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UnifiedModel.h"

#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include "objModelLoader.h"
#include "plyModelLoader.h"

#ifdef SAIGA_USE_ASSIMP
#    include "saiga/core/model/assimpLoader.h"
#endif

namespace Saiga
{
std::ostream& operator<<(std::ostream& strm, const UnifiedMaterial& material)
{
    std::cout << "[Mat] " << std::setw(20) << material.name << material.color_diffuse.transpose()
              << ", tex: " << material.texture_diffuse;
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

UnifiedModel& UnifiedModel::Normalize()
{
    auto box = BoundingBox();
    float s  = 2.0 / box.maxSize();
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
            throw std::runtime_error("File not found!");
        }
        return result;
    };

    for (auto& mat : materials)
    {
        mat.texture_diffuse = search(mat.texture_diffuse);
        mat.texture_normal  = search(mat.texture_normal);
        mat.texture_bump    = search(mat.texture_bump);
        mat.texture_alpha   = search(mat.texture_alpha);
    }
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
