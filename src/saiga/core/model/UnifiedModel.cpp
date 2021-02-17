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

namespace Saiga
{
UnifiedModel::UnifiedModel(const std::string& file_name)
{
    std::cout << "Loading Unified Model " << file_name << std::endl;

    auto full_file = SearchPathes::model(file_name);
    if (full_file.empty())
    {
        throw std::runtime_error("Could not open file " + file_name);
    }

    std::string type = fileEnding(file_name);

    if (type == "obj")
    {
        std::cout << "load obj" << std::endl;
        ObjModelLoader loader(full_file);

        *this = loader.out_model;

        for (auto& v : loader.outVertices)
        {
            position.push_back(v.position.head<3>());
            normal.push_back(v.normal.head<3>());
            texture_coordinates.push_back(v.texture);
        }

        //        for (auto& c : loader.vertexColors)
        //        {
        //            color.push_back(c);
        //        }

        //        triangles = loader.out_model.triangles;
        //        for (auto& f : loader.outTriangles)
        //        {
        //            triangles.push_back(f);
        //        }

        //        materials       = loader.out_model.materials;
        //        material_groups = loader.out_model.material_groups;

        //        for (int i = 0; i < loader.triangleGroups.size(); ++i)
        //        {
        //            auto& tg = loader.triangleGroups[i];
        //            //            materials.push_back(tg.material);

        //            UnifiedMaterialGroup umg;
        //            umg.startFace  = tg.startFace;
        //            umg.numFaces   = tg.numFaces;
        //            umg.materialId = i;
        //            material_groups.push_back(umg);
        //        }
    }
    else
    {
        throw std::runtime_error("Unknown model file format " + to_string(type));
    }


    std::cout << "type " << type << std::endl;
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
        std::cout << "  " << m.name << ", " << m.color_diffuse.transpose() << ", tex: " << m.texture_diffuse << "\n";
    }

    return strm;
}

}  // namespace Saiga
