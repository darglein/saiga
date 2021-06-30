/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UnifiedModel.h"

#include "saiga/core/image/templatedImage.h"
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

    if (type == "obj" && false)
    {
#if 0
        ObjModelLoader loader(full_file);
        *this = loader.out_model;
        for (auto& v : loader.outVertices)
        {
            position.push_back(v.position.head<3>());
            normal.push_back(v.normal.head<3>());
            texture_coordinates.push_back(v.texture);
        }
        LocateTextures(full_file);
#endif
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
#if 0
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
#endif



void UnifiedModel::LocateTextures(const std::string& base)
{
    auto get_embedded_id = [base](std::string str) -> int {
        if (str.size() < 2) return -1;
        if (str.front() == '*')
        {
            auto remaining = str.substr(1);
            return to_int(remaining);
        }
        return -1;
    };

    auto search = [this, base](std::string str) -> std::string {
        if (str.empty()) return "";
        if (texture_name_to_id.count(str) > 0) return str;

        std::replace(str.begin(), str.end(), '\\', '/');

        // first search relative to the parent
        std::string result;
        result = SearchPathes::model.getRelative(base, str);

        // no search in the image dir
        if (result.empty()) result = SearchPathes::image.getRelative(base, str);

        if (result.empty())
        {
            std::cout << "Could not find image " << str << std::endl;
            // throw std::runtime_error("File not found!");
        }
        else
        {
            std::cout << "load " << result << std::endl;
            Image img(result);
            texture_name_to_id[result] = textures.size();
            textures.push_back(img);
        }

        return result;
    };

    for (int i = 0; i < textures.size(); ++i)
    {
        texture_name_to_id["*" + std::to_string(i)] = i;
    }

    // std::cout << "Embedded Textures " << textures.size() << std::endl;
    for (auto& mat : materials)
    {
        mat.texture_diffuse  = search(mat.texture_diffuse);
        mat.texture_normal   = search(mat.texture_normal);
        mat.texture_bump     = search(mat.texture_bump);
        mat.texture_alpha    = search(mat.texture_alpha);
        mat.texture_emissive = search(mat.texture_emissive);
    }
    // std::cout << "Total Textures " << textures.size() << std::endl;
}



std::ostream& operator<<(std::ostream& strm, const UnifiedModel& model)
{
    std::cout << "[UnifiedModel] " << model.name << "\n";
    // std::cout << "  Vertices " << model.position.size() << " Triangles " << model.triangles.size() << "\n";
    // std::cout << "  Bounding Box " << model.BoundingBox() << "\n";
    std::cout << "Materials\n";
    for (auto& m : model.materials)
    {
        strm << " " << m << std::endl;
    }

    return strm;
}
UnifiedModel& UnifiedModel::Normalize(float dimensions)
{
    AABB total_aabb;
    total_aabb.makeNegative();
    for (auto& m : mesh)
    {
        total_aabb.growBox(m.BoundingBox());
    }

    float s = dimensions / total_aabb.maxSize();
    vec3 p  = total_aabb.getPosition();

    mat4 S = scale(vec3(s, s, s));
    mat4 T = translate(-p);

    mat4 trans = S * T;


    for (auto& m : mesh)
    {
        m.transform(trans);
    }
    return *this;
}


UnifiedModel& UnifiedModel::AddMissingDummyTextures()
{
    std::cout << "AddMissingDummyTextures" << std::endl;
    //    textures
    bool need_dummy = false;
    for (auto& m : materials)
    {
        if (m.texture_diffuse.empty())
        {
            std::cout << "Add dummy for " << m.name << std::endl;
            m.texture_diffuse                     = "dummy";
            texture_name_to_id[m.texture_diffuse] = textures.size();
            need_dummy                            = true;
        }
    }

    if (need_dummy)
    {
        TemplatedImage<ucvec4> dummy(10, 10);
        dummy.getImageView().set(ucvec4(100, 100, 100, 255));
        textures.push_back(dummy);
    }
    return *this;
}

std::pair<UnifiedMesh, std::vector<UnifiedMaterialGroup>> UnifiedModel::CombinedMesh(int vertex_flags) const
{
    SAIGA_ASSERT(vertex_flags & VERTEX_POSITION);

    UnifiedMesh combined;
    std::vector<UnifiedMaterialGroup> groups;

    combined.triangles.reserve(TotalTriangles());

    int nv = TotalVertices();
    if (vertex_flags & VERTEX_POSITION) combined.position.reserve(nv);
    if (vertex_flags & VERTEX_NORMAL) combined.normal.reserve(nv);
    if (vertex_flags & VERTEX_COLOR) combined.color.reserve(nv);
    if (vertex_flags & VERTEX_TEXTURE_COORDINATES) combined.texture_coordinates.reserve(nv);
    if (vertex_flags & VERTEX_EXTRA_DATA) combined.data.reserve(nv);
    if (vertex_flags & VERTEX_BONE_INFO) combined.bone_info.reserve(nv);

    for (auto& m : mesh)
    {
        UnifiedMaterialGroup umg;
        umg.numFaces   = m.NumFaces();
        umg.startFace  = combined.NumFaces();
        umg.materialId = m.material_id;
        groups.push_back(umg);

        for (auto t : m.triangles)
        {
            t(0) += combined.NumVertices();
            t(1) += combined.NumVertices();
            t(2) += combined.NumVertices();
            combined.triangles.push_back(t);
        }
        for (auto t : m.lines)
        {
            t(0) += combined.NumVertices();
            t(1) += combined.NumVertices();
            combined.lines.push_back(t);
        }

        if (vertex_flags & VERTEX_POSITION)
        {
            SAIGA_ASSERT(m.HasPosition());
            combined.position.insert(combined.position.end(), m.position.begin(), m.position.end());
        }
        if (vertex_flags & VERTEX_NORMAL)
        {
            SAIGA_ASSERT(m.HasNormal());
            combined.normal.insert(combined.normal.end(), m.normal.begin(), m.normal.end());
        }
        if (vertex_flags & VERTEX_COLOR)
        {
            SAIGA_ASSERT(m.HasColor(),
                         "Missing vertex color. Call UnifiedModel::ComputeColor() before creating the asset.");
            combined.color.insert(combined.color.end(), m.color.begin(), m.color.end());
        }
        if (vertex_flags & VERTEX_TEXTURE_COORDINATES)
        {
            SAIGA_ASSERT(m.HasTC());
            combined.texture_coordinates.insert(combined.texture_coordinates.end(), m.texture_coordinates.begin(),
                                                m.texture_coordinates.end());
        }
        if (vertex_flags & VERTEX_EXTRA_DATA)
        {
            SAIGA_ASSERT(m.HasData());
            combined.data.insert(combined.data.end(), m.data.begin(), m.data.end());
        }
        if (vertex_flags & VERTEX_BONE_INFO)
        {
            SAIGA_ASSERT(m.HasBones());
            combined.bone_info.insert(combined.bone_info.end(), m.bone_info.begin(), m.bone_info.end());
        }
    }

    return {combined, groups};
}
UnifiedModel& UnifiedModel::ComputeColor()
{
    for (auto& m : mesh)
    {
        if (m.HasColor()) continue;

        auto& mat = materials[m.material_id];


        if (!m.HasTC() || texture_name_to_id.count(mat.texture_diffuse) == 0)
        {
            // don't use texture color
            m.color.resize(m.NumVertices());
            for (auto& c : m.color)
            {
                c = mat.color_diffuse;
            }
        }
        else
        {
            // sample color from texture
            Image& img = textures[texture_name_to_id[mat.texture_diffuse]];
            m.color.resize(m.NumVertices());

            for (int i = 0; i < m.NumVertices(); ++i)
            {
                vec2 tc = m.texture_coordinates[i];

                vec4 c = img.texture(tc);

                m.color[i] = c;
            }
        }
    }
    return *this;
}

UnifiedModel& UnifiedModel::SetVertexColor(const vec4& color)
{
    for (auto& m : mesh)
    {
        m.SetVertexColor(color);
    }
    return *this;
}
AABB UnifiedModel::BoundingBox() const
{
    AABB combined;
    combined.makeNegative();
    for (auto& m : mesh)
    {
        combined.growBox(m.BoundingBox());
    }
    return combined;
}

}  // namespace Saiga
