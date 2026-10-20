/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "model_loader_obj.h"

#include "saiga/core/math/String.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>

namespace Saiga
{
static constexpr std::string_view line_delims = "\t ,\n";


struct ObjLine
{
    std::string key;
    std::vector<std::string> values;

    ObjLine() {}
    ObjLine(const std::string& line)
    {
        StringViewParser lineParser(line_delims, true);
        lineParser.set(line);
        key = lineParser.next();

        while (true)
        {
            auto value = lineParser.next();
            if (!value.empty())
            {
                values.push_back(std::string(value));
            }
            else
            {
                break;
            }
        }
    }
};



std::vector<UnifiedMaterial> LoadMTL(const std::string& file)
{
    std::vector<UnifiedMaterial> materials;
    UnifiedMaterial* currentMaterial = nullptr;

    std::cout << "ObjMaterialLoader: loading file " << file << std::endl;
    auto file_strs = File::loadFileStringArray(file);
    File::removeWindowsLineEnding(file_strs);
    for (auto& raw_line : file_strs)
    {
        //        parseLine(line);
        ObjLine line(raw_line);

        if (line.key == "newmtl")
        {
            SAIGA_ASSERT(line.values.size() == 1);
            UnifiedMaterial m(line.values.front());
            materials.push_back(m);
            currentMaterial = &materials[materials.size() - 1];
        }
        if (currentMaterial == nullptr)
        {
            continue;
        }

        if (line.key == "Ka")
        {
            for (int i = 0; i < std::min<int>(4, line.values.size()); ++i)
            {
                currentMaterial->color_ambient(i) = to_double(line.values[i]);
            }
        }
        else if (line.key == "Kd")
        {
            for (int i = 0; i < std::min<int>(4, line.values.size()); ++i)
            {
                currentMaterial->color_diffuse(i) = to_double(line.values[i]);
            }
        }
        else if (line.key == "Ks")
        {
            for (int i = 0; i < std::min<int>(4, line.values.size()); ++i)
            {
                currentMaterial->color_specular(i) = to_double(line.values[i]);
            }
        }
        else if (line.key == "Ke")
        {
            for (int i = 0; i < std::min<int>(4, line.values.size()); ++i)
            {
                currentMaterial->color_emissive(i) = to_double(line.values[i]);
            }
        }
        else if (line.key == "map_Kd")
        {
            // SAIGA_ASSERT(line.values.size() == 1);
            currentMaterial->texture_diffuse = line.values.back();
        }
        else if (line.key == "map_d")
        {
            SAIGA_ASSERT(line.values.size() == 1);
            currentMaterial->texture_alpha = line.values.front();
        }
        else if (line.key == "map_bump" || line.key == "bump")
        {
            SAIGA_ASSERT(line.values.size() == 1);
            currentMaterial->texture_bump = line.values.front();
        }
    }
    return materials;
}



// parsing index vertex
// examples:
// v1/vt1/vn1        12/51/1
// v1//vn1           51//4
// The returned indices are zero based, relative (negative) indices are resolved in addFace.
static ObjModelLoader::IndexedVertex2 parseIV(std::string_view line)
{
    ObjModelLoader::IndexedVertex2 iv;
    StringViewParser faceParser("/", false);
    faceParser.set(line);
    std::string_view tmp;
    tmp = faceParser.next();
    if (!tmp.empty()) iv.v = to_long(tmp) - 1;
    tmp = faceParser.next();
    if (!tmp.empty()) iv.t = to_long(tmp) - 1;
    tmp = faceParser.next();
    if (!tmp.empty()) iv.n = to_long(tmp) - 1;
    return iv;
}


ObjModelLoader::ObjModelLoader(const std::string& file) : file(file)
{
    loadFile(file);
}


bool ObjModelLoader::loadFile(const std::string& _file)
{
    this->file = SearchPathes::model(_file).string();
    if (file == "")
    {
        std::cerr << "Could not open file " << _file << std::endl;
        std::cerr << SearchPathes::model << std::endl;
        return false;
    }

    out_model = UnifiedModel();
    vertices.clear();
    normals.clear();
    texCoords.clear();
    faces.clear();
    material_groups.clear();

    UnifiedMaterialGroup tg;
    tg.startFace  = 0;
    tg.numFaces   = 0;
    tg.materialId = -1;
    material_groups.push_back(tg);

    std::vector<std::string> data = File::loadFileStringArray(file);
    File::removeWindowsLineEnding(data);
    for (auto& line : data)
    {
        parseLine(line);
    }

    // finish last group
    UnifiedMaterialGroup& lastGroup = material_groups.back();
    lastGroup.numFaces              = faces.size() - lastGroup.startFace;

    createMeshes();
    return true;
}

void ObjModelLoader::createMeshes()
{
    auto check_index = [this](int index, size_t size, const char* name)
    {
        if (index < 0 || index >= (int)size)
        {
            throw std::runtime_error("Invalid " + std::string(name) + " index in OBJ file " + file);
        }
    };

    for (auto& group : material_groups)
    {
        if (group.numFaces == 0) continue;

        int material_id = group.materialId;
        if (material_id < 0)
        {
            // Faces without (or with an unknown) material get a default material, so that every mesh has a valid
            // material id like the meshes of the other loaders.
            auto it = std::find_if(out_model.materials.begin(), out_model.materials.end(),
                                   [](const UnifiedMaterial& m) { return m.name == "default"; });
            if (it == out_model.materials.end())
            {
                UnifiedMaterial m("default");
                m.color_diffuse = vec4(1, 1, 1, 1);
                out_model.materials.push_back(m);
                it = out_model.materials.end() - 1;
            }
            material_id = it - out_model.materials.begin();
        }

        UnifiedMesh mesh;
        mesh.material_id = material_id;

        bool has_tc = false, has_normal = false, missing_normal = false;
        for (int f = group.startFace; f < group.startFace + group.numFaces; ++f)
        {
            for (auto& iv : faces[f])
            {
                has_tc |= iv.t != INVALID_VERTEX_ID;
                has_normal |= iv.n != INVALID_VERTEX_ID;
                missing_normal |= iv.n == INVALID_VERTEX_ID;
            }
        }

        // One output vertex per distinct (position, texture, normal) index triple.
        std::map<std::tuple<int, int, int>, int> vertex_map;
        std::vector<bool> normal_given;
        for (int f = group.startFace; f < group.startFace + group.numFaces; ++f)
        {
            ivec3 tri;
            for (int i = 0; i < 3; ++i)
            {
                const IndexedVertex2& iv = faces[f][i];
                auto key                 = std::make_tuple(iv.v, iv.t, iv.n);
                auto it                  = vertex_map.find(key);
                if (it != vertex_map.end())
                {
                    tri(i) = it->second;
                    continue;
                }

                check_index(iv.v, vertices.size(), "vertex");
                mesh.position.push_back(vertices[iv.v]);
                if (has_tc)
                {
                    vec2 tc = vec2(0, 0);
                    if (iv.t != INVALID_VERTEX_ID)
                    {
                        check_index(iv.t, texCoords.size(), "texture coordinate");
                        tc = texCoords[iv.t];
                    }
                    mesh.texture_coordinates.push_back(tc);
                }
                vec3 n = vec3(0, 0, 0);
                if (iv.n != INVALID_VERTEX_ID)
                {
                    check_index(iv.n, normals.size(), "normal");
                    n = normals[iv.n];
                }
                mesh.normal.push_back(n);
                normal_given.push_back(iv.n != INVALID_VERTEX_ID);

                tri(i)          = mesh.NumVertices() - 1;
                vertex_map[key] = tri(i);
            }
            mesh.triangles.push_back(tri);
        }

        if (missing_normal)
        {
            // Compute the area weighted normals and use them only for the vertices without a given normal.
            std::vector<vec3> given_normals = mesh.normal;
            mesh.CalculateVertexNormals();
            if (has_normal)
            {
                for (int i = 0; i < mesh.NumVertices(); ++i)
                {
                    if (normal_given[i]) mesh.normal[i] = given_normals[i];
                }
            }
        }

        out_model.mesh.push_back(mesh);
    }
}

std::vector<std::vector<ObjModelLoader::IndexedVertex2>> ObjModelLoader::triangulateFace(
    const std::vector<IndexedVertex2>& f)
{
    // Fan triangulation (0, 1, 2), (0, 2, 3), ... keeps the orientation of the polygon.
    std::vector<std::vector<IndexedVertex2>> newFaces;
    for (size_t i = 2; i < f.size(); ++i)
    {
        newFaces.push_back({f[0], f[i - 1], f[i]});
    }
    return newFaces;
}

void ObjModelLoader::parseLine(std::string_view line)
{
    StringViewParser lineParser(line_delims, true);
    lineParser.set(line);
    auto header = lineParser.next();

    if (header == "usemtl")
    {
        // finish current group and create new one
        UnifiedMaterialGroup& currentGroup = material_groups.back();
        currentGroup.numFaces              = faces.size() - currentGroup.startFace;

        UnifiedMaterialGroup newGroup;
        newGroup.startFace = faces.size();

        std::string mtl_name = std::string(lineParser.next());
        int mtl_id           = -1;
        for (size_t i = 0; i < out_model.materials.size(); ++i)
        {
            if (out_model.materials[i].name == mtl_name)
            {
                mtl_id = i;
                break;
            }
        }
        if (mtl_id == -1)
        {
            std::cerr << "[ObjModelLoader] Unknown material " << mtl_name << " in " << file << std::endl;
        }
        newGroup.materialId = mtl_id;

        material_groups.push_back(newGroup);
    }
    else if (header == "mtllib")
    {
        FileChecker fc;
        for (auto name = lineParser.next(); !name.empty(); name = lineParser.next())
        {
            std::string mtl_file = fc.getRelative(file, std::string(name)).string();
            if (mtl_file.empty())
            {
                std::cerr << "[ObjModelLoader] Could not find material file " << name << std::endl;
                continue;
            }
            auto mats = LoadMTL(mtl_file);
            out_model.materials.insert(out_model.materials.end(), mats.begin(), mats.end());
        }
    }
    else if (header == "v")
    {
        vec3 v;
        v(0) = to_double(lineParser.next());
        v(1) = to_double(lineParser.next());
        v(2) = to_double(lineParser.next());
        vertices.push_back(v);
    }
    else if (header == "vt")
    {
        vec2 v;
        v(0) = to_double(lineParser.next());
        v(1) = to_double(lineParser.next());
        texCoords.push_back(v);
    }
    else if (header == "vn")
    {
        vec3 v;
        v(0) = to_double(lineParser.next());
        v(1) = to_double(lineParser.next());
        v(2) = to_double(lineParser.next());
        normals.push_back(v);
    }
    else if (header == "f")
    {
        std::vector<IndexedVertex2> ivs;
        for (auto f = lineParser.next(); !f.empty(); f = lineParser.next())
        {
            ivs.push_back(parseIV(f));
        }
        addFace(std::move(ivs));
    }
}



void ObjModelLoader::addFace(std::vector<IndexedVertex2> ivs)
{
    // Relative indexing: the OBJ index -k (stored as -k-1) refers to the k-th last element read so far.
    // The OBJ index 0 is invalid. It is stored as -1 and rejected as out of range in createMeshes.
    auto resolve = [](int& index, size_t size)
    {
        if (index < -1 && index != INVALID_VERTEX_ID) index = (int)size + index + 1;
    };
    for (auto& i : ivs)
    {
        resolve(i.v, vertices.size());
        resolve(i.n, normals.size());
        resolve(i.t, texCoords.size());
    }

    auto nf = triangulateFace(ivs);
    faces.insert(faces.end(), nf.begin(), nf.end());
}



}  // namespace Saiga
