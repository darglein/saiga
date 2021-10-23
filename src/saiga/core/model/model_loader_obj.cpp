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
#include <sstream>

namespace Saiga
{
static StringViewParser lineParser = {"\t ,\n", true};
static StringViewParser faceParser = {"/", false};


struct ObjLine
{
    std::string key;
    std::vector<std::string> values;

    ObjLine() {}
    ObjLine(const std::string& line)
    {
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
inline ObjModelLoader::IndexedVertex2 parseIV(std::string_view line)
{
    ObjModelLoader::IndexedVertex2 iv;
#if 0
    std::vector<std::string> s = split(line, '/');
    if (s.size() > 0 && s[0].size() > 0) iv.v = std::atoi(s[0].c_str()) - 1;
    if (s.size() > 1 && s[1].size() > 0) iv.t = std::atoi(s[1].c_str()) - 1;
    if (s.size() > 2 && s[2].size() > 0) iv.n = std::atoi(s[2].c_str()) - 1;
#else
    faceParser.set(line);
    std::string_view tmp;
    tmp = faceParser.next();
    if (!tmp.empty()) iv.v = to_long(tmp) - 1;
    tmp = faceParser.next();
    if (!tmp.empty()) iv.t = to_long(tmp) - 1;
    tmp = faceParser.next();
    if (!tmp.empty()) iv.n = to_long(tmp) - 1;
#endif
    return iv;
}


ObjModelLoader::ObjModelLoader(const std::string& file) : file(file)
{
    loadFile(file);
}


bool ObjModelLoader::loadFile(const std::string& _file)
{
    this->file = SearchPathes::model(_file);
    if (file == "")
    {
        std::cerr << "Could not open file " << _file << std::endl;
        std::cerr << SearchPathes::model << std::endl;
        return false;
    }

    //    std::ifstream stream(file, std::ios::in);
    //    if (!stream.is_open())
    //    {
    //        throw std::runtime_error("Search Pathes broken!");
    //    }


    std::cout << "[ObjModelLoader] Loading " << file << std::endl;

    UnifiedMaterialGroup tg;
    tg.startFace = 0;
    tg.numFaces  = 0;
    out_model.material_groups.push_back(tg);

    std::vector<std::string> data;
    {
        //        SAIGA_BLOCK_TIMER();
        data = File::loadFileStringArray(file);
        File::removeWindowsLineEnding(data);
    }
    {
        //        SAIGA_BLOCK_TIMER();
        for (auto& line : data)
        {
            lineParser.set(line);
            parseLine();
        }
    }

    // finish last group
    UnifiedMaterialGroup& lastGroup = out_model.material_groups[out_model.material_groups.size() - 1];
    lastGroup.numFaces              = faces.size() - lastGroup.startFace;


    // remove groups with 0 faces
    out_model.material_groups.erase(std::remove_if(out_model.material_groups.begin(), out_model.material_groups.end(),
                                                   [](const UnifiedMaterialGroup& otg) { return otg.numFaces == 0; }),
                                    out_model.material_groups.end());

    std::cout << "[ObjModelLoader] Done.  "
              << "V " << vertices.size() << " N " << normals.size() << " T " << texCoords.size() << " F "
              << faces.size() << " Material Groups " << out_model.material_groups.size() << std::endl;



    //    std::cout<<"number of vertices "<<vertices.size()<<" number of faces "<<faces.size()<<endl;
    createVertexIndexList();
    //    std::cout<<"number of vertices "<<outVertices.size()<<" number of faces "<<outTriangles.size()<<endl;
    separateVerticesByGroup();
    //    std::cout<<"number of vertices "<<outVertices.size()<<" number of faces "<<outTriangles.size()<<endl;

    calculateMissingNormals();

    //    std::cout<<"objloader finished :)"<<endl;
    return true;
}

void ObjModelLoader::separateVerticesByGroup()
{
    // make sure faces from different triangle groups do not reference the same vertex
    // needs to be called after createVertexIndexList()
#if 0

    std::vector<int> vertexReference(outVertices.size(), INVALID_VERTEX_ID);

    for (int t = 0; t < (int)out_model.material_groups.size(); ++t)
    {
        UnifiedMaterialGroup& tg = out_model.material_groups[t];
        for (int i = 0; i < tg.numFaces; ++i)
        {
            ivec3& face = out_model.triangles[i + tg.startFace];

            for (int j = 0; j < 3; ++j)
            {
                int index = face[j];
                if (vertexReference[index] == INVALID_VERTEX_ID) vertexReference[index] = t;
                if (vertexReference[index] != t)
                {
                    // duplicate vertices
                    VertexNT v   = outVertices[index];
                    int newIndex = outVertices.size();
                    outVertices.push_back(v);
                    face[j] = newIndex;
                    vertexReference.push_back(t);
                    //                    SAIGA_ASSERT(0);
                }
            }
        }
    }
#endif
}

void ObjModelLoader::calculateMissingNormals()
{
#if 0
    for (auto tri : out_model.triangles)
    {
        auto& v1 = outVertices[tri[0]];
        auto& v2 = outVertices[tri[1]];
        auto& v3 = outVertices[tri[2]];

        vec3 normal = normalize(cross(make_vec3(v3.position - v1.position), make_vec3(v2.position - v1.position)));

        if (v1.normal == vec4(0, 0, 0, 0)) v1.normal = make_vec4(normal, 0);
        if (v2.normal == vec4(0, 0, 0, 0)) v2.normal = make_vec4(normal, 0);
        if (v3.normal == vec4(0, 0, 0, 0)) v3.normal = make_vec4(normal, 0);
    }
#endif
}

#if 0
void ObjModelLoader::computeVertexColorAndData()
{
    vertexColors.resize(outVertices.size(), make_vec4(1));
    vertexData.resize(outVertices.size(), make_vec4(0));

    for (UnifiedMaterialGroup& tg : out_model.material_groups)
    {
        for (int i = 0; i < tg.numFaces; ++i)
        {
            ivec3& face = out_model.triangles[i + tg.startFace];
            for (int f = 0; f < 3; ++f)
            {
                int index            = face[f];
                vertexColors[index]  = out_model.materials[tg.materialId].color_diffuse;
                float spec           = dot(out_model.materials[tg.materialId].color_specular, make_vec4(1)) / 4.0f;
                vertexData[index][0] = spec;
            }
        }
    }
}
void ObjModelLoader::toTriangleMesh(TriangleMesh<VertexNC, uint32_t>& mesh)
{
    if (vertexColors.empty())
    {
        computeVertexColorAndData();
    }

    SAIGA_ASSERT(vertexColors.size() == outVertices.size());
    mesh.faces.reserve( out_model.triangles.size());
    for (ivec3& oj :  out_model.triangles)
    {
        mesh.addFace(oj(0), oj(1), oj(2));
    }

    mesh.vertices.reserve(outTriangles.size());
    for (unsigned int i = 0; i < outVertices.size(); ++i)
    {
        auto& v = outVertices[i];
        VertexNC vn;
        vn.position = v.position;
        vn.normal   = v.normal;
        vn.color    = vertexColors[i];
        vn.data     = vertexData[i];
        mesh.addVertex(vn);
    }
}

void ObjModelLoader::toTriangleMesh(TriangleMesh<VertexNTD, uint32_t>& mesh)
{
    //    SAIGA_ASSERT(texCoords.size() == outVertices.size());
    if (vertexColors.empty())
    {
        computeVertexColorAndData();
    }
    SAIGA_ASSERT(vertexData.size() == outVertices.size());

    mesh.faces.reserve(outTriangles.size());
    for (ivec3& oj : outTriangles)
    {
        mesh.addFace(oj(0), oj(1), oj(2));
    }

    mesh.vertices.reserve(outTriangles.size());
    for (unsigned int i = 0; i < outVertices.size(); ++i)
    {
        auto& v = outVertices[i];
        VertexNTD vn;
        vn.position = v.position;
        vn.normal   = v.normal;
        vn.texture  = v.texture;
        vn.data     = vertexData[i];
        mesh.addVertex(vn);
    }
}
#endif

void ObjModelLoader::createVertexIndexList()
{
#if 0
    std::vector<bool> vertices_used(vertices.size(), false);

    outVertices.resize(vertices.size());

    for (std::vector<IndexedVertex2>& f : faces)
    {
        ivec3 fa;
        for (int i = 0; i < 3; i++)
        {
            IndexedVertex2& currentVertex = f[i];
            int vert                      = currentVertex.v;
            int norm                      = currentVertex.n;
            int tex                       = currentVertex.t;

            VertexNT verte;
            if (vert >= 0)
            {
                SAIGA_ASSERT(vert < (int)vertices.size());
                verte.position = make_vec4(vertices[vert], 1);
            }
            if (norm >= 0)
            {
                SAIGA_ASSERT(norm < (int)normals.size());
                verte.normal = make_vec4(normals[norm], 0);
            }
            if (tex >= 0)
            {
                SAIGA_ASSERT(tex < (int)texCoords.size());
                verte.texture = texCoords[tex];
            }


            int index = INVALID_VERTEX_ID;
            if (vertices_used[vert])
            {
                if (verte == outVertices[vert])
                {
                    index = vert;
                }
            }
            else
            {
                outVertices[vert]   = verte;
                index               = vert;
                vertices_used[vert] = true;
            }

            if (index == INVALID_VERTEX_ID)
            {
                index = outVertices.size();
                outVertices.push_back(verte);
            }
            fa[i] = index;
        }

        out_model.triangles.push_back(fa);
    }
#endif
}

std::vector<std::vector<ObjModelLoader::IndexedVertex2>> ObjModelLoader::triangulateFace(
    const std::vector<IndexedVertex2>& f)
{
    std::vector<std::vector<IndexedVertex2>> newFaces;

    // more than 3 indices -> triangulate
    std::vector<IndexedVertex2> face;
    int cornerCount = 1;
    IndexedVertex2 startVertex, lastVertex;
    for (const IndexedVertex2& currentVertex : f)
    {
        if (cornerCount <= 3)
        {
            if (cornerCount == 1) startVertex = currentVertex;
            face.push_back(currentVertex);
            if (cornerCount == 3) newFaces.push_back(face);
        }
        else
        {
            face.resize(3);
            face[0] = lastVertex;
            face[1] = currentVertex;
            face[2] = startVertex;
            newFaces.push_back(face);
        }

        lastVertex = currentVertex;
        cornerCount++;
    }
    return newFaces;
}

void ObjModelLoader::parseLine()
{
    auto header = lineParser.next();

    if (header == "#")
    {
    }
    else if (header == "usemtl")
    {
        // finish current group and create new one
        if (!out_model.material_groups.empty())
        {
            UnifiedMaterialGroup& currentGroup = out_model.material_groups[out_model.material_groups.size() - 1];
            currentGroup.numFaces              = faces.size() - currentGroup.startFace;
        }
        UnifiedMaterialGroup newGroup;
        newGroup.startFace = faces.size();

        //         newGroup.material  = materialLoader.getMaterial(std::string(lineParser.next()));
        //        newGroup.materialId = materialLoader.getMaterialId(std::string(lineParser.next()));

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
        newGroup.materialId = mtl_id;

        out_model.material_groups.push_back(newGroup);
    }
    else if (header == "mtllib")
    {
        FileChecker fc;

        std::string mtl_file = fc.getRelative(file, std::string(lineParser.next()));

        out_model.materials = LoadMTL(mtl_file);
    }
    else if (header == "g")
    {
    }
    else if (header == "o")
    {
    }
    else if (header == "s")
    {
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
        parseF();
    }
}



void ObjModelLoader::parseF()
{
    ivs.clear();

    std::string_view f = lineParser.next();
    while (!f.empty())
    {
        ivs.push_back(parseIV(f));
        f = lineParser.next();
    }

    auto nf = triangulateFace(ivs);

    // relative indexing, when the index is negativ
    for (auto& f : nf)
    {
        for (auto& i : f)
        {
            if (i.v < 0 && i.v != INVALID_VERTEX_ID)
            {
                int absIndex = vertices.size() + i.v + 1;
                i.v          = absIndex;
            }
            if (i.n < 0 && i.n != INVALID_VERTEX_ID)
            {
                int absIndex = normals.size() + i.n + 1;
                i.n          = absIndex;
            }
            if (i.t < 0 && i.t != INVALID_VERTEX_ID)
            {
                int absIndex = texCoords.size() + i.t + 1;
                i.t          = absIndex;
            }
        }
    }

    faces.insert(faces.end(), nf.begin(), nf.end());
}



}  // namespace Saiga
