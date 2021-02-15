/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "objModelLoader.h"

#include "saiga/core/math/String.h"
#include "saiga/core/time/all.h"
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
ObjMaterialLoader::ObjMaterialLoader(const std::string& file) : file(file)
{
    loadFile(file);
}

bool ObjMaterialLoader::loadFile(const std::string& _file)
{
    file = _file;
    std::ifstream stream(file, std::ios::in);
    if (!stream.is_open())
    {
        return false;
    }


    std::cout << "ObjMaterialLoader: loading file " << file << std::endl;


    while (!stream.eof())
    {
        std::string line;
        std::getline(stream, line);
        // remove carriage return from windows
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        parseLine(line);
    }
    return true;
}

UnifiedMaterial ObjMaterialLoader::getMaterial(const std::string& name)
{
    for (UnifiedMaterial& m : materials)
    {
        if (m.name == name) return m;
    }
    std::cout << "Warning material '" << name << "' not found!" << std::endl;
    return UnifiedMaterial("default");
}


void ObjMaterialLoader::parseLine(const std::string& line)
{
    std::stringstream sstream(line);

    std::string header;
    sstream >> header;

    std::string rest;
    std::getline(sstream, rest);

    // remove first white space
    if (rest[0] == ' ' && rest.size() > 1)
    {
        rest = rest.substr(1);
    }

    std::stringstream restStream(rest);

    if (header == "newmtl")
    {
        UnifiedMaterial m(rest);
        materials.push_back(m);
        currentMaterial = &materials[materials.size() - 1];
    }
    if (currentMaterial == nullptr)
    {
        return;
    }

    //    if(header == "Kd"){
    //        restStream >> currentMaterial->color;
    //    }

#if 0
    if (header == "Ns")
    {
        restStream >> currentMaterial->Ns;
    }
    else if (header == "Ni")
    {
        restStream >> currentMaterial->Ni;
    }
    else if (header == "d")
    {
        restStream >> currentMaterial->d;
    }
    else if (header == "Tr")
    {
        restStream >> currentMaterial->Tr;
    }
    else if (header == "Tf")
    {
        restStream >> currentMaterial->Tf;
    }
    else if (header == "illum")
    {
        restStream >> currentMaterial->illum;
    }
#endif
    if (header == "Ka")
    {
        restStream >> currentMaterial->color_ambient;
    }
    else if (header == "Kd")
    {
        restStream >> currentMaterial->color_diffuse;
    }
    else if (header == "Ks")
    {
        restStream >> currentMaterial->color_specular;
    }
    else if (header == "Ke")
    {
        restStream >> currentMaterial->color_emissive;
    }
    else if (header == "map_Kd")
    {
        currentMaterial->texture_diffuse = getPathImage(rest);
    }
    else if (header == "map_d")
    {
        //        TextureParameters tp;
        //        tp.srgb = false;
        currentMaterial->texture_alpha = getPathImage(rest);
        //        if(currentMaterial->map_d) currentMaterial->map_d->setWrap(GL_REPEAT);
    }
    else if (header == "map_bump" || header == "bump")
    {
        //        TextureParameters tp;
        //        tp.srgb = false;
        currentMaterial->texture_normal = getPathImage(rest);
        //        if(currentMaterial->map_bump) currentMaterial->map_bump->setWrap(GL_REPEAT);
    }
}

std::string ObjMaterialLoader::getPathImage(const std::string& str)
{
    std::string result;

    // first search relative to the parent
    result = SearchPathes::model.getRelative(file, str);
    if (!result.empty()) return result;


    // no search in the image dir
    result = SearchPathes::image.getRelative(file, str);
    if (!result.empty()) return result;

    std::cout << str << " " << file << " " << result << std::endl;
    SAIGA_ASSERT(!result.empty());
    return result;
}


static StringViewParser lineParser = {" ,\n", true};
static StringViewParser faceParser = {"/", false};


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

    ObjTriangleGroup tg;
    tg.startFace = 0;
    tg.faces     = 0;
    triangleGroups.push_back(tg);

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
    ObjTriangleGroup& lastGroup = triangleGroups[triangleGroups.size() - 1];
    lastGroup.faces             = faces.size() - lastGroup.startFace;


    // remove groups with 0 faces
    triangleGroups.erase(std::remove_if(triangleGroups.begin(), triangleGroups.end(),
                                        [](const ObjTriangleGroup& otg) { return otg.faces == 0; }),
                         triangleGroups.end());

    std::cout << "[ObjModelLoader] Done.  "
              << "V " << vertices.size() << " N " << normals.size() << " T " << texCoords.size() << " F "
              << faces.size() << " Material Groups " << triangleGroups.size() << std::endl;



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

    std::vector<int> vertexReference(outVertices.size(), INVALID_VERTEX_ID);

    for (int t = 0; t < (int)triangleGroups.size(); ++t)
    {
        ObjTriangleGroup& tg = triangleGroups[t];
        for (int i = 0; i < tg.faces; ++i)
        {
            ivec3& face = outTriangles[i + tg.startFace];

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
}

void ObjModelLoader::calculateMissingNormals()
{
    for (auto tri : outTriangles)
    {
        auto& v1 = outVertices[tri[0]];
        auto& v2 = outVertices[tri[1]];
        auto& v3 = outVertices[tri[2]];

        vec3 normal = normalize(cross(make_vec3(v3.position - v1.position), make_vec3(v2.position - v1.position)));

        if (v1.normal == vec4(0, 0, 0, 0)) v1.normal = make_vec4(normal, 0);
        if (v2.normal == vec4(0, 0, 0, 0)) v2.normal = make_vec4(normal, 0);
        if (v3.normal == vec4(0, 0, 0, 0)) v3.normal = make_vec4(normal, 0);
    }
}

void ObjModelLoader::computeVertexColorAndData()
{
    vertexColors.resize(outVertices.size(), make_vec4(1));
    vertexData.resize(outVertices.size(), make_vec4(0));

    for (ObjTriangleGroup& tg : triangleGroups)
    {
        for (int i = 0; i < tg.faces; ++i)
        {
            ivec3& face = outTriangles[i + tg.startFace];
            for (int f = 0; f < 3; ++f)
            {
                int index            = face[f];
                vertexColors[index]  = tg.material.color_diffuse;
                float spec           = dot(tg.material.color_specular, make_vec4(1)) / 4.0f;
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
    mesh.faces.reserve(outTriangles.size());
    for (ivec3& oj : outTriangles)
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

void ObjModelLoader::createVertexIndexList()
{
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

        outTriangles.push_back(fa);
    }
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
        if (!triangleGroups.empty())
        {
            ObjTriangleGroup& currentGroup = triangleGroups[triangleGroups.size() - 1];
            currentGroup.faces             = faces.size() - currentGroup.startFace;
        }
        ObjTriangleGroup newGroup;
        newGroup.startFace = faces.size();
        newGroup.material  = materialLoader.getMaterial(std::string(lineParser.next()));
        triangleGroups.push_back(newGroup);
    }
    else if (header == "mtllib")
    {
        FileChecker fc;
        materialLoader.loadFile(fc.getRelative(file, std::string(lineParser.next())));
    }
    else if (header == "g")
    {
        //        std::cout<<"Found Group: "<<line<<endl;
    }
    else if (header == "o")
    {
        //        std::cout<<"Found Object: "<<line<<endl;
    }
    else if (header == "s")
    {
        // smooth shading
    }
    else if (header == "v")
    {
        //        parseV(rest);
        vec3 v;
        v(0) = to_double(lineParser.next());
        v(1) = to_double(lineParser.next());
        v(2) = to_double(lineParser.next());
        vertices.push_back(v);
        //        to_double()
    }
    else if (header == "vt")
    {
        //        parseVT(rest);
        vec2 v;
        v(0) = to_double(lineParser.next());
        v(1) = to_double(lineParser.next());
        texCoords.push_back(v);
    }
    else if (header == "vn")
    {
        //        parseVN(rest);
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
