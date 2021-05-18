/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "model_loader_ply.h"

#include "saiga/core/util/color.h"
#include "saiga/core/util/fileChecker.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>

namespace Saiga
{
PLYLoader::PLYLoader(const std::string& _file)
{
    auto file = SearchPathes::model(_file);
    std::ifstream stream(file, std::ios::binary);

    if (!stream.is_open())
    {
        std::cerr << "Could not open file " << file << std::endl;
        throw std::runtime_error("invalid file: " + file + ", " + _file);
    }

    // open the file:
    //         std::ifstream file(filename, std::ios::binary);

    // read the data:
    data = std::vector<char>((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    //         std::cout << "filesize: " << data.size() << std::endl;

    parseHeader();
    parseMeshBinary();
}

int PLYLoader::sizeoftype(std::string t)
{
    if (t == "float" || t == "int")
    {
        return 4;
    }
    if (t == "uchar")
    {
        return 1;
    }
    SAIGA_ASSERT(0);
    return -1;
}

void PLYLoader::parseHeader()
{
    // Note: the header is always in ascii and ends with the string end_header
    std::string str2(data.begin(), data.end());
    auto pos = str2.find("end_header");

    SAIGA_ASSERT(pos != std::string::npos);

    dataStart = pos;
    // go until next newline
    while (data[dataStart] != '\n')
    {
        dataStart++;
    }
    dataStart++;

    std::string header(str2.begin(), str2.begin() + pos);
    header.erase(std::remove(header.begin(), header.end(), '\r'), header.end());

    std::vector<std::string> headerLines = split(header, '\n');

    SAIGA_ASSERT(headerLines[0] == "ply");

    int elementStatus = -1;
    for (auto l : headerLines)
    {
        //        std::cout << l << std::endl;
        std::vector<std::string> splitLine = split(l, ' ');

        if (splitLine.size() == 0) continue;

        std::string& type = splitLine[0];
        //            std::cout << "type " << type << std::endl;


        if (type == "format")
        {
            SAIGA_ASSERT(splitLine[1] == "binary_little_endian");
        }

        if (type == "element")
        {
            std::string& ident = splitLine[1];

            if (ident == "vertex")
            {
                vertexCount   = to_int(splitLine[2]);
                elementStatus = 1;
            }
            if (ident == "face")
            {
                faceCount     = to_int(splitLine[2]);
                elementStatus = 2;
            }
        }

        if (type == "property")
        {
            SAIGA_ASSERT(elementStatus != -1);

            if (elementStatus == 1)
            {
                // vertex property
                VertexProperty vp;
                vp.name = splitLine[2];
                vp.type = splitLine[1];
                vertexProperties.push_back(vp);
            }
            if (elementStatus == 2)
            {
                SAIGA_ASSERT(splitLine[1] == "list");
                SAIGA_ASSERT(splitLine[4] == "vertex_indices");

                faceVertexCountType = splitLine[2];
                faceVertexIndexType = splitLine[3];
            }
        }
    }


    //    std::cout << std::endl;
    //    std::cout << "V " << vertexCount << " F " << faceCount << std::endl;
    //    std::cout << "Face: " << faceVertexCountType << " - " << faceVertexIndexType << std::endl;

    offsetType.resize(6, std::pair<int, int>(0, -1));

    vertexSize = 0;
    for (auto vp : vertexProperties)
    {
        int t = sizeoftype(vp.type);


        if (vp.name == "x") offsetType[0] = std::make_pair(vertexSize, t);
        if (vp.name == "y") offsetType[1] = std::make_pair(vertexSize, t);
        if (vp.name == "z") offsetType[2] = std::make_pair(vertexSize, t);
        if (vp.name == "red") offsetType[3] = std::make_pair(vertexSize, t);
        if (vp.name == "green") offsetType[4] = std::make_pair(vertexSize, t);
        if (vp.name == "blue") offsetType[5] = std::make_pair(vertexSize, t);

        //        std::cout << "Prop: " << vp.type << " " << vp.name << std::endl;
        vertexSize += sizeoftype(vp.type);
    }

    //    std::cout << "vertex size: " << vertexSize << std::endl;


    SAIGA_ASSERT(vertexCount > 0 && faceCount > 0);
    //        std::cout << header << std::endl;
}

void PLYLoader::parseMeshBinary()
{
    for (int i = 0; i < vertexCount; ++i)
    {
        char* start = data.data() + dataStart + i * vertexSize;



        float* x = reinterpret_cast<float*>(start + offsetType[0].first);
        vec3 pos(x[0], x[1], x[2]);


        vec3 color(1);
        if (offsetType[3].second == 1)
        {
            unsigned char* c = reinterpret_cast<unsigned char*>(start + offsetType[3].first);
            color            = vec3(c[0], c[1], c[2]);
            color /= 255.0f;
        }
        else if (offsetType[3].second == 4)
        {
            float* c = reinterpret_cast<float*>(start + offsetType[3].first);
            color    = vec3(c[0], c[1], c[2]);
        }


        VertexNC v;
        v.position = make_vec4(pos, 1);
        v.color    = make_vec4(color, 1);
        mesh.addVertex(v);
    }

    int faceStart = dataStart + vertexCount * vertexSize;
    char* start   = data.data() + faceStart;

    int countSize = sizeoftype(faceVertexCountType);

    for (int i = 0; i < faceCount; ++i)
    {
        int c = 0;
        if (countSize == 1)
        {
            c = *reinterpret_cast<unsigned char*>(start);
        }
        else if (countSize == 4)
        {
            c = *reinterpret_cast<int*>(start);
        }
        start += countSize;

        SAIGA_ASSERT(c == 3);

        uint32_t* face = reinterpret_cast<uint32_t*>(start);
        mesh.addFace(face);

        start += sizeof(uint32_t) * c;
        //            std::cout << c << std::endl;
    }

    mesh.computePerVertexNormal();

    std::cout << "Loaded Ply mesh: V " << mesh.vertices.size() << " F " << mesh.faces.size() << std::endl;
}

}  // namespace Saiga
