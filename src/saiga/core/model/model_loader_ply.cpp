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
#include <cstring>
#include <stdexcept>

namespace Saiga
{
PLYLoader::PLYLoader(const std::string& _file)
{
    auto file = SearchPathes::model(_file);
    std::ifstream stream(file, std::ios::binary);

    if (!stream.is_open())
    {
        std::cerr << "Could not open file " << file << std::endl;
        throw std::runtime_error("invalid file: " + file.string() + ", " + _file);
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
    throw std::runtime_error("PLYLoader: unsupported property type " + t);
}

void PLYLoader::parseHeader()
{
    // Note: the header is always in ascii and ends with the string end_header
    // The file content is untrusted, so malformed headers throw like the checks in parseMeshBinary.
    auto check = [](bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(std::string("PLYLoader: ") + message);
    };

    std::string str2(data.begin(), data.end());
    auto pos = str2.find("end_header");

    check(pos != std::string::npos, "no end_header");

    dataStart = pos;
    // go until next newline
    while (size_t(dataStart) < data.size() && data[dataStart] != '\n')
    {
        dataStart++;
    }
    check(size_t(dataStart) < data.size(), "unexpected end of header");
    dataStart++;

    std::string header(str2.begin(), str2.begin() + pos);
    header.erase(std::remove(header.begin(), header.end(), '\r'), header.end());

    std::vector<std::string> headerLines = split(header, '\n');

    check(!headerLines.empty() && headerLines[0] == "ply", "missing ply magic");

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
            check(splitLine.size() > 1 && splitLine[1] == "binary_little_endian",
                  "only binary_little_endian is supported");
        }

        if (type == "element")
        {
            check(splitLine.size() > 2, "invalid element line");
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
            check(elementStatus != -1, "property before element");
            check(splitLine.size() > 2, "invalid property line");

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
                check(splitLine.size() > 4 && splitLine[1] == "list" && splitLine[4] == "vertex_indices",
                      "invalid face property");

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


    check(vertexCount > 0 && faceCount > 0, "no vertices or faces");
    //        std::cout << header << std::endl;
}

void PLYLoader::parseMeshBinary()
{
    // The file content is untrusted: check every read against the end of the data and every face index.
    auto read = [this](size_t offset, auto& value)
    {
        if (offset > data.size() || data.size() - offset < sizeof(value))
        {
            throw std::runtime_error("PLYLoader: unexpected end of file");
        }
        std::memcpy(&value, data.data() + offset, sizeof(value));
    };

    for (int c = 0; c < 3; ++c)
    {
        if (offsetType[c].second != 4) throw std::runtime_error("PLYLoader: the vertex positions must be float");
    }

    for (int i = 0; i < vertexCount; ++i)
    {
        size_t start = size_t(dataStart) + size_t(i) * vertexSize;

        vec3 pos;
        for (int c = 0; c < 3; ++c)
        {
            read(start + offsetType[c].first, pos(c));
        }

        vec3 color(1, 1, 1);
        for (int c = 0; c < 3; ++c)
        {
            if (offsetType[3 + c].second == 1)
            {
                unsigned char uc;
                read(start + offsetType[3 + c].first, uc);
                color(c) = uc / 255.0f;
            }
            else if (offsetType[3 + c].second == 4)
            {
                read(start + offsetType[3 + c].first, color(c));
            }
        }

        VertexNC v;
        v.position = make_vec4(pos, 1);
        v.color    = make_vec4(color, 1);
        mesh.addVertex(v);
    }

    size_t start = size_t(dataStart) + size_t(vertexCount) * vertexSize;

    int countSize = sizeoftype(faceVertexCountType);
    if (sizeoftype(faceVertexIndexType) != 4) throw std::runtime_error("PLYLoader: the face indices must be int");

    for (int i = 0; i < faceCount; ++i)
    {
        int c = 0;
        if (countSize == 1)
        {
            unsigned char uc;
            read(start, uc);
            c = uc;
        }
        else if (countSize == 4)
        {
            read(start, c);
        }
        start += countSize;

        if (c != 3) throw std::runtime_error("PLYLoader: only triangles are supported");

        uint32_t face[3];
        for (int j = 0; j < 3; ++j)
        {
            read(start + j * sizeof(uint32_t), face[j]);
            if (face[j] >= uint32_t(vertexCount)) throw std::runtime_error("PLYLoader: invalid vertex index");
        }
        mesh.addFace(face);

        start += sizeof(uint32_t) * c;
    }

    mesh.computePerVertexNormal();

    std::cout << "Loaded Ply mesh: V " << mesh.vertices.size() << " F " << mesh.faces.size() << std::endl;
}

}  // namespace Saiga
