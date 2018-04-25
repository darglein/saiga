/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/tostring.h"
#include "saiga/geometry/triangle_mesh.h"
#include <fstream>

namespace Saiga {


class SAIGA_GLOBAL PLYLoader
{
public:

    struct VertexProperty
    {
        std::string name;
        std::string type;
    };

    int vertexSize;
    int dataStart;
    std::vector<char> data;
    std::vector<VertexProperty> vertexProperties;

    std::string faceVertexCountType;
    std::string faceVertexIndexType;

    TriangleMesh<VertexNC,GLuint> mesh;

    int vertexCount = -1, faceCount = -1;

    PLYLoader(const std::string& file);


    int sizeoftype(std::string t);

    void parseHeader();

    void parseMeshBinary();
};


}
