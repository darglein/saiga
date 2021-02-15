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

        for (auto& v : loader.outVertices)
        {
            position.push_back(v.position.head<3>());
            normal.push_back(v.normal.head<3>());
            texture_coordinates.push_back(v.texture);
        }

        for (auto& c : loader.vertexColors)
        {
            color.push_back(c);
        }

        for (auto& f : loader.outTriangles)
        {
            triangles.push_back(f);
        }
    }
    else
    {
        throw std::runtime_error("Unknown model file format " + to_string(type));
    }


    std::cout << "type " << type << std::endl;
}

}  // namespace Saiga
