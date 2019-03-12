/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ModelLoader.h"

#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include "objModelLoader.h"
#include "plyModelLoader.h"
namespace Saiga
{
bool GenericModel::load(const std::string& _file)
{
    auto file = SearchPathes::model(_file);

    if (file.empty())
    {
        cout << "Can not open file " << _file << endl;
        return false;
    }


    std::string type = fileEnding(file);


    if (type == "obj")
    {
        cout << "load obj" << endl;
    }
    else if (type == "ply")
    {
        cout << "load ply" << endl;
    }
    else
    {
        throw std::runtime_error("Unknown model file format " + to_string(type));
    }
    return true;
}

}  // namespace Saiga
