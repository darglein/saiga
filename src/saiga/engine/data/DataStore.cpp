/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// DataStore and the data file readers (docs/engine_design.md §14, S11).

#include "saiga/engine/data/DataStore.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace Saiga
{
namespace Engine
{
bool readJsonFile(const std::string& path, json11::Json& out, std::string& error)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        error = path + ": cannot be read";
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string parseError;
    json11::Json j = json11::Json::parse(ss.str(), parseError);
    if (!parseError.empty())
    {
        error = path + ": " + parseError;
        return false;
    }
    out = std::move(j);
    return true;
}

std::string snakeCase(const std::string& name)
{
    std::string out;
    for (size_t i = 0; i < name.size(); ++i)
    {
        const unsigned char c = (unsigned char)name[i];
        if (std::isupper(c))
        {
            if (i > 0 && name[i - 1] != '_') out += '_';
            out += char(std::tolower(c));
        }
        else
            out += char(c);
    }
    return out;
}

DataStore::DataStore(DataTypeList t, std::string directory) : types(std::move(t)), dir(std::move(directory)) {}

bool DataStore::readAll(std::string& error)
{
    for (auto& t : types.tables)
    {
        if (dir.empty())
        {
            error = std::string("the data type ") + t->name + " is registered, but WorldSetup::dataDirectory is empty";
            return false;
        }
        if (!t->read(dir + "/" + t->file, error)) return false;
    }
    return true;
}

bool DataStore::load(std::string& error)
{
    if (!readAll(error)) return false;
    for (auto& t : types.tables) t->applyPending();
    return true;
}

bool DataStore::reload(std::string& error)
{
    if (!readAll(error)) return false;
    for (auto& t : types.tables)
        if (!t->checkPending(error)) return false;
    for (auto& t : types.tables) t->applyPending();
    return true;
}

const DataTableBase* DataStore::find(const std::string& name) const
{
    for (auto& t : types.tables)
        if (name == t->name) return t.get();
    return nullptr;
}

}  // namespace Engine
}  // namespace Saiga
