/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/export.h"

#include <map>
#include <stdexcept>
#include <string>

// Reading data files (docs/engine_design.md §14, R-DATA-1; S11): JSON, missing keys keep the struct defaults, unknown
// keys, wrong types and out-of-range values are errors naming the file and the key path. These readers report errors;
// the World makes them fatal at construction.

namespace Saiga
{
namespace Engine
{
// A data entry that does not exist (World::data with an unknown name): fatal when it escapes to the top, an error
// result when it happens during a load or an API call
class DataError : public std::runtime_error
{
   public:
    using std::runtime_error::runtime_error;
};

// The file as JSON; false and an error naming the file if it cannot be read or parsed
SAIGA_ENGINE_API bool readJsonFile(const std::string& path, json11::Json& out, std::string& error);

// "CreatureStats" -> "creature_stats" (the file name of a data type)
SAIGA_ENGINE_API std::string snakeCase(const std::string& name);

// An object of named entries, each merged into a default T. On an error out is unchanged.
template <class T>
bool readDataEntries(const json11::Json& json, const std::string& path, std::map<std::string, T>& out,
                     std::string& error)
{
    if (!json.is_object())
    {
        error = path + ": expected an object of named entries";
        return false;
    }
    std::map<std::string, T> entries;
    for (const auto& kv : json.object_items())
    {
        T value{};
        std::string e;
        if (!fromJson(kv.second, value, &e, true))
        {
            // fromJson names the key path ("speedd: unknown field ..."; "document: ..." for the entry itself)
            if (e.compare(0, 10, "document: ") == 0)
                error = path + ": " + kv.first + ": " + e.substr(10);
            else
                error = path + ": " + kv.first + "." + e;
            return false;
        }
        entries.emplace(kv.first, std::move(value));
    }
    out = std::move(entries);
    return true;
}

template <class T>
bool readDataFile(const std::string& path, std::map<std::string, T>& out, std::string& error)
{
    json11::Json json;
    return readJsonFile(path, json, error) && readDataEntries(json, path, out, error);
}

}  // namespace Engine
}  // namespace Saiga
