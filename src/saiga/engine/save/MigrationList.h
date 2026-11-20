/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/export.h"

#include <cstdint>
#include <string>
#include <vector>

// Migrations of saved types (docs/engine_design.md §15.2, R-SAVE-1; S11): a function per (type, version) that rewrites
// the fields of a value saved with that version into the next version, and entries for removed types (drop the
// entity, drop the component, or load it as another type). Registered in WorldSetup::migrations.

namespace Saiga
{
namespace Engine
{
// The fields of one saved value in the JSON form of VisitJson.h (exact: 64-bit integers beyond 2^53 and non-finite
// doubles as strings, enums by name, nested structs as objects), changed in place by a migration
struct SAIGA_ENGINE_API FieldMap
{
    json11::Json::object fields;

    bool has(const std::string& name) const { return fields.count(name) != 0; }
    // The value, or a null Json if the field is missing
    json11::Json get(const std::string& name) const;
    void set(const std::string& name, json11::Json value) { fields[name] = std::move(value); }
    // Renames a field (no-op if it is missing)
    void rename(const std::string& from, const std::string& to);
    void remove(const std::string& name) { fields.erase(name); }
};

enum class RemovedPolicy : uint8_t
{
    DropEntity,     // entities with the type are dropped with their descendants (handles to them become stale)
    DropComponent,  // the component is dropped (a removed class: the entity stays without a game object)
    Replace,        // loaded as the replacement type (fields by name, after the removed type's own migrations)
};

class SAIGA_ENGINE_API MigrationList
{
   public:
    using Fn = void (*)(FieldMap& fields);

    // T's fields saved with version fromVersion are rewritten into version fromVersion + 1
    template <class T>
    void add(uint32_t fromVersion, Fn fn)
    {
        add(T::name, fromVersion, fn);
    }
    // By name: for types that are no longer registered (a Replace runs the removed type's migrations first)
    void add(const char* typeName, uint32_t fromVersion, Fn fn);
    // A component, class, event or nested struct that no longer exists. replacement: the type (or class) to load it
    // as (Replace only)
    void addRemoved(const char* name, RemovedPolicy policy, const char* replacement = nullptr);

    // Engine-internal
    struct Removed
    {
        std::string name;
        RemovedPolicy policy;
        std::string replacement;
    };
    Fn find(const std::string& typeName, uint32_t fromVersion) const;
    const Removed* findRemoved(const std::string& name) const;

   private:
    struct Migration
    {
        std::string typeName;
        uint32_t fromVersion;
        Fn fn;
    };
    std::vector<Migration> migrations;
    std::vector<Removed> removed;
};

}  // namespace Engine
}  // namespace Saiga
