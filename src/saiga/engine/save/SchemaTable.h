/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/TypeSchema.h"
#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/export.h"

#include <string>
#include <utility>
#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
class World;

/**
 * The schema table of a save (docs/engine_design.md §15.2, S11): every saved type by name with its version and field
 * list (components with Saved::Yes incl. Transform, events, the nested structs reachable from them) and the classes as
 * (ID, name). A load maps the saved types to the registered ones by name.
 */
struct SAIGA_ENGINE_API SchemaTable
{
    std::vector<SchemaType> types;
    std::vector<std::pair<uint32_t, std::string>> classes;
    // Rebuilds the lookup index (after types were changed directly)
    void buildIndex();

    static SchemaTable fromWorld(const World& world);

    const SchemaType* find(SchemaKind kind, const std::string& name) const;
    // The struct type of a nested field (by name), nullptr if none
    const SchemaType* findStruct(const std::string& name) const { return find(SchemaKind::Struct, name); }
    const std::string* className(uint32_t id) const;

    void write(BinaryWriter& w) const;
    bool read(BinaryReader& r, std::string& error);

   private:
    std::unordered_map<std::string, size_t> index;  // "kind:name" -> types index
};

// How a saved type relates to the registered one of the same name
enum class SchemaMatch
{
    Same,          // the same version and identical field lists (nested structs too): read the binary form directly
    Compatible,    // the same version, fields added, removed or enum values changed: by name
    Migrate,       // an older version: migrations, then by name
    Incompatible,  // a field changed its type without a version increment, or a newer version than registered
};
SAIGA_ENGINE_API SchemaMatch compareTypes(const SchemaTable& saved, const SchemaType& savedType,
                                          const SchemaTable& current, const SchemaType& currentType,
                                          std::string& reason);

// The struct names a field type string refers to ("vector<Item>" -> Item; none for scalars and math types)
SAIGA_ENGINE_API std::string structNameOf(const std::string& fieldType);

}  // namespace Engine
}  // namespace Saiga
