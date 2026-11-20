/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/engine/save/MigrationList.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/export.h"

#include <string>

namespace Saiga
{
namespace Engine
{
/**
 * Decodes the binary form of a saved value by its saved field list into the JSON form of VisitJson.h (S11): the slow
 * path of loads (types that changed since the save) and the comparison of saves. With a current schema and the
 * migrations, a struct is migrated to the registered version (nested structs first) and its fields that no longer
 * exist are dropped, so VisitJson's fromJson can merge the result into a default value.
 */
class SAIGA_ENGINE_API SaveDecoder
{
   public:
    // current and migrations may be null: the values are decoded as saved (comparisons)
    SaveDecoder(const SchemaTable& saved, const SchemaTable* current, const MigrationList* migrations)
        : saved(saved), current(current), migrations(migrations)
    {
    }

    // A value of the saved struct type savedType. target: the registered type to migrate to (nullptr: as saved; a
    // removed type is migrated as far as its own migrations go)
    bool decode(const SchemaType& savedType, BinaryReader& r, const SchemaType* target, json11::Json& out,
                std::string& error);

    // Moves past a value of the saved struct type without building JSON (finds value boundaries fast)
    bool skip(const SchemaType& savedType, BinaryReader& r, std::string& error);

    // Migrates fields saved with fromVersion to target's version and drops the fields target does not have
    bool migrate(const std::string& typeName, uint32_t fromVersion, const SchemaType* target, FieldMap& fields,
                 std::string& error) const;

   private:
    static bool runMigration(MigrationList::Fn fn, const std::string& typeName, uint32_t version, FieldMap& fields,
                             std::string& error);
    bool skipValue(const std::string& type, const FieldInfo& field, BinaryReader& r, std::string& error);
    bool decodeValue(const std::string& type, const FieldInfo& field, BinaryReader& r, json11::Json& out,
                     std::string& error);

    const SchemaTable& saved;
    const SchemaTable* current;
    const MigrationList* migrations;
    int depth = 0;
    // Elements that took no bytes (empty structs, zero-length arrays): bounded for the whole decoder, so nested
    // containers of them cannot multiply into an unbounded amount of work (untrusted input)
    uint64_t zeroByteBudget = uint64_t(1) << 20;
    bool spendZeroByte(size_t before, const BinaryReader& r, std::string& error);
};

}  // namespace Engine
}  // namespace Saiga
