/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// MigrationList: the registered migrations and removed types of a world (S11).

#include "saiga/engine/save/MigrationList.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Engine
{
json11::Json FieldMap::get(const std::string& name) const
{
    auto it = fields.find(name);
    return it == fields.end() ? json11::Json() : it->second;
}

void FieldMap::rename(const std::string& from, const std::string& to)
{
    auto it = fields.find(from);
    if (it == fields.end()) return;
    json11::Json v = it->second;
    fields.erase(it);
    fields[to] = std::move(v);
}

void MigrationList::add(const char* typeName, uint32_t fromVersion, Fn fn)
{
    SAIGA_ASSERT(typeName && fn, "MigrationList::add: a type name and a function");
    // In every build: two migrations of one version would make the result depend on the registration order
    if (find(typeName, fromVersion))
        SAIGA_EXIT_ERROR(std::string("MigrationList::add: ") + typeName + " already has a migration from version " +
                         std::to_string(fromVersion));
    migrations.push_back(Migration{typeName, fromVersion, fn});
}

void MigrationList::addRemoved(const char* name, RemovedPolicy policy, const char* replacement)
{
    SAIGA_ASSERT(name, "MigrationList::addRemoved: a name");
    if (findRemoved(name))
        SAIGA_EXIT_ERROR(std::string("MigrationList::addRemoved: ") + name + " is already registered as removed");
    if ((policy == RemovedPolicy::Replace) != (replacement != nullptr))
        SAIGA_EXIT_ERROR(std::string("MigrationList::addRemoved: ") + name +
                         ": a replacement is given exactly for RemovedPolicy::Replace");
    removed.push_back(Removed{name, policy, replacement ? replacement : ""});
}

MigrationList::Fn MigrationList::find(const std::string& typeName, uint32_t fromVersion) const
{
    for (const Migration& m : migrations)
        if (m.fromVersion == fromVersion && m.typeName == typeName) return m.fn;
    return nullptr;
}

const MigrationList::Removed* MigrationList::findRemoved(const std::string& name) const
{
    for (const Removed& r : removed)
        if (r.name == name) return &r;
    return nullptr;
}

}  // namespace Engine
}  // namespace Saiga
